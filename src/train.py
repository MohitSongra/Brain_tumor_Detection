"""Train an audited MRI classifier and evaluate its validation-selected checkpoint."""
from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from models.registry import MODEL_NAMES, get_model_module, is_transfer_model
from src.data_loader import compute_class_weights, create_dataset, load_manifest, verify_manifest_integrity
from src.utils import (configure_runtime, get_class_names, load_config, manifest_fingerprint,
                       resolve_path, select_best_phase, setup_logging, write_json)


class EpochJournal(tf.keras.callbacks.Callback):
    """Persist progress every epoch so a long CPU run remains inspectable."""
    def __init__(self, path: Path, phase: str):
        super().__init__()
        self.path, self.phase = path, phase
        self.started = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self.started = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        record = {"phase": self.phase, "epoch": epoch + 1,
                  "seconds": time.perf_counter() - self.started,
                  **{key: float(value) for key, value in (logs or {}).items()}}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        logging.info("%s epoch %d: loss=%.4f val_loss=%.4f accuracy=%.4f val_accuracy=%.4f (%.1fs)",
                     self.phase, epoch + 1, record.get("loss", np.nan), record.get("val_loss", np.nan),
                     record.get("accuracy", np.nan), record.get("val_accuracy", np.nan), record["seconds"])


def plot_history(history: dict, output_dir: Path) -> None:
    """Plot all measured phases with explicit fine-tuning boundaries."""
    for metric in ("accuracy", "loss"):
        fig, ax = plt.subplots(figsize=(8, 5))
        values = history.get(metric, [])
        ax.plot(range(1, len(values) + 1), values, label="Training")
        validation = history.get(f"val_{metric}", [])
        ax.plot(range(1, len(validation) + 1), validation, label="Validation")
        for boundary in history.get("phase_boundaries", []):
            ax.axvline(boundary + 0.5, color="gray", linestyle="--", label="Fine-tuning begins")
        ax.set(xlabel="Epoch", ylabel=metric.title(), title=f"Training and validation {metric}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{metric}.png", dpi=160)
        plt.close(fig)


def _fit_phase(model, training, validation, config, output_dir, phase, epochs, weights):
    checkpoint = output_dir / f"{phase}.keras"
    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=config["early_stopping_patience"],
                                         restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=config["reduce_lr_factor"],
                                            patience=config["reduce_lr_patience"],
                                            min_lr=config["min_learning_rate"]),
        EpochJournal(output_dir / "epochs.jsonl", phase),
    ]
    started = time.perf_counter()
    history = model.fit(training, validation_data=validation, epochs=epochs,
                        callbacks=callbacks, class_weight=weights, verbose=0)
    measured = {key: [float(v) for v in values] for key, values in history.history.items()}
    validation_losses = measured.get("val_loss", [])
    if not validation_losses or not checkpoint.is_file() or not all(np.isfinite(validation_losses)):
        raise RuntimeError(f"{phase} failed to produce finite validation losses and a checkpoint.")
    return {"phase": phase, "checkpoint": str(checkpoint), "history": measured,
            "best_val_loss": min(validation_losses),
            "best_epoch": int(np.argmin(validation_losses)) + 1,
            "epochs_run": len(validation_losses), "seconds": time.perf_counter() - started,
            "stopping_reason": "early_stopping" if len(validation_losses) < epochs else "epoch_limit"}


def train(config: dict, *, overwrite: bool = False, smoke: bool = False, evaluate: bool = True) -> dict:
    """Train one model; test data is read only after checkpoint selection."""
    runtime = configure_runtime(config)
    architecture = config["model_name"]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    output_dir = resolve_path(config["training_dir"], config) / architecture
    model_dir = resolve_path(config["saved_models_dir"], config) / architecture
    if smoke:
        output_dir = resolve_path("results/smoke", config) / f"{architecture}_{timestamp}"
        model_dir = output_dir / "model"
    existing = model_dir / "best_model.keras"
    if existing.exists() and not overwrite:
        raise FileExistsError(f"A trained model already exists at {existing}. Use --overwrite to intentionally retrain.")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "training.log")
    journal = model_dir / "epochs.jsonl"
    if journal.exists():
        journal.unlink()
    manifest = load_manifest(config)
    if not smoke:
        integrity = verify_manifest_integrity(config)
        write_json(output_dir / "integrity.json", integrity)
    fingerprint = manifest_fingerprint(resolve_path(config["manifest_path"], config))
    write_json(output_dir / "config.json", {key: value for key, value in config.items() if not key.startswith("_")})
    write_json(output_dir / "environment.json", runtime)
    training = create_dataset(config, "train", training=True, batch_size=runtime["batch_size"])
    validation = create_dataset(config, "validation", training=False, batch_size=runtime["batch_size"])
    weights = compute_class_weights(config) if config.get("class_weights", True) else None
    if smoke:
        training, validation = training.take(3), validation.take(2)
    module = get_model_module(architecture)
    transfer = is_transfer_model(architecture)
    pretrained_weights = config.get("weights", "imagenet")
    if transfer and not smoke:
        if pretrained_weights != "imagenet":
            raise ValueError("Real transfer-learning runs require weights: imagenet. For offline use, populate the standard Keras weights cache; arbitrary weight paths can omit ImageNet-specific preprocessing.")
    model = module.build_model(config, weights=None if smoke else pretrained_weights) if transfer else module.build_model(config)
    module.compile_model(model, config)
    with (output_dir / "model_summary.txt").open("w", encoding="utf-8") as stream:
        model.summary(print_fn=lambda line: stream.write(line + "\n"))
    # Benchmark on development data, then restore model AND optimizer state.
    initial_weights = model.get_weights()
    timings = []
    for images, labels in training.take(config.get("benchmark_batches", 3)):
        started = time.perf_counter()
        model.train_on_batch(images, labels, class_weight=weights)
        timings.append(time.perf_counter() - started)
    # Rebuild from the same seed after benchmarking, including layer RNGs and
    # dataset shuffle/augmentation state, so timing does not alter training.
    del model
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config["random_seed"])
    training = create_dataset(config, "train", training=True, batch_size=runtime["batch_size"])
    validation = create_dataset(config, "validation", training=False, batch_size=runtime["batch_size"])
    if smoke:
        training, validation = training.take(3), validation.take(2)
    model = module.build_model(config, weights=None if smoke else pretrained_weights) if transfer else module.build_model(config)
    model.set_weights(initial_weights)
    del initial_weights
    module.compile_model(model, config)
    steps = int(np.ceil((manifest.split == "train").sum() / runtime["batch_size"]))
    estimate = float(np.mean(timings[1:] or timings) * steps)
    write_json(output_dir / "benchmark.json", {"batch_seconds": timings, "estimated_training_seconds_per_epoch": estimate,
                                                "excludes_validation_and_finetuning": True})
    logging.info("Measured batch benchmark suggests %.1f minutes per training epoch, excluding validation.", estimate / 60)
    started = time.perf_counter()
    phases = [_fit_phase(model, training, validation, config, model_dir,
                         "head" if transfer else "baseline",
                         1 if smoke else config["epochs"], weights)]
    if transfer and config.get("fine_tune", True):
        model = tf.keras.models.load_model(phases[0]["checkpoint"], compile=False)
        module.unfreeze_for_finetuning(model, config)
        module.compile_model(model, config, learning_rate=config["fine_tune_learning_rate"])
        phases.append(_fit_phase(model, training, validation, config, model_dir,
                                 "finetune", 1 if smoke else config["fine_tune_epochs"], weights))
    selected = select_best_phase(phases)
    shutil.copy2(selected["checkpoint"], existing)
    combined: dict = {"phase_boundaries": []}
    completed = 0
    for phase in phases:
        if completed:
            combined["phase_boundaries"].append(completed)
        for key, values in phase["history"].items():
            combined.setdefault(key, []).extend(values)
        completed += phase["epochs_run"]
    write_json(output_dir / "history.json", combined)
    plot_history(combined, output_dir)
    metadata = {"task": config["task"], "class_names": get_class_names(config),
                "image_size": config["image_size"], "input_range": [0, 255],
                "threshold": config["prediction_threshold"], "model_name": architecture,
                "gradcam_layer": "gradcam_features", "dataset_fingerprint": fingerprint,
                "trained_on_real_data": not smoke, "created_utc": timestamp,
                "selected_phase": selected["phase"], "selected_val_loss": selected["best_val_loss"],
                "selected_epoch": selected["best_epoch"], "total_epochs": completed,
                "training_seconds": time.perf_counter() - started,
                "class_weights": weights, "runtime": runtime}
    write_json(model_dir / "metadata.json", metadata)
    write_json(model_dir / "class_names.json", get_class_names(config))
    summaries = [{key: value for key, value in phase.items() if key not in {"history", "checkpoint"}} for phase in phases]
    write_json(output_dir / "run.json", {**metadata, "phases": summaries})
    if architecture == "efficientnet" and not smoke:
        published_model = resolve_path(config["model_path"], config)
        published_model.parent.mkdir(parents=True, exist_ok=True)
        if existing.resolve() != published_model.resolve():
            shutil.copy2(existing, published_model)
        for name in ("metadata.json", "class_names.json"):
            destination = published_model.parent / name
            if (model_dir / name).resolve() != destination.resolve():
                shutil.copy2(model_dir / name, destination)
        shutil.copy2(output_dir / "history.json", resolve_path(config["training_dir"], config) / "history.json")
    if evaluate and not smoke:
        from src.evaluate import evaluate_model
        evaluate_model(existing, config, resolve_path(config["evaluation_dir"], config) / architecture)
    logging.info("Saved %s, selected by validation loss %.5f", existing, selected["best_val_loss"])
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--model", choices=MODEL_NAMES)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Small isolated run; never publish academic metrics or the application model.")
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()
    setup_logging()
    try:
        config = load_config(args.config)
        if args.model:
            config["model_name"] = args.model
        train(config, overwrite=args.overwrite, smoke=args.smoke, evaluate=not args.skip_evaluation)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, OSError) as error:
        logging.error("Training failed: %s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
