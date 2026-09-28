"""Build the academic results document only from measured experiment artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.registry import DISPLAY_NAMES, MODEL_NAMES
from src.compare import compare_experiments
from src.utils import get_class_names, load_config, manifest_fingerprint, resolve_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_artifacts(config: dict, audit: dict, training: Path, evaluation: Path) -> None:
    """Reject stale or mismatched cohorts/checkpoints before writing the report."""
    fingerprint = manifest_fingerprint(resolve_path(config["manifest_path"], config))
    if audit.get("status") != "passed" or audit.get("dataset_fingerprint") != fingerprint:
        raise ValueError("The dataset audit does not match the current manifest; regenerate the audit before reporting.")
    for architecture in MODEL_NAMES:
        metrics_path = evaluation / architecture / "metrics.json"
        if not metrics_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("dataset_fingerprint") != fingerprint:
            raise ValueError(f"{architecture} evaluation belongs to a different manifest than the current audit.")
        for field, expected in (("task", config["task"]), ("class_names", get_class_names(config)),
                                ("threshold", config["prediction_threshold"])):
            if metrics.get(field) != expected:
                raise ValueError(f"{architecture} evaluation {field} differs from the reporting configuration.")
        if metrics.get("n_samples") != sum(audit["split_class_counts"]["test"].values()):
            raise ValueError(f"{architecture} evaluation sample count differs from the audited test count.")
        run_path = training / architecture / "run.json"
        if not run_path.is_file():
            raise ValueError(f"{architecture} has evaluation metrics but no completed training record; restore the matching run.json.")
        run = json.loads(run_path.read_text(encoding="utf-8"))
        for field in ("dataset_fingerprint", "model_name", "task", "class_names"):
            if run.get(field) != metrics.get(field):
                raise ValueError(f"{architecture} training and evaluation disagree on {field}.")
        model_path = resolve_path(metrics["model_path"], config)
        if not model_path.is_file() or _sha256(model_path) != metrics.get("model_sha256"):
            raise ValueError(f"{architecture} evaluated checkpoint is missing or changed; re-evaluate the matching checkpoint.")
        metadata = json.loads((model_path.parent / "metadata.json").read_text(encoding="utf-8"))
        for field in ("dataset_fingerprint", "model_name", "task", "class_names", "selected_phase", "selected_epoch", "selected_val_loss", "total_epochs"):
            if field not in run or run[field] != metadata.get(field):
                raise ValueError(f"{architecture} training record and saved checkpoint disagree on {field}.")


def build_report(config: dict) -> Path:
    """Validate comparison provenance, then render factual results and curve summaries."""
    evaluation = resolve_path(config["evaluation_dir"], config)
    training = resolve_path(config["training_dir"], config)
    processed = resolve_path(config["processed_dir"], config)
    audit = json.loads((processed / "audit.json").read_text(encoding="utf-8"))
    _validate_artifacts(config, audit, training, evaluation)
    compare_experiments(config)
    destination = resolve_path("docs/results.md", config)

    def link(path: Path) -> str:
        return quote(Path(os.path.relpath(path, destination.parent)).as_posix(), safe="/.-_")

    patient_note = ("Supplied patient identifiers were used for group separation; completeness and correctness of those identifiers remain source limitations."
                    if audit.get("patient_mapping_available") else "Patient independence cannot be established because no patient mapping is available.")
    split_note = ("The eligible official Testing assignment was preserved." if audit.get("layout") == "official"
                  else "Held-out assignments follow the audited grouping and splitting policy for the supplied dataset layout.")
    lines = ["# Measured results", "", "## Dataset and evaluation protocol", "",
             f"The audit inspected {audit['total_inspected']:,} images and retained {audit['included']:,}. "
             f"It excluded {audit['excluded']:,} images for the reasons recorded below.", ""]
    lines += [f"- {reason}: {count:,}." for reason, count in audit["exclusion_counts"].items()]
    observed = set().union(*(set(counts) for counts in audit["split_class_counts"].values()))
    categories = [name for name in config["source_classes"] if name in observed]
    categories += sorted(observed - set(categories))
    headings = [name.replace("notumor", "no tumor").replace("_", " ").title() for name in categories]
    lines += ["", "| Split | " + " | ".join(headings) + " | Total |",
              "|---|" + "---:|" * (len(categories) + 1)]
    for split in ("train", "validation", "test"):
        counts = audit["split_class_counts"][split]
        values = [counts.get(name, 0) for name in categories]
        lines.append(f"| {split.title()} | " + " | ".join(str(value) for value in [*values, sum(counts.values())]) + " |")
    threshold_note = (f"The binary decision threshold is {config['prediction_threshold']}; it must be selected independently of test performance. "
                      if config["task"] == "binary" else "Multiclass predictions use the maximum class probability. ")
    lines += ["", split_note + " Checkpoint selection uses validation loss. " + threshold_note +
              "EfficientNet is the designated application model. Perceptual filtering is conservative. " + patient_note, "",
              f"Manifest SHA-256: `{audit['dataset_fingerprint']}`.", "", "## Held-out comparison", "",
              (evaluation / "comparison.md").read_text(encoding="utf-8").strip(), "",
              "Values are fractions on a 0–1 scale. These are measured image-classification results on this dataset, "
              "not clinical validation or calibrated diagnostic probabilities.", ""]
    for architecture in MODEL_NAMES:
        title = DISPLAY_NAMES[architecture]
        run_path = training / architecture / "run.json"
        metrics_path = evaluation / architecture / "metrics.json"
        lines += [f"## {title}", ""]
        if not run_path.is_file() or not metrics_path.is_file():
            lines += ["Training required — no measured result available.", ""]
            continue
        run = json.loads(run_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        history = json.loads((training / architecture / "history.json").read_text(encoding="utf-8"))
        run_config_path = training / architecture / "config.json"
        run_config = json.loads(run_config_path.read_text(encoding="utf-8")) if run_config_path.is_file() else {}
        lines += [f"Selected checkpoint: **{run['selected_phase']}, epoch {run['selected_epoch']}**; "
                  f"validation loss **{run['selected_val_loss']:.5f}**. "
                  f"The run completed {run['total_epochs']} epochs across its phases in "
                  f"{run['training_seconds'] / 60:.1f} minutes (recorded training wall time).", ""]
        for phase in run["phases"]:
            lines.append(f"- {phase['phase']}: {phase['epochs_run']} epochs; stopped by {phase['stopping_reason']}; "
                         f"best validation loss {phase['best_val_loss']:.5f} at epoch {phase['best_epoch']}.")
        lines += ["", "### Training/validation behavior", "",
                  f"Across the recorded run, training accuracy changed from {history['accuracy'][0]:.4f} to "
                  f"{history['accuracy'][-1]:.4f}, and validation accuracy from {history['val_accuracy'][0]:.4f} "
                  f"to {history['val_accuracy'][-1]:.4f}. Training loss changed from {history['loss'][0]:.4f} "
                  f"to {history['loss'][-1]:.4f}; validation loss from {history['val_loss'][0]:.4f} "
                  f"to {history['val_loss'][-1]:.4f}.", ""]
        lines += [f"Validation loss ranged from {min(history['val_loss']):.4f} to {max(history['val_loss']):.4f}. " +
                  ("The final validation loss exceeded the selected checkpoint's loss, so retaining the earlier best checkpoint avoids using the later degraded state. "
                   if history["val_loss"][-1] > run["selected_val_loss"] else "The final validation loss matched the selected checkpoint's loss. ") +
                  "Variation or spikes in validation loss should be inspected directly; a smooth overfitting trend is not assumed.", ""]
        regressions = sum(history["loss"][i] < history["loss"][i - 1] and
                          history["val_loss"][i] > history["val_loss"][i - 1]
                          for i in range(1, len(history["loss"])) if i not in history.get("phase_boundaries", []))
        lines += [f"There were {regressions} within-phase epoch transitions with falling training loss and rising validation loss. "
                  "This is a signal to inspect generalization, not a standalone diagnosis of overfitting. "
                  "Training uses augmentation and dropout; validation uses deterministic inputs and inference mode. "
                  + ("Training also uses class-weighted loss while validation is unweighted, so the absolute losses are not identical objectives. "
                     if run_config.get("class_weights", bool(run.get("class_weights"))) else "") +
                  "Early stopping, dropout, "
                  "augmentation, and validation-triggered learning-rate reduction were applied. The last epoch is not "
                  "automatically the deployed checkpoint.", "",
                  f"![{title} accuracy]({link(training / architecture / 'accuracy.png')})", "",
                  f"![{title} loss]({link(training / architecture / 'loss.png')})", ""]
        if metrics["task"] == "binary":
            tumor = metrics["per_class"]["tumor"]
            lines += [f"Held-out outcomes: TN={tumor['tn']}, FP={tumor['fp']}, FN={tumor['fn']}, TP={tumor['tp']}. "
                      "False negatives are labeled tumor images classified as no tumor; false positives are labeled "
                      "no-tumor images classified as tumor.", ""]
        lines += [f"![{title} confusion matrix]({link(evaluation / architecture / 'confusion_matrix.png')})", "",
                  f"Detailed metrics, probabilities, and model SHA-256 are in [the {title} evaluation folder]({link(evaluation / architecture)}/).", ""]
    lines += ["## Interpretation limits", "", "These experiments use one public collection and one seeded split. " +
              patient_note + " There is no independent external validation, calibration study, or clinical approval. "
              "Filtering substantially changes class proportions and may exclude visually similar but distinct images. "
              "Inspect specificity and sensitivity together; high accuracy can obscure class imbalance. "
              "Grad-CAM is an interpretability aid and does not validate anatomical localization.", "",
              "CPU experiment wall times may overlap and are not controlled performance benchmarks. "
              "See [the verification record](verification.md) for the actual execution environment and software checks.", ""]
    if (evaluation / "mobilenet" / "metrics.json").is_file():
        lines += ["The three-model study is a sequential extension: the original CNN/EfficientNet test results "
                  "and sample mistakes had already been inspected before MobileNetV2 was added. MobileNetV2 "
                  "used the recorded fixed training defaults, seed 42, threshold 0.5, and validation-only "
                  "checkpoint selection. Reusing this previously inspected holdout makes the comparison "
                  "exploratory; it is not fresh external confirmation or evidence of statistical significance. "
                  "See [the recorded protocol](../paper/experiment_protocol.md).", ""]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    report = build_report(load_config(parser.parse_args().config))
    # ASCII output also works in legacy Windows terminals when the workspace
    # directory contains characters outside the console code page.
    print("Saved the measured results report to docs/results.md")
