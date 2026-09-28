"""Training orchestration checks with small fixtures, isolated from real results."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from models import custom_cnn
from src import train as training_module
from src.utils import load_config


def test_phase_checkpoint_preserves_best_epoch(monkeypatch, tmp_path):
    """Replay deterministic losses through real callbacks to test best-only saving."""
    config = load_config()
    model = tf.keras.Sequential([tf.keras.Input((1,)), tf.keras.layers.Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    losses = [0.5, 0.2, 0.8]
    def replay_fit(training, validation_data, epochs, callbacks, class_weight, verbose):
        for callback in callbacks:
            callback.set_model(model)
            callback.on_train_begin()
        for epoch, loss in enumerate(losses):
            for callback in callbacks:
                callback.on_epoch_begin(epoch)
            model.set_weights([np.full((1, 1), epoch + 1, dtype=np.float32), np.zeros(1, np.float32)])
            logs = {"loss": loss, "val_loss": loss, "accuracy": 0.5, "val_accuracy": 0.5}
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs)
        for callback in callbacks:
            callback.on_train_end()
        history = tf.keras.callbacks.History()
        history.history = {"loss": losses, "val_loss": losses}
        return history
    monkeypatch.setattr(model, "fit", replay_fit)
    phase = training_module._fit_phase(model, None, None, config, tmp_path, "test_phase", 3, None)
    assert phase["best_epoch"] == 2 and phase["best_val_loss"] == 0.2
    restored = tf.keras.models.load_model(phase["checkpoint"], compile=False)
    np.testing.assert_array_equal(restored.get_weights()[0], [[2]])
    journal = [json.loads(line) for line in (tmp_path / "epochs.jsonl").read_text().splitlines()]
    assert [entry["epoch"] for entry in journal] == [1, 2, 3]


def test_benchmark_restores_weights_optimizer_and_smoke_isolation(monkeypatch, tmp_path):
    config = load_config()
    config.update(_project_root=str(tmp_path), model_name="custom_cnn", image_size=32,
                  cnn_filters=[4, 8, 16], dense_units=8, cpu_batch_size=2,
                  benchmark_batches=2, class_weights=False, fine_tune=False)
    manifest_path = tmp_path / config["manifest_path"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("isolated test manifest", encoding="utf-8")
    manifest = pd.DataFrame({"split": ["train"] * 4 + ["validation"] * 4})
    monkeypatch.setattr(training_module, "load_manifest", lambda config: manifest)
    pixels = np.random.default_rng(42).uniform(0, 255, size=(4, 32, 32, 3)).astype(np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((pixels, np.array([0, 1, 0, 1], np.int32))).batch(2)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    dataset = dataset.with_options(options)
    monkeypatch.setattr(training_module, "create_dataset", lambda *args, **kwargs: dataset)
    original_build = custom_cnn.build_model
    original_phase = training_module._fit_phase
    initial = {}
    def capture_model(config):
        model = original_build(config)
        initial["weights"] = model.get_weights()
        return model
    monkeypatch.setattr(custom_cnn, "build_model", capture_model)
    checked = []
    def check_restored(model, *args, **kwargs):
        assert int(model.optimizer.iterations.numpy()) == 0
        for actual, expected in zip(model.get_weights(), initial["weights"]):
            np.testing.assert_array_equal(actual, expected)
        checked.append(True)
        return original_phase(model, *args, **kwargs)
    monkeypatch.setattr(training_module, "_fit_phase", check_restored)
    metadata = training_module.train(config, smoke=True, evaluate=False)
    assert checked == [True]
    assert metadata["trained_on_real_data"] is False
    assert len(list((tmp_path / "results" / "smoke").glob("*/model/best_model.keras"))) == 1
    assert not (tmp_path / "models" / "saved" / "best_model.keras").exists()
    assert not (tmp_path / "results" / "evaluation").exists()


@pytest.mark.parametrize("finetune_loss,expected_phase", [
    (0.4, "head"),
    (0.1, "finetune"),
    (0.2, "head"),
])
def test_mobilenet_selects_checkpoint_across_phases_without_republishing_application(
    monkeypatch, tmp_path, finetune_loss, expected_phase
):
    """Exercise real orchestration/copying with controlled, temporary checkpoints.

    Lightweight stand-ins replace fitting and serialization, already covered by
    separate tests. The non-smoke branch is intentional: smoke runs cannot catch
    accidental publication over the existing EfficientNet application bundle.
    All fixture artifacts are confined to pytest's temporary project directory.
    """
    config = load_config()
    config.update(_project_root=str(tmp_path), model_name="mobilenet",
                  cpu_batch_size=2, benchmark_batches=1, class_weights=False,
                  fine_tune=True)
    manifest_path = tmp_path / config["manifest_path"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("temporary orchestration fixture", encoding="utf-8")
    manifest = pd.DataFrame({"split": ["train", "train", "validation"]})

    application_model = tmp_path / config["model_path"]
    protected_paths = [
        application_model,
        application_model.parent / "metadata.json",
        application_model.parent / "class_names.json",
        tmp_path / config["training_dir"] / "history.json",
    ]
    protected = {}
    for index, path in enumerate(protected_paths):
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        content = f"Existing EfficientNet application artifact {index}".encode()
        path.write_bytes(content)
        protected[path] = content

    class TinyModel:
        def __init__(self, checkpoint=None):
            self.checkpoint = checkpoint
            self.weights = [np.array([0.0], dtype=np.float32)]

        def summary(self, print_fn):
            print_fn("Temporary orchestration stand-in")

        def get_weights(self):
            return [weight.copy() for weight in self.weights]

        def set_weights(self, weights):
            self.weights = [weight.copy() for weight in weights]

        def train_on_batch(self, images, labels, class_weight):
            self.weights[0] += 1

    class TinyDataset:
        def take(self, count):
            return [(np.zeros((2, 1)), np.array([0, 1]))][:count]

    builds, loaded_paths, phase_calls, compile_rates = [], [], [], []

    def build_model(config, weights):
        builds.append(weights)
        return TinyModel()

    def compile_model(model, config, learning_rate=None):
        compile_rates.append(learning_rate or config["learning_rate"])

    def load_model(path, compile):
        path = Path(path)
        assert compile is False
        assert path.read_bytes() == b"best head checkpoint"
        loaded_paths.append(path)
        return TinyModel(checkpoint="head")

    def unfreeze_for_finetuning(model, config):
        # Fine-tuning must begin from the saved best head, not its final epoch.
        assert model.checkpoint == "head"

    def fit_phase(model, training, validation, config, output_dir, phase, epochs, weights):
        phase_calls.append(phase)
        checkpoint = output_dir / f"{phase}.keras"
        checkpoint.write_bytes(f"best {phase} checkpoint".encode())
        losses = [0.6, 0.2, 0.5] if phase == "head" else [0.7, finetune_loss, 0.8]
        # The live end-of-phase model deliberately differs from the saved best.
        model.checkpoint = f"last {phase} epoch"
        return {"phase": phase, "checkpoint": str(checkpoint),
                "history": {"loss": losses, "val_loss": losses},
                "best_val_loss": min(losses), "best_epoch": 2,
                "epochs_run": 3, "seconds": 0.01, "stopping_reason": "early_stopping"}

    module = SimpleNamespace(build_model=build_model, compile_model=compile_model,
                             unfreeze_for_finetuning=unfreeze_for_finetuning)
    monkeypatch.setattr(training_module, "configure_runtime", lambda config: {"batch_size": 2})
    monkeypatch.setattr(training_module, "setup_logging", lambda *args: None)
    monkeypatch.setattr(training_module, "load_manifest", lambda config: manifest)
    monkeypatch.setattr(training_module, "verify_manifest_integrity",
                        lambda config: {"checked_images": 3})
    monkeypatch.setattr(training_module, "create_dataset", lambda *args, **kwargs: TinyDataset())
    monkeypatch.setattr(training_module, "get_model_module", lambda architecture: module)
    monkeypatch.setattr(training_module.tf.keras.models, "load_model", load_model)
    monkeypatch.setattr(training_module, "_fit_phase", fit_phase)
    monkeypatch.setattr(training_module, "plot_history", lambda *args: None)

    metadata = training_module.train(config, smoke=False, evaluate=False)
    model_dir = tmp_path / config["saved_models_dir"] / "mobilenet"
    output_dir = tmp_path / config["training_dir"] / "mobilenet"
    assert builds == ["imagenet", "imagenet"]
    assert phase_calls == ["head", "finetune"]
    assert loaded_paths == [model_dir / "head.keras"]
    assert compile_rates[-1] == config["fine_tune_learning_rate"]
    assert (model_dir / "best_model.keras").read_bytes() == f"best {expected_phase} checkpoint".encode()
    assert metadata["selected_phase"] == expected_phase
    assert metadata["selected_val_loss"] == min(0.2, finetune_loss)
    assert metadata["selected_epoch"] == 2 and metadata["total_epochs"] == 6
    assert json.loads((model_dir / "metadata.json").read_text())["selected_phase"] == expected_phase
    assert json.loads((output_dir / "run.json").read_text())["selected_phase"] == expected_phase
    history = json.loads((output_dir / "history.json").read_text())
    assert history["phase_boundaries"] == [3] and len(history["val_loss"]) == 6
    for path, original_content in protected.items():
        assert path.read_bytes() == original_content, f"MobileNet replaced application artifact: {path.name}"
