"""Validate reporting contracts with temporary fixtures, without TensorFlow."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

from scripts.build_results_report import build_report
from scripts.verify_notebooks import execute_notebooks
from src.utils import manifest_fingerprint, write_json


def _report_fixture(root: Path) -> tuple[dict, dict]:
    manifest = root / "processed" / "manifest.csv"
    manifest.parent.mkdir()
    manifest.write_text("Reporting test fixture, not MRI experiment data\n", encoding="utf-8")
    audit = {"status": "passed", "dataset_fingerprint": manifest_fingerprint(manifest), "total_inspected": 6,
             "included": 6, "excluded": 0, "exclusion_counts": {}, "layout": "unsplit", "patient_mapping_available": True,
             "split_class_counts": {split: {"notumor": 1, "glioma": 1} for split in ("train", "validation", "test")}}
    write_json(manifest.parent / "audit.json", audit)
    config = {"_project_root": str(root), "manifest_path": "processed/manifest.csv", "processed_dir": "processed",
              "evaluation_dir": "custom evaluation", "training_dir": "custom training", "task": "binary",
              "prediction_threshold": 0.5, "source_classes": ["glioma", "notumor"]}
    return config, audit


def _add_measured_fixture(root: Path, config: dict, audit: dict) -> Path:
    """Generate arithmetic-only report inputs isolated in pytest's temp folder."""
    model = root / "model" / "best_model.keras"
    model.parent.mkdir()
    model.write_bytes(b"Not a neural model: temporary reporting test fixture")
    run = {"dataset_fingerprint": audit["dataset_fingerprint"], "model_name": "custom_cnn", "task": "binary",
           "class_names": ["no_tumor", "tumor"], "selected_phase": "baseline", "selected_epoch": 1,
           "selected_val_loss": 0.6, "total_epochs": 1, "training_seconds": 1.0, "class_weights": None,
           "phases": [{"phase": "baseline", "epochs_run": 1, "stopping_reason": "epoch_limit", "best_val_loss": 0.6, "best_epoch": 1}]}
    write_json(model.parent / "metadata.json", run)
    training = root / config["training_dir"] / "custom_cnn"
    write_json(training / "run.json", run)
    write_json(training / "history.json", {"accuracy": [0.5], "val_accuracy": [0.5], "loss": [0.7], "val_loss": [0.6]})
    metrics = {**run, "trained_on_real_data": True, "evaluation_split": "test", "threshold": 0.5, "n_samples": 2,
               "model_path": "model/best_model.keras", "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
               "accuracy": 0.5, "precision": 0.5, "recall": 1.0, "f1": 2 / 3, "specificity": 0.0, "roc_auc": 0.5,
               "per_class": {"tumor": {"tn": 0, "fp": 1, "fn": 0, "tp": 1}}}
    write_json(root / config["evaluation_dir"] / "custom_cnn" / "metrics.json", metrics)
    return model


def test_report_missing_training_stays_unavailable(tmp_path: Path) -> None:
    config, _ = _report_fixture(tmp_path)
    text = build_report(config).read_text(encoding="utf-8")
    assert text.count("Training required — no measured result available.") == 3
    assert "## MobileNetV2" in text
    assert "eligible official Testing assignment was preserved" not in text
    assert "Supplied patient identifiers" in text
    assert "no patient mapping is available" not in text


def test_report_rejects_stale_current_manifest(tmp_path: Path) -> None:
    config, _ = _report_fixture(tmp_path)
    (tmp_path / config["manifest_path"]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="audit does not match"):
        build_report(config)


def test_report_rejects_evaluation_from_another_cohort(tmp_path: Path) -> None:
    config, audit = _report_fixture(tmp_path)
    _add_measured_fixture(tmp_path, config, audit)
    path = tmp_path / config["evaluation_dir"] / "custom_cnn" / "metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    metrics["dataset_fingerprint"] = "old-manifest"
    write_json(path, metrics)
    with pytest.raises(ValueError, match="different manifest"):
        build_report(config)


def test_report_links_configured_paths_and_respects_loss_configuration(tmp_path: Path) -> None:
    config, audit = _report_fixture(tmp_path)
    _add_measured_fixture(tmp_path, config, audit)
    text = build_report(config).read_text(encoding="utf-8")
    assert "../custom%20training/custom_cnn/accuracy.png" in text
    assert "../custom%20evaluation/custom_cnn/confusion_matrix.png" in text
    assert "class-weighted loss" not in text


def test_report_rejects_changed_evaluated_checkpoint(tmp_path: Path) -> None:
    config, audit = _report_fixture(tmp_path)
    model = _add_measured_fixture(tmp_path, config, audit)
    model.write_bytes(b"changed model")
    with pytest.raises(ValueError, match="checkpoint is missing or changed"):
        build_report(config)


def test_notebook_output_cannot_overwrite_sources() -> None:
    from scripts.verify_notebooks import ROOT
    with pytest.raises(ValueError, match="must not overwrite source notebooks"):
        execute_notebooks(Path(sys.executable), ROOT / "notebooks", 1, ["01_EDA.ipynb"])
