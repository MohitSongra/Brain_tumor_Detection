"""Three-model comparison regressions use isolated arithmetic-only fixtures."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from models.registry import DISPLAY_NAMES, MODEL_NAMES
from src.compare import COLUMNS, compare_experiments


def _record(root: Path, name: str, **changes) -> dict:
    """Write temporary provenance and known arithmetic; never touch real results."""
    record = {
        "model_name": name, "trained_on_real_data": True, "evaluation_split": "test",
        "dataset_fingerprint": "temporary-fixture-manifest", "task": "binary",
        "class_names": ["no_tumor", "tumor"], "threshold": 0.5, "n_samples": 10,
        "accuracy": 0.8, "precision": 0.75, "recall": 0.9,
        "f1": 0.82, "specificity": 0.7, "roc_auc": 0.85,
    }
    record.update(changes)
    destination = root / name / "metrics.json"
    destination.parent.mkdir(parents=True)
    destination.write_text(json.dumps(record), encoding="utf-8")
    return record


def test_missing_third_model_is_explicit_and_existing_metrics_stay_unchanged(tmp_path):
    first = _record(tmp_path, "custom_cnn", accuracy=0.7)
    second = _record(tmp_path, "efficientnet", accuracy=0.8)
    originals = {name: (tmp_path / name / "metrics.json").read_bytes()
                 for name in ("custom_cnn", "efficientnet")}
    frame = compare_experiments({"evaluation_dir": str(tmp_path)})
    assert frame["Model"].tolist() == [DISPLAY_NAMES[name] for name in MODEL_NAMES]
    assert frame.iloc[0]["Accuracy"] == first["accuracy"]
    assert frame.iloc[1]["Accuracy"] == second["accuracy"]
    third = frame.loc[frame["Model"] == "MobileNetV2"].iloc[0]
    assert set(third[list(COLUMNS.values())]) == {"Not available"}
    markdown = (tmp_path / "comparison.md").read_text(encoding="utf-8")
    assert "| MobileNetV2 | Not available |" in markdown
    assert "Training required" in markdown
    for name, original in originals.items():
        assert (tmp_path / name / "metrics.json").read_bytes() == original


@pytest.mark.parametrize(("field", "mismatched"), [
    ("dataset_fingerprint", "different-manifest"),
    ("task", "multiclass"),
    ("class_names", ["tumor", "no_tumor"]),
    ("threshold", 0.7),
    ("n_samples", 11),
])
def test_third_model_provenance_mismatch_is_rejected(tmp_path, field, mismatched):
    for name in MODEL_NAMES:
        _record(tmp_path, name, **({field: mismatched} if name == "mobilenet" else {}))
    with pytest.raises(ValueError, match=field):
        compare_experiments({"evaluation_dir": str(tmp_path)})
    assert not (tmp_path / "comparison.csv").exists()


def test_available_transfer_models_checked_when_baseline_is_missing(tmp_path):
    _record(tmp_path, "efficientnet")
    _record(tmp_path, "mobilenet", dataset_fingerprint="different-manifest")
    with pytest.raises(ValueError, match="dataset_fingerprint"):
        compare_experiments({"evaluation_dir": str(tmp_path)})


def test_three_matching_models_are_reported_in_registry_order(tmp_path):
    for name in MODEL_NAMES:
        _record(tmp_path, name)
    frame = compare_experiments({"evaluation_dir": str(tmp_path)})
    assert frame["Model"].tolist() == [DISPLAY_NAMES[name] for name in MODEL_NAMES]
    assert frame["Accuracy"].tolist() == [0.8] * len(MODEL_NAMES)
    markdown = (tmp_path / "comparison.md").read_text(encoding="utf-8")
    assert "All available measured rows" in markdown
    assert "Training required" not in markdown


def test_registry_and_comparison_imports_do_not_load_tensorflow():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-c", "import sys; import src.compare; import models.registry; assert 'tensorflow' not in sys.modules"],
        cwd=root, capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
