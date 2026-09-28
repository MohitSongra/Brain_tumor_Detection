"""Final verification rejects stale evaluations without running a model."""
from __future__ import annotations

import hashlib

import pytest

from scripts.verify_trained_models import _verify_evaluation_provenance, _write_effective_config
from src.utils import load_config


def test_effective_config_roundtrip_preserves_small_float_types(tmp_path):
    config = load_config()
    path = tmp_path / "effective_config.yaml"
    _write_effective_config(config, path)
    restored = load_config(path)
    for field in ("fine_tune_learning_rate", "min_learning_rate", "learning_rate"):
        assert isinstance(restored[field], float)
        assert restored[field] == config[field]
    assert "_project_root" not in path.read_text(encoding="utf-8")


@pytest.fixture
def provenance(tmp_path):
    path = tmp_path / "fixture.keras"
    path.write_bytes(b"isolated provenance fixture, not a trained model")
    metadata = {"task": "binary", "class_names": ["no_tumor", "tumor"], "threshold": 0.5}
    measured = {**metadata, "trained_on_real_data": True, "n_samples": 10,
                "dataset_fingerprint": "current-manifest", "evaluation_split": "test",
                "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return path, metadata, measured


def test_final_verifier_accepts_matching_provenance(provenance):
    path, metadata, measured = provenance
    assert _verify_evaluation_provenance(measured, metadata, path, "current-manifest", 10) == measured["model_sha256"]


@pytest.mark.parametrize("field,replacement", [
    ("dataset_fingerprint", "old-manifest"), ("model_sha256", "old-checkpoint"),
    ("task", "multiclass"), ("class_names", ["tumor", "no_tumor"]),
    ("threshold", 0.7), ("n_samples", 9), ("trained_on_real_data", False),
])
def test_final_verifier_rejects_stale_or_incompatible_provenance(provenance, field, replacement):
    path, metadata, measured = provenance
    measured[field] = replacement
    with pytest.raises(ValueError, match=field):
        _verify_evaluation_provenance(measured, metadata, path, "current-manifest", 10)
