"""Configuration and checkpoint selection invariants."""
import json

import pytest
import yaml

import src.utils as utils
from src.utils import get_class_names, load_config, select_best_phase, write_json


def test_binary_order_is_explicit():
    assert get_class_names({"task": "binary"}) == ["no_tumor", "tumor"]


def test_worse_finetune_does_not_replace_head():
    phases = [{"phase": "head", "best_val_loss": 0.2}, {"phase": "finetune", "best_val_loss": 0.3}]
    assert select_best_phase(phases)["phase"] == "head"
    phases[1]["best_val_loss"] = 0.2
    assert select_best_phase(phases)["phase"] == "head"


def test_invalid_config_is_actionable(tmp_path):
    config = tmp_path / "invalid.yaml"
    config.write_text("prediction_threshold: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prediction_threshold"):
        load_config(config)


def test_missing_config_is_not_ignored(tmp_path):
    with pytest.raises(FileNotFoundError, match="Configuration not found"):
        load_config(tmp_path / "absent.yaml")


def test_json_refuses_nan(tmp_path):
    with pytest.raises(ValueError):
        write_json(tmp_path / "invalid.json", {"accuracy": float("nan")})


def test_valid_defaults_are_preserved():
    defaults = yaml.safe_load((utils.ROOT / "config/config.yaml").read_text(encoding="utf-8"))
    loaded = load_config()
    assert {key: loaded[key] for key in defaults} == defaults


def test_malformed_yaml_reports_override_path_and_location(tmp_path):
    path = tmp_path / "malformed.yaml"
    path.write_text("augmentation: [unclosed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid YAML configuration") as error:
        load_config(path)
    assert str(path) in str(error.value)
    assert "line" in str(error.value)


def test_malformed_default_yaml_is_also_actionable(tmp_path, monkeypatch):
    directory = tmp_path / "config"
    directory.mkdir()
    path = directory / "config.yaml"
    path.write_text("image_size: [unclosed\n", encoding="utf-8")
    monkeypatch.setattr(utils, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="Invalid YAML configuration") as error:
        load_config()
    assert str(path) in str(error.value)


@pytest.mark.parametrize("content", ["- item\n", "null\n", "true: value\n"])
def test_nonmapping_or_nonstring_config_keys_fail_cleanly(tmp_path, content):
    path = tmp_path / "shape.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_config(path)


@pytest.mark.parametrize(("key", "value"), [
    ("prediction_threshold", "0.5"), ("prediction_threshold", None),
    ("learning_rate", "fast"), ("fine_tune_learning_rate", None),
    ("learning_rate", float("nan")), ("min_learning_rate", float("inf")),
    ("validation_split", [0.2]), ("test_split", True), ("dropout", {}),
    ("reduce_lr_factor", 1), ("max_upload_mb", "20"),
    ("batch_size", True), ("image_size", 224.5), ("random_seed", -1),
    ("random_seed", 2**32), ("benchmark_batches", 0), ("early_stopping_patience", None),
    ("phash_distance", 4.5), ("intra_op_threads", "auto"),
    ("fine_tune", "false"), ("deterministic", 1), ("class_weights", None),
    ("model_name", []), ("task", {}), ("device", None),
    ("cnn_filters", "32,64,128"), ("cnn_filters", [32, True]),
    ("source_classes", None), ("source_classes", [["a"], "b"]),
    ("binary_mapping", None), ("binary_mapping", {"glioma": True}),
    ("model_path", None), ("raw_dir", []), ("manifest_path", " "),
    ("patient_metadata", 12), ("patient_id_regex", 5),
    ("patient_id_regex", "["), ("weights", False),
    ("augmentation", None), ("augmentation", {"horizontal_flip": "false"}),
    ("augmentation", {"contrast": None}), ("augmentation", {"zoom": float("nan")}),
])
def test_invalid_pipeline_field_types_and_ranges_are_actionable(tmp_path, key, value):
    path = tmp_path / "invalid_field.yaml"
    path.write_text(yaml.safe_dump({key: value}), encoding="utf-8")
    with pytest.raises(ValueError, match=key):
        load_config(path)


def test_valid_override_keeps_merge_and_boundary_behavior(tmp_path):
    path = tmp_path / "valid.yaml"
    path.write_text(yaml.safe_dump({"prediction_threshold": 0, "min_learning_rate": 0,
                                    "intra_op_threads": 0, "early_stopping_patience": 0,
                                    "fine_tune": False, "weights": None,
                                    "augmentation": {"contrast": 0}}), encoding="utf-8")
    loaded = load_config(path)
    assert loaded["prediction_threshold"] == 0
    assert loaded["min_learning_rate"] == loaded["intra_op_threads"] == 0
    assert loaded["fine_tune"] is False and loaded["weights"] is None
    assert loaded["augmentation"]["contrast"] == 0
    assert loaded["augmentation"]["rotation_degrees"] == 10


@pytest.mark.parametrize("names", [None, "glioma,notumor", [["glioma"], "notumor"], ["", "tumor"]])
def test_direct_multiclass_mapping_validation_is_type_safe(names):
    with pytest.raises(ValueError, match="source_classes"):
        get_class_names({"task": "multiclass", "source_classes": names})
