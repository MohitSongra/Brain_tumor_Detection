"""Small generated fixtures verify mechanics; these are never research results."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from src.data_loader import compute_class_weights, create_dataset, load_manifest, verify_manifest_integrity
from src.prepare_data import audit_groups, inspect_source, prepare_dataset
from src.preprocessing import load_rgb_image, preprocess_image
from src.utils import load_config


@pytest.fixture
def config(tmp_path):
    settings = load_config()
    settings.update({"_project_root": str(tmp_path), "image_size": 32, "cpu_batch_size": 2,
                     "parallel_calls": 1, "phash_distance": 0})
    return settings


def image_file(path: Path, seed: int, mode: str = "RGB") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    shape = (36, 48) if mode == "L" else (36, 48, 4 if mode == "RGBA" else 3)
    Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8), mode=mode).save(path)
    return path


def make_official(root: Path, count: int = 12) -> None:
    for class_index, source_class in enumerate(["notumor", "glioma"]):
        for index in range(count):
            image_file(root / "Training" / source_class / f"train_{index}.png", class_index * 1000 + index)
        for index in range(4):
            image_file(root / "Testing" / source_class / f"test_{index}.png", class_index * 1000 + 500 + index)


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA"])
def test_preprocessing_rgb_shape_range_and_consistency(tmp_path, mode):
    path = image_file(tmp_path / "image.png", 1, mode)
    result = preprocess_image(path, 32)
    assert result.shape == (32, 32, 3)
    assert result.dtype == np.float32
    assert np.isfinite(result).all() and result.min() >= 0 and result.max() <= 255
    np.testing.assert_array_equal(result, preprocess_image(path.read_bytes(), 32))
    np.testing.assert_array_equal(result, preprocess_image(io.BytesIO(path.read_bytes()), 32))
    assert load_rgb_image(path).mode == "RGB"


def test_invalid_and_unsupported_images(tmp_path):
    corrupt = tmp_path / "broken.png"
    corrupt.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="corrupt"):
        load_rgb_image(corrupt)
    gif = io.BytesIO()
    Image.new("RGB", (20, 20)).save(gif, format="GIF")
    with pytest.raises(ValueError, match="Unsupported"):
        load_rgb_image(gif.getvalue())
    with pytest.raises(FileNotFoundError):
        load_rgb_image(tmp_path / "absent.jpg")
    with pytest.raises(ValueError, match="image_size"):
        preprocess_image(Image.new("RGB", (20, 20)), 0)


def test_prepare_is_reproducible_and_excludes_augmented_and_duplicates(config, tmp_path):
    raw = tmp_path / "data/raw"
    make_official(raw)
    test_source = raw / "Testing/notumor/test_0.png"
    duplicate = raw / "Training/notumor/duplicate.png"
    duplicate.write_bytes(test_source.read_bytes())
    image_file(raw / "Testing/glioma/test_aug_01.png", 9001)
    report = prepare_dataset(config)
    manifest = load_manifest(config)
    assert report["total_inspected"] == 34
    assert report["excluded"] == 2
    assert report["exact_duplicates_removed"] == 1
    assert set(manifest["split"]) == {"train", "validation", "test"}
    assert set(manifest["label"]) == {0, 1}
    assert (manifest.groupby("group_id")["split"].nunique() == 1).all()
    assert not manifest["source_path"].str.contains("aug|duplicate").any()
    original_test = manifest.loc[manifest["original_split"] == "test"]
    assert (original_test["split"] == "test").all()
    again = prepare_dataset(config)
    assert again["dataset_fingerprint"] == report["dataset_fingerprint"]
    pd.testing.assert_frame_equal(manifest, load_manifest(config))


@pytest.mark.parametrize("input_names", [("notumor", "glioma"), ("no_tumor", "tumor")])
def test_presplit_in_place_rerun_preserves_inputs(config, tmp_path, input_names):
    source = tmp_path / "data"
    original_paths = []
    for split_index, split in enumerate(["train", "validation", "test"]):
        for class_index, source_class in enumerate(input_names):
            for image_index in range(4):
                path = image_file(source / split / source_class / f"image_{image_index}.png",
                                  split_index * 1000 + class_index * 100 + image_index)
                original_paths.append(path)
    first = prepare_dataset(config, source=source)
    second = prepare_dataset(config, source=source)
    assert first["dataset_fingerprint"] == second["dataset_fingerprint"]
    assert first["total_inspected"] == second["total_inspected"] == 24
    assert all(path.is_file() for path in original_paths)
    import hashlib
    assert second["dataset_fingerprint"] == hashlib.sha256((tmp_path / "data/processed/manifest.csv").read_bytes()).hexdigest()


def test_exact_conflicting_source_labels_stop_audit(config, tmp_path):
    raw = tmp_path / "data/raw"
    make_official(raw)
    conflict = raw / "Training/glioma/conflict.png"
    conflict.write_bytes((raw / "Training/notumor/train_0.png").read_bytes())
    with pytest.raises(ValueError, match="label conflict"):
        prepare_dataset(config)
    assert (tmp_path / "data/processed/audit_failure.json").is_file()
    assert not (tmp_path / "data/processed/manifest.csv").exists()


def test_patient_groups_stay_together_and_test_overlap_excluded(config, tmp_path):
    raw = tmp_path / "data/raw"
    make_official(raw)
    frame, layout, _ = inspect_source(config, raw)
    first = frame.index[frame["source_path"].str.endswith("Training/notumor/train_0.png")][0]
    second = frame.index[frame["source_path"].str.endswith("Testing/notumor/test_0.png")][0]
    frame.loc[[first, second], "patient_id"] = "known_patient"
    audited, report = audit_groups(frame, config, layout)
    assert audited.at[first, "excluded"]
    assert audited.at[first, "exclusion_reason"] == "development_group_overlaps_official_test"
    assert not audited.at[second, "excluded"]
    assert report["development_overlapping_test_removed"] == 1


def test_patient_csv_and_unsplit_groups(config, tmp_path):
    raw = tmp_path / "data/raw"
    records = []
    for class_index, name in enumerate(["glioma", "notumor"]):
        for patient in range(12):
            for slice_id in range(2):
                path = image_file(raw / name / f"patient_{patient}_{slice_id}.png", class_index * 1000 + patient * 10 + slice_id)
                records.append({"path": str(path.relative_to(raw)), "patient_id": f"{name}_{patient}"})
    metadata = tmp_path / "patients.csv"
    pd.DataFrame(records).to_csv(metadata, index=False)
    config["patient_metadata"] = str(metadata)
    report = prepare_dataset(config)
    manifest = load_manifest(config)
    assert report["patient_mapping_available"]
    assert (manifest.groupby("patient_id")["split"].nunique() == 1).all()
    assert set(manifest["split"]) == {"train", "validation", "test"}


def test_loader_order_labels_validation_not_augmented_and_weights(config, tmp_path):
    raw = tmp_path / "data/raw"
    make_official(raw)
    prepare_dataset(config)
    frame = load_manifest(config)
    selected = frame.loc[frame["split"] == "validation"]
    first_batch, first_labels = next(iter(create_dataset(config, "validation", batch_size=2)))
    second_batch, second_labels = next(iter(create_dataset(config, "validation", batch_size=2)))
    np.testing.assert_array_equal(first_batch.numpy(), second_batch.numpy())
    np.testing.assert_array_equal(first_labels.numpy(), second_labels.numpy())
    np.testing.assert_array_equal(first_batch[0].numpy(), preprocess_image(selected.iloc[0]["path"], 32))
    assert first_labels.shape == (2, 1)
    np.testing.assert_array_equal(first_labels.numpy().ravel(), selected["label"].to_numpy()[:2])
    assert set(compute_class_weights(config)) == {0, 1}
    with pytest.raises(ValueError, match="only allowed"):
        create_dataset(config, "test", training=True)


def test_training_augmentation_and_multiclass_mapping(config, tmp_path):
    make_official(tmp_path / "data/raw")
    prepare_dataset(config)
    pixels, labels = next(iter(create_dataset(config, "train", training=True, batch_size=3)))
    assert pixels.shape == (3, 32, 32, 3)
    assert np.isfinite(pixels.numpy()).all()
    assert float(np.min(pixels)) >= 0 and float(np.max(pixels)) <= 255
    assert labels.shape == (3, 1)
    config.update(task="multiclass", source_classes=["glioma", "notumor"])
    frame = load_manifest(config)
    assert set(frame.loc[frame["source_class"] == "glioma", "label"]) == {0}
    assert set(frame.loc[frame["source_class"] == "notumor", "label"]) == {1}
    _, multiclass_labels = next(iter(create_dataset(config, "test", batch_size=3)))
    assert multiclass_labels.shape == (3,)
    assert multiclass_labels.dtype.name == "int32"


def test_missing_empty_and_incorrect_dataset_errors(config, tmp_path):
    with pytest.raises(FileNotFoundError, match="source directory"):
        prepare_dataset(config)
    (tmp_path / "data/raw").mkdir(parents=True)
    with pytest.raises(ValueError, match="Incorrect dataset structure"):
        prepare_dataset(config)
    (tmp_path / "data/raw/notumor").mkdir()
    with pytest.raises(ValueError, match="no supported images"):
        prepare_dataset(config)


def test_loader_missing_and_leaking_manifests(config, tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest"):
        load_manifest(config)
    raw = tmp_path / "data/raw"
    make_official(raw)
    prepare_dataset(config)
    manifest_path = tmp_path / "data/processed/manifest.csv"
    frame = pd.read_csv(manifest_path)
    train_index = frame.index[frame["split"] == "train"][0]
    test_index = frame.index[frame["split"] == "test"][0]
    frame.at[train_index, "group_id"] = frame.at[test_index, "group_id"]
    frame.to_csv(manifest_path, index=False)
    with pytest.raises(ValueError, match="leakage"):
        load_manifest(config)


def test_integrity_verifies_fingerprint_and_detects_modified_image(config, tmp_path):
    make_official(tmp_path / "data/raw")
    audit = prepare_dataset(config)
    result = verify_manifest_integrity(config)
    assert result == {"checked_images": audit["included"], "dataset_fingerprint": audit["dataset_fingerprint"]}
    modified = Path(load_manifest(config).iloc[0]["path"])
    modified.write_bytes(b"changed after the original audit")
    with pytest.raises(ValueError, match="integrity check failed"):
        verify_manifest_integrity(config)


@pytest.mark.parametrize("field", ["byte_sha256", "pixel_sha256", "group_id"])
@pytest.mark.parametrize("change", ["drop", "empty"])
def test_integrity_rejects_incomplete_audit_manifest(config, tmp_path, field, change):
    make_official(tmp_path / "data/raw")
    prepare_dataset(config)
    manifest_path = tmp_path / "data/processed/manifest.csv"
    frame = pd.read_csv(manifest_path, keep_default_na=False)
    if change == "drop":
        frame = frame.drop(columns=field)
    else:
        frame.at[0, field] = ""
    frame.to_csv(manifest_path, index=False)
    with pytest.raises(ValueError, match="missing fields|empty"):
        verify_manifest_integrity(config)


def test_integrity_hash_formatting_cannot_hide_cross_split_duplicates(config, tmp_path):
    make_official(tmp_path / "data/raw")
    prepare_dataset(config)
    manifest_path = tmp_path / "data/processed/manifest.csv"
    frame = pd.read_csv(manifest_path, keep_default_na=False)
    train_index = frame.index[frame["split"] == "train"][0]
    test_index = frame.index[frame["split"] == "test"][0]
    frame.at[train_index, "byte_sha256"] = f" {frame.at[test_index, 'byte_sha256'].upper()} "
    frame.to_csv(manifest_path, index=False)
    with pytest.raises(ValueError, match="leakage"):
        verify_manifest_integrity(config)


def test_integrity_import_supports_cached_loader_without_new_export(config, tmp_path, monkeypatch):
    import src.data_loader as loader
    from src.integrity import verify_manifest_integrity as direct_verify

    make_official(tmp_path / "data/raw")
    audit = prepare_dataset(config)
    monkeypatch.delattr(loader, "verify_manifest_integrity")
    assert direct_verify(config)["checked_images"] == audit["included"]
