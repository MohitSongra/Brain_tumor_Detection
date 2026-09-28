"""Manifest-backed, bounded-memory TensorFlow input pipelines."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing import build_augmentation, preprocess_image
from src.integrity import verify_manifest_integrity
from src.utils import get_class_names, resolve_path


def load_manifest(config: dict) -> pd.DataFrame:
    """Read and validate the prepared, included-image manifest."""
    path = resolve_path(config.get("manifest_path", "data/processed/manifest.csv"), config)
    if not path.is_file():
        raise FileNotFoundError(f"Prepared dataset manifest not found: {path}. Run python src/prepare_data.py first.")
    try:
        frame = pd.read_csv(path, keep_default_na=False)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Dataset manifest is empty: {path}") from exc
    required = {"path", "source_class", "split", "label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Manifest must contain columns: {', '.join(sorted(required))}.")
    if frame.empty:
        raise ValueError("The prepared dataset is empty.")
    if not set(frame["split"]).issubset({"train", "validation", "test"}):
        raise ValueError("Manifest contains an unknown split; expected train, validation, or test.")
    if frame["path"].duplicated().any():
        raise ValueError("Manifest repeats image paths; regenerate the dataset audit.")
    for field in ("byte_sha256", "pixel_sha256", "group_id"):
        if field in frame:
            # CSV formatting must not disguise identical hashes/groups.
            frame[field] = frame[field].astype(str).str.strip()
            if field.endswith("sha256"):
                frame[field] = frame[field].str.lower()
            nonempty = frame.loc[frame[field] != ""]
            if (nonempty.groupby(field)["split"].nunique() > 1).any():
                raise ValueError(f"Data leakage detected: {field} crosses dataset splits.")
    classes = get_class_names(config)
    if config.get("task", "binary") == "binary":
        mapping = config.get("binary_mapping", {})
        try:
            frame["label"] = frame["source_class"].map(lambda name: int(mapping[name]))
        except KeyError as exc:
            raise ValueError(f"No binary mapping for source class {exc.args[0]!r}.") from exc
    else:
        lookup = {name: index for index, name in enumerate(classes)}
        try:
            frame["label"] = frame["source_class"].map(lookup.__getitem__)
        except KeyError as exc:
            raise ValueError(f"Source class {exc.args[0]!r} is absent from source_classes.") from exc
    if not frame["label"].isin(range(len(classes))).all():
        raise ValueError("Manifest contains a label outside the configured class mapping.")
    return frame.sort_values(["split", "path"]).reset_index(drop=True)


def create_dataset(config: dict, split: str, training: bool = False, batch_size: int | None = None):
    """Build a deterministic dataset without caching all decoded images.

    Labels are (batch, 1) float32 for sigmoid training and (batch,) int32
    for multiclass training. Only the training split can be augmented.
    """
    import tensorflow as tf

    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test.")
    if training and split != "train":
        raise ValueError("Augmentation/shuffling is only allowed for the training split.")
    frame = load_manifest(config)
    selected = frame.loc[frame["split"] == split].copy()
    if selected.empty:
        raise ValueError(f"The {split} dataset is empty. Inspect the preparation audit.")
    paths = [str(resolve_path(value, config)) for value in selected["path"]]
    missing = next((path for path in paths if not Path(path).is_file()), None)
    if missing:
        raise FileNotFoundError(f"A prepared image is missing: {missing}. Rerun dataset preparation.")
    if batch_size is None:
        batch_size = int(config.get("batch_size", 32) if tf.config.list_physical_devices("GPU") else config.get("cpu_batch_size", 8))
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    image_size = int(config.get("image_size", 224))
    binary = config.get("task", "binary") == "binary"
    labels = selected["label"].to_numpy(dtype=np.float32 if binary else np.int32)
    if binary:
        labels = labels[:, None]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = max(1, int(config.get("parallel_calls", 2)))
    dataset = dataset.with_options(options)
    if training:
        dataset = dataset.shuffle(len(selected), seed=int(config.get("random_seed", 42)), reshuffle_each_iteration=True)

    def decode(path, label):
        def read_numpy(value):
            encoded = value.item() if hasattr(value, "item") else value
            return preprocess_image(encoded.decode("utf-8"), image_size)
        pixels = tf.numpy_function(read_numpy, [path], tf.float32)
        pixels.set_shape((image_size, image_size, 3))
        return pixels, label

    dataset = dataset.map(decode, num_parallel_calls=max(1, int(config.get("parallel_calls", 2))))
    dataset = dataset.batch(batch_size, drop_remainder=False)
    if training:
        augmenter = build_augmentation(config)
        dataset = dataset.map(
            lambda pixels, labels: (tf.clip_by_value(augmenter(pixels, training=True), 0.0, 255.0), labels),
            num_parallel_calls=1,
        )
    return dataset.prefetch(max(1, int(config.get("prefetch_batches", 1))))


def compute_class_weights(config: dict) -> dict[int, float]:
    """Compute balanced weights from training labels only; fail on absent classes."""
    from sklearn.utils.class_weight import compute_class_weight

    frame = load_manifest(config)
    labels = frame.loc[frame["split"] == "train", "label"].to_numpy(dtype=int)
    classes = np.arange(len(get_class_names(config)))
    if not set(classes).issubset(set(labels)):
        raise ValueError("Every configured output class must be represented in training.")
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return {int(label): float(weight) for label, weight in zip(classes, weights)}
