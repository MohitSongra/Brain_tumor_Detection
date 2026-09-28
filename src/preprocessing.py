"""Validated image decoding and shared preprocessing for training and inference.

All models accept RGB float32 pixels in [0, 255]. Saved models normalize inputs:
the baseline embeds Rescaling(1/255) to [0, 1], EfficientNetB0 keeps its built-in
preprocessing, and MobileNetV2 embeds Rescaling(1/127.5, offset=-1) to [-1, 1].
Do not apply a second normalization outside the saved model.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED_FORMATS = frozenset({"JPEG", "PNG", "BMP"})
SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
ImageSource = str | Path | bytes | BinaryIO | Image.Image


def load_rgb_image(source: ImageSource) -> Image.Image:
    """Decode an actual JPEG/PNG/BMP image, orient it, and return independent RGB data.

    A PIL image is accepted for already-decoded internal calls. File extensions
    alone are never trusted. Corrupt, unsupported, and oversized inputs receive
    useful errors instead of an opaque TensorFlow decoding failure.
    """
    if isinstance(source, Image.Image):
        if source.width <= 0 or source.height <= 0:
            raise ValueError("The image is empty.")
        return ImageOps.exif_transpose(source).convert("RGB").copy()
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Image does not exist: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError("Unsupported image format. Please use JPEG, PNG, or BMP.")
        payload: str | Path | BinaryIO = path
    elif isinstance(source, bytes):
        payload = io.BytesIO(source)
    elif hasattr(source, "read"):
        # Copy uploaded streams so verify/reopen does not close the caller's stream.
        try:
            source.seek(0)
        except (AttributeError, OSError):
            pass
        payload = io.BytesIO(source.read())
    else:
        raise TypeError("Expected an image path, bytes, file-like object, or PIL image.")
    try:
        with Image.open(payload) as candidate:
            if candidate.format not in SUPPORTED_FORMATS:
                raise ValueError("Unsupported image format. Please use JPEG, PNG, or BMP.")
            candidate.verify()
        if hasattr(payload, "seek"):
            payload.seek(0)
        with Image.open(payload) as candidate:
            candidate.load()
            return ImageOps.exif_transpose(candidate).convert("RGB").copy()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ValueError("The image is corrupt, unreadable, or too large. Use a valid JPEG, PNG, or BMP.") from exc


def preprocess_image(source: ImageSource, image_size: int = 224) -> np.ndarray:
    """Return one resized H×W×3 float32 image using TensorFlow bilinear resizing."""
    if not isinstance(image_size, int) or image_size < 16:
        raise ValueError("image_size must be an integer of at least 16 pixels.")
    import tensorflow as tf

    pixels = np.asarray(load_rgb_image(source), dtype=np.float32)
    return tf.image.resize(pixels, (image_size, image_size), method="bilinear", antialias=True).numpy()


def build_augmentation(config: dict):
    """Create mild seeded training-only augmentation; never persist augmented data."""
    import tensorflow as tf

    settings = config.get("augmentation", {})
    seed = int(config.get("random_seed", 42))
    layers = []
    if settings.get("horizontal_flip", True):
        layers.append(tf.keras.layers.RandomFlip("horizontal", seed=seed))
    if settings.get("rotation_degrees", 10):
        layers.append(tf.keras.layers.RandomRotation(
            float(settings.get("rotation_degrees", 10)) / 360, fill_mode="reflect", seed=seed + 1))
    if settings.get("zoom", 0.05):
        layers.append(tf.keras.layers.RandomZoom(
            float(settings.get("zoom", 0.05)), fill_mode="reflect", seed=seed + 2))
    if settings.get("translation", 0.05):
        translation = float(settings.get("translation", 0.05))
        layers.append(tf.keras.layers.RandomTranslation(
            translation, translation, fill_mode="reflect", seed=seed + 3))
    if settings.get("contrast", 0.1):
        layers.append(tf.keras.layers.RandomContrast(
            float(settings.get("contrast", 0.1)), seed=seed + 4))
    return tf.keras.Sequential(layers, name="training_augmentation")
