"""MobileNetV2 transfer learning with serializable RGB preprocessing."""
from __future__ import annotations

from typing import Any

import tensorflow as tf

from models.custom_cnn import compile_model


def build_model(config: dict[str, Any], weights: str | None = "imagenet") -> tf.keras.Model:
    """Build a frozen MobileNetV2 and classifier accepting RGB pixels in [0,255].

    MobileNetV2 expects pixels in [-1,1], so a built-in Rescaling layer performs
    that transformation inside the saved model. ``weights=None`` supports
    isolated architecture tests; production runs use ImageNet weights and never
    fall back to random initialization when a download fails.
    """
    size = int(config.get("image_size", 224))
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights=weights,
        input_shape=(size, size, 3),
        name="mobilenetv2",
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=(size, size, 3), name="mri_image")
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_preprocessing")(inputs)
    # Preserve ImageNet batch-normalization statistics during both training phases.
    x = base(x, training=False)
    x = tf.keras.layers.Activation("linear", name="gradcam_features")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")(x)
    dropout = float(config.get("dropout", 0.3))
    x = tf.keras.layers.Dropout(dropout, name="pool_dropout")(x)
    x = tf.keras.layers.Dense(
        int(config.get("dense_units", 128)), activation="relu", name="dense_head"
    )(x)
    x = tf.keras.layers.Dropout(dropout, name="head_dropout")(x)
    task = config.get("task", "binary")
    if not isinstance(task, str) or task not in {"binary", "multiclass"}:
        raise ValueError("task must be binary or multiclass.")
    classes = config.get("source_classes", ["glioma", "meningioma", "notumor", "pituitary"])
    outputs = tf.keras.layers.Dense(
        1 if task == "binary" else len(classes),
        activation="sigmoid" if task == "binary" else "softmax",
        name="predictions",
    )(x)
    return tf.keras.Model(inputs, outputs, name="mobilenet_classifier")


def unfreeze_for_finetuning(model: tf.keras.Model, config: dict[str, Any]) -> tf.keras.Model:
    """Unfreeze trailing backbone layers, keep BN frozen, and recompile at low LR."""
    base = model.get_layer("mobilenetv2")
    count = int(config.get("finetune_layers", 20))
    if count < 1 or count >= len(base.layers):
        raise ValueError(
            f"finetune_layers must be between 1 and {len(base.layers) - 1}; "
            "full-backbone unfreezing is disabled."
        )
    base.trainable = True
    cutoff = len(base.layers) - count
    for index, layer in enumerate(base.layers):
        layer.trainable = index >= cutoff and not isinstance(layer, tf.keras.layers.BatchNormalization)
    return compile_model(model, config, learning_rate=float(config.get("fine_tune_learning_rate", 1e-5)))
