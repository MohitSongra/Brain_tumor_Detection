"""ImageNet EfficientNetB0 transfer learning with controlled fine-tuning."""

from __future__ import annotations

from typing import Any

import tensorflow as tf

from models.custom_cnn import compile_model


def build_model(config: dict[str, Any], weights: str | None = "imagenet") -> tf.keras.Model:
    """Build a frozen EfficientNetB0 and trainable classifier.

    EfficientNet includes its own normalization: inputs must be RGB float pixels
    in [0,255]. ``weights=None`` is reserved for isolated architecture tests, not
    a substitute if downloading ImageNet weights fails.
    """
    size = int(config.get("image_size", 224))
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights=weights, input_shape=(size, size, 3)
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=(size, size, 3), name="mri_image")
    # Keep batch-normalization statistics fixed even after selected convolutions
    # are unfrozen. This argument is retained by Keras serialization.
    x = base(inputs, training=False)
    x = tf.keras.layers.Activation("linear", name="gradcam_features")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = tf.keras.layers.Dropout(float(config.get("dropout", 0.3)), name="pool_dropout")(x)
    x = tf.keras.layers.Dense(int(config.get("dense_units", 128)), activation="relu", name="dense_head")(x)
    x = tf.keras.layers.Dropout(float(config.get("dropout", 0.3)), name="head_dropout")(x)
    task = config.get("task", "binary")
    if task not in {"binary", "multiclass"}:
        raise ValueError("task must be binary or multiclass.")
    units = 1 if task == "binary" else len(config.get("source_classes", ["glioma", "meningioma", "notumor", "pituitary"]))
    outputs = tf.keras.layers.Dense(units, activation="sigmoid" if task == "binary" else "softmax", name="predictions")(x)
    return tf.keras.Model(inputs, outputs, name="efficientnet_classifier")


def unfreeze_for_finetuning(model: tf.keras.Model, config: dict[str, Any]) -> tf.keras.Model:
    """Unfreeze trailing backbone layers except BN, then recompile at a small LR."""
    base = model.get_layer("efficientnetb0")
    count = int(config.get("finetune_layers", 20))
    if count < 1 or count >= len(base.layers):
        raise ValueError(f"finetune_layers must be between 1 and {len(base.layers) - 1}; full-backbone unfreezing is disabled.")
    base.trainable = True
    cutoff = len(base.layers) - count
    for index, layer in enumerate(base.layers):
        layer.trainable = index >= cutoff and not isinstance(layer, tf.keras.layers.BatchNormalization)
    return compile_model(model, config, learning_rate=float(config.get("fine_tune_learning_rate", 1e-5)))
