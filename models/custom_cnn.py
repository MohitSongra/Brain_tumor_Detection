"""A compact convolutional baseline with preprocessing inside the saved model."""

from __future__ import annotations

from typing import Any

import tensorflow as tf


def build_model(config: dict[str, Any]) -> tf.keras.Model:
    """Build the baseline; callers supply RGB float pixels in the range [0,255]."""
    size = int(config.get("image_size", 224))
    inputs = tf.keras.Input(shape=(size, size, 3), name="mri_image")
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="normalize_pixels")(inputs)
    filters = config.get("cnn_filters", [32, 64, 128])
    if not filters or any(int(n) < 1 for n in filters):
        raise ValueError("cnn_filters must contain positive filter counts.")
    for index, count in enumerate(filters, start=1):
        x = tf.keras.layers.Conv2D(
            int(count), 3, padding="same", use_bias=False, name=f"conv_{index}"
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"batch_norm_{index}")(x)
        x = tf.keras.layers.ReLU(name=f"relu_{index}")(x)
        x = tf.keras.layers.MaxPooling2D(name=f"pool_{index}")(x)
    x = tf.keras.layers.Activation("linear", name="gradcam_features")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = tf.keras.layers.Dense(int(config.get("dense_units", 128)), activation="relu", name="dense_head")(x)
    x = tf.keras.layers.Dropout(float(config.get("dropout", 0.3)), name="head_dropout")(x)
    task = config.get("task", "binary")
    if task not in {"binary", "multiclass"}:
        raise ValueError("task must be binary or multiclass.")
    units = 1 if task == "binary" else len(config.get("source_classes", ["glioma", "meningioma", "notumor", "pituitary"]))
    outputs = tf.keras.layers.Dense(units, activation="sigmoid" if task == "binary" else "softmax", name="predictions")(x)
    return tf.keras.Model(inputs, outputs, name="custom_cnn")


def compile_model(
    model: tf.keras.Model, config: dict[str, Any], learning_rate: float | None = None
) -> tf.keras.Model:
    """Compile a model with Adam and metrics appropriate to the task."""
    binary = config.get("task", "binary") == "binary"
    metrics = (
        [tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=float(config.get("prediction_threshold", 0.5))),
         tf.keras.metrics.AUC(name="auc")]
        if binary else [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(learning_rate if learning_rate is not None else config.get("learning_rate", 1e-3))),
        loss=tf.keras.losses.BinaryCrossentropy() if binary else tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=metrics,
    )
    return model

