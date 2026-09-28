"""Architecture/inference checks use isolated random weights, never research data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import tensorflow as tf

from models import custom_cnn, efficientnet_model
from src.gradcam import generate_gradcam, save_gradcam
from src.predict import load_model_bundle, predict_image


def small_config(task: str = "binary") -> dict:
    return {"task": task, "image_size": 32, "cnn_filters": [4, 8, 16],
            "dense_units": 8, "dropout": 0.1, "learning_rate": 1e-3,
            "fine_tune_learning_rate": 1e-5, "finetune_layers": 20,
            "source_classes": ["glioma", "meningioma", "notumor", "pituitary"]}


def metadata_for(task: str = "binary") -> dict:
    classes = ["no_tumor", "tumor"] if task == "binary" else small_config(task)["source_classes"]
    return {"task": task, "class_names": classes, "image_size": 32,
            "input_range": [0, 255], "threshold": 0.5,
            "gradcam_layer": "gradcam_features", "trained_on_real_data": False}


def write_bundle(model, target: Path, metadata: dict) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    path = target / "best_model.keras"
    model.save(path)
    (target / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (target / "class_names.json").write_text(json.dumps(metadata["class_names"]), encoding="utf-8")
    return path


def test_imagenet_channel_scaling_preserves_rgb_shape():
    """Catch the Keras 3.11.3 bug that broke real ImageNet weight loading."""
    inputs = tf.keras.Input((32, 32, 3))
    scaled = tf.keras.layers.Rescaling([1.1, 1.2, 1.3])(inputs)
    assert tuple(scaled.shape) == (None, 32, 32, 3)


@pytest.mark.parametrize("architecture", ["custom_cnn", "efficientnet"])
@pytest.mark.parametrize("task", ["binary", "multiclass"])
def test_outputs_reload_and_gradcam(tmp_path, architecture, task):
    tf.keras.backend.clear_session()
    config = small_config(task)
    model = custom_cnn.build_model(config) if architecture == "custom_cnn" else efficientnet_model.build_model(config, weights=None)
    image = np.random.default_rng(42).integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    before = np.asarray(model(image[None].astype(np.float32), training=False))
    assert before.shape == (1, 1 if task == "binary" else 4)
    assert np.isfinite(before).all() and ((before >= 0) & (before <= 1)).all()
    if task == "multiclass":
        np.testing.assert_allclose(before.sum(axis=1), 1, atol=1e-6)
    path = write_bundle(model, tmp_path, metadata_for(task))
    restored, metadata = load_model_bundle(path)
    after = np.asarray(restored(image[None].astype(np.float32), training=False))
    np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-6)
    prediction = predict_image(Image.fromarray(image), restored, metadata)
    assert 0 <= prediction["confidence"] <= 1
    assert len(prediction["probabilities"]) == (2 if task == "binary" else 4)
    assert sum(prediction["probabilities"]) == pytest.approx(1, abs=1e-5)
    assert prediction["class_name"] == metadata["class_names"][prediction["class_index"]]
    heatmap = generate_gradcam(restored, image.astype(np.float32), prediction["class_index"])
    assert heatmap.ndim == 2 and np.isfinite(heatmap).all()
    assert heatmap.min() >= 0 and heatmap.max() <= 1


@pytest.mark.parametrize("task", ["binary", "multiclass"])
def test_baseline_training_step(task):
    config = small_config(task)
    model = custom_cnn.compile_model(custom_cnn.build_model(config), config)
    pixels = np.random.default_rng(7).uniform(0, 255, size=(2, 32, 32, 3)).astype(np.float32)
    result = model.train_on_batch(pixels, np.array([0, 1]), return_dict=True)
    assert all(np.isfinite(value) for value in result.values())


def test_finetuning_freezes_bn_and_early_layers(tmp_path):
    config = small_config()
    model = efficientnet_model.build_model(config, weights=None)
    base = model.get_layer("efficientnetb0")
    assert not base.trainable and not base.trainable_weights
    efficientnet_model.unfreeze_for_finetuning(model, config)
    assert base.trainable and base.trainable_weights
    assert all(not layer.trainable for layer in base.layers[:-20])
    assert all(not layer.trainable for layer in base.layers if isinstance(layer, tf.keras.layers.BatchNormalization))
    assert float(model.optimizer.learning_rate.numpy()) == pytest.approx(1e-5)
    before = [layer.moving_mean.numpy().copy() for layer in base.layers if isinstance(layer, tf.keras.layers.BatchNormalization)]
    model(np.ones((1, 32, 32, 3), np.float32) * 100, training=True)
    after = [layer.moving_mean.numpy() for layer in base.layers if isinstance(layer, tf.keras.layers.BatchNormalization)]
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)
    restored, _ = load_model_bundle(write_bundle(model, tmp_path, metadata_for()))
    restored_base = restored.get_layer("efficientnetb0")
    assert all(not layer.trainable for layer in restored_base.layers if isinstance(layer, tf.keras.layers.BatchNormalization))
    with pytest.raises(ValueError, match="full-backbone"):
        efficientnet_model.unfreeze_for_finetuning(model, {**config, "finetune_layers": len(base.layers)})


def test_binary_gradcam_uses_class_logit_without_mutation(tmp_path):
    inputs = tf.keras.Input(shape=(32, 32, 3))
    features = tf.keras.layers.Activation("linear", name="gradcam_features")(inputs)
    pooled = tf.keras.layers.GlobalAveragePooling2D()(features)
    output = tf.keras.layers.Dense(1, activation="sigmoid", kernel_initializer="ones", bias_initializer="zeros", name="predictions")(pooled)
    model = tf.keras.Model(inputs, output)
    model.trainable = False
    pixels = np.ones((32, 32, 3), np.float32)
    prediction_before = np.asarray(model(pixels[None]))
    positive = generate_gradcam(model, pixels, 1)
    negative = generate_gradcam(model, pixels, 0)
    np.testing.assert_allclose(positive, 1)
    np.testing.assert_allclose(negative, 0)
    np.testing.assert_array_equal(prediction_before, np.asarray(model(pixels[None])))
    paths = save_gradcam(Image.fromarray(pixels.astype(np.uint8)), model, metadata_for(), tmp_path, class_index=0)
    assert set(paths) == {"original", "heatmap", "overlay", "combined"}
    assert all(Path(path).is_file() for path in paths.values())
    np.testing.assert_array_equal(np.asarray(Image.open(paths["original"])), np.asarray(Image.open(paths["overlay"])))


def test_bundle_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="Trained model not found"):
        load_model_bundle(tmp_path / "missing.keras")
    corrupt = tmp_path / "corrupt.keras"
    corrupt.write_bytes(b"not a model")
    with pytest.raises(FileNotFoundError, match="sidecar"):
        load_model_bundle(corrupt)
    metadata = metadata_for()
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (tmp_path / "class_names.json").write_text(json.dumps(metadata["class_names"]), encoding="utf-8")
    with pytest.raises(ValueError, match="Unable to load"):
        load_model_bundle(corrupt)
    (tmp_path / "class_names.json").write_text('["tumor", "no_tumor"]', encoding="utf-8")
    with pytest.raises(ValueError, match="disagree"):
        load_model_bundle(corrupt)


@pytest.mark.parametrize("field,value,message", [
    ("task", [], "task"),
    ("task", {}, "task"),
    ("threshold", 10 ** 400, "threshold"),
])
def test_malformed_metadata_fails_before_model_loading(tmp_path, monkeypatch, field, value, message):
    path = tmp_path / "dummy.keras"
    path.write_bytes(b"validation must reject sidecars before loading these bytes")
    metadata = {**metadata_for(), field: value}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (tmp_path / "class_names.json").write_text(json.dumps(metadata["class_names"]), encoding="utf-8")
    def unexpected_load(*args, **kwargs):
        raise AssertionError("TensorFlow model loading must not run for malformed metadata.")
    monkeypatch.setattr(tf.keras.models, "load_model", unexpected_load)
    with pytest.raises(ValueError, match=message):
        load_model_bundle(path)


def test_prediction_rejects_invalid_probabilities_and_inputs():
    image = Image.new("RGB", (32, 32))
    metadata = metadata_for()
    for value in (-0.1, 1.1, np.nan, np.inf):
        with pytest.raises(ValueError, match="invalid probabilities"):
            predict_image(image, lambda batch, training: np.array([[value]]), metadata)
    with pytest.raises(ValueError, match="threshold"):
        predict_image(image, None, metadata, threshold=2)
    with pytest.raises(ValueError, match="no_tumor=0"):
        predict_image(image, None, {**metadata, "class_names": ["tumor", "no_tumor"]})
    with pytest.raises((ValueError, OSError)):
        predict_image(b"invalid image bytes", custom_cnn.build_model(small_config()), metadata)


def test_threshold_controls_label_and_class_probability():
    fake_model = lambda batch, training: np.array([[0.4]], dtype=np.float32)
    image = Image.new("RGB", (32, 32))
    normal = predict_image(image, fake_model, metadata_for())
    changed = predict_image(image, fake_model, metadata_for(), threshold=0.3)
    assert normal["class_index"] == 0 and normal["confidence"] == pytest.approx(0.6)
    assert changed["class_index"] == 1 and changed["confidence"] == pytest.approx(0.4)
