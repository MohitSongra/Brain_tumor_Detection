"""MobileNetV2 checks use temporary random-weight models, never research outputs."""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import tensorflow as tf

from models import mobilenet_model
from models.registry import DISPLAY_NAMES, MODEL_NAMES, get_model_module, is_transfer_model
from src.gradcam import generate_gradcam
from src.predict import load_model_bundle, predict_image
from src.utils import ROOT


def small_config(task="binary"):
    return {"task": task, "image_size": 32, "dense_units": 8, "dropout": 0.1,
            "learning_rate": 1e-3, "fine_tune_learning_rate": 1e-5, "finetune_layers": 20,
            "source_classes": ["glioma", "meningioma", "notumor", "pituitary"]}


@pytest.mark.parametrize("task", ["binary", "multiclass"])
def test_mobilenet_preprocessing_prediction_reload_and_gradcam(tmp_path, task):
    tf.keras.backend.clear_session()
    config = small_config(task)
    model = mobilenet_model.build_model(config, weights=None)
    base = model.get_layer("mobilenetv2")
    assert not base.trainable and not base.trainable_weights
    preprocessing = model.get_layer("mobilenet_preprocessing")
    assert isinstance(preprocessing, tf.keras.layers.Rescaling)
    pixels = np.array([0, 127.5, 255], dtype=np.float32).reshape(1, 1, 1, 3)
    np.testing.assert_allclose(preprocessing(pixels).numpy().reshape(-1), [-1, 0, 1], atol=1e-6)
    image = np.random.default_rng(42).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    before = np.asarray(model(image[None].astype(np.float32), training=False))
    assert before.shape == (1, 1 if task == "binary" else 4)
    assert np.isfinite(before).all() and ((before >= 0) & (before <= 1)).all()
    classes = ["no_tumor", "tumor"] if task == "binary" else config["source_classes"]
    metadata = {"task": task, "class_names": classes, "image_size": 32,
                "input_range": [0, 255], "threshold": 0.5, "model_name": "mobilenet",
                "gradcam_layer": "gradcam_features", "trained_on_real_data": False}
    model_path = tmp_path / "best_model.keras"
    model.save(model_path)
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (tmp_path / "class_names.json").write_text(json.dumps(classes), encoding="utf-8")
    restored, restored_metadata = load_model_bundle(model_path)
    after = np.asarray(restored(image[None].astype(np.float32), training=False))
    np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-6)
    prediction = predict_image(Image.fromarray(image), restored, restored_metadata)
    assert 0 <= prediction["confidence"] <= 1
    assert len(prediction["probabilities"]) == len(classes)
    assert sum(prediction["probabilities"]) == pytest.approx(1, abs=1e-5)
    heatmap = generate_gradcam(restored, image, prediction["class_index"])
    assert heatmap.ndim == 2 and np.isfinite(heatmap).all()
    assert heatmap.min() >= 0 and heatmap.max() <= 1
    np.testing.assert_array_equal(after, np.asarray(restored(image[None].astype(np.float32), training=False)))


def test_mobilenet_finetuning_keeps_batchnorm_frozen(tmp_path):
    config = small_config()
    model = mobilenet_model.build_model(config, weights=None)
    mobilenet_model.unfreeze_for_finetuning(model, config)
    base = model.get_layer("mobilenetv2")
    assert base.trainable and base.trainable_weights
    assert all(not layer.trainable for layer in base.layers[:-20])
    batch_norm = [layer for layer in base.layers if isinstance(layer, tf.keras.layers.BatchNormalization)]
    assert all(not layer.trainable for layer in batch_norm)
    assert float(model.optimizer.learning_rate.numpy()) == pytest.approx(1e-5)
    before = [layer.moving_mean.numpy().copy() for layer in batch_norm]
    model(np.full((1, 32, 32, 3), 100, dtype=np.float32), training=True)
    for expected, layer in zip(before, batch_norm):
        np.testing.assert_array_equal(expected, layer.moving_mean.numpy())
    path = tmp_path / "finetuned_fixture.keras"
    model.save(path)
    restored = tf.keras.models.load_model(path, compile=False, safe_mode=True)
    assert all(not layer.trainable for layer in restored.get_layer("mobilenetv2").layers
               if isinstance(layer, tf.keras.layers.BatchNormalization))
    with pytest.raises(ValueError, match="full-backbone"):
        mobilenet_model.unfreeze_for_finetuning(model, {**config, "finetune_layers": len(base.layers)})


def test_registry_exposes_three_algorithms_and_rejects_invalid_names():
    assert MODEL_NAMES == ("custom_cnn", "efficientnet", "mobilenet")
    assert set(DISPLAY_NAMES) == set(MODEL_NAMES)
    assert get_model_module("mobilenet") is mobilenet_model
    assert is_transfer_model("mobilenet") and is_transfer_model("efficientnet")
    assert not is_transfer_model("custom_cnn")
    for name in ("unknown", [], None):
        with pytest.raises(ValueError, match="Unknown model"):
            get_model_module(name)
        with pytest.raises(ValueError, match="Unknown model"):
            is_transfer_model(name)


def test_registry_import_does_not_initialize_tensorflow():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import models.registry; assert 'tensorflow' not in sys.modules"],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
