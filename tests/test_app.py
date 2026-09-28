"""Streamlit behavior tests; all uploads and model fixtures remain temporary."""
from __future__ import annotations

import io
from types import SimpleNamespace

from PIL import Image
import streamlit as st
from streamlit.testing.v1 import AppTest

from models.custom_cnn import build_model
from src import utils
from src.utils import DISCLAIMER, ROOT


def app_config(monkeypatch, tmp_path):
    config = utils.load_config()
    config.update(model_path=str(tmp_path / "test_model.keras"), image_size=32)
    monkeypatch.setattr(utils, "load_config", lambda: config)
    monkeypatch.setattr(utils, "configure_runtime", lambda config: {"device": "cpu"})
    return config


def upload_bytes(color="gray"):
    stream = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(stream, format="PNG")
    return stream.getvalue()


def test_missing_model_keeps_disclaimer_and_disables_prediction(monkeypatch, tmp_path):
    app_config(monkeypatch, tmp_path)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert DISCLAIMER in [element.value for element in app.info]
    assert any("Training required" in element.value for element in app.warning)
    assert app.button[0].disabled


def test_corrupt_upload_reports_useful_error(monkeypatch, tmp_path):
    app_config(monkeypatch, tmp_path)
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: SimpleNamespace(getvalue=lambda: b"invalid PNG"))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert any("Cannot read this image" in element.value for element in app.error)
    assert app.button[0].disabled
    assert DISCLAIMER in [element.value for element in app.info]


def test_configuration_error_keeps_disclaimer(monkeypatch, tmp_path):
    app_config(monkeypatch, tmp_path)
    def missing():
        raise FileNotFoundError("Missing test configuration")
    monkeypatch.setattr(utils, "load_config", missing)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert any("Configuration could not be loaded" in element.value for element in app.error)
    assert DISCLAIMER in [element.value for element in app.info]


def test_prediction_flow_matches_shared_inference_and_clears_stale_upload(monkeypatch, tmp_path):
    from pathlib import Path
    from src import predict
    config = app_config(monkeypatch, tmp_path)
    config.update(cnn_filters=[4, 8, 16], dense_units=8)
    Path(config["model_path"]).write_bytes(b"mock readiness marker, not a production model")
    model = build_model(config)
    # Mock a trained bundle contract solely to exercise the rendering branch;
    # this random model is never saved, published, or reported as research data.
    metadata = {"task": "binary", "class_names": ["no_tumor", "tumor"],
                "image_size": 32, "input_range": [0, 255], "threshold": 0.5,
                "model_name": "isolated_test_fixture", "trained_on_real_data": True,
                "gradcam_layer": "gradcam_features"}
    monkeypatch.setattr(predict, "load_model_bundle", lambda path: (model, metadata))
    current = {"payload": upload_bytes()}
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: SimpleNamespace(getvalue=lambda: current["payload"]))
    expected = predict.predict_image(current["payload"], model, metadata)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception and not app.button[0].disabled
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert len(app.metric) == 2
    assert app.metric[1].value == f"{expected['confidence']:.2%}"
    assert app.session_state["prediction"]["probabilities"] == expected["probabilities"]
    # A replaced checkpoint must clear old visible predictions before the next click.
    Path(config["model_path"]).write_bytes(b"replaced mock readiness marker")
    app.run()
    assert not app.exception and len(app.metric) == 0
    app.button[0].click().run()
    assert not app.exception and len(app.metric) == 2
    current["payload"] = upload_bytes("white")
    app.run()
    assert not app.exception and len(app.metric) == 0
    assert DISCLAIMER in [element.value for element in app.info]
