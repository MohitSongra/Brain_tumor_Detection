"""Verify real saved artifacts, CLI/app agreement, and Grad-CAM after training."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from models.registry import DISPLAY_NAMES, MODEL_NAMES
from src.data_loader import load_manifest
from src.gradcam import save_gradcam
from src.integrity import verify_manifest_integrity
from src.predict import load_model_bundle, predict_image
from src.utils import ROOT, configure_runtime, load_config, resolve_path, write_json


def _require(condition: bool, message: str) -> None:
    """Keep artifact verification active even when Python runs with -O."""
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_effective_config(config: dict, path: Path) -> None:
    """Preserve numeric types when passing the effective configuration to a CLI.

    JSON exponent literals such as 1e-05 can be parsed as strings by PyYAML's
    YAML 1.1 resolver. Its own safe serializer emits compatible float literals.
    """
    public_config = {key: value for key, value in config.items() if not key.startswith("_")}
    path.write_text(yaml.safe_dump(public_config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _verify_evaluation_provenance(
    measured: dict, metadata: dict, model_path: Path, fingerprint: str, sample_count: int
) -> str:
    """Reject stale metrics even when an earlier evaluation had the same size."""
    model_hash = _sha256(model_path)
    required = {
        "trained_on_real_data": True, "n_samples": sample_count,
        "dataset_fingerprint": fingerprint, "model_sha256": model_hash,
        "evaluation_split": "test", "task": metadata["task"],
        "class_names": metadata["class_names"], "threshold": metadata.get("threshold", 0.5),
    }
    if "model_name" in metadata:
        required["model_name"] = metadata["model_name"]
    for field, expected in required.items():
        _require(measured.get(field) == expected,
                 f"Evaluation {field!r} does not match the finalized model/test manifest: {model_path}. Rerun evaluation with the training configuration.")
    return model_hash


def verify(config: dict) -> dict:
    """Use real files and weights; mock only the AppTest upload interaction."""
    configure_runtime(config)
    integrity = verify_manifest_integrity(config)
    frame = load_manifest(config)
    examples = frame[frame.split == "test"].groupby("label", sort=True).head(1)
    _require(not examples.empty, "The audited manifest has no test images to verify.")
    output = resolve_path("results/verification", config)
    output.mkdir(parents=True, exist_ok=True)
    effective_config = output / "effective_config.yaml"
    _write_effective_config(config, effective_config)
    verified = {"dataset_integrity": integrity, "models": {}, "application": {}}
    app_example = None
    for architecture in MODEL_NAMES:
        model_path = resolve_path(config["saved_models_dir"], config) / architecture / "best_model.keras"
        model, metadata = load_model_bundle(model_path)
        _require(metadata.get("model_name") == architecture,
                 f"The {DISPLAY_NAMES[architecture]} bundle declares a different model_name; restore the matching trained model.")
        if not metadata.get("trained_on_real_data") or metadata.get("dataset_fingerprint") != integrity["dataset_fingerprint"]:
            raise ValueError(f"{architecture} is not a real-data model for the current audited manifest.")
        measured = json.loads((resolve_path(config["evaluation_dir"], config) / architecture / "metrics.json").read_text(encoding="utf-8"))
        model_hash = _verify_evaluation_provenance(measured, metadata, model_path,
            integrity["dataset_fingerprint"], int((frame.split == "test").sum()))
        if architecture == "efficientnet":
            published_path = resolve_path(config["model_path"], config)
            _require(published_path.is_file() and _sha256(published_path) == model_hash,
                     "The application model does not match the finalized EfficientNet checkpoint.")
            for sidecar in ("metadata.json", "class_names.json"):
                _require(_sha256(published_path.parent / sidecar) == _sha256(model_path.parent / sidecar),
                         f"The application {sidecar} differs from the finalized EfficientNet bundle.")
        items = []
        for row in examples.itertuples(index=False):
            source = resolve_path(row.path, config)
            result = predict_image(source, model, metadata)
            values = np.asarray(result["probabilities"])
            _require(np.isfinite(values).all() and np.all((values >= 0) & (values <= 1)),
                     f"{architecture} returned invalid probabilities for {source}.")
            _require(np.isclose(values.sum(), 1), f"{architecture} probabilities do not sum to one.")
            visualizations = save_gradcam(source, model, metadata,
                resolve_path(config["gradcam_dir"], config) / architecture / f"true_class_{row.label}", result["class_index"])
            _require(all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in visualizations.values()),
                     f"{architecture} did not save all Grad-CAM visualizations.")
            item = {"image": row.path, "true_label": int(row.label), "prediction": result, "gradcam_saved": True}
            items.append(item)
            if architecture == "efficientnet" and app_example is None:
                app_example = (source, result)
        # Exercise the documented CLI against the first deterministically chosen image.
        source = resolve_path(examples.iloc[0]["path"], config)
        cli_output = output / f"cli_{architecture}"
        command = [sys.executable, str(ROOT / "src/predict.py"), "--image", str(source),
                   "--model", str(model_path), "--config", str(effective_config),
                   "--no-gradcam", "--output-dir", str(cli_output)]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=180)
        (output / f"cli_{architecture}.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"{architecture} CLI failed: {completed.stderr}")
        cli_result = json.loads((cli_output / "prediction.json").read_text(encoding="utf-8"))
        _require(np.allclose(cli_result["probabilities"], items[0]["prediction"]["probabilities"], atol=1e-6),
                 f"{architecture} CLI and shared inference probabilities differ.")
        verified["models"][architecture] = {"status": "passed", "display_name": DISPLAY_NAMES[architecture], "model_sha256": model_hash,
                                            "evaluation_provenance_verified": True,
                                            "cli_agreement": True, "examples": items}
    if app_example is None:
        raise ValueError("No application verification image available.")
    from streamlit.testing.v1 import AppTest
    source, expected = app_example
    payload = source.read_bytes()
    with patch("streamlit.file_uploader", return_value=SimpleNamespace(getvalue=lambda: payload)), \
         patch("src.utils.load_config", return_value=config):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
        if app.exception or not app.button or app.button[0].disabled:
            raise RuntimeError("Trained application did not reach its enabled prediction state.")
        app.button[0].click().run()
        if app.exception or app.error:
            raise RuntimeError(f"Trained application failed: {app.exception} {app.error}")
        actual = app.session_state["prediction"]
        _require(actual["class_name"] == expected["class_name"], "The application and CLI predicted different classes.")
        _require(np.allclose(actual["probabilities"], expected["probabilities"], atol=1e-6),
                 "The application and CLI probabilities differ.")
        _require("visualization" in app.session_state, "The application did not generate a Grad-CAM visualization.")
    verified["application"] = {"status": "passed", "real_model_loaded": True,
                                 "cli_app_probability_agreement": True,
                                 "image": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source)}
    write_json(output / "trained_models.json", verified)
    return verified


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    result = verify(load_config(parser.parse_args().config))
    print(json.dumps({"models": list(result["models"]), "application": result["application"]["status"]}, indent=2))
