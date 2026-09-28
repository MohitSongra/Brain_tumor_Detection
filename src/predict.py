"""Shared, metadata-validated inference for the CLI and Streamlit application."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import tensorflow as tf

from src.preprocessing import preprocess_image
from src.utils import configure_runtime, load_config, resolve_path, setup_logging, write_json

LOGGER = logging.getLogger(__name__)


def _validate_metadata(metadata: dict[str, Any], class_names: Any) -> None:
    """Reject ambiguous preprocessing and label contracts before inference."""
    if not isinstance(class_names, list) or len(class_names) < 2 or not all(isinstance(name, str) and name for name in class_names):
        raise ValueError("class_names.json must contain an ordered list of at least two class names.")
    if len(set(class_names)) != len(class_names):
        raise ValueError("Class names must be unique.")
    if metadata.get("class_names") != class_names:
        raise ValueError("Class names in metadata.json and class_names.json disagree.")
    task = metadata.get("task")
    if not isinstance(task, str) or task not in {"binary", "multiclass"}:
        raise ValueError("Model metadata must declare task: binary or multiclass.")
    if metadata["task"] == "binary" and class_names != ["no_tumor", "tumor"]:
        raise ValueError("Binary models require the mapping no_tumor=0, tumor=1.")
    if metadata.get("input_range") != [0, 255]:
        raise ValueError("Unsupported model input_range; expected [0,255] RGB pixels.")
    size = metadata.get("image_size")
    if isinstance(size, bool) or not isinstance(size, int) or size < 32:
        raise ValueError("Model metadata image_size must be an integer of at least 32.")
    _validate_threshold(metadata.get("threshold", 0.5))


def _validate_threshold(value: Any) -> float:
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Prediction threshold must be a number in [0,1].") from exc
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError("Prediction threshold must be a finite number in [0,1].")
    return result


def load_model_bundle(model_path: str | Path) -> tuple[tf.keras.Model, dict[str, Any]]:
    """Load a .keras model and its required metadata/ordered-class sidecars.

    Saved metadata, rather than the current training configuration, controls the
    preprocessing and class order used for inference.
    """
    path = resolve_path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Trained model not found: {path}. Run python src/train.py first.")
    if path.suffix.lower() != ".keras":
        raise ValueError("Expected a .keras model file.")
    metadata_path = path.parent / "metadata.json"
    names_path = path.parent / "class_names.json"
    for sidecar in (metadata_path, names_path):
        if not sidecar.is_file():
            raise FileNotFoundError(f"Missing model sidecar: {sidecar}. Use a complete training output bundle.")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        class_names = json.loads(names_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError("Model metadata is not valid UTF-8 JSON.") from exc
    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must contain a JSON object.")
    _validate_metadata(metadata, class_names)
    try:
        model = tf.keras.models.load_model(path, compile=False, safe_mode=True)
    except Exception as exc:
        raise ValueError(f"Unable to load model {path.name}: {exc}") from exc
    expected_shape = (None, metadata["image_size"], metadata["image_size"], 3)
    if tuple(model.input_shape) != expected_shape:
        raise ValueError(f"Model input shape {model.input_shape} disagrees with metadata {expected_shape}.")
    expected_outputs = 1 if metadata["task"] == "binary" else len(class_names)
    if tuple(model.output_shape) != (None, expected_outputs):
        raise ValueError("Model output shape disagrees with its task and ordered class names.")
    return model, metadata


def predict_image(
    source: Any,
    model: tf.keras.Model,
    metadata: dict[str, Any],
    threshold: float | None = None,
) -> dict[str, Any]:
    """Predict one image; confidence is the uncalibrated selected-class probability."""
    _validate_metadata(metadata, metadata.get("class_names"))
    cutoff = _validate_threshold(metadata.get("threshold", 0.5) if threshold is None else threshold)
    array = preprocess_image(source, image_size=metadata["image_size"])
    output = np.asarray(model(np.expand_dims(array, axis=0), training=False), dtype=np.float64)
    classes = metadata["class_names"]
    expected = (1, 1 if metadata["task"] == "binary" else len(classes))
    if output.shape != expected:
        raise ValueError(f"Unexpected prediction shape {output.shape}; expected {expected}.")
    if not np.isfinite(output).all() or np.any(output < -1e-7) or np.any(output > 1 + 1e-7):
        raise ValueError("Model returned invalid probabilities; all outputs must be finite and in [0,1].")
    output = np.clip(output, 0, 1)
    if metadata["task"] == "binary":
        tumor = float(output[0, 0])
        probabilities = [1.0 - tumor, tumor]
        index = int(tumor >= cutoff)
    else:
        probabilities = output[0].tolist()
        if not np.isclose(sum(probabilities), 1.0, atol=1e-5):
            raise ValueError("Multiclass probabilities must sum to one.")
        index = int(np.argmax(probabilities))
    return {
        "class_name": classes[index],
        "class_index": index,
        "confidence": float(probabilities[index]),
        "probabilities": probabilities,
        "threshold": cutoff,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify one MRI image with a trained academic prototype.")
    parser.add_argument("--image", required=True, help="Path to a JPEG, PNG, or BMP image.")
    parser.add_argument("--model", help="Path to a .keras model with metadata sidecars.")
    parser.add_argument("--config", help="Path to YAML configuration.")
    parser.add_argument("--threshold", type=float, help="Binary threshold override in [0,1].")
    parser.add_argument("--output-dir", help="Directory for prediction JSON and visualizations; defaults to configured prediction_dir.")
    parser.add_argument("--no-gradcam", action="store_true", help="Skip interpretation image generation.")
    args = parser.parse_args()
    setup_logging()
    try:
        config = load_config(args.config)
        configure_runtime(config)
        model, metadata = load_model_bundle(args.model or config.get("model_path", "models/saved/best_model.keras"))
        source = resolve_path(args.image)
        result = predict_image(source, model, metadata, threshold=args.threshold)
        output = resolve_path(args.output_dir or config.get("prediction_dir", "results/prediction"), config)
        output.mkdir(parents=True, exist_ok=True)
        result["image"] = str(source)
        result["class_names"] = metadata["class_names"]
        if not args.no_gradcam:
            from src.gradcam import save_gradcam
            result["gradcam"] = save_gradcam(source, model, metadata, output, class_index=result["class_index"])
        write_json(output / "prediction.json", result)
        label = result["class_name"].replace("_", " ").title()
        print(f"Prediction: {label}\nConfidence: {result['confidence']:.2%} (uncalibrated model probability)")
        print("This application is an academic/research prototype for brain MRI image classification. It is not a medical diagnostic device and should not be used to make medical decisions. Predictions should be reviewed by qualified medical professionals.")
        LOGGER.info("Saved prediction to %s", output / "prediction.json")
    except (FileNotFoundError, ValueError, RuntimeError, OSError, tf.errors.OpError) as exc:
        LOGGER.error("Prediction failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
