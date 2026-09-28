"""Configuration, reproducibility, paths, and artifact utilities."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import platform
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from models.registry import MODEL_NAMES

ROOT = Path(__file__).resolve().parents[1]
DISCLAIMER = (
    "This application is an academic/research prototype for brain MRI image "
    "classification. It is not a medical diagnostic device and should not be "
    "used to make medical decisions. Predictions should be reviewed by "
    "qualified medical professionals."
)


def resolve_path(path: str | Path, config: dict | None = None) -> Path:
    """Resolve relative artifact paths against the configured project root."""
    candidate = Path(path).expanduser()
    root = Path((config or {}).get("_project_root", ROOT))
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load defaults plus an optional YAML override and validate core settings."""
    def read_yaml_mapping(selected: Path) -> dict[str, Any]:
        try:
            with selected.open(encoding="utf-8") as stream:
                data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            location = getattr(exc, "problem_mark", None)
            context = f" at line {location.line + 1}, column {location.column + 1}" if location else ""
            raise ValueError(f"Invalid YAML configuration in {selected}{context}. Check indentation, brackets, and scalar values.") from exc
        except UnicodeError as exc:
            raise ValueError(f"Configuration must be UTF-8 text: {selected}") from exc
        if not isinstance(data, dict) or not all(isinstance(key, str) for key in data):
            raise ValueError(f"Configuration must be a YAML mapping with string field names: {selected}")
        return data

    def finite_number(value: Any, field: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field} must be a finite numeric value, not a string, boolean, or null.")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{field} must be a finite numeric value.") from exc
        if not math.isfinite(number):
            raise ValueError(f"{field} must be a finite numeric value.")
        return number

    def integer(value: Any, field: str, minimum: int = 0) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            description = "positive" if minimum == 1 else "nonnegative"
            raise ValueError(f"{field} must be a {description} integer.")

    def path_string(value: Any, field: str, nullable: bool = False) -> None:
        if nullable and value is None:
            return
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise ValueError(f"{field} must be a nonempty path string" + (" or null." if nullable else "."))

    default_path = ROOT / "config" / "config.yaml"
    if not default_path.is_file():
        raise FileNotFoundError(f"Missing configuration: {default_path}")
    config = read_yaml_mapping(default_path)
    if path is not None:
        if not isinstance(path, (str, Path)) or (isinstance(path, str) and (not path.strip() or "\x00" in path)):
            raise ValueError("Configuration path must be a nonempty path string or Path.")
        selected = resolve_path(path)
        if not selected.is_file():
            raise FileNotFoundError(f"Configuration not found: {selected}")
        override = read_yaml_mapping(selected)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key] = {**config[key], **value}
            else:
                config[key] = value
    for key, allowed in (("task", {"binary", "multiclass"}),
                         ("model_name", set(MODEL_NAMES)),
                         ("device", {"auto", "cpu", "gpu"})):
        if not isinstance(config.get(key), str) or config[key] not in allowed:
            raise ValueError(f"{key} must be one of: {', '.join(sorted(allowed))}.")
    for key in ("raw_dir", "processed_dir", "manifest_path", "train_dir", "validation_dir", "test_dir",
                "saved_models_dir", "model_path", "eda_dir", "training_dir", "evaluation_dir", "gradcam_dir",
                "prediction_dir"):
        path_string(config.get(key), key)
    path_string(config.get("patient_metadata"), "patient_metadata", nullable=True)
    pattern = config.get("patient_id_regex")
    if pattern is not None:
        if not isinstance(pattern, str):
            raise ValueError("patient_id_regex must be a regular-expression string or null.")
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"Invalid patient_id_regex: {exc}") from exc
    path_string(config.get("weights"), "weights", nullable=True)
    for key in ("fine_tune", "class_weights", "deterministic"):
        if not isinstance(config.get(key), bool):
            raise ValueError(f"{key} must be true or false (a YAML boolean).")
    for key in ("image_size", "batch_size", "cpu_batch_size", "epochs", "fine_tune_epochs",
                "dense_units", "finetune_layers", "parallel_calls", "prefetch_batches", "benchmark_batches"):
        integer(config.get(key), key, minimum=1)
    for key in ("random_seed", "early_stopping_patience", "reduce_lr_patience", "intra_op_threads", "inter_op_threads", "phash_distance"):
        integer(config.get(key), key)
    if config["random_seed"] > 2**32 - 1:
        raise ValueError("random_seed must be between 0 and 2**32 - 1 for NumPy reproducibility.")
    if config["image_size"] < 32:
        raise ValueError("image_size must be at least 32 for the transfer-learning backbones.")
    if not 0 <= finite_number(config.get("prediction_threshold"), "prediction_threshold") <= 1:
        raise ValueError("prediction_threshold must be finite and between 0 and 1.")
    for key in ("validation_split", "unsplit_validation_split", "test_split"):
        if not 0 < finite_number(config.get(key), key) < 1:
            raise ValueError(f"{key} must be between 0 and 1, exclusive.")
    if config["unsplit_validation_split"] + config["test_split"] >= 1:
        raise ValueError("Validation and test fractions must leave training data.")
    if not 0 <= finite_number(config.get("dropout"), "dropout") < 1:
        raise ValueError("dropout must be in [0, 1).")
    for key in ("learning_rate", "fine_tune_learning_rate", "max_upload_mb"):
        if finite_number(config.get(key), key) <= 0:
            raise ValueError(f"{key} must be finite and positive.")
    if finite_number(config.get("min_learning_rate"), "min_learning_rate") < 0:
        raise ValueError("min_learning_rate must be finite and nonnegative.")
    if not 0 < finite_number(config.get("reduce_lr_factor"), "reduce_lr_factor") < 1:
        raise ValueError("reduce_lr_factor must be between 0 and 1, exclusive.")
    if not 0 <= config["phash_distance"] <= 16:
        raise ValueError("phash_distance must be between 0 and 16.")
    filters = config.get("cnn_filters")
    if not isinstance(filters, list) or not filters:
        raise ValueError("cnn_filters must be a nonempty list of positive integer filter counts.")
    for count in filters:
        integer(count, "cnn_filters", minimum=1)
    names = config.get("source_classes")
    if (not isinstance(names, list) or len(names) < 2
            or not all(isinstance(name, str) and name.strip() for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("source_classes must be a list of at least two distinct nonempty class names.")
    mapping = config.get("binary_mapping")
    if (not isinstance(mapping, dict) or not mapping
            or not all(isinstance(name, str) and name.strip() for name in mapping)
            or not all(type(label) is int and label in {0, 1} for label in mapping.values())):
        raise ValueError("binary_mapping must map class-name strings to integer 0 or 1; quote YAML keys 'yes' and 'no'.")
    if config["task"] == "binary" and not set(names).issubset(mapping):
        raise ValueError("binary_mapping must define 0 or 1 for every configured source_classes entry.")
    augmentation = config.get("augmentation")
    if not isinstance(augmentation, dict):
        raise ValueError("augmentation must be a YAML mapping of augmentation settings.")
    if not isinstance(augmentation.get("horizontal_flip"), bool):
        raise ValueError("augmentation.horizontal_flip must be true or false (a YAML boolean).")
    for key, maximum in (("rotation_degrees", 360), ("zoom", 1), ("translation", 1), ("contrast", 1)):
        if not 0 <= finite_number(augmentation.get(key), f"augmentation.{key}") <= maximum:
            raise ValueError(f"augmentation.{key} must be between 0 and {maximum}.")
    get_class_names(config)
    config["_project_root"] = str(ROOT)
    return config


def get_class_names(config: dict) -> list[str]:
    """Return the fixed output order, independent of folder enumeration order."""
    if config.get("task", "binary") == "binary":
        return ["no_tumor", "tumor"]
    names = config.get("source_classes", [])
    if (not isinstance(names, list) or len(names) < 2
            or not all(isinstance(x, str) and x.strip() for x in names)
            or len(set(names)) != len(names)):
        raise ValueError("source_classes must contain at least two distinct class names.")
    return list(names)


def write_json(path: str | Path, data: Any) -> None:
    """Write UTF-8 JSON atomically and refuse non-standard NaN values."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False,
                                    default=lambda value: value.item() if isinstance(value, np.generic) else str(value)),
                         encoding="utf-8")
    temporary.replace(destination)


def manifest_fingerprint(path: str | Path) -> str:
    """Fingerprint the exact audited split manifest used by an experiment."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup_logging(log_file: str | Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=handlers, force=True)


def configure_runtime(config: dict) -> dict[str, Any]:
    """Set seeds and resource limits; use small batches without an available GPU."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if config.get("device") == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import tensorflow as tf

    seed = int(config.get("random_seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(int(config.get("intra_op_threads", 8)))
        tf.config.threading.set_inter_op_parallelism_threads(int(config.get("inter_op_threads", 2)))
    except RuntimeError:
        logging.debug("TensorFlow was already initialized; retaining existing thread limits.")
    if config.get("device") == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError:
            if tf.config.get_visible_devices("GPU"):
                raise RuntimeError("Restart this process to switch an initialized GPU to CPU.")
    gpus = tf.config.get_visible_devices("GPU")
    if config.get("device") == "gpu" and not gpus:
        raise RuntimeError("GPU requested, but TensorFlow cannot access a GPU. Use device: cpu/auto or configure Linux/WSL.")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    if config.get("deterministic", True):
        tf.config.experimental.enable_op_determinism()
    runtime = {
        "python": platform.python_version(), "tensorflow": tf.__version__,
        "keras": tf.keras.__version__, "platform": platform.platform(),
        "processor": platform.processor(), "device": "gpu" if gpus else "cpu",
        "gpu_devices": [gpu.name for gpu in gpus],
        "batch_size": int(config["batch_size"] if gpus else config["cpu_batch_size"]),
        "random_seed": seed,
    }
    logging.info("Runtime: %s; batch size %d", runtime["device"], runtime["batch_size"])
    return runtime


def select_best_phase(phases: list[dict]) -> dict:
    """Select lowest validation loss, preferring the earlier phase on ties."""
    eligible = [phase for phase in phases if np.isfinite(phase["best_val_loss"])]
    if not eligible:
        raise ValueError("Training produced no finite validation loss.")
    return min(eligible, key=lambda phase: phase["best_val_loss"])
