"""Lightweight architecture registry shared by configuration and command tools.

Importing this module does not initialize TensorFlow. Model modules are imported
only when a caller explicitly requests an architecture.
"""
from __future__ import annotations

import importlib
from types import ModuleType

MODEL_NAMES = ("custom_cnn", "efficientnet", "mobilenet")
DISPLAY_NAMES = {
    "custom_cnn": "Custom CNN",
    "efficientnet": "EfficientNetB0",
    "mobilenet": "MobileNetV2",
}
_MODULES = {
    "custom_cnn": "models.custom_cnn",
    "efficientnet": "models.efficientnet_model",
    "mobilenet": "models.mobilenet_model",
}


def _validate_name(name: str) -> None:
    if not isinstance(name, str) or name not in MODEL_NAMES:
        raise ValueError(f"Unknown model name {name!r}. Choose from: {', '.join(MODEL_NAMES)}.")


def get_model_module(name: str) -> ModuleType:
    """Import the selected model implementation, rejecting unsupported names."""
    _validate_name(name)
    return importlib.import_module(_MODULES[name])


def is_transfer_model(name: str) -> bool:
    """Return whether a registered architecture uses an ImageNet backbone."""
    _validate_name(name)
    return name != "custom_cnn"
