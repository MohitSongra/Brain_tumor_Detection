"""Class-specific Grad-CAM visualizations; an aid, not a medical explanation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf

from src.preprocessing import load_rgb_image, preprocess_image
from src.utils import configure_runtime, load_config, resolve_path, setup_logging

LOGGER = logging.getLogger(__name__)


def generate_gradcam(
    model: tf.keras.Model,
    image_array: np.ndarray,
    class_index: int,
    layer_name: str = "gradcam_features",
) -> np.ndarray:
    """Return a finite, normalized 2-D map without changing model weights/activation.

    Gradients target pre-activation logits. For a sigmoid classifier, tumor uses
    logit z and no-tumor uses -z. An all-zero result means this method did not
    produce a positive class-support map, not that the image has no tumor.
    """
    array = np.asarray(image_array, dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] != 3 or not np.isfinite(array).all():
        raise ValueError("Grad-CAM expects one finite H×W×3 RGB image array.")
    try:
        spatial = model.get_layer(layer_name)
        classifier = model.get_layer("predictions")
    except ValueError as exc:
        raise ValueError("Model does not expose the expected Grad-CAM features and classifier layers.") from exc
    if not isinstance(classifier, tf.keras.layers.Dense) or len(spatial.output.shape) != 4:
        raise ValueError("Grad-CAM requires spatial feature maps and a Dense predictions layer.")
    units = int(classifier.units)
    classes = 2 if units == 1 else units
    if isinstance(class_index, bool) or not isinstance(class_index, (int, np.integer)) or not 0 <= class_index < classes:
        raise ValueError(f"class_index must be an integer between 0 and {classes - 1}.")
    gradient_model = tf.keras.Model(model.inputs, [spatial.output, classifier.input])
    tensor = tf.convert_to_tensor(array[None, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        # Explicitly watching inputs also supports fully frozen loaded models.
        tape.watch(tensor)
        features, head = gradient_model(tensor, training=False)
        logits = tf.linalg.matmul(head, classifier.kernel)
        if classifier.use_bias:
            logits = tf.nn.bias_add(logits, classifier.bias)
        score = logits[:, 0] * (1.0 if class_index == 1 else -1.0) if units == 1 else logits[:, class_index]
    gradients = tape.gradient(score, features)
    if gradients is None:
        raise ValueError("Grad-CAM gradients are disconnected from the selected feature layer.")
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.nn.relu(tf.reduce_sum(features * weights, axis=-1))[0]
    heatmap = tf.math.divide_no_nan(heatmap, tf.reduce_max(heatmap))
    result = np.asarray(heatmap, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Grad-CAM produced non-finite values.")
    return np.clip(result, 0, 1)


def render_gradcam(
    source: Any,
    model: tf.keras.Model,
    metadata: dict[str, Any],
    class_index: int | None = None,
) -> dict[str, Any]:
    """Render original/heatmap/overlay PIL images for shared UI and file output."""
    from src.predict import predict_image
    original = load_rgb_image(source)
    if class_index is None:
        class_index = predict_image(original, model, metadata)["class_index"]
    array = preprocess_image(original, image_size=metadata["image_size"])
    heatmap = generate_gradcam(model, array, class_index, layer_name=metadata.get("gradcam_layer", "gradcam_features"))
    resized = cv2.resize(heatmap, original.size, interpolation=cv2.INTER_LINEAR)
    colored = cv2.cvtColor(cv2.applyColorMap(np.uint8(np.clip(resized * 255, 0, 255)), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    rgb = np.asarray(original)
    # Opacity follows attribution strength, so zero attribution leaves the image intact.
    alpha = 0.4 * resized[..., None]
    overlay = np.uint8(np.clip((1 - alpha) * rgb + alpha * colored, 0, 255))
    return {"original": original, "heatmap": Image.fromarray(colored),
            "overlay": Image.fromarray(overlay), "zero_map": bool(float(heatmap.max()) == 0)}


def save_gradcam(
    source: Any,
    model: tf.keras.Model,
    metadata: dict[str, Any],
    output_dir: str | Path,
    class_index: int | None = None,
) -> dict[str, str]:
    """Save original, heatmap, overlay, and a labeled comparison figure."""
    from src.predict import predict_image
    original = load_rgb_image(source)
    if class_index is None:
        class_index = predict_image(original, model, metadata)["class_index"]
    rendered = render_gradcam(original, model, metadata, class_index)
    output = resolve_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {name: str(output / f"{name}.png") for name in ("original", "heatmap", "overlay", "combined")}
    for name in ("original", "heatmap", "overlay"):
        rendered[name].save(paths[name])
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.4), constrained_layout=True)
    for axis, values, title in zip(axes, (rendered["original"], rendered["heatmap"], rendered["overlay"]), ("Original MRI", "Grad-CAM heatmap", "Overlay")):
        axis.imshow(values)
        axis.set_title(title)
        axis.axis("off")
    label = metadata["class_names"][class_index].replace("_", " ")
    note = "No positive attribution map was produced. " if rendered["zero_map"] else ""
    figure.suptitle(f"Explained class: {label}\n{note}Interpretability aid; not a medically validated localization.", fontsize=11)
    figure.savefig(paths["combined"], dpi=160)
    plt.close(figure)
    if rendered["zero_map"]:
        LOGGER.warning("Grad-CAM produced a zero map; this is not evidence of tumor absence.")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM for a trained MRI classifier.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model")
    parser.add_argument("--config")
    parser.add_argument("--class-index", type=int)
    parser.add_argument("--output-dir", help="Defaults to configured gradcam_dir.")
    args = parser.parse_args()
    setup_logging()
    try:
        from src.predict import load_model_bundle
        config = load_config(args.config)
        configure_runtime(config)
        model, metadata = load_model_bundle(args.model or config.get("model_path", "models/saved/best_model.keras"))
        paths = save_gradcam(resolve_path(args.image), model, metadata, args.output_dir or config.get("gradcam_dir", "results/gradcam"), args.class_index)
        LOGGER.info("Saved Grad-CAM comparison to %s", paths["combined"])
    except (FileNotFoundError, ValueError, RuntimeError, OSError, tf.errors.OpError) as exc:
        LOGGER.error("Grad-CAM failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
