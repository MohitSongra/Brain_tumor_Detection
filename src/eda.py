"""Generate reproducible exploratory figures from the audited dataset manifest."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.data_loader import load_manifest
from src.preprocessing import build_augmentation, load_rgb_image, preprocess_image
from src.utils import configure_runtime, get_class_names, load_config, resolve_path, setup_logging, write_json

LOGGER = logging.getLogger(__name__)


def run_eda(config: dict) -> dict:
    """Save counts, representative images, dimensions, intensities, and augmentation.

    Pixel histograms sample up to 200 images to avoid loading the entire dataset.
    All image examples are drawn from training; test images are not inspected for
    model-development decisions.
    """
    configure_runtime(config)
    frame = load_manifest(config)
    training = frame.loc[frame["split"] == "train"]
    if training.empty:
        raise ValueError("The training split is empty; EDA needs prepared training images.")
    output = resolve_path(config.get("eda_dir", "results/eda"), config)
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="colorblind")
    paths = {}

    def save(fig, name):
        path = output / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths[name] = str(path)

    classes = get_class_names(config)
    frame["display_class"] = frame["label"].map(dict(enumerate(classes)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(data=frame, x="display_class", order=classes, hue="display_class", legend=False, ax=axes[0])
    axes[0].set(title="Included images by output class", xlabel="Class", ylabel="Images")
    counts = frame.groupby(["split", "display_class"]).size().unstack(fill_value=0).reindex(["train", "validation", "test"])
    counts.plot.bar(ax=axes[1], rot=0)
    axes[1].set(title="Audited split distribution", xlabel="Split", ylabel="Images")
    fig.tight_layout()
    save(fig, "class_distribution")

    source_classes = sorted(training["source_class"].unique())
    fig, axes = plt.subplots(len(source_classes), 4, figsize=(10, 2.7 * len(source_classes)), squeeze=False)
    for row, source_class in enumerate(source_classes):
        available = training.loc[training["source_class"] == source_class]
        samples = available.sample(min(4, len(available)), random_state=int(config.get("random_seed", 42)))
        for col, axis in enumerate(axes[row]):
            axis.axis("off")
            if col < len(samples):
                axis.imshow(load_rgb_image(resolve_path(samples.iloc[col]["path"], config)))
                axis.set_title(source_class.replace("_", " "))
    fig.suptitle("Training MRI examples from the audited manifest")
    fig.tight_layout()
    save(fig, "sample_mri_grid")

    if {"width", "height"}.issubset(frame.columns):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        sns.scatterplot(data=frame, x="width", y="height", hue="split", alpha=0.35, s=15, ax=axes[0])
        axes[0].set_title("Original image dimensions")
        axes[1].hist([frame["width"], frame["height"]], bins=25, label=["Width", "Height"])
        axes[1].set(title="Image size distribution", xlabel="Pixels", ylabel="Images")
        axes[1].legend()
        fig.tight_layout()
        save(fig, "image_dimensions")

    fig, axis = plt.subplots(figsize=(8, 4))
    histogram = np.zeros(256, dtype=np.int64)
    intensity_sample = training.sample(min(200, len(training)), random_state=int(config.get("random_seed", 42)))
    for row in intensity_sample.itertuples(index=False):
        grayscale = np.asarray(load_rgb_image(resolve_path(row.path, config)).convert("L"))
        histogram += np.bincount(grayscale.ravel(), minlength=256)
    axis.plot(np.arange(256), histogram / max(int(histogram.sum()), 1))
    axis.set(title=f"Training pixel intensity distribution ({len(intensity_sample)} sampled images)", xlabel="Grayscale pixel intensity", ylabel="Pixel fraction")
    save(fig, "pixel_intensity")

    example = preprocess_image(resolve_path(training.iloc[0]["path"], config), int(config.get("image_size", 224)))
    augmentation = build_augmentation(config)
    fig, axes = plt.subplots(1, 5, figsize=(13, 3))
    axes[0].imshow(example.astype(np.uint8))
    axes[0].set_title("Resized original")
    for index, axis in enumerate(axes[1:], start=1):
        augmented = augmentation(example[None], training=True).numpy()[0]
        axis.imshow(np.clip(augmented, 0, 255).astype(np.uint8))
        axis.set_title(f"Training variant {index}")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    save(fig, "augmentation_examples")
    summary = {"total_images": len(frame), "source_classes": frame["source_class"].value_counts().to_dict(),
               "output_classes": frame["display_class"].value_counts().to_dict(),
               "split_counts": frame["split"].value_counts().to_dict(),
               "split_class_counts": counts.to_dict(orient="index"), "plots": paths,
               "intensity_histogram_sample_size": len(intensity_sample),
               "note": "Representative images, intensity histograms, and augmentations use training data only."}
    write_json(output / "summary.json", summary)
    frame.groupby(["split", "source_class"]).size().rename("images").to_csv(output / "class_counts.csv")
    LOGGER.info("Saved EDA figures and counts to %s", output)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    setup_logging()
    try:
        run_eda(load_config(args.config))
    except (FileNotFoundError, ValueError, OSError) as exc:
        LOGGER.error("EDA failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
