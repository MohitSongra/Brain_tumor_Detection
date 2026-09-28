"""Evaluate an actual saved classifier on the audited test manifest.

Undefined statistics are JSON null, never invented or replaced by perfect scores.
Binary statistics use tumor (class 1) as the positive class. Multiclass summary
statistics are macro averages; per-class and support-weighted values are included.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import logging
import os
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve

from models.registry import MODEL_NAMES
from src.utils import ROOT, get_class_names, load_config, manifest_fingerprint, resolve_path, setup_logging, write_json

LOGGER = logging.getLogger(__name__)
METRIC_NAMES = ("accuracy", "precision", "recall", "sensitivity", "specificity", "f1", "roc_auc")


def _divide(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _average(values: list[float | None], weights: list[int] | None = None) -> float | None:
    """Do not silently drop undefined supported classes from an average."""
    if weights is None:
        return None if any(v is None for v in values) else float(np.mean(values))
    supported = [(value, weight) for value, weight in zip(values, weights) if weight > 0]
    if not supported or any(value is None for value, _ in supported):
        return None
    return float(sum(value * weight for value, weight in supported) / sum(weight for _, weight in supported))


def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    threshold: float = 0.5,
    task: str = "binary",
) -> dict[str, Any]:
    """Calculate reproducible classification metrics without loading TensorFlow.

    Binary probabilities must have shape (N,) or (N, 1). Multiclass inputs must
    have shape (N, K), sum to one, and use the recorded class order.
    """
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or not len(labels):
        raise ValueError("Evaluation requires a nonempty one-dimensional label array.")
    if len(class_names) < 2 or len(set(class_names)) != len(class_names):
        raise ValueError("At least two distinct class names are required.")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Labels must be integer class indices.")
    if np.any(labels < 0) or np.any(labels >= len(class_names)):
        raise ValueError("Evaluation label is outside the recorded class mapping.")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Prediction probabilities must be finite and between 0 and 1.")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Prediction threshold must be between 0 and 1, inclusive.")
    if task == "binary":
        if len(class_names) != 2 or probabilities.shape not in {(len(labels),), (len(labels), 1)}:
            raise ValueError("Binary evaluation expects two classes and probabilities of shape (N,) or (N, 1).")
        positive = probabilities.reshape(-1)
        matrix = np.column_stack((1.0 - positive, positive))
        predicted = (positive >= threshold).astype(int)
    elif task == "multiclass":
        if probabilities.shape != (len(labels), len(class_names)):
            raise ValueError("Multiclass probability shape must match sample count and class mapping.")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError("Multiclass probabilities must sum to 1 for each image.")
        matrix = probabilities
        predicted = probabilities.argmax(axis=1)
    else:
        raise ValueError("Task must be 'binary' or 'multiclass'.")

    confusion = confusion_matrix(labels, predicted, labels=np.arange(len(class_names)))
    per_class: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(class_names):
        tp = int(confusion[index, index])
        fn = int(confusion[index, :].sum() - tp)
        fp = int(confusion[:, index].sum() - tp)
        tn = int(confusion.sum() - tp - fn - fp)
        one_vs_rest = (labels == index).astype(int)
        auc = float(roc_auc_score(one_vs_rest, matrix[:, index])) if len(np.unique(one_vs_rest)) == 2 else None
        per_class[name] = {
            "precision": _divide(tp, tp + fp), "recall": _divide(tp, tp + fn),
            "sensitivity": _divide(tp, tp + fn), "specificity": _divide(tn, tn + fp),
            "f1": _divide(2 * tp, 2 * tp + fp + fn), "roc_auc": auc,
            "support": tp + fn, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        }
    averaged_names = METRIC_NAMES[1:]
    supports = [per_class[name]["support"] for name in class_names]
    macro = {metric: _average([per_class[name][metric] for name in class_names]) for metric in averaged_names}
    weighted = {
        metric: _average([per_class[name][metric] for name in class_names], supports)
        for metric in averaged_names
    }
    selected = per_class[class_names[1]] if task == "binary" else macro
    return {
        "accuracy": float(np.mean(labels == predicted)),
        **{metric: selected[metric] for metric in averaged_names},
        "task": task, "class_names": list(class_names), "threshold": float(threshold),
        "n_samples": int(len(labels)), "positive_class": class_names[1] if task == "binary" else None,
        "averaging": "positive_class" if task == "binary" else "macro",
        "per_class": per_class, "macro": macro, "weighted": weighted,
        "confusion_matrix": confusion.tolist(),
        "undefined_metric_policy": "null; macro is null if any class is undefined; weighted ignores only zero-support classes",
    }


def _plot_evaluation(labels: np.ndarray, matrix: np.ndarray, metrics: dict[str, Any], output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    names = metrics["class_names"]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(np.asarray(metrics["confusion_matrix"]), annot=True, fmt="d", cmap="Blues",
                xticklabels=names, yticklabels=names, ax=ax, cbar=False)
    ax.set(xlabel="Predicted class", ylabel="True class", title="Test confusion matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    indices = [1] if metrics["task"] == "binary" else list(range(len(names)))
    for curve_type in ("roc", "precision_recall"):
        fig, ax = plt.subplots(figsize=(6.5, 5.0))
        available = False
        for index in indices:
            target = (labels == index).astype(int)
            if len(np.unique(target)) < 2:
                continue
            available = True
            if curve_type == "roc":
                fpr, tpr, _ = roc_curve(target, matrix[:, index])
                auc = metrics["per_class"][names[index]]["roc_auc"]
                ax.plot(fpr, tpr, label=f"{names[index]} (AUC {auc:.3f})")
            else:
                precision, recall, _ = precision_recall_curve(target, matrix[:, index])
                ax.plot(recall, precision, label=names[index])
        if curve_type == "roc":
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
            ax.set(xlabel="False positive rate", ylabel="Sensitivity / true positive rate", title="Test ROC curve (one-vs-rest)")
        else:
            ax.set(xlabel="Recall / sensitivity", ylabel="Precision", title="Test precision–recall curve (one-vs-rest)")
        if available:
            ax.legend(loc="best")
        else:
            ax.text(0.5, 0.5, "Not available: test labels need both classes", ha="center", va="center", transform=ax.transAxes)
        ax.set(xlim=(0, 1), ylim=(0, 1.02))
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / f"{curve_type}_curve.png", dpi=160)
        plt.close(fig)


def _model_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_model(model_path: str | Path, config: dict[str, Any], output_dir: str | Path | None = None) -> dict[str, Any]:
    """Load a saved model and write metrics, curves, predictions and provenance."""
    from src.data_loader import create_dataset, load_manifest
    from src.integrity import verify_manifest_integrity
    from src.predict import load_model_bundle
    from src.utils import configure_runtime

    runtime = configure_runtime(config)
    resolved_model = resolve_path(model_path, config)
    model, metadata = load_model_bundle(resolved_model)
    if metadata.get("trained_on_real_data", False):
        verify_manifest_integrity(config)
    class_names = get_class_names(config)
    fingerprint = manifest_fingerprint(resolve_path(config["manifest_path"], config))
    expected = {"task": config["task"], "class_names": class_names, "image_size": config["image_size"],
                "dataset_fingerprint": fingerprint}
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(f"Model metadata {field!r} does not match this evaluation configuration/manifest. Use the training configuration and manifest.")
    threshold = float(config["prediction_threshold"])
    manifest = load_manifest(config)
    test_rows = manifest.loc[manifest["split"] == "test"].reset_index(drop=True)
    if test_rows.empty:
        raise ValueError("Test dataset is empty. Prepare and audit the dataset before evaluation.")
    labels = test_rows["label"].to_numpy(dtype=np.int64)
    dataset = create_dataset(config, "test", training=False, batch_size=runtime["batch_size"])
    raw_probabilities = np.asarray(model.predict(dataset, verbose=0))
    metrics = calculate_metrics(labels, raw_probabilities, class_names, threshold, config["task"])
    matrix = np.column_stack((1 - raw_probabilities.reshape(-1), raw_probabilities.reshape(-1))) if config["task"] == "binary" else raw_probabilities
    predicted = (matrix[:, 1] >= threshold).astype(int) if config["task"] == "binary" else matrix.argmax(axis=1)
    if output_dir is None:
        model_name = metadata.get("model_name")
        if model_name not in MODEL_NAMES:
            raise ValueError("Model metadata needs a recognized model_name, or supply --output-dir explicitly.")
        output = resolve_path(config["evaluation_dir"], config) / model_name
    else:
        output = resolve_path(output_dir, config)
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": metadata.get("model_name", config.get("model_name")),
        "model_path": Path(os.path.relpath(resolved_model, ROOT)).as_posix(),
        "model_sha256": _model_hash(resolved_model), "dataset_fingerprint": fingerprint,
        "trained_on_real_data": bool(metadata.get("trained_on_real_data", False)),
        "evaluation_split": "test", "threshold_source": "configuration, fixed independently of test predictions",
        "image_size": config["image_size"], "input_range": metadata.get("input_range"),
        "tensorflow_version": __import__("tensorflow").__version__,
    }
    metrics.update(provenance)
    write_json(output / "metrics.json", metrics)
    write_json(output / "provenance.json", provenance)
    pd.DataFrame([{key: metrics[key] for key in METRIC_NAMES}]).to_csv(output / "metrics.csv", index=False)
    report = pd.DataFrame.from_dict(metrics["per_class"], orient="index")
    report.index.name = "class"
    report.to_csv(output / "classification_report.csv")
    write_json(output / "classification_report.json", {"per_class": metrics["per_class"], "macro": metrics["macro"], "weighted": metrics["weighted"]})
    report_text = report[["precision", "recall", "specificity", "f1", "roc_auc", "support"]].to_string(na_rep="Not available", float_format=lambda value: f"{value:.4f}")
    (output / "classification_report.txt").write_text(report_text + "\n", encoding="utf-8")
    paths = [Path(os.path.relpath(path, ROOT)).as_posix() if Path(path).is_absolute() else Path(path).as_posix() for path in test_rows["path"].astype(str)]
    predictions = pd.DataFrame({"path": paths, "true_label": labels, "true_class": [class_names[x] for x in labels],
                                "predicted_label": predicted, "predicted_class": [class_names[x] for x in predicted]})
    for index, name in enumerate(class_names):
        predictions[f"probability_{name}"] = matrix[:, index]
    predictions.to_csv(output / "predictions.csv", index=False)
    _plot_evaluation(labels, matrix, metrics, output)
    LOGGER.info("Test classification report:\n%s", report_text)
    LOGGER.info("Evaluation saved to %s", output)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    setup_logging()
    try:
        config = load_config(args.config)
        evaluate_model(args.model or config["model_path"], config, args.output_dir)
    except (ValueError, FileNotFoundError, OSError) as exc:
        LOGGER.error("Evaluation failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
