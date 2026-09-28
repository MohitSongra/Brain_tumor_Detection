"""Generate publication figures from completed, matching three-model experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

from models.registry import DISPLAY_NAMES, MODEL_NAMES
from scripts.build_results_report import _validate_artifacts
from src.compare import compare_experiments
from src.evaluate import calculate_metrics
from src.utils import ROOT, load_config, resolve_path, write_json


def build_figures() -> None:
    """Refuse missing/changed experiments and never manufacture plotted values."""
    config = load_config()
    training = resolve_path(config["training_dir"], config)
    evaluation = resolve_path(config["evaluation_dir"], config)
    audit = json.loads((resolve_path(config["processed_dir"], config) / "audit.json").read_text())
    for name in MODEL_NAMES:
        if not (evaluation / name / "metrics.json").is_file():
            raise FileNotFoundError(f"Complete real-data training/evaluation for {name} before generating paper figures.")
    _validate_artifacts(config, audit, training, evaluation)
    compare_experiments(config)
    output = ROOT / "paper" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    records, predictions, histories, provenance = {}, {}, {}, {}
    reference_paths = None
    for name in MODEL_NAMES:
        metrics_path = evaluation / name / "metrics.json"
        prediction_path = evaluation / name / "predictions.csv"
        history_path = training / name / "history.json"
        records[name] = json.loads(metrics_path.read_text())
        predictions[name] = pd.read_csv(prediction_path)
        histories[name] = json.loads(history_path.read_text())
        rows = predictions[name][["path", "true_label"]]
        if reference_paths is not None and not rows.equals(reference_paths):
            raise ValueError("Model prediction rows refer to different test images or label order.")
        reference_paths = rows
        measured = calculate_metrics(rows.true_label.to_numpy(), predictions[name].probability_tumor.to_numpy(),
                                     ["no_tumor", "tumor"], 0.5, "binary")
        for key in ("accuracy", "precision", "recall", "f1", "specificity", "roc_auc"):
            if not np.isclose(measured[key], records[name][key], rtol=0, atol=1e-12):
                raise ValueError(f"Stored {name} {key} disagrees with saved per-image predictions.")
        provenance[name] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (metrics_path, prediction_path, history_path)}
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
                         "legend.fontsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 300})
    styles = [("#1c4e80", "-"), ("#9e3b22", "--"), ("#26734d", "-.")]
    fig, ax = plt.subplots(figsize=(3.35, 2.5), layout="constrained")
    for name, (color, pattern) in zip(MODEL_NAMES, styles):
        frame = predictions[name]
        fpr, tpr, _ = roc_curve(frame.true_label, frame.probability_tumor)
        ax.plot(fpr, tpr, color=color, linestyle=pattern, linewidth=1.4,
                label=f"{DISPLAY_NAMES[name]} ({auc(fpr, tpr):.4f})")
    ax.plot([0, 1], [0, 1], color="0.6", linestyle=":", linewidth=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.015))
    ax.legend(loc="lower right", frameon=False, title="Model (ROC-AUC)", title_fontsize=7.5)
    ax.grid(alpha=0.18)
    fig.savefig(output / "roc_comparison.png")
    fig.savefig(output / "roc_comparison.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 3.55), sharex=True, layout="constrained")
    for name, (color, pattern) in zip(MODEL_NAMES, styles):
        history = histories[name]
        epochs = np.arange(1, len(history["val_loss"]) + 1)
        axes[0].plot(epochs, history["val_loss"], color=color, linestyle=pattern, linewidth=1.2, label=DISPLAY_NAMES[name])
        axes[1].plot(epochs, history["val_accuracy"], color=color, linestyle=pattern, linewidth=1.2)
        for boundary in history.get("phase_boundaries", []):
            for axis, key in zip(axes, ("val_loss", "val_accuracy")):
                axis.scatter([boundary + 1], [history[key][boundary]], s=17, marker="o", color=color, zorder=4)
    axes[0].set(ylabel="Validation loss (log scale)", yscale="log")
    axes[0].legend(frameon=False, loc="upper right")
    axes[1].set(xlabel="Epoch across completed phases", ylabel="Validation accuracy", ylim=(0, 1.015))
    for ax in axes:
        ax.grid(alpha=0.18)
    fig.savefig(output / "validation_history.png")
    fig.savefig(output / "validation_history.pdf")
    plt.close(fig)
    write_json(output / "provenance.json", {"source_files": provenance,
        "figure_sha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                          for name in ("roc_comparison.png", "validation_history.png")},
        "dataset_fingerprint": audit["dataset_fingerprint"], "test_images": len(reference_paths),
        "all_reported_metrics_recomputed_from_saved_predictions": True,
        "figure_data": "Original measured predictions and histories; no smoothing or simulated values."})
    print("Verified paper figures saved under paper/figures.")


if __name__ == "__main__":
    build_figures()
