"""Compare measured model results only when their evaluation conditions match."""
from __future__ import annotations

import argparse
from itertools import combinations
import json
import logging
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from models.registry import DISPLAY_NAMES, MODEL_NAMES
from src.utils import load_config, resolve_path, setup_logging

LOGGER = logging.getLogger(__name__)
COLUMNS = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1", "specificity": "Specificity", "roc_auc": "ROC-AUC"}


def compare_experiments(config: dict[str, Any], output_dir: str | Path | None = None) -> pd.DataFrame:
    """Write a comparison without filling missing experiments or mixing cohorts."""
    root = resolve_path(config["evaluation_dir"], config)
    destination = resolve_path(output_dir, config) if output_dir else root
    records: dict[str, dict[str, Any] | None] = {}
    for model in MODEL_NAMES:
        path = root / model / "metrics.json"
        records[model] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        if records[model] is not None and records[model].get("model_name") != model:
            raise ValueError(f"Evaluation record in {model}/metrics.json must declare model_name={model!r}; refusing to mislabel copied or incomplete results.")
    available = [record for record in records.values() if record is not None]
    for record in available:
        if not record.get("trained_on_real_data") or record.get("evaluation_split") != "test":
            raise ValueError("Comparison accepts only models trained on real data and evaluated on the test split.")
        for field in ("dataset_fingerprint", "task", "class_names", "threshold", "n_samples"):
            if field not in record:
                raise ValueError(f"Evaluation record is missing required provenance: {field}.")
    for first, second in combinations(available, 2):
        for field in ("dataset_fingerprint", "task", "class_names", "threshold", "n_samples"):
            if first[field] != second[field]:
                raise ValueError(
                    f"Cannot compare {first['model_name']} and {second['model_name']} with different {field}. "
                    "Re-evaluate under matching conditions."
                )
    rows = []
    for model, record in records.items():
        row = {"Model": DISPLAY_NAMES[model]}
        row.update({title: record.get(metric) if record is not None and record.get(metric) is not None else "Not available" for metric, title in COLUMNS.items()})
        rows.append(row)
    frame = pd.DataFrame(rows)
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / "comparison.csv", index=False)
    header = "| " + " | ".join(frame.columns) + " |"
    lines = [header, "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(f"{value:.4f}" if isinstance(value, (int, float)) else str(value) for value in row.values()) + " |")
    if len(available) < len(MODEL_NAMES):
        lines.extend(["", "Training required — no measured result available for missing experiments."])
    if available:
        lines.extend(["", f"Task: {available[0]['task']}. Threshold: {available[0]['threshold']}. Test images: {available[0]['n_samples']}.",
                      "", "All available measured rows use the same saved split manifest. "
                      "This is an image-level research comparison, not clinical validation."])
    (destination / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Comparison saved to %s", destination)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    setup_logging()
    try:
        compare_experiments(load_config(args.config), args.output_dir)
    except (ValueError, FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        LOGGER.error("Comparison failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
