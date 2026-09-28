"""Check metric arithmetic and reject invalid or misleading result records."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.compare import compare_experiments
from src.evaluate import calculate_metrics


def test_known_binary_confusion_and_metrics() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.3, 0.7, 0.9])
    result = calculate_metrics(labels, probabilities, ["no_tumor", "tumor"])
    assert result["confusion_matrix"] == [[2, 1], [1, 2]]
    for key in ("accuracy", "precision", "recall", "sensitivity", "specificity", "f1"):
        assert result[key] == pytest.approx(2 / 3)
    assert result["roc_auc"] == pytest.approx(7 / 9)
    assert result["per_class"]["tumor"]["tp"] == 2
    assert result["per_class"]["tumor"]["tn"] == 2


def test_sensitivity_and_specificity_have_correct_denominators() -> None:
    result = calculate_metrics(np.array([0, 0, 0, 1]), np.array([0.1, 0.2, 0.8, 0.9]), ["no_tumor", "tumor"])
    assert result["sensitivity"] == 1.0
    assert result["specificity"] == pytest.approx(2 / 3)
    assert result["precision"] == 0.5


def test_single_class_and_no_positive_predictions_are_undefined() -> None:
    result = calculate_metrics(np.array([0, 0]), np.array([0.1, 0.2]), ["no_tumor", "tumor"])
    assert result["accuracy"] == 1.0
    assert result["precision"] is None
    assert result["recall"] is None
    assert result["f1"] is None
    assert result["roc_auc"] is None
    assert result["specificity"] == 1.0
    json.dumps(result, allow_nan=False)


def test_missed_positive_cases_have_zero_recall_and_f1() -> None:
    result = calculate_metrics(np.array([0, 1]), np.array([0.1, 0.2]), ["no_tumor", "tumor"])
    assert result["precision"] is None
    assert result["recall"] == 0
    assert result["f1"] == 0


def test_threshold_boundary_predicts_positive() -> None:
    result = calculate_metrics(np.array([0, 1]), np.array([[0.49], [0.50]]), ["no_tumor", "tumor"])
    assert result["accuracy"] == 1


@pytest.mark.parametrize("threshold,expected", [(0.0, [[0, 1], [0, 1]]), (1.0, [[1, 0], [0, 1]])])
def test_threshold_endpoints_are_supported(threshold: float, expected: list[list[int]]) -> None:
    result = calculate_metrics(np.array([0, 1]), np.array([0.0, 1.0]), ["no_tumor", "tumor"], threshold=threshold)
    assert result["confusion_matrix"] == expected
    assert result["threshold"] == threshold


@pytest.mark.parametrize("threshold", [-0.01, 1.01, np.nan, np.inf])
def test_invalid_thresholds_are_rejected(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        calculate_metrics(np.array([0, 1]), np.array([0.1, 0.9]), ["no_tumor", "tumor"], threshold=threshold)


@pytest.mark.parametrize("probabilities", [np.array([-0.1, 0.2]), np.array([0.1, 1.1]), np.array([np.nan, 0.2]), np.array([0.1, np.inf])])
def test_invalid_probabilities_rejected(probabilities: np.ndarray) -> None:
    with pytest.raises(ValueError, match="finite and between"):
        calculate_metrics(np.array([0, 1]), probabilities, ["no_tumor", "tumor"])


def test_output_shape_rejected() -> None:
    with pytest.raises(ValueError, match="shape"):
        calculate_metrics(np.array([0, 1]), np.array([[0.1, 0.9], [0.2, 0.8]]), ["no_tumor", "tumor"])


def test_perfect_multiclass_one_vs_rest_metrics() -> None:
    probabilities = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    result = calculate_metrics(np.array([0, 1, 2]), probabilities, ["a", "b", "c"], task="multiclass")
    for key in ("accuracy", "precision", "recall", "sensitivity", "specificity", "f1", "roc_auc"):
        assert result[key] == 1.0
    assert result["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_multiclass_missing_class_auc_remains_undefined() -> None:
    result = calculate_metrics(np.array([0, 1]), np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]), ["a", "b", "c"], task="multiclass")
    assert result["roc_auc"] is None
    assert result["per_class"]["c"]["support"] == 0
    assert result["weighted"]["roc_auc"] == 1.0


def test_multiclass_sum_and_labels_validated() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        calculate_metrics(np.array([0, 1]), np.array([[0.8, 0.8], [0.1, 0.9]]), ["a", "b"], task="multiclass")
    with pytest.raises(ValueError, match="mapping"):
        calculate_metrics(np.array([0, 2]), np.array([0.1, 0.9]), ["a", "b"])


def test_missing_comparison_results_are_not_fabricated(tmp_path: Path) -> None:
    frame = compare_experiments({"evaluation_dir": str(tmp_path)})
    assert set(frame["Accuracy"]) == {"Not available"}
    assert "Training required" in (tmp_path / "comparison.md").read_text(encoding="utf-8")


def test_comparison_rejects_different_datasets(tmp_path: Path) -> None:
    for name, fingerprint in (("custom_cnn", "one"), ("efficientnet", "two")):
        directory = tmp_path / name
        directory.mkdir()
        record = {"model_name": name, "trained_on_real_data": True, "evaluation_split": "test", "dataset_fingerprint": fingerprint,
                  "task": "binary", "class_names": ["no_tumor", "tumor"], "threshold": 0.5, "n_samples": 10}
        (directory / "metrics.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="dataset_fingerprint"):
        compare_experiments({"evaluation_dir": str(tmp_path)})


@pytest.mark.parametrize("declared_name", [None, "efficientnet"])
def test_comparison_rejects_missing_or_mislabeled_model_name(tmp_path: Path, declared_name: str | None) -> None:
    directory = tmp_path / "custom_cnn"
    directory.mkdir()
    record = {"trained_on_real_data": True, "evaluation_split": "test", "dataset_fingerprint": "same",
              "task": "binary", "class_names": ["no_tumor", "tumor"], "threshold": 0.5, "n_samples": 10}
    if declared_name is not None:
        record["model_name"] = declared_name
    (directory / "metrics.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="model_name"):
        compare_experiments({"evaluation_dir": str(tmp_path)})


@pytest.mark.parametrize("model_name", ["custom_cnn", "mobilenet"])
def test_saved_model_evaluation_pipeline_in_temporary_fixture(tmp_path: Path, model_name: str) -> None:
    """Exercise real loading/prediction/plots; fixtures are not MRI results."""
    import pandas as pd
    from PIL import Image
    import tensorflow as tf
    from src.evaluate import evaluate_model
    from src.utils import load_config, manifest_fingerprint, write_json

    rows = []
    for index, (value, source_class) in enumerate(((0, "notumor"), (255, "glioma"))):
        image_path = tmp_path / f"fixture_{index}.png"
        Image.new("RGB", (32, 32), (value, value, value)).save(image_path)
        rows.append({"path": str(image_path), "source_class": source_class, "split": "test", "label": index})
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    inputs = tf.keras.Input((32, 32, 3))
    features = tf.keras.layers.GlobalAveragePooling2D()(tf.keras.layers.Rescaling(1 / 255)(inputs))
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", kernel_initializer=tf.keras.initializers.Constant(10 / 3),
                                    bias_initializer=tf.keras.initializers.Constant(-5))(features)
    model = tf.keras.Model(inputs, outputs)
    model_path = model_dir / "best_model.keras"
    model.save(model_path)
    metadata = {"task": "binary", "class_names": ["no_tumor", "tumor"], "image_size": 32,
                "input_range": [0, 255], "threshold": 0.5, "model_name": model_name,
                "dataset_fingerprint": manifest_fingerprint(manifest), "trained_on_real_data": False}
    write_json(model_dir / "metadata.json", metadata)
    write_json(model_dir / "class_names.json", metadata["class_names"])
    config = load_config()
    config.update({"manifest_path": str(manifest), "image_size": 32, "cpu_batch_size": 2,
                   "device": "cpu", "evaluation_dir": str(tmp_path / "evaluation")})
    output = tmp_path / "evaluation" / model_name
    result = evaluate_model(model_path, config)
    assert result["accuracy"] == 1.0
    assert result["trained_on_real_data"] is False
    assert result["model_name"] == model_name
    assert result["n_samples"] == 2
    for filename in ("metrics.json", "metrics.csv", "classification_report.json", "classification_report.csv",
                     "classification_report.txt", "predictions.csv", "provenance.json", "confusion_matrix.png",
                     "roc_curve.png", "precision_recall_curve.png"):
        assert (output / filename).is_file()
    predictions = pd.read_csv(output / "predictions.csv")
    assert not any(Path(path).is_absolute() for path in predictions["path"])
    assert predictions["predicted_label"].tolist() == [0, 1]
    metadata["dataset_fingerprint"] = "incorrect"
    write_json(model_dir / "metadata.json", metadata)
    with pytest.raises(ValueError, match="dataset_fingerprint"):
        evaluate_model(model_path, config, output)
