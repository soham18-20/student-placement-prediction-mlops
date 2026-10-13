import pytest
from ml.evaluate import load_evaluation_report


def test_evaluation_report_structure():
    report = load_evaluation_report()

    # Check top-level keys
    assert "model_version" in report
    assert "algorithm" in report
    assert "primary_metrics" in report
    assert "baseline_comparison" in report
    assert "dataset" in report


def test_primary_metrics_validity():
    report = load_evaluation_report()
    primary = report["primary_metrics"]

    assert 0.0 <= primary["accuracy"] <= 1.0
    assert 0.0 <= primary["precision"] <= 1.0
    assert 0.0 <= primary["recall"] <= 1.0
    assert 0.0 <= primary["f1_score"] <= 1.0

    # Primary model should have solid predictive power on this domain dataset
    assert primary["accuracy"] >= 0.80
    assert primary["f1_score"] >= 0.80


def test_confusion_matrix_elements():
    report = load_evaluation_report()
    cm = report["primary_metrics"]["confusion_matrix"]

    assert "true_negatives" in cm
    assert "false_positives" in cm
    assert "false_negatives" in cm
    assert "true_positives" in cm

    total_test = cm["true_negatives"] + cm["false_positives"] + cm["false_negatives"] + cm["true_positives"]
    assert total_test == report["split"]["test_samples"]


def test_baseline_comparison_outperformance():
    report = load_evaluation_report()
    baselines = report["baseline_comparison"]

    assert "dummy_majority_class" in baselines
    dummy = baselines["dummy_majority_class"]

    # Model must outperform dummy baseline accuracy
    assert dummy["accuracy_delta"] > 0
    assert dummy["f1_delta"] > 0
