import json
import logging
import sys
from typing import Dict, Any
from pathlib import Path

# Fix Windows console encoding if needed
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ml.config import METRICS_PATH, MODEL_PATH

logger = logging.getLogger("ml.evaluate")


def load_evaluation_report(metrics_path: Path = METRICS_PATH) -> Dict[str, Any]:
    """Loads saved metrics and baseline comparison from disk."""
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found at {metrics_path}. Train the pipeline first.")
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_evaluation_summary(report: Dict[str, Any]) -> str:
    """Formats metrics, confusion matrix, and baseline comparison into a readable text report."""
    primary = report.get("primary_metrics", {})
    cm = primary.get("confusion_matrix", {})
    baselines = report.get("baseline_comparison", {})
    dataset = report.get("dataset", {})

    tn = cm.get("true_negatives", 0)
    fp = cm.get("false_positives", 0)
    fn = cm.get("false_negatives", 0)
    tp = cm.get("true_positives", 0)

    lines = [
        "=" * 70,
        "STUDENT PLACEMENT PREDICTION - MODEL EVALUATION REPORT",
        "=" * 70,

        f"Model Version     : {report.get('model_version', 'N/A')}",
        f"Algorithm         : {report.get('algorithm', 'N/A')}",
        f"Training Timestamp: {report.get('training_timestamp', 'N/A')}",
        f"Data Mode         : {dataset.get('demo_mode_label', 'N/A')}",
        f"Total Samples     : {dataset.get('total_records', 'N/A')} (Train: {report.get('split', {}).get('train_samples')}, Test: {report.get('split', {}).get('test_samples')})",
        "",
        "-" * 70,
        "1. PRIMARY MODEL PERFORMANCE (Holdout Test Set)",
        "-" * 70,
        f"  Accuracy  : {primary.get('accuracy', 0.0) * 100:.2f}%",
        f"  Precision : {primary.get('precision', 0.0) * 100:.2f}%",
        f"  Recall    : {primary.get('recall', 0.0) * 100:.2f}%",
        f"  F1-Score  : {primary.get('f1_score', 0.0) * 100:.2f}%",
        f"  ROC-AUC   : {primary.get('roc_auc', 0.0) * 100:.2f}%" if primary.get('roc_auc') is not None else "  ROC-AUC   : N/A",
        "",
        "-" * 70,
        "2. CONFUSION MATRIX",
        "-" * 70,
        "                     Predicted Unlikely      Predicted Likely",
        f"  Actual Unlikely    TN = {tn:<15}   FP = {fp:<15}",
        f"  Actual Likely      FN = {fn:<15}   TP = {tp:<15}",
        "",
        "-" * 70,
        "3. BASELINE COMPARISON",
        "-" * 70,
        f"{'Model / Baseline':<32} {'Accuracy':<12} {'F1-Score':<12} {'Acc Delta':<12}",
        "-" * 70,
        f"{'Primary (Random Forest Pipeline)':<32} {primary.get('accuracy', 0.0)*100:.2f}%{'':<5} {primary.get('f1_score', 0.0)*100:.2f}%{'':<5} {'Base Reference':<12}",
    ]

    for key, b_info in baselines.items():
        b_name = b_info.get("name", key)
        b_m = b_info.get("metrics", {})
        delta = b_info.get("accuracy_delta", 0.0)
        delta_str = f"+{delta*100:.2f}%" if delta >= 0 else f"{delta*100:.2f}%"
        lines.append(
            f"{b_name:<32} {b_m.get('accuracy', 0.0)*100:.2f}%{'':<5} {b_m.get('f1_score', 0.0)*100:.2f}%{'':<5} {delta_str:<12}"
        )

    lines.extend([
        "",
        "-" * 70,
        "4. FEATURE IMPORTANCES",
        "-" * 70,
    ])

    for feat, imp in report.get("feature_importances", {}).items():
        bar = "#" * int(imp * 30)
        lines.append(f"  {feat:<22} : {imp:.4f} {bar}")

    lines.extend([
        "",
        "-" * 70,
        "5. DATASET & DEMO MODE DISCLOSURE",
        "-" * 70,
        f"  {dataset.get('disclosure', 'No disclosure specified.')}",
        "=" * 70,
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    try:
        rep = load_evaluation_report()
        print(format_evaluation_summary(rep))
    except Exception as e:
        print(f"Error loading evaluation report: {e}")
