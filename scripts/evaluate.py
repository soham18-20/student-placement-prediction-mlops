"""
Script: scripts/evaluate.py
Loads model evaluation artifacts from models/metrics.json and displays
holdout test set performance, confusion matrix, and baseline comparison.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.evaluate import load_evaluation_report, format_evaluation_summary

if __name__ == "__main__":
    try:
        report = load_evaluation_report()
        print(format_evaluation_summary(report))
    except FileNotFoundError as e:
        print(f"[Error] {e}")
        print("Run 'python scripts/train.py' first to train the pipeline.")
        sys.exit(1)
