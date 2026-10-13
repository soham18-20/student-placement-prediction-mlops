"""
Script: scripts/train.py
Trains the scikit-learn student placement pipeline, computes evaluation metrics,
compares against baselines, and saves serialized artifacts to models/.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.pipeline import train_pipeline
from ml.evaluate import format_evaluation_summary

if __name__ == "__main__":
    print("[Pipeline] Initiating model training pipeline...")
    pipeline, metadata = train_pipeline()
    print("[Pipeline] Training completed successfully!")
    print(format_evaluation_summary(metadata))
