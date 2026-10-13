"""
Script: scripts/generate_data.py
Generates realistic student placement data for synthetic demo mode or inspects current dataset.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.dataset import load_or_create_dataset
from ml.config import RAW_DATA_PATH

if __name__ == "__main__":
    print(f"[Dataset] Loading/generating placement dataset at: {RAW_DATA_PATH}")
    df, meta = load_or_create_dataset(force_regenerate=True)
    print(f"[Dataset] Generated {len(df)} samples.")
    print(f"[Dataset] Synthetic Demo Mode: {meta['is_synthetic']}")
    print(f"[Dataset] Class balance: {meta['class_balance']}")
    print("\nFeature Summary:")
    print(df.describe().round(2))
