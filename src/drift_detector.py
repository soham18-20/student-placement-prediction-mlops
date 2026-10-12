import json
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from src.config import RAW_DATA_PATH, PREDICTIONS_LOG_PATH, FEATURE_COLUMNS


class DataDriftDetector:
    """
    Monitors inference data drift against baseline training data.
    Computes statistical Z-score shifts and identifies features experiencing drift.
    """

    def __init__(self, baseline_path=RAW_DATA_PATH):
        self.baseline_path = baseline_path
        self.baseline_stats: Dict[str, Dict[str, float]] = {}
        self._load_baseline()

    def _load_baseline(self):
        if self.baseline_path.exists():
            df = pd.read_csv(self.baseline_path)
            for col in FEATURE_COLUMNS:
                if col in df.columns:
                    self.baseline_stats[col] = {
                        "mean": float(df[col].mean()),
                        "std": float(df[col].std()) if df[col].std() > 0 else 1.0,
                        "min": float(df[col].min()),
                        "max": float(df[col].max()),
                    }

    def evaluate_recent_data(self, window_size: int = 50) -> Dict[str, Any]:
        """
        Examines recent production predictions logged in JSONL format.
        Compares recent feature means to baseline means.
        """
        if not PREDICTIONS_LOG_PATH.exists():
            return {
                "status": "HEALTHY",
                "sample_count": 0,
                "drift_detected": False,
                "message": "No production predictions logged yet.",
                "feature_drift": {},
            }

        records: List[Dict[str, Any]] = []
        try:
            with open(PREDICTIONS_LOG_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
        except Exception:
            pass

        if len(records) < 5:
            return {
                "status": "HEALTHY",
                "sample_count": len(records),
                "drift_detected": False,
                "message": f"Insufficient sample size ({len(records)} < 5) for statistical drift test.",
                "feature_drift": {},
            }

        recent = records[-window_size:]
        feature_drift = {}
        drift_detected = False

        for col in FEATURE_COLUMNS:
            vals = [r.get("inputs", {}).get(col) for r in recent if col in r.get("inputs", {})]
            if len(vals) < 5 or col not in self.baseline_stats:
                continue

            current_mean = float(np.mean(vals))
            baseline_mean = self.baseline_stats[col]["mean"]
            baseline_std = self.baseline_stats[col]["std"]

            # Standard score (z-shift)
            z_shift = abs(current_mean - baseline_mean) / baseline_std
            has_drift = z_shift > 1.25

            if has_drift:
                drift_detected = True

            feature_drift[col] = {
                "baseline_mean": round(baseline_mean, 2),
                "production_mean": round(current_mean, 2),
                "z_shift": round(z_shift, 3),
                "drift_status": "DRIFTING" if has_drift else "STABLE",
            }

        return {
            "status": "DRIFT_DETECTED" if drift_detected else "HEALTHY",
            "drift_detected": drift_detected,
            "sample_count": len(recent),
            "recommendation": (
                "Trigger automated model retraining"
                if drift_detected
                else "Model operating within expected baseline parameters."
            ),
            "feature_drift": feature_drift,
        }
