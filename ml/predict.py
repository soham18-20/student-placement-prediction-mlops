import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

import joblib
import numpy as np
import pandas as pd

from ml.config import (
    MODEL_PATH,
    METRICS_PATH,
    PREDICTIONS_LOG_PATH,
    FEATURE_COLUMNS,
    CLASSIFICATION_THRESHOLD,
    MODEL_VERSION,
    SYNTHETIC_DISCLOSURE,
)

logger = logging.getLogger("ml.predict")


class ModelArtifactNotFoundError(Exception):
    """Raised when model artifact files cannot be found on disk."""
    pass


class PlacementPredictor:
    def __init__(self, model_path: Path = MODEL_PATH, metrics_path: Path = METRICS_PATH):
        self.model_path = model_path
        self.metrics_path = metrics_path
        self.pipeline = None
        self.metrics: Dict[str, Any] = {}
        self.load_model(strict=False)

    def load_model(self, strict: bool = False):
        """Loads serialized scikit-learn pipeline and metrics from disk."""
        if not self.model_path.exists():
            self.pipeline = None
            if strict:
                raise ModelArtifactNotFoundError(
                    f"Model artifact not found at '{self.model_path}'. "
                    "Run 'python scripts/train.py' to generate the model pipeline."
                )
            logger.warning(f"Model file not found at {self.model_path}")
            return

        try:
            self.pipeline = joblib.load(self.model_path)
            logger.info(f"Loaded scikit-learn pipeline from {self.model_path}")
        except Exception as e:
            self.pipeline = None
            if strict:
                raise ModelArtifactNotFoundError(f"Failed to load model artifact: {e}")
            logger.error(f"Failed to load model artifact: {e}")

        if self.metrics_path.exists():
            try:
                with open(self.metrics_path, "r", encoding="utf-8") as f:
                    self.metrics = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read metrics file: {e}")
                self.metrics = {"model_version": MODEL_VERSION}
        else:
            self.metrics = {"model_version": MODEL_VERSION}

    def is_ready(self) -> bool:
        """Checks if the model pipeline is loaded and ready for inference."""
        return self.pipeline is not None

    def _generate_factors_and_recommendations(
        self, data: Dict[str, Any], prob: float
    ) -> Tuple[List[str], List[str]]:
        factors = []
        recommendations = []

        cgpa = float(data["cgpa"])
        tech = float(data["technical_score"])
        apt = float(data["aptitude_score"])
        comm = float(data["communication_score"])
        intern = int(data["internship_exp"])
        proj = int(data["projects_completed"])
        att = float(data["attendance"])

        # Key Factors (strengths)
        if cgpa >= 8.5:
            factors.append("Exceptional CGPA (>= 8.5) provides a strong academic threshold.")
        elif cgpa >= 7.5:
            factors.append("Healthy CGPA clearing recruitment screening criteria.")

        if tech >= 80.0:
            factors.append("High technical skills score demonstrates coding/problem-solving readiness.")

        if intern == 1:
            factors.append("Practical internship experience distinguishes the profile in interviews.")

        if proj >= 3:
            factors.append("Strong portfolio of completed practical software projects.")

        if comm >= 80.0:
            factors.append("High communication proficiency aids technical and HR interviews.")

        if att >= 85.0:
            factors.append("Consistent lecture and lab attendance record.")

        if not factors:
            factors.append("Balanced baseline parameters across all placement dimensions.")

        # Recommendations (areas to improve)
        if cgpa < 7.0:
            recommendations.append("CGPA is below 7.0. Strive to maintain >= 7.5 to satisfy premier campus recruitment cutoffs.")

        if tech < 70.0:
            recommendations.append("Technical score is under 70. Strengthen Data Structures, Algorithms, and Core CS subjects.")

        if intern == 0:
            recommendations.append("Gain practical exposure: apply for at least 1 verified internship or industrial training program.")

        if proj < 2:
            recommendations.append("Expand project portfolio: build and document at least 2 full-stack or domain projects on GitHub.")

        if apt < 65.0:
            recommendations.append("Practice quantitative aptitude and logical reasoning mock assessments regularly.")

        if comm < 65.0:
            recommendations.append("Participate in group discussions, technical presentations, and communication workshops.")

        if att < 75.0:
            recommendations.append("Attendance is below the mandatory 75% threshold; increase presence to prevent disqualification.")

        if not recommendations:
            recommendations.append("Outstanding academic profile! Focus on behavioral mock interviews and system design practice.")

        return factors, recommendations

    def predict(self, student_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs inference on validated student features.
        Raises ModelArtifactNotFoundError if model is not loaded.
        """
        if not self.is_ready():
            self.load_model(strict=True)

        # Standardize features
        row = {
            "cgpa": float(student_data["cgpa"]),
            "technical_score": float(student_data["technical_score"]),
            "aptitude_score": float(student_data["aptitude_score"]),
            "communication_score": float(student_data["communication_score"]),
            "internship_exp": int(student_data["internship_exp"]),
            "projects_completed": int(student_data["projects_completed"]),
            "attendance": float(student_data["attendance"]),
        }

        input_df = pd.DataFrame([row], columns=FEATURE_COLUMNS)

        # Predict probability
        prob_arr = self.pipeline.predict_proba(input_df)[0]
        prob_likely = float(prob_arr[1])
        prob_unlikely = float(prob_arr[0])

        classification = "Likely" if prob_likely >= CLASSIFICATION_THRESHOLD else "Unlikely"

        # Determine confidence level
        if prob_likely >= 0.75 or prob_likely <= 0.25:
            confidence = "High"
        elif prob_likely >= 0.60 or prob_likely <= 0.40:
            confidence = "Moderate"
        else:
            confidence = "Borderline"

        factors, recommendations = self._generate_factors_and_recommendations(row, prob_likely)

        pred_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        dataset_meta = self.metrics.get("dataset", {})
        is_synthetic = dataset_meta.get("is_synthetic", True)

        result = {
            "prediction_id": pred_id,
            "prediction": classification,
            "probability": round(prob_likely, 4),
            "probability_percentage": f"{round(prob_likely * 100, 1)}%",
            "probability_unlikely": round(prob_unlikely, 4),
            "confidence": confidence,
            "threshold_used": CLASSIFICATION_THRESHOLD,
            "key_factors": factors,
            "recommendations": recommendations,
            "model_version": self.metrics.get("model_version", MODEL_VERSION),
            "algorithm": self.metrics.get("algorithm", "RandomForestClassifier"),
            "timestamp": timestamp,
            "demo_mode": {
                "is_synthetic": is_synthetic,
                "label": dataset_meta.get("demo_mode_label", "Synthetic Educational Simulation Data"),
                "disclosure": dataset_meta.get("disclosure", SYNTHETIC_DISCLOSURE),
            },
        }

        # Log prediction asynchronously / append to file
        self._log_prediction(pred_id, row, result)

        return result

    def _log_prediction(self, pred_id: str, inputs: Dict[str, Any], output: Dict[str, Any]):
        try:
            PREDICTIONS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            log_record = {
                "prediction_id": pred_id,
                "timestamp": output["timestamp"],
                "model_version": output["model_version"],
                "inputs": inputs,
                "output": {
                    "prediction": output["prediction"],
                    "probability": output["probability"],
                    "confidence": output["confidence"],
                },
            }
            with open(PREDICTIONS_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_record) + "\n")
        except Exception as e:
            logger.warning(f"Failed to append to prediction audit log: {e}")


# Singleton predictor instance for application use
predictor = PlacementPredictor()
