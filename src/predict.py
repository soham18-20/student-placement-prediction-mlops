import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.config import (
    MODEL_PATH,
    SCALER_PATH,
    METADATA_PATH,
    PREDICTIONS_LOG_PATH,
    FEATURE_COLUMNS,
    PLACEMENT_PROBABILITY_THRESHOLD,
    CURRENT_MODEL_VERSION,
)

logger = logging.getLogger("MLOps-Predict")


class StudentInput(BaseModel):
    cgpa: float = Field(..., ge=0.0, le=10.0, description="Cumulative Grade Point Average (0-10)", json_schema_extra={"example": 8.2})
    technical_score: float = Field(..., ge=0.0, le=100.0, description="Technical Skills Score (0-100)", json_schema_extra={"example": 85.0})
    aptitude_score: float = Field(..., ge=0.0, le=100.0, description="Aptitude & Logical Reasoning (0-100)", json_schema_extra={"example": 75.0})
    communication_score: float = Field(..., ge=0.0, le=100.0, description="Communication & Soft Skills (0-100)", json_schema_extra={"example": 80.0})
    internships: int = Field(..., ge=0, le=10, description="Number of internships completed", json_schema_extra={"example": 2})
    projects: int = Field(..., ge=0, le=20, description="Number of major/minor projects completed", json_schema_extra={"example": 3})
    attendance: float = Field(..., ge=0.0, le=100.0, description="Classroom attendance percentage", json_schema_extra={"example": 88.5})


class PredictionResponse(BaseModel):
    prediction_id: str
    prediction: str  # "Likely" or "Unlikely"
    probability: float
    probability_percentage: str
    confidence: str
    key_factors: List[str]
    recommendations: List[str]
    model_version: str
    timestamp: str


class PlacementPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.metadata: Dict[str, Any] = {}
        self.load_artifacts()

    def load_artifacts(self):
        """Loads or reloads serialized model artifacts."""
        if MODEL_PATH.exists() and SCALER_PATH.exists():
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            if METADATA_PATH.exists():
                with open(METADATA_PATH, "r") as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {"model_version": CURRENT_MODEL_VERSION}
            logger.info("Successfully loaded ML model and scaler artifacts.")
        else:
            logger.warning("Model artifacts not found on disk. Run src.train to generate them.")

    def is_ready(self) -> bool:
        return self.model is not None and self.scaler is not None

    def _generate_recommendations(self, data: StudentInput, prob: float) -> List[str]:
        tips: List[str] = []
        if data.cgpa < 7.0:
            tips.append("CGPA is below 7.0. Strive to maintain at least 7.5 to clear campus recruitment eligibility cutoffs.")
        if data.technical_score < 70:
            tips.append("Technical score is below 70. Strengthen Data Structures, Algorithms, and Core CS fundamentals.")
        if data.internships == 0:
            tips.append("Complete at least 1 industry internship to gain competitive practical experience.")
        if data.projects < 2:
            tips.append("Build and showcase at least 2 full-stack or domain-specific capstone projects on GitHub.")
        if data.aptitude_score < 65:
            tips.append("Practice quantitative aptitude and logical reasoning mock tests.")
        if data.attendance < 75:
            tips.append("Attendance is below 75%; improve lecture attendance to prevent eligibility disqualification.")

        if not tips:
            tips.append("Outstanding academic and technical profile! Prepare for behavioral and system design interview rounds.")
        return tips

    def _generate_key_factors(self, data: StudentInput) -> List[str]:
        factors = []
        if data.cgpa >= 8.5:
            factors.append("Exceptional CGPA gives strong academic standing")
        if data.technical_score >= 85:
            factors.append("High technical competence significantly boosts placement odds")
        if data.internships >= 2:
            factors.append("Substantial internship track record valued by hiring managers")
        if data.projects >= 4:
            factors.append("Rich portfolio of completed practical projects")
        if not factors:
            factors.append("Balanced general profile across key placement parameters")
        return factors

    def predict(self, student: StudentInput) -> PredictionResponse:
        if not self.is_ready():
            self.load_artifacts()
            if not self.is_ready():
                raise RuntimeError("ML model artifacts are not initialized. Please train the model first.")

        input_df = pd.DataFrame([[
            student.cgpa,
            student.technical_score,
            student.aptitude_score,
            student.communication_score,
            student.internships,
            student.projects,
            student.attendance,
        ]], columns=FEATURE_COLUMNS)

        scaled_features = self.scaler.transform(input_df)
        prob = float(self.model.predict_proba(scaled_features)[0][1])
        prediction_label = "Likely" if prob >= PLACEMENT_PROBABILITY_THRESHOLD else "Unlikely"

        # Determine confidence level
        if prob >= 0.75 or prob <= 0.25:
            confidence = "High"
        elif prob >= 0.60 or prob <= 0.40:
            confidence = "Moderate"
        else:
            confidence = "Low"

        pred_id = str(uuid.uuid4())
        now_ts = datetime.now(timezone.utc).isoformat()
        version = self.metadata.get("model_version", CURRENT_MODEL_VERSION)

        response = PredictionResponse(
            prediction_id=pred_id,
            prediction=prediction_label,
            probability=round(prob, 4),
            probability_percentage=f"{round(prob * 100, 1)}%",
            confidence=confidence,
            key_factors=self._generate_key_factors(student),
            recommendations=self._generate_recommendations(student, prob),
            model_version=version,
            timestamp=now_ts,
        )

        # Log prediction asynchronously for MLOps drift & audit
        self._log_prediction(pred_id, student, response)

        return response

    def _log_prediction(self, pred_id: str, student: StudentInput, response: PredictionResponse):
        try:
            PREDICTIONS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            log_entry = {
                "prediction_id": pred_id,
                "timestamp": response.timestamp,
                "model_version": response.model_version,
                "inputs": student.model_dump(),
                "output": {
                    "prediction": response.prediction,
                    "probability": response.probability,
                },
            }
            with open(PREDICTIONS_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.warning(f"Failed to log prediction entry: {e}")


# Singleton predictor
predictor = PlacementPredictor()
