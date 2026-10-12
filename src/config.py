import os
from pathlib import Path

# Base Paths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "student_placement_data.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "placement_model.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

LOGS_DIR = PROJECT_ROOT / "logs"
PREDICTIONS_LOG_PATH = LOGS_DIR / "predictions.jsonl"
SYSTEM_LOG_PATH = LOGS_DIR / "app.log"

STATIC_DIR = PROJECT_ROOT / "static"

# Ensure essential directories exist
for folder in [DATA_DIR / "raw", PROCESSED_DATA_DIR, MODEL_DIR, LOGS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Feature Configuration
FEATURE_COLUMNS = [
    "cgpa",
    "technical_score",
    "aptitude_score",
    "communication_score",
    "internships",
    "projects",
    "attendance",
]

TARGET_COLUMN = "placement_status"

FEATURE_RANGES = {
    "cgpa": {"min": 0.0, "max": 10.0, "default": 8.2},
    "technical_score": {"min": 0.0, "max": 100.0, "default": 78.0},
    "aptitude_score": {"min": 0.0, "max": 100.0, "default": 74.0},
    "communication_score": {"min": 0.0, "max": 100.0, "default": 80.0},
    "internships": {"min": 0, "max": 10, "default": 2},
    "projects": {"min": 0, "max": 20, "default": 3},
    "attendance": {"min": 0.0, "max": 100.0, "default": 88.0},
}

# Model Settings
CURRENT_MODEL_VERSION = "v1.2.0"
PLACEMENT_PROBABILITY_THRESHOLD = 0.50
RANDOM_STATE = 42
TEST_SIZE = 0.20
