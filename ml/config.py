import os
from pathlib import Path

# Base Paths
ML_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ML_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "student_placement_data.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "placement_pipeline.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

LOGS_DIR = PROJECT_ROOT / "logs"
PREDICTIONS_LOG_PATH = LOGS_DIR / "predictions.jsonl"
APP_LOG_PATH = LOGS_DIR / "app.log"

# Create directories if they do not exist
for directory in [DATA_DIR / "raw", PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Standardized Feature Columns
FEATURE_COLUMNS = [
    "cgpa",
    "technical_score",
    "aptitude_score",
    "communication_score",
    "internship_exp",
    "projects_completed",
    "attendance",
]

TARGET_COLUMN = "placement_status"

# Feature constraints for validation & documentation
FEATURE_SPECS = {
    "cgpa": {"min": 0.0, "max": 10.0, "type": "float", "label": "CGPA"},
    "technical_score": {"min": 0.0, "max": 100.0, "type": "float", "label": "Technical Skills Score"},
    "aptitude_score": {"min": 0.0, "max": 100.0, "type": "float", "label": "Aptitude Score"},
    "communication_score": {"min": 0.0, "max": 100.0, "type": "float", "label": "Communication Score"},
    "internship_exp": {"valid_values": [0, 1], "type": "binary", "label": "Internship Experience (yes/no)"},
    "projects_completed": {"min": 0, "max": 50, "type": "int", "label": "Projects Completed"},
    "attendance": {"min": 0.0, "max": 100.0, "type": "float", "label": "Attendance Percentage"},
}

# Model Settings
RANDOM_STATE = 42
TEST_SIZE = 0.20
CLASSIFICATION_THRESHOLD = 0.50
MODEL_VERSION = "v1.0.0"

# Synthetic Data Disclosure Note
SYNTHETIC_DISCLOSURE = (
    "DEMO MODE NOTICE: Trained on synthetic educational simulation data for Agile and DevOps Lab demonstrations. "
    "Predictions are algorithmic estimations based on domain rules and do not represent guaranteed real-world placement outcomes."
)
