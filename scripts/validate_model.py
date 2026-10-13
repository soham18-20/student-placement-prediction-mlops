"""
Script: scripts/validate_model.py
Validates that serialized machine learning models can be loaded into memory
and successfully execute inference on sample candidate data.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Ensure UTF-8 output across Windows and POSIX shells
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def validate_models():
    print("=======================================================")
    print("🤖 Machine Learning Model Artifact Validation")
    print("=======================================================")

    errors = []

    # 1. Validate primary model (src/predict.py)
    try:
        from src.predict import predictor, StudentInput
        predictor.load_artifacts()
        if not predictor.is_ready():
            raise RuntimeError("FastAPI PlacementPredictor reports artifacts are not ready.")

        sample_input = StudentInput(
            cgpa=8.5,
            technical_score=85.0,
            aptitude_score=80.0,
            communication_score=78.0,
            internships=2,
            projects=3,
            attendance=90.0,
        )
        res = predictor.predict(sample_input)
        print("✅ Primary Model (FastAPI): Loaded successfully.")
        print(f"   Sample inference -> Prediction: {res.prediction}, Probability: {res.probability_percentage}, Conf: {res.confidence}")
    except Exception as e:
        print(f"❌ Primary Model (FastAPI) Validation Failed: {e}")
        errors.append(f"Primary model error: {e}")

    # 2. Validate pipeline model (ml/predict.py)
    try:
        from ml.predict import predictor as ml_predictor
        from app.validators import validate_student_input
        ml_predictor.load_model(strict=True)
        if not ml_predictor.is_ready():
            raise RuntimeError("Flask ML Pipeline Predictor reports artifacts are not ready.")

        ml_sample = {
            "cgpa": 8.5,
            "technical_score": 85.0,
            "aptitude_score": 80.0,
            "communication_score": 78.0,
            "internship_experience": "yes",
            "projects_completed": 3,
            "attendance": 90.0,
        }
        validated_input, val_errors = validate_student_input(ml_sample)
        if val_errors:
            raise ValueError(f"Input validation errors: {val_errors}")

        ml_res = ml_predictor.predict(validated_input)
        print("✅ Secondary Pipeline Model (Flask): Loaded successfully.")
        print(f"   Sample inference -> Prediction: {ml_res['prediction']}, Probability: {ml_res['probability_percentage']}, Conf: {ml_res['confidence']}")
    except Exception as e:
        print(f"❌ Secondary Pipeline Model (Flask) Validation Failed: {e}")
        errors.append(f"Pipeline model error: {e}")

    print("=======================================================")
    if errors:
        print(f"❌ Validation failed with {len(errors)} error(s).")
        sys.exit(1)
    else:
        print("🎉 All ML model artifacts validated successfully!")
        sys.exit(0)


if __name__ == "__main__":
    validate_models()
