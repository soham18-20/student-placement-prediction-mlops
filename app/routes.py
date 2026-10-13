import time
import os
from flask import Blueprint, request, jsonify, send_from_directory, current_app

from ml.predict import predictor, ModelArtifactNotFoundError
from ml.evaluate import load_evaluation_report
from ml.config import MODEL_VERSION, SYNTHETIC_DISCLOSURE
from app.validators import validate_student_input

api_bp = Blueprint("api", __name__)
APP_START_TIME = time.time()


@api_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint indicating service status, model availability, and uptime."""
    uptime_seconds = round(time.time() - APP_START_TIME, 2)
    model_ready = predictor.is_ready()

    return jsonify({
        "status": "healthy" if model_ready else "degraded",
        "service": "Student Placement Prediction System with MLOps",
        "version": predictor.metrics.get("model_version", MODEL_VERSION),
        "uptime_seconds": uptime_seconds,
        "model_loaded": model_ready,
        "algorithm": predictor.metrics.get("algorithm", "RandomForestClassifier"),
        "demo_mode": predictor.metrics.get("dataset", {}).get("is_synthetic", True),
    }), 200


@api_bp.route("/api/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint.
    Accepts JSON with student parameters, validates fields, and returns placement probability and classification.
    """
    if not request.is_json:
        return jsonify({
            "status": "error",
            "error": "Invalid Content-Type",
            "message": "Request payload must be JSON with Content-Type: application/json.",
        }), 400

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({
            "status": "error",
            "error": "Malformed JSON",
            "message": "Failed to parse JSON body. Ensure the JSON syntax is valid.",
        }), 400

    validated_data, errors = validate_student_input(data)
    if errors:
        return jsonify({
            "status": "error",
            "error": "Validation Failed",
            "message": "One or more input fields failed validation.",
            "errors": errors,
        }), 400

    try:
        prediction_result = predictor.predict(validated_data)
        return jsonify({
            "status": "success",
            "data": prediction_result,
        }), 200
    except ModelArtifactNotFoundError as e:
        return jsonify({
            "status": "error",
            "error": "Model Not Ready",
            "message": str(e),
            "instructions": "Execute 'python scripts/train.py' to train and serialize the model pipeline.",
        }), 503
    except Exception as e:
        current_app.logger.error(f"Inference error: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": "Inference Error",
            "message": "An unexpected error occurred during prediction.",
            "details": str(e),
        }), 500


@api_bp.route("/api/metrics", methods=["GET"])
def get_metrics():
    """Returns model evaluation metrics, confusion matrix, and baseline comparison."""
    try:
        report = load_evaluation_report()
        return jsonify({
            "status": "success",
            "data": report,
        }), 200
    except FileNotFoundError as e:
        return jsonify({
            "status": "error",
            "error": "Metrics Not Found",
            "message": str(e),
            "instructions": "Execute 'python scripts/train.py' to generate metrics.",
        }), 404
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": "Failed to Load Metrics",
            "details": str(e),
        }), 500


@api_bp.route("/api/model-info", methods=["GET"])
def get_model_info():
    """Returns model metadata, algorithm specs, and demo disclosure notice."""
    return jsonify({
        "status": "success",
        "model_version": predictor.metrics.get("model_version", MODEL_VERSION),
        "algorithm": predictor.metrics.get("algorithm", "RandomForestClassifier"),
        "features": predictor.metrics.get("features", []),
        "demo_mode": predictor.metrics.get("dataset", {}).get("is_synthetic", True),
        "disclosure": predictor.metrics.get("dataset", {}).get("disclosure", SYNTHETIC_DISCLOSURE),
    }), 200
