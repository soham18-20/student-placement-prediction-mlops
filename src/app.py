import time
import os
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from src.config import (
    STATIC_DIR,
    CURRENT_MODEL_VERSION,
    RAW_DATA_PATH,
    MODEL_PATH,
    FEATURE_COLUMNS,
)
from src.predict import (
    StudentInput,
    PredictionResponse,
    predictor,
)
from src.train import train_model
from src.drift_detector import DataDriftDetector

from contextlib import asynccontextmanager

# App startup time for uptime tracking
START_TIME = time.time()

# Request counters for Prometheus-style monitoring
METRICS_STORE = {
    "total_predictions": 0,
    "likely_placements": 0,
    "unlikely_placements": 0,
    "retrain_count": 0,
    "total_latency_seconds": 0.0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure data is generated and model is trained if not already present."""
    if not MODEL_PATH.exists():
        print("[Startup] Initializing first-time model training...")
        train_model()
    predictor.load_artifacts()
    yield


app = FastAPI(
    title="Student Placement Prediction API",
    description="Production-ready MLOps API for Student Placement Prediction with Automated Retraining, CI/CD, and Drift Detection.",
    version=CURRENT_MODEL_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

drift_detector = DataDriftDetector()


# Mount Static UI
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Student Placement Prediction API is running. Visit /docs for Swagger UI."}


@app.get("/health", tags=["Monitoring & Health"])
async def health_check():
    """System health check and component status."""
    uptime_seconds = round(time.time() - START_TIME, 2)
    model_ready = predictor.is_ready()

    return {
        "status": "healthy" if model_ready else "degraded",
        "service": "student-placement-mlops",
        "version": predictor.metadata.get("model_version", CURRENT_MODEL_VERSION),
        "uptime_seconds": uptime_seconds,
        "model_loaded": model_ready,
        "algorithm": predictor.metadata.get("algorithm", "RandomForestClassifier"),
    }


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_single(student: StudentInput):
    """Predict campus placement likelihood for an individual student."""
    t0 = time.time()
    try:
        result = predictor.predict(student)
        latency = time.time() - t0

        METRICS_STORE["total_predictions"] += 1
        METRICS_STORE["total_latency_seconds"] += latency
        if result.prediction == "Likely":
            METRICS_STORE["likely_placements"] += 1
        else:
            METRICS_STORE["unlikely_placements"] += 1

        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction inference failed: {str(e)}",
        )


@app.post("/api/v1/predict/batch", response_model=List[PredictionResponse], tags=["Inference"])
async def predict_batch(students: List[StudentInput]):
    """Batch prediction endpoint for processing cohorts of students."""
    if not students:
        raise HTTPException(status_code=400, detail="Student list cannot be empty.")
    if len(students) > 500:
        raise HTTPException(status_code=400, detail="Maximum batch limit is 500 records.")

    results: List[PredictionResponse] = []
    t0 = time.time()
    for student in students:
        res = predictor.predict(student)
        results.append(res)
        METRICS_STORE["total_predictions"] += 1
        if res.prediction == "Likely":
            METRICS_STORE["likely_placements"] += 1
        else:
            METRICS_STORE["unlikely_placements"] += 1

    METRICS_STORE["total_latency_seconds"] += (time.time() - t0)
    return results


@app.post("/api/v1/retrain", tags=["MLOps Pipeline"])
async def retrain_pipeline():
    """
    Triggers automated model retraining on updated dataset.
    Hot-reloads model in memory without server downtime.
    """
    try:
        old_version = predictor.metadata.get("model_version", CURRENT_MODEL_VERSION)
        old_acc = predictor.metadata.get("metrics", {}).get("accuracy", 0.0)

        # Generate bumped version tag
        try:
            parts = old_version.lstrip("v").split(".")
            new_patch = int(parts[-1]) + 1
            new_version = f"v{parts[0]}.{parts[1]}.{new_patch}"
        except Exception:
            new_version = f"{old_version}-retrained"

        _, _, new_metadata = train_model(version=new_version)
        predictor.load_artifacts()
        METRICS_STORE["retrain_count"] += 1

        new_acc = new_metadata["metrics"]["accuracy"]
        return {
            "status": "success",
            "message": "Model retrained and hot-reloaded successfully.",
            "previous_version": old_version,
            "new_version": new_version,
            "accuracy_change": {
                "before": old_acc,
                "after": new_acc,
                "delta": round(new_acc - old_acc, 4),
            },
            "metrics": new_metadata["metrics"],
            "trained_at": new_metadata["training_timestamp"],
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retraining pipeline failed: {str(e)}",
        )


@app.get("/api/v1/model/metadata", tags=["MLOps Pipeline"])
async def get_model_metadata():
    """Retrieve active model version, performance metrics, and feature importances."""
    if not predictor.metadata:
        predictor.load_artifacts()
    return predictor.metadata


@app.get("/api/v1/monitoring/drift", tags=["Monitoring & Health"])
async def get_data_drift():
    """Inspect production inference features for statistical distribution drift."""
    return drift_detector.evaluate_recent_data()


@app.get("/api/v1/monitoring/metrics", tags=["Monitoring & Health"])
async def get_prometheus_metrics():
    """Return operational metrics for monitoring & viva demonstrations."""
    total = METRICS_STORE["total_predictions"]
    avg_latency = (
        round(METRICS_STORE["total_latency_seconds"] / total, 4) if total > 0 else 0.0
    )
    placement_rate = (
        round((METRICS_STORE["likely_placements"] / total) * 100, 1) if total > 0 else 0.0
    )

    return {
        "total_requests": total,
        "likely_predictions": METRICS_STORE["likely_placements"],
        "unlikely_predictions": METRICS_STORE["unlikely_placements"],
        "predicted_placement_rate_percent": placement_rate,
        "average_latency_seconds": avg_latency,
        "automated_retrains_triggered": METRICS_STORE["retrain_count"],
        "model_version": predictor.metadata.get("model_version", CURRENT_MODEL_VERSION),
        "status": "OPERATIONAL",
    }


@app.get("/api/v1/sample-data", tags=["Testing & Demo"])
async def get_sample_students():
    """Preset profiles to test directly in the UI or during viva voce."""
    return [
        {
            "name": "High Achiever (Top Tier)",
            "data": {
                "cgpa": 9.2,
                "technical_score": 92.0,
                "aptitude_score": 88.0,
                "communication_score": 85.0,
                "internships": 2,
                "projects": 4,
                "attendance": 92.0,
            },
            "expected": "Likely (High Probability)",
        },
        {
            "name": "Average Student (Borderline)",
            "data": {
                "cgpa": 7.1,
                "technical_score": 68.0,
                "aptitude_score": 64.0,
                "communication_score": 65.0,
                "internships": 1,
                "projects": 2,
                "attendance": 78.0,
            },
            "expected": "Borderline / Moderate",
        },
        {
            "name": "At-Risk Student",
            "data": {
                "cgpa": 5.8,
                "technical_score": 45.0,
                "aptitude_score": 48.0,
                "communication_score": 52.0,
                "internships": 0,
                "projects": 1,
                "attendance": 64.0,
            },
            "expected": "Unlikely (Low Probability)",
        },
    ]
