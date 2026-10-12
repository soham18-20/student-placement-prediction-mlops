import pytest
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "version" in data
    assert "uptime_seconds" in data
    assert data["model_loaded"] is True


def test_predict_endpoint_success():
    payload = {
        "cgpa": 8.5,
        "technical_score": 88.0,
        "aptitude_score": 82.0,
        "communication_score": 80.0,
        "internships": 2,
        "projects": 3,
        "attendance": 90.0,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in ["Likely", "Unlikely"]
    assert "probability" in data
    assert "confidence" in data
    assert "recommendations" in data


def test_predict_endpoint_validation_error():
    # CGPA exceeding 10.0 should trigger HTTP 422 Unprocessable Entity
    invalid_payload = {
        "cgpa": 14.5,
        "technical_score": 88.0,
        "aptitude_score": 82.0,
        "communication_score": 80.0,
        "internships": 2,
        "projects": 3,
        "attendance": 90.0,
    }
    response = client.post("/api/v1/predict", json=invalid_payload)
    assert response.status_code == 422


def test_batch_predict_endpoint():
    cohort = [
        {
            "cgpa": 9.0,
            "technical_score": 90.0,
            "aptitude_score": 85.0,
            "communication_score": 85.0,
            "internships": 2,
            "projects": 3,
            "attendance": 90.0,
        },
        {
            "cgpa": 5.5,
            "technical_score": 42.0,
            "aptitude_score": 45.0,
            "communication_score": 50.0,
            "internships": 0,
            "projects": 1,
            "attendance": 65.0,
        },
    ]
    response = client.post("/api/v1/predict/batch", json=cohort)
    assert response.status_code == 200
    results = response.json()
    assert len(results) == 2
    assert results[0]["prediction"] == "Likely"
    assert results[1]["prediction"] == "Unlikely"


def test_metadata_endpoint():
    response = client.get("/api/v1/model/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert "metrics" in data
    assert "accuracy" in data["metrics"]


def test_metrics_endpoint():
    response = client.get("/api/v1/monitoring/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "status" in data


def test_drift_endpoint():
    response = client.get("/api/v1/monitoring/drift")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "drift_detected" in data
