import pytest
from app import create_app


@pytest.fixture
def client():
    app = create_app({"TESTING": True})
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] in ["healthy", "degraded"]
    assert "version" in data
    assert "uptime_seconds" in data
    assert "model_loaded" in data


def test_predict_success(client):
    payload = {
        "cgpa": 8.8,
        "technical_score": 85.0,
        "aptitude_score": 80.0,
        "communication_score": 78.0,
        "internship_experience": "yes",
        "projects_completed": 3,
        "attendance": 90.0,
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    pred_data = data["data"]
    assert pred_data["prediction"] in ["Likely", "Unlikely"]
    assert 0.0 <= pred_data["probability"] <= 1.0
    assert "probability_percentage" in pred_data
    assert "recommendations" in pred_data
    assert "key_factors" in pred_data


def test_predict_validation_error_400(client):
    # Invalid CGPA (> 10.0)
    payload = {
        "cgpa": 14.5,
        "technical_score": 85.0,
        "aptitude_score": 80.0,
        "communication_score": 78.0,
        "internship_experience": "yes",
        "projects_completed": 3,
        "attendance": 90.0,
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert "errors" in data
    assert any(e["field"] == "cgpa" for e in data["errors"])


def test_predict_non_json_payload(client):
    response = client.post("/api/predict", data="not json", content_type="text/plain")
    assert response.status_code == 400
    data = response.get_json()
    assert data["error"] == "Invalid Content-Type"


def test_metrics_endpoint(client):
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    report = data["data"]
    assert "primary_metrics" in report
    assert "accuracy" in report["primary_metrics"]
    assert "confusion_matrix" in report["primary_metrics"]
    assert "baseline_comparison" in report


def test_model_info_endpoint(client):
    response = client.get("/api/model-info")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "model_version" in data
    assert "features" in data
    assert "demo_mode" in data
    assert "disclosure" in data


def test_dashboard_route(client):
    response = client.get("/")
    assert response.status_code == 200
    # Should serve HTML dashboard
    assert b"Student Placement Prediction System" in response.data
