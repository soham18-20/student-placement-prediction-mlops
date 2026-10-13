import pytest
from pathlib import Path
from ml.predict import PlacementPredictor, ModelArtifactNotFoundError
from app import create_app
from app.routes import predictor as route_predictor


def test_predictor_missing_model_raises_error(tmp_path):
    fake_model_path = tmp_path / "non_existent_model.joblib"
    fake_metrics_path = tmp_path / "non_existent_metrics.json"

    custom_predictor = PlacementPredictor(
        model_path=fake_model_path, metrics_path=fake_metrics_path
    )
    assert custom_predictor.is_ready() is False

    with pytest.raises(ModelArtifactNotFoundError) as exc_info:
        custom_predictor.predict({
            "cgpa": 8.0,
            "technical_score": 80.0,
            "aptitude_score": 75.0,
            "communication_score": 75.0,
            "internship_exp": 1,
            "projects_completed": 2,
            "attendance": 85.0,
        })
    assert "Model artifact not found" in str(exc_info.value)
    assert "scripts/train.py" in str(exc_info.value)


def test_api_handles_missing_artifacts_gracefully(monkeypatch):
    app = create_app({"TESTING": True})
    client = app.test_client()

    # Mock route_predictor to simulate missing model
    monkeypatch.setattr(route_predictor, "pipeline", None)

    def mock_predict(*args, **kwargs):
        raise ModelArtifactNotFoundError("Simulated missing model artifact for test.")

    monkeypatch.setattr(route_predictor, "predict", mock_predict)

    payload = {
        "cgpa": 8.5,
        "technical_score": 80.0,
        "aptitude_score": 75.0,
        "communication_score": 75.0,
        "internship_experience": "yes",
        "projects_completed": 2,
        "attendance": 85.0,
    }

    response = client.post("/api/predict", json=payload)
    assert response.status_code == 503
    data = response.get_json()
    assert data["status"] == "error"
    assert data["error"] == "Model Not Ready"
    assert "instructions" in data
    assert "scripts/train.py" in data["instructions"]
