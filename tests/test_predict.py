import pytest
from ml.predict import predictor


def test_predictor_loaded_and_ready():
    assert predictor.is_ready() is True


def test_predict_high_achiever():
    student_data = {
        "cgpa": 9.4,
        "technical_score": 92.0,
        "aptitude_score": 88.0,
        "communication_score": 86.0,
        "internship_exp": 1,
        "projects_completed": 4,
        "attendance": 94.0,
    }
    result = predictor.predict(student_data)
    assert result["prediction"] == "Likely"
    assert result["probability"] >= 0.70
    assert 0.0 <= result["probability"] <= 1.0
    assert result["confidence"] == "High"
    assert "prediction_id" in result
    assert len(result["key_factors"]) > 0
    assert "demo_mode" in result


def test_predict_at_risk_student():
    student_data = {
        "cgpa": 5.4,
        "technical_score": 42.0,
        "aptitude_score": 45.0,
        "communication_score": 48.0,
        "internship_exp": 0,
        "projects_completed": 0,
        "attendance": 62.0,
    }
    result = predictor.predict(student_data)
    assert result["prediction"] == "Unlikely"
    assert result["probability"] <= 0.40
    assert result["confidence"] in ["High", "Moderate"]
    assert any("CGPA" in r or "Technical" in r for r in result["recommendations"])


def test_probability_format_and_range():
    student_data = {
        "cgpa": 7.5,
        "technical_score": 75.0,
        "aptitude_score": 72.0,
        "communication_score": 70.0,
        "internship_exp": 0,
        "projects_completed": 2,
        "attendance": 80.0,
    }
    result = predictor.predict(student_data)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["probability_percentage"].endswith("%")
    assert round(result["probability"] + result["probability_unlikely"], 2) == 1.00


def test_synthetic_demo_disclosure_present():
    student_data = {
        "cgpa": 8.0,
        "technical_score": 80.0,
        "aptitude_score": 75.0,
        "communication_score": 75.0,
        "internship_exp": 1,
        "projects_completed": 2,
        "attendance": 85.0,
    }
    result = predictor.predict(student_data)
    demo_info = result["demo_mode"]
    assert "is_synthetic" in demo_info
    assert "disclosure" in demo_info
    assert "DEMO MODE NOTICE" in demo_info["disclosure"]
