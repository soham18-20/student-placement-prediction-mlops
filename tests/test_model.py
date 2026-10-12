import pytest
from src.predict import predictor, StudentInput


def test_predictor_is_ready():
    predictor.load_artifacts()
    assert predictor.is_ready() is True


def test_predict_high_achiever():
    student = StudentInput(
        cgpa=9.5,
        technical_score=95.0,
        aptitude_score=90.0,
        communication_score=90.0,
        internships=3,
        projects=5,
        attendance=95.0,
    )
    result = predictor.predict(student)
    assert result.prediction == "Likely"
    assert result.probability >= 0.70
    assert result.confidence == "High"
    assert len(result.recommendations) > 0


def test_predict_at_risk_student():
    student = StudentInput(
        cgpa=5.2,
        technical_score=40.0,
        aptitude_score=45.0,
        communication_score=45.0,
        internships=0,
        projects=0,
        attendance=62.0,
    )
    result = predictor.predict(student)
    assert result.prediction == "Unlikely"
    assert result.probability <= 0.40
    assert any("CGPA" in tip or "Technical" in tip for tip in result.recommendations)


def test_probability_bounds():
    student = StudentInput(
        cgpa=7.5,
        technical_score=72.0,
        aptitude_score=70.0,
        communication_score=70.0,
        internships=1,
        projects=2,
        attendance=80.0,
    )
    result = predictor.predict(student)
    assert 0.0 <= result.probability <= 1.0
    assert result.probability_percentage.endswith("%")
