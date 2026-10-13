import pytest
from app.validators import validate_student_input


def test_valid_student_input():
    payload = {
        "cgpa": 8.5,
        "technical_score": 85.0,
        "aptitude_score": 80.0,
        "communication_score": 75.0,
        "internship_experience": "yes",
        "projects_completed": 3,
        "attendance": 90.0,
    }
    validated, errors = validate_student_input(payload)
    assert errors == []
    assert validated is not None
    assert validated["cgpa"] == 8.5
    assert validated["internship_exp"] == 1
    assert validated["projects_completed"] == 3


def test_invalid_cgpa_out_of_range():
    # CGPA greater than 10.0
    payload = {
        "cgpa": 12.5,
        "technical_score": 80.0,
        "aptitude_score": 70.0,
        "communication_score": 70.0,
        "internship_experience": "no",
        "projects_completed": 2,
        "attendance": 80.0,
    }
    validated, errors = validate_student_input(payload)
    assert validated is None
    assert any(e["field"] == "cgpa" and "between 0.0 and 10.0" in e["message"] for e in errors)


def test_invalid_cgpa_negative():
    payload = {
        "cgpa": -1.0,
        "technical_score": 80.0,
        "aptitude_score": 70.0,
        "communication_score": 70.0,
        "internship_experience": "no",
        "projects_completed": 2,
        "attendance": 80.0,
    }
    validated, errors = validate_student_input(payload)
    assert validated is None
    assert any(e["field"] == "cgpa" for e in errors)


def test_invalid_technical_score():
    payload = {
        "cgpa": 8.0,
        "technical_score": 105.0,
        "aptitude_score": 70.0,
        "communication_score": 70.0,
        "internship_experience": "yes",
        "projects_completed": 2,
        "attendance": 80.0,
    }
    validated, errors = validate_student_input(payload)
    assert validated is None
    assert any(e["field"] == "technical_score" for e in errors)


def test_invalid_internship_experience_string():
    payload = {
        "cgpa": 8.0,
        "technical_score": 80.0,
        "aptitude_score": 70.0,
        "communication_score": 70.0,
        "internship_experience": "maybe",
        "projects_completed": 2,
        "attendance": 80.0,
    }
    validated, errors = validate_student_input(payload)
    assert validated is None
    assert any(e["field"] == "internship_experience" and "'yes' or 'no'" in e["message"] for e in errors)


def test_valid_internship_boolean_and_variations():
    for val, expected in [("YES", 1), ("No", 0), (True, 1), (False, 0), (1, 1), (0, 0)]:
        payload = {
            "cgpa": 8.0,
            "technical_score": 80.0,
            "aptitude_score": 70.0,
            "communication_score": 70.0,
            "internship_experience": val,
            "projects_completed": 2,
            "attendance": 80.0,
        }
        validated, errors = validate_student_input(payload)
        assert errors == []
        assert validated["internship_exp"] == expected


def test_invalid_projects_negative():
    payload = {
        "cgpa": 8.0,
        "technical_score": 80.0,
        "aptitude_score": 70.0,
        "communication_score": 70.0,
        "internship_experience": "yes",
        "projects_completed": -3,
        "attendance": 80.0,
    }
    validated, errors = validate_student_input(payload)
    assert validated is None
    assert any(e["field"] == "projects_completed" and "non-negative integer" in e["message"] for e in errors)


def test_invalid_projects_float():
    payload = {
        "cgpa": 8.0,
        "technical_score": 80.0,
        "aptitude_score": 70.0,
        "communication_score": 70.0,
        "internship_experience": "yes",
        "projects_completed": 2.7,
        "attendance": 80.0,
    }
    validated, errors = validate_student_input(payload)
    assert validated is None
    assert any(e["field"] == "projects_completed" and "integer count" in e["message"] for e in errors)


def test_invalid_attendance():
    payload = {
        "cgpa": 8.0,
        "technical_score": 80.0,
        "aptitude_score": 70.0,
        "communication_score": 70.0,
        "internship_experience": "yes",
        "projects_completed": 2,
        "attendance": 105.0,
    }
    validated, errors = validate_student_input(payload)
    assert validated is None
    assert any(e["field"] == "attendance" for e in errors)


def test_missing_required_fields():
    payload = {"cgpa": 8.0}
    validated, errors = validate_student_input(payload)
    assert validated is None
    fields = [e["field"] for e in errors]
    assert "technical_score" in fields
    assert "aptitude_score" in fields
    assert "communication_score" in fields
    assert "internship_experience" in fields
    assert "projects_completed" in fields
    assert "attendance" in fields


def test_non_dict_payload():
    validated, errors = validate_student_input(["not", "a", "dict"])
    assert validated is None
    assert errors[0]["field"] == "body"
