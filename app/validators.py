from typing import Dict, Any, Tuple, List, Optional


class ValidationError(Exception):
    def __init__(self, errors: List[Dict[str, str]]):
        self.errors = errors
        super().__init__(str(errors))


def validate_student_input(data: Any) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, str]]]:
    """
    Validates payload against student placement feature requirements.
    Returns:
        (validated_dict, error_list)
    If error_list is not empty, validated_dict is None.
    """
    errors: List[Dict[str, str]] = []

    if not isinstance(data, dict):
        return None, [{"field": "body", "message": "Request body must be a JSON object."}]

    validated: Dict[str, Any] = {}

    # 1. CGPA: 0.0 to 10.0
    if "cgpa" not in data or data["cgpa"] is None:
        errors.append({"field": "cgpa", "message": "CGPA is required."})
    else:
        try:
            val = float(data["cgpa"])
            if not (0.0 <= val <= 10.0):
                errors.append({
                    "field": "cgpa",
                    "message": f"CGPA must be a number between 0.0 and 10.0 (received {val})."
                })
            else:
                validated["cgpa"] = round(val, 2)
        except (ValueError, TypeError):
            errors.append({
                "field": "cgpa",
                "message": f"CGPA must be a valid numeric value between 0.0 and 10.0 (received '{data.get('cgpa')}')."
            })

    # 2. Technical Skills Score: 0.0 to 100.0
    tech_val = data.get("technical_score") or data.get("technical_skills_score")
    if tech_val is None:
        errors.append({"field": "technical_score", "message": "Technical skills score is required."})
    else:
        try:
            val = float(tech_val)
            if not (0.0 <= val <= 100.0):
                errors.append({
                    "field": "technical_score",
                    "message": f"Technical skills score must be between 0.0 and 100.0 (received {val})."
                })
            else:
                validated["technical_score"] = round(val, 1)
        except (ValueError, TypeError):
            errors.append({
                "field": "technical_score",
                "message": f"Technical skills score must be a valid number between 0.0 and 100.0 (received '{tech_val}')."
            })

    # 3. Aptitude Score: 0.0 to 100.0
    if "aptitude_score" not in data or data["aptitude_score"] is None:
        errors.append({"field": "aptitude_score", "message": "Aptitude score is required."})
    else:
        try:
            val = float(data["aptitude_score"])
            if not (0.0 <= val <= 100.0):
                errors.append({
                    "field": "aptitude_score",
                    "message": f"Aptitude score must be between 0.0 and 100.0 (received {val})."
                })
            else:
                validated["aptitude_score"] = round(val, 1)
        except (ValueError, TypeError):
            errors.append({
                "field": "aptitude_score",
                "message": f"Aptitude score must be a valid number between 0.0 and 100.0 (received '{data.get('aptitude_score')}')."
            })

    # 4. Communication Score: 0.0 to 100.0
    if "communication_score" not in data or data["communication_score"] is None:
        errors.append({"field": "communication_score", "message": "Communication score is required."})
    else:
        try:
            val = float(data["communication_score"])
            if not (0.0 <= val <= 100.0):
                errors.append({
                    "field": "communication_score",
                    "message": f"Communication score must be between 0.0 and 100.0 (received {val})."
                })
            else:
                validated["communication_score"] = round(val, 1)
        except (ValueError, TypeError):
            errors.append({
                "field": "communication_score",
                "message": f"Communication score must be a valid number between 0.0 and 100.0 (received '{data.get('communication_score')}')."
            })

    # 5. Internship Experience: (yes/no, boolean, or 1/0)
    intern_val = data.get("internship_exp")
    if intern_val is None:
        intern_val = data.get("internship_experience")
    if intern_val is None:
        intern_val = data.get("internship")

    if intern_val is None:
        errors.append({
            "field": "internship_experience",
            "message": "Internship experience is required ('yes' or 'no')."
        })
    else:
        if isinstance(intern_val, bool):
            validated["internship_exp"] = 1 if intern_val else 0
        elif isinstance(intern_val, (int, float)) and intern_val in [0, 1]:
            validated["internship_exp"] = int(intern_val)
        elif isinstance(intern_val, str):
            cleaned = intern_val.strip().lower()
            if cleaned in ["yes", "y", "true", "1"]:
                validated["internship_exp"] = 1
            elif cleaned in ["no", "n", "false", "0"]:
                validated["internship_exp"] = 0
            else:
                errors.append({
                    "field": "internship_experience",
                    "message": f"Internship experience must be 'yes' or 'no' (received '{intern_val}')."
                })
        else:
            errors.append({
                "field": "internship_experience",
                "message": f"Internship experience must be 'yes' or 'no' (received '{intern_val}')."
            })

    # 6. Projects Completed: Non-negative integer (>= 0)
    proj_val = data.get("projects_completed")
    if proj_val is None:
        proj_val = data.get("projects")

    if proj_val is None:
        errors.append({"field": "projects_completed", "message": "Projects completed count is required."})
    else:
        try:
            # Check for non-integer floats like 2.5
            float_val = float(proj_val)
            if not float_val.is_integer():
                errors.append({
                    "field": "projects_completed",
                    "message": f"Projects completed must be an integer count (received {proj_val})."
                })
            elif int(float_val) < 0:
                errors.append({
                    "field": "projects_completed",
                    "message": f"Projects completed must be a non-negative integer (received {proj_val})."
                })
            else:
                validated["projects_completed"] = int(float_val)
        except (ValueError, TypeError):
            errors.append({
                "field": "projects_completed",
                "message": f"Projects completed must be a non-negative integer (received '{proj_val}')."
            })

    # 7. Attendance: 0.0 to 100.0
    if "attendance" not in data or data["attendance"] is None:
        errors.append({"field": "attendance", "message": "Attendance percentage is required."})
    else:
        try:
            val = float(data["attendance"])
            if not (0.0 <= val <= 100.0):
                errors.append({
                    "field": "attendance",
                    "message": f"Attendance must be between 0.0 and 100.0 (received {val})."
                })
            else:
                validated["attendance"] = round(val, 1)
        except (ValueError, TypeError):
            errors.append({
                "field": "attendance",
                "message": f"Attendance must be a valid number between 0.0 and 100.0 (received '{data.get('attendance')}')."
            })

    if errors:
        return None, errors

    return validated, []
