import numpy as np
import pandas as pd
from pathlib import Path
from src.config import RAW_DATA_PATH, FEATURE_COLUMNS, RANDOM_STATE


def generate_placement_dataset(
    n_samples: int = 1500, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """
    Generates a realistic synthetic dataset for student campus placement prediction.
    Features:
      - cgpa: Continuous 5.0 to 10.0
      - technical_score: 30 to 100
      - aptitude_score: 30 to 100
      - communication_score: 30 to 100
      - internships: Count 0 to 4
      - projects: Count 0 to 8
      - attendance: Percentage 60.0 to 100.0
    Target:
      - placement_status: 1 (Likely/Placed) or 0 (Unlikely/Not Placed)
    """
    rng = np.random.default_rng(random_state)

    # 1. CGPA: Normal distribution centered at 7.4
    cgpa = np.clip(rng.normal(loc=7.4, scale=1.1, size=n_samples), 5.0, 9.9)
    cgpa = np.round(cgpa, 2)

    # 2. Technical Score: Positively correlated with CGPA
    tech_base = (cgpa / 10.0) * 55 + rng.normal(loc=25, scale=10, size=n_samples)
    technical_score = np.clip(np.round(tech_base, 1), 35.0, 99.0)

    # 3. Aptitude Score: Independent cognitive evaluation
    aptitude_score = np.clip(
        np.round(rng.normal(loc=70.0, scale=13.0, size=n_samples), 1),
        30.0,
        98.0,
    )

    # 4. Communication Score
    communication_score = np.clip(
        np.round(rng.normal(loc=72.0, scale=12.5, size=n_samples), 1),
        35.0,
        99.0,
    )

    # 5. Internships: Poisson distribution biased towards 0-3
    internships = np.clip(rng.poisson(lam=1.1, size=n_samples), 0, 4)

    # 6. Projects Completed: Related to technical engagement
    projects = np.clip(
        rng.poisson(lam=2.5 + (technical_score > 75).astype(int), size=n_samples),
        0,
        7,
    )

    # 7. Attendance Percentage
    attendance = np.clip(
        np.round(rng.normal(loc=82.0, scale=8.5, size=n_samples), 1),
        60.0,
        99.5,
    )

    # Calculate realistic logit for placement probability
    # Weights reflect industry campus recruitment criteria
    z = (
        (cgpa - 7.0) * 1.8
        + (technical_score - 70.0) * 0.065
        + (aptitude_score - 65.0) * 0.045
        + (communication_score - 65.0) * 0.035
        + (internships * 0.75)
        + (projects * 0.45)
        + (attendance - 75.0) * 0.03
        + rng.normal(0, 0.45, size=n_samples)  # Real-world noise
    )

    # Sigmoid function
    probabilities = 1 / (1 + np.exp(-z))
    placement_status = (probabilities >= 0.50).astype(int)

    df = pd.DataFrame(
        {
            "cgpa": cgpa,
            "technical_score": technical_score,
            "aptitude_score": aptitude_score,
            "communication_score": communication_score,
            "internships": internships,
            "projects": projects,
            "attendance": attendance,
            "placement_probability": np.round(probabilities, 4),
            "placement_status": placement_status,
        }
    )

    return df


def save_dataset_if_not_exists(
    output_path: Path = RAW_DATA_PATH, n_samples: int = 1500
) -> Path:
    """Generates and saves the raw dataset if not present."""
    if not output_path.exists():
        df = generate_placement_dataset(n_samples=n_samples)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Save training data without the leak probability column
        training_df = df[FEATURE_COLUMNS + ["placement_status"]]
        training_df.to_csv(output_path, index=False)
        print(f"[Data Generator] Created dataset at {output_path} ({len(training_df)} records)")
    return output_path


if __name__ == "__main__":
    save_dataset_if_not_exists()
