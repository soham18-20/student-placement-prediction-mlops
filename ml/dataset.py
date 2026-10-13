import logging
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd

from ml.config import (
    RAW_DATA_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    RANDOM_STATE,
    SYNTHETIC_DISCLOSURE,
)

logger = logging.getLogger("ml.dataset")


def generate_synthetic_dataset(
    n_samples: int = 1500, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """
    Generates a realistic synthetic dataset for student campus placement prediction.
    Features:
      - cgpa: Continuous (5.0 to 10.0)
      - technical_score: Continuous (30.0 to 100.0)
      - aptitude_score: Continuous (30.0 to 100.0)
      - communication_score: Continuous (30.0 to 100.0)
      - internship_exp: Binary (0 = No, 1 = Yes)
      - projects_completed: Non-negative integer (0 to 8)
      - attendance: Continuous (60.0 to 100.0)
    Target:
      - placement_status: Binary (0 = Unlikely, 1 = Likely)
    """
    rng = np.random.default_rng(random_state)

    # 1. CGPA: Realistic academic normal distribution
    cgpa = np.clip(rng.normal(loc=7.4, scale=1.1, size=n_samples), 5.0, 9.9)
    cgpa = np.round(cgpa, 2)

    # 2. Technical Score: Positively correlated with CGPA with realistic variance
    tech_mean = (cgpa / 10.0) * 55 + 25
    technical_score = np.clip(
        np.round(rng.normal(loc=tech_mean, scale=10.0, size=n_samples), 1),
        30.0,
        99.0,
    )

    # 3. Aptitude Score: Standard cognitive evaluation
    aptitude_score = np.clip(
        np.round(rng.normal(loc=70.0, scale=13.0, size=n_samples), 1),
        30.0,
        99.0,
    )

    # 4. Communication Score: Soft skills & verbal proficiency
    communication_score = np.clip(
        np.round(rng.normal(loc=71.0, scale=12.0, size=n_samples), 1),
        30.0,
        99.0,
    )

    # 5. Internship Experience: Binary (0 = No, 1 = Yes), ~45% students have completed an internship
    internship_prob = np.clip(0.20 + (cgpa - 6.5) * 0.15 + (technical_score - 60) * 0.005, 0.05, 0.85)
    internship_exp = rng.binomial(n=1, p=internship_prob, size=n_samples)

    # 6. Projects Completed: Non-negative integer count
    projects_completed = np.clip(
        rng.poisson(lam=2.2 + 0.8 * internship_exp + (technical_score > 75).astype(int), size=n_samples),
        0,
        8,
    )

    # 7. Attendance Percentage: Lecture presence
    attendance = np.clip(
        np.round(rng.normal(loc=82.0, scale=8.5, size=n_samples), 1),
        60.0,
        99.5,
    )

    # Latent recruitment score (logit) reflecting campus criteria + natural stochastic variance
    logit = (
        (cgpa - 7.0) * 1.70
        + (technical_score - 70.0) * 0.055
        + (aptitude_score - 65.0) * 0.040
        + (communication_score - 65.0) * 0.035
        + (internship_exp * 1.10)
        + (projects_completed * 0.35)
        + (attendance - 75.0) * 0.025
        + rng.normal(0, 0.50, size=n_samples)  # Real-world variance/noise
    )

    probabilities = 1.0 / (1.0 + np.exp(-logit))
    placement_status = (probabilities >= 0.50).astype(int)

    df = pd.DataFrame(
        {
            "cgpa": cgpa,
            "technical_score": technical_score,
            "aptitude_score": aptitude_score,
            "communication_score": communication_score,
            "internship_exp": internship_exp,
            "projects_completed": projects_completed,
            "attendance": attendance,
            "placement_status": placement_status,
        }
    )

    return df


def load_or_create_dataset(
    filepath: Path = RAW_DATA_PATH, force_regenerate: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads an existing dataset from disk or generates a synthetic demo dataset.
    Returns:
        (dataframe, dataset_metadata)
    """
    is_synthetic = False

    if filepath.exists() and not force_regenerate:
        logger.info(f"Loading existing placement dataset from: {filepath}")
        df = pd.read_csv(filepath)
        # Check if columns match expected features
        missing_cols = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
        if missing_cols:
            logger.warning(
                f"Dataset at {filepath} lacks columns {missing_cols}. Generating synthetic dataset."
            )
            df = generate_synthetic_dataset()
            filepath.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(filepath, index=False)
            is_synthetic = True
    else:
        logger.info(f"Generating synthetic placement dataset (Demo Mode) at: {filepath}")
        df = generate_synthetic_dataset()
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        is_synthetic = True

    metadata = {
        "dataset_path": str(filepath),
        "total_records": len(df),
        "is_synthetic": is_synthetic,
        "demo_mode_label": "Synthetic Educational Simulation Data" if is_synthetic else "Genuine Placement Dataset",
        "disclosure": SYNTHETIC_DISCLOSURE if is_synthetic else "Real-world institutional placement dataset.",
        "class_balance": {
            "unlikely_count": int((df[TARGET_COLUMN] == 0).sum()),
            "likely_count": int((df[TARGET_COLUMN] == 1).sum()),
            "placement_rate": round(float((df[TARGET_COLUMN] == 1).mean()), 4),
        },
    }

    return df, metadata


if __name__ == "__main__":
    df, meta = load_or_create_dataset()
    print("Dataset generated/loaded successfully:")
    print(meta)
    print(df.head())
