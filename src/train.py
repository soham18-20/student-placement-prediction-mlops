import datetime
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_DIR,
    MODEL_PATH,
    SCALER_PATH,
    METADATA_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    CURRENT_MODEL_VERSION,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.data_generator import save_dataset_if_not_exists

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MLOps-Train")


def train_model(
    data_path: Path = RAW_DATA_PATH,
    version: str = CURRENT_MODEL_VERSION,
    save_artifacts: bool = True,
) -> Tuple[RandomForestClassifier, StandardScaler, Dict[str, Any]]:
    """
    Trains, evaluates, and versions the student placement prediction model.
    Returns:
        (trained_model, scaler, metadata_dict)
    """
    logger.info("Initializing model training pipeline...")
    save_dataset_if_not_exists(data_path)

    df = pd.read_csv(data_path)
    # Support both column naming conventions seamlessly
    rename_map = {
        "internship_exp": "internships",
        "projects_completed": "projects",
    }
    df = df.rename(columns=rename_map)
    logger.info(f"Loaded dataset: {len(df)} samples, features: {FEATURE_COLUMNS}")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Preprocessing: StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save processed splits for reproducibility
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DATA_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)

    # Train RandomForestClassifier
    model_params = {
        "n_estimators": 120,
        "max_depth": 7,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": RANDOM_STATE,
    }
    model = RandomForestClassifier(**model_params)
    model.fit(X_train_scaled, y_train)

    # Evaluate on holdout test set
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Feature Importance
    feature_importances = {
        feat: round(float(imp), 4)
        for feat, imp in sorted(
            zip(FEATURE_COLUMNS, model.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )
    }

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }

    metadata: Dict[str, Any] = {
        "model_version": version,
        "algorithm": "RandomForestClassifier",
        "training_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset_size": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "hyperparameters": model_params,
        "features": FEATURE_COLUMNS,
        "metrics": metrics,
        "feature_importances": feature_importances,
    }

    logger.info(
        f"Training complete. Accuracy: {metrics['accuracy']:.4f}, "
        f"F1-Score: {metrics['f1_score']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}"
    )

    if save_artifacts:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        with open(METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Model artifacts saved to {MODEL_PATH.parent}")

    return model, scaler, metadata


if __name__ == "__main__":
    train_model()
