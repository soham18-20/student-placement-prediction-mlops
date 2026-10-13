import datetime
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_DIR,
    MODEL_PATH,
    METRICS_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    MODEL_VERSION,
    RANDOM_STATE,
    TEST_SIZE,
)
from ml.dataset import load_or_create_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ml.pipeline")


def calculate_metrics(y_true, y_pred, y_prob=None) -> Dict[str, Any]:
    """Calculates standard classification metrics and confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics


def train_pipeline(
    data_path: Path = RAW_DATA_PATH,
    save_artifacts: bool = True,
    random_state: int = RANDOM_STATE,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Trains a reproducible scikit-learn pipeline, evaluates against a baseline,
    and saves serialized model and metrics artifacts.
    """
    logger.info("Starting model training pipeline...")
    df, dataset_meta = load_or_create_dataset(data_path)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Documented Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=random_state,
        stratify=y,
    )
    logger.info(
        f"Split dataset: {len(X_train)} train samples ({(1 - TEST_SIZE) * 100:.0f}%), "
        f"{len(X_test)} test samples ({TEST_SIZE * 100:.0f}%), random_state={random_state}"
    )

    # Save processed splits for reproducibility
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DATA_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)

    # 1. Primary Model: Pipeline(StandardScaler, RandomForestClassifier)
    primary_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=120,
                    max_depth=7,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=random_state,
                ),
            ),
        ]
    )

    primary_pipeline.fit(X_train, y_train)

    y_test_pred = primary_pipeline.predict(X_test)
    y_test_prob = primary_pipeline.predict_proba(X_test)[:, 1]
    primary_metrics = calculate_metrics(y_test, y_test_pred, y_test_prob)

    # 2. Baseline Model 1: DummyClassifier (Majority Class Baseline)
    dummy_baseline = DummyClassifier(strategy="most_frequent")
    dummy_baseline.fit(X_train, y_train)
    dummy_pred = dummy_baseline.predict(X_test)
    dummy_metrics = calculate_metrics(y_test, dummy_pred)

    # 3. Baseline Model 2: Logistic Regression Baseline
    logreg_baseline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=random_state)),
        ]
    )
    logreg_baseline.fit(X_train, y_train)
    logreg_pred = logreg_baseline.predict(X_test)
    logreg_prob = logreg_baseline.predict_proba(X_test)[:, 1]
    logreg_metrics = calculate_metrics(y_test, logreg_pred, logreg_prob)

    # Extract Feature Importances from Random Forest
    rf_model = primary_pipeline.named_steps["classifier"]
    feature_importances = {
        feat: round(float(imp), 4)
        for feat, imp in sorted(
            zip(FEATURE_COLUMNS, rf_model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
    }

    # Baseline comparison deltas
    baseline_comparison = {
        "dummy_majority_class": {
            "name": "DummyClassifier (Most Frequent)",
            "metrics": dummy_metrics,
            "accuracy_delta": round(primary_metrics["accuracy"] - dummy_metrics["accuracy"], 4),
            "f1_delta": round(primary_metrics["f1_score"] - dummy_metrics["f1_score"], 4),
        },
        "logistic_regression": {
            "name": "Standard Logistic Regression",
            "metrics": logreg_metrics,
            "accuracy_delta": round(primary_metrics["accuracy"] - logreg_metrics["accuracy"], 4),
            "f1_delta": round(primary_metrics["f1_score"] - logreg_metrics["f1_score"], 4),
        },
    }

    metadata: Dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "algorithm": "RandomForestClassifier (scikit-learn Pipeline)",
        "pipeline_steps": ["StandardScaler", "RandomForestClassifier"],
        "training_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "random_state": random_state,
        "split": {
            "test_size": TEST_SIZE,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "stratified": True,
        },
        "features": FEATURE_COLUMNS,
        "dataset": dataset_meta,
        "primary_metrics": primary_metrics,
        "baseline_comparison": baseline_comparison,
        "feature_importances": feature_importances,
    }

    logger.info(
        f"Primary Model Trained: Accuracy={primary_metrics['accuracy']:.4f}, "
        f"F1={primary_metrics['f1_score']:.4f}, ROC-AUC={primary_metrics['roc_auc']:.4f}"
    )
    logger.info(
        f"Vs Dummy Baseline Accuracy: {dummy_metrics['accuracy']:.4f} "
        f"(Delta: +{baseline_comparison['dummy_majority_class']['accuracy_delta']:.4f})"
    )

    if save_artifacts:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(primary_pipeline, MODEL_PATH)
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved model pipeline to {MODEL_PATH}")
        logger.info(f"Saved evaluation metrics & metadata to {METRICS_PATH}")

    return primary_pipeline, metadata


if __name__ == "__main__":
    train_pipeline()
