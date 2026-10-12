import pytest
import pandas as pd
from src.data_generator import generate_placement_dataset
from src.config import FEATURE_COLUMNS, TARGET_COLUMN


def test_dataset_generation_shape():
    df = generate_placement_dataset(n_samples=200, random_state=42)
    assert len(df) == 200
    for col in FEATURE_COLUMNS:
        assert col in df.columns
    assert TARGET_COLUMN in df.columns


def test_dataset_no_null_values():
    df = generate_placement_dataset(n_samples=150, random_state=123)
    assert df.isnull().sum().sum() == 0


def test_dataset_ranges():
    df = generate_placement_dataset(n_samples=300, random_state=99)
    assert df["cgpa"].min() >= 5.0
    assert df["cgpa"].max() <= 10.0
    assert df["technical_score"].min() >= 30.0
    assert df["technical_score"].max() <= 100.0
    assert df["internships"].min() >= 0
    assert df["projects"].min() >= 0
    assert df["attendance"].min() >= 50.0
    assert df["attendance"].max() <= 100.0


def test_target_is_binary():
    df = generate_placement_dataset(n_samples=250, random_state=77)
    unique_targets = set(df[TARGET_COLUMN].unique())
    assert unique_targets.issubset({0, 1})
    # Both classes should be present in a reasonable sample
    assert len(unique_targets) == 2
