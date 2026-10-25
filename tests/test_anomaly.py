"""
Unit tests for the Machine Learning Anomaly Detector.
"""

import numpy as np
import pandas as pd
import pytest
from models.anomaly_detector import ApplianceAnomalyDetector

@pytest.fixture
def sample_events_df():
    return pd.read_csv("data/device_consumption.csv")

def test_feature_extraction_strictly_excludes_labels(sample_events_df):
    """Verify that synthetic ground truth labels are NEVER included in features."""
    detector = ApplianceAnomalyDetector()
    X = detector._extract_features(sample_events_df)
    
    assert "is_synthetic_anomaly_label" not in X.columns
    assert "scenario_type" not in X.columns
    assert "case_id" not in X.columns
    assert "timestamp" not in X.columns
    assert "device_id" not in X.columns
    assert "data_source" not in X.columns

    # Check physical features
    assert "power_watts" in X.columns
    assert "duration_minutes" in X.columns
    assert "energy_kwh" in X.columns
    assert "power_deviation_ratio" in X.columns

def test_anomaly_detector_training_and_prediction(sample_events_df):
    """Verify training, binary output, and normalized anomaly scores."""
    detector = ApplianceAnomalyDetector(contamination=0.06, random_state=42)
    detector.fit(sample_events_df)

    preds, scores = detector.predict(sample_events_df)
    assert len(preds) == len(sample_events_df)
    assert len(scores) == len(sample_events_df)

    # Predictions must be binary 0 or 1
    assert set(np.unique(preds)).issubset({0, 1})
    # Scores must be normalized between 0 and 1
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0

    # Ensure model actually flagged some anomalies (~6% contamination)
    flagged = int(np.sum(preds))
    assert 20 <= flagged <= 100, f"Expected ~30-70 flagged anomalies, got {flagged}"

def test_model_evaluation_metrics(sample_events_df):
    """Verify computation of confusion matrix, precision, recall, and F1."""
    detector = ApplianceAnomalyDetector(contamination=0.06, random_state=42)
    preds, _ = detector.predict(sample_events_df)
    ground_truth = sample_events_df["is_synthetic_anomaly_label"].values

    metrics = detector.evaluate_against_ground_truth(preds, ground_truth)
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "true_positives" in metrics
    assert metrics["total_samples"] == len(sample_events_df)
    # Model should have decent recall and precision on physical outliers
    assert metrics["f1_score"] > 0.40

def test_insufficient_samples_raises():
    """Verify that fitting with fewer than 10 samples raises ValueError."""
    detector = ApplianceAnomalyDetector()
    small_df = pd.DataFrame({
        "power_watts": [500.0, 750.0],
        "duration_minutes": [30, 45],
        "energy_kwh": [0.25, 0.56],
        "device_type": ["washing_machine", "water_pump"]
    })
    with pytest.raises(ValueError):
        detector.fit(small_df)
