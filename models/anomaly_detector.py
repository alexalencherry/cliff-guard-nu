"""
Isolation Forest Machine Learning Anomaly Detector for Cliff Guard.
Detects electrical anomalies (power surges, excessive runtime, abnormal energy signatures)
in appliance consumption events without data leakage.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

class ApplianceAnomalyDetector:
    def __init__(
        self,
        contamination: float = 0.06,
        random_state: int = 42
    ):
        """
        Initialize the Isolation Forest anomaly detector.
        contamination: expected proportion of outliers in the dataset (default 6%).
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=150,
            max_samples="auto"
        )
        self.feature_names: List[str] = []
        self.is_fitted = False

        # Reference nominal power specs for deviation feature
        self.nominal_power = {
            "washing_machine": 500.0,
            "water_pump": 750.0,
            "water_heater": 2000.0
        }

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract defensible physical features for unsupervised anomaly detection.
        STRICT PROHIBITION: Never includes 'is_synthetic_anomaly_label', 'scenario_type',
        or record identifiers.
        """
        feats = pd.DataFrame()
        feats["power_watts"] = df["power_watts"].astype(float)
        feats["duration_minutes"] = df["duration_minutes"].astype(float)
        feats["energy_kwh"] = df["energy_kwh"].astype(float)

        # Power deviation from nominal engineering baseline
        dev_nom = []
        for _, row in df.iterrows():
            dev_type = str(row.get("device_type", "water_pump"))
            nom = self.nominal_power.get(dev_type, 750.0)
            dev_ratio = (float(row["power_watts"]) - nom) / nom
            dev_nom.append(dev_ratio)
        feats["power_deviation_ratio"] = dev_nom

        # Energy rate (kWh per minute)
        feats["kwh_per_minute"] = feats["energy_kwh"] / np.maximum(feats["duration_minutes"], 1.0)

        # One-hot encoding of supported devices
        feats["is_washing_machine"] = (df["device_type"] == "washing_machine").astype(int)
        feats["is_water_pump"] = (df["device_type"] == "water_pump").astype(int)
        feats["is_water_heater"] = (df["device_type"] == "water_heater").astype(int)

        return feats

    def fit(self, df: pd.DataFrame):
        """Fit Isolation Forest on feature matrix."""
        if len(df) < 10:
            raise ValueError("Insufficient records to train Isolation Forest (minimum 10 required).")

        X = self._extract_features(df)
        self.feature_names = list(X.columns)
        self.model.fit(X)
        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference on dataset.
        Returns:
            predictions: binary 1 (anomaly) or 0 (normal)
            scores: normalized anomaly score where higher = more anomalous (0.0 to 1.0)
        """
        if not self.is_fitted:
            self.fit(df)

        X = self._extract_features(df)
        # raw decision_function: lower means more anomalous
        raw_scores = self.model.decision_function(X)
        
        # Convert to 0 (normal) or 1 (anomaly)
        # sklearn outputs -1 for anomaly, 1 for inlier
        raw_preds = self.model.predict(X)
        binary_preds = np.where(raw_preds == -1, 1, 0)

        # Normalize score into [0, 1] range where 1.0 is highest anomaly risk
        min_s, max_s = raw_scores.min(), raw_scores.max()
        if max_s > min_s:
            norm_scores = 1.0 - ((raw_scores - min_s) / (max_s - min_s))
        else:
            norm_scores = np.zeros_like(raw_scores)

        return binary_preds, np.round(norm_scores, 4)

    def evaluate_against_ground_truth(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray
    ) -> Dict[str, Any]:
        """
        Strict evaluation against synthetic evaluation labels.
        These metrics are calculated AFTER inference and never used for training.
        """
        y_true = np.array(ground_truth).astype(int)
        y_pred = np.array(predictions).astype(int)

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        precision = round(tp / (tp + fp), 3) if (tp + fp) > 0 else 0.0
        recall = round(tp / (tp + fn), 3) if (tp + fn) > 0 else 0.0
        f1 = round(2 * precision * recall / (precision + recall), 3) if (precision + recall) > 0 else 0.0
        accuracy = round((tp + tn) / len(y_true), 3) if len(y_true) > 0 else 0.0

        return {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "total_samples": len(y_true)
        }
