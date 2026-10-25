"""
Anomaly Detection Service for Cliff Guard.
Wraps the ML model, handles inference caching, and generates evidence-based explanations.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from models.anomaly_detector import ApplianceAnomalyDetector

class AnomalyService:
    def __init__(self, contamination: float = 0.06):
        self.contamination = contamination
        self.detector = ApplianceAnomalyDetector(contamination=contamination)
        self.cached_results: Optional[pd.DataFrame] = None

    def run_detection(self, events_df: pd.DataFrame, force_retrain: bool = False) -> pd.DataFrame:
        """
        Run anomaly detection on the events dataframe.
        Appends:
          - ml_predicted_anomaly (0 or 1)
          - anomaly_score (0.0 to 1.0)
          - severity ('Normal', 'Moderate', 'Critical')
          - evidence_summary (str)
        """
        if self.cached_results is not None and not force_retrain and len(self.cached_results) == len(events_df):
            return self.cached_results.copy()

        df = events_df.copy()
        self.detector.fit(df)
        preds, scores = self.detector.predict(df)

        df["ml_predicted_anomaly"] = preds
        df["anomaly_score"] = scores

        # Generate evidence and severity
        severities = []
        evidences = []
        for _, row in df.iterrows():
            is_anom = int(row["ml_predicted_anomaly"])
            score = float(row["anomaly_score"])
            p_watts = float(row["power_watts"])
            dur_mins = float(row["duration_minutes"])
            dev_type = str(row["device_type"])

            if is_anom == 1:
                if score >= 0.85:
                    sev = "Critical"
                else:
                    sev = "Elevated"
                ev = self._generate_evidence_summary(dev_type, p_watts, dur_mins, score)
            else:
                sev = "Normal"
                ev = "Operating within expected bounds."

            severities.append(sev)
            evidences.append(ev)

        df["severity"] = severities
        df["evidence_summary"] = evidences
        self.cached_results = df
        return df

    def _generate_evidence_summary(
        self,
        device_type: str,
        power_watts: float,
        duration_minutes: float,
        score: float
    ) -> str:
        """Generate human-readable evidence for the anomaly."""
        nom_map = {
            "washing_machine": (500.0, 60),
            "water_pump": (750.0, 45),
            "water_heater": (2000.0, 30)
        }
        nom_p, typ_dur = nom_map.get(device_type, (750.0, 45))
        reasons = []

        if power_watts > nom_p * 1.5:
            pct_p = int(round((power_watts - nom_p) / nom_p * 100))
            reasons.append(f"Power surge (+{pct_p}% above nominal {int(nom_p)}W)")
        elif power_watts < nom_p * 0.5:
            reasons.append(f"Unusually low power draw ({power_watts:.0f}W)")

        if duration_minutes > typ_dur * 1.8:
            pct_d = int(round((duration_minutes - typ_dur) / typ_dur * 100))
            reasons.append(f"Excessive duration (+{pct_d}% beyond typical {typ_dur}m)")

        if not reasons:
            reasons.append(f"Unusual joint power-duration signature (Score: {score:.2f})")

        return "; ".join(reasons)

    def get_inspection_checklist(self, device_type: str, evidence: str) -> List[str]:
        """Provide domain-specific electrical inspection steps."""
        if device_type == "water_pump":
            return [
                "1. Check overhead tank overflow sensor and automatic cut-off switch.",
                "2. Inspect foot valve and suction line for air suction or water leakage.",
                "3. Check motor capacitor and pump impeller for debris or bearing drag.",
                "4. Ensure pump is not running dry in open-well or borewell."
            ]
        elif device_type == "water_heater":
            return [
                "1. Check thermostat contactor for sticking or calibration drift.",
                "2. Inspect heating element for heavy mineral/hard-water scaling.",
                "3. Verify pressure relief safety valve and earth leakage circuit breaker (ELCB).",
                "4. Confirm hot water taps are closed and not continuously dripping."
            ]
        elif device_type == "washing_machine":
            return [
                "1. Inspect drum drive motor and belt tension for motor jamming.",
                "2. Check drain pump filter for trapped lint, coins, or obstructions.",
                "3. Verify water inlet solenoid valve and water supply pressure.",
                "4. Confirm wash cycle timer completed normally without hanging in rinse cycle."
            ]
        return ["Inspect appliance power cord and circuit breaker."]

    def get_model_evaluation(self, events_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute performance metrics against synthetic ground truth."""
        df_pred = self.run_detection(events_df)
        return self.detector.evaluate_against_ground_truth(
            predictions=df_pred["ml_predicted_anomaly"].values,
            ground_truth=df_pred["is_synthetic_anomaly_label"].values
        )
