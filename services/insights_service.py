"""
Evidence-based Insights and Recommendations Engine for Cliff Guard.
Generates deterministic, quantified energy intelligence cards combining
tariff calculations, machine learning anomaly flags, and load profile analytics.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
from services.tariff_engine import TariffEngine
from utils.formatting import format_inr, format_kwh, format_power, decimal_hour_to_hhmm

class InsightsService:
    def __init__(self, tariff_engine: TariffEngine):
        self.tariff_engine = tariff_engine

    def generate_household_insights(
        self,
        household_profile: Dict[str, Any],
        hourly_df: pd.DataFrame,
        events_df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Generate ranked energy recommendations for a specific household.
        """
        insights = []

        # 1. Evening Peak Concentration Insight
        # Check percentage of energy in peak window (18:00 - 22:00)
        peak_hours_mask = hourly_df["hour"].between(18, 21)
        peak_energy = hourly_df[peak_hours_mask]["energy_kwh"].sum()
        total_energy = hourly_df["energy_kwh"].sum()
        
        if total_energy > 0:
            peak_pct = (peak_energy / total_energy) * 100.0
            if peak_pct >= 32.0:
                peak_rate = self.tariff_engine.get_rate_for_hour(19)
                off_peak_rate = self.tariff_engine.get_rate_for_hour(23)
                diff_rate = peak_rate - off_peak_rate
                # Shifting 30% of peak energy to off-peak
                shiftable_kwh = peak_energy * 0.35
                daily_potential_save = shiftable_kwh * diff_rate
                monthly_potential_save = daily_potential_save * 30.0

                insights.append({
                    "id": "INS_PEAK_CONCENTRATION",
                    "title": "High Evening Peak Demand Concentration",
                    "severity": "High",
                    "category": "Tariff Optimization",
                    "evidence": (
                        f"{peak_energy:.2f} kWh ({peak_pct:.1f}% of total daily energy) is consumed "
                        f"during the premium evening peak (6:00 PM – 10:00 PM)."
                    ),
                    "explanation": (
                        f"Evening power is billed at the highest ToD rate of {format_inr(peak_rate)}/kWh. "
                        "Shifting non-essential loads (e.g. water pumping or laundry) directly reduces peak charges."
                    ),
                    "suggested_action": (
                        "Reschedule water pumping to morning off-peak (05:00–06:00 AM) and run laundry "
                        "during off-peak hours before 07:00 AM or after 10:00 PM."
                    ),
                    "calculated_impact_inr": round(monthly_potential_save, 2),
                    "impact_label": f"Potential monthly savings: ~{format_inr(monthly_potential_save)}",
                    "limitations": "Assumes flexible appliance availability and adequate overhead tank capacity."
                })

        # 2. Standby / Baseline Power Leakage
        baseline_power = float(household_profile.get("baseline_power_watts", 0.0))
        if baseline_power > 250.0:
            daily_standby_kwh = (baseline_power * 24.0) / 1000.0
            avg_rate = 6.0
            monthly_standby_cost = daily_standby_kwh * 30.0 * avg_rate

            insights.append({
                "id": "INS_STANDBY_LEAKAGE",
                "title": "Elevated Continuous Baseload / Vampire Draw",
                "severity": "Medium",
                "category": "Efficiency",
                "evidence": (
                    f"Measured baseline continuous power is {format_power(baseline_power)}, "
                    f"contributing {daily_standby_kwh:.2f} kWh/day in continuous idling loads."
                ),
                "explanation": (
                    "High baseline draw typically stems from continuous set-top boxes, older UPS inverter idling, "
                    "refrigerator door seal wear, or unattended sub-meters and outdoor security lights."
                ),
                "suggested_action": (
                    "Audit always-on appliances using smart plugs. Turn off entertainment consoles at wall switches, "
                    "and inspect inverter/UPS efficiency rating."
                ),
                "calculated_impact_inr": round(monthly_standby_cost * 0.25, 2),
                "impact_label": f"Projected monthly reduction: ~{format_inr(monthly_standby_cost * 0.25)} (25% reduction)",
                "limitations": "Some baseline draw is unavoidable (refrigerators, WiFi routers, security devices)."
            })

        # 3. Machine Learning Anomaly Detections in Appliance Logs
        if "ml_predicted_anomaly" in events_df.columns:
            anom_df = events_df[events_df["ml_predicted_anomaly"] == 1]
            if len(anom_df) > 0:
                # Group by device
                dev_counts = anom_df["device_type"].value_counts().to_dict()
                top_dev = anom_df["device_type"].iloc[0]
                sample_anom = anom_df.iloc[0]

                insights.append({
                    "id": "INS_ML_ANOMALY_ALERT",
                    "title": f"Appliance Anomaly Alert: {len(anom_df)} Unusual Events Detected",
                    "severity": "High",
                    "category": "Safety & Health",
                    "evidence": (
                        f"Isolation Forest identified {len(anom_df)} outlier cycles in {top_dev.replace('_', ' ')} logs. "
                        f"Sample event logged {sample_anom['power_watts']}W for {sample_anom['duration_minutes']} min "
                        f"(Score: {sample_anom['anomaly_score']:.2f})."
                    ),
                    "explanation": (
                        "Severe deviations in power or runtime point towards mechanical binding, defective heating coils, "
                        "or malfunctioning automated sensors."
                    ),
                    "suggested_action": (
                        f"Perform preventive maintenance on the {top_dev.replace('_', ' ')}. "
                        "Review the detailed inspection checklist in the AI Anomaly Detection module."
                    ),
                    "calculated_impact_inr": None,
                    "impact_label": "Prevents catastrophic motor burn-out and fire hazard",
                    "limitations": "Model flags statistical outliers and requires physical technician verification."
                })

        # 4. Water Pump Load Shifting Feasibility
        pump_events = events_df[events_df["device_type"] == "water_pump"]
        if len(pump_events) > 0:
            # Check how many pump runs occurred during peak (18:00 - 22:00)
            pump_peak = pump_events[pump_events["preferred_start_hour_simulated"].between(18.0, 21.9)]
            if len(pump_peak) > 0:
                p_run = pump_peak.iloc[0]
                p_watts = float(p_run["power_watts"])
                dur_m = float(p_run["duration_minutes"])
                
                # Compare baseline at 19:00 vs optimal morning at 06:00
                from services.scheduler import SmartScheduler
                sched = SmartScheduler(self.tariff_engine)
                opt_res = sched.schedule_appliance(
                    device_type="water_pump",
                    power_watts=p_watts,
                    duration_minutes=dur_m,
                    earliest_start_hour=5.0,
                    latest_finish_hour=20.0,
                    preferred_start_hour=19.0,
                    flexible_load=True
                )
                if opt_res["savings_inr"] > 0:
                    monthly_pump_save = opt_res["savings_inr"] * 30.0
                    insights.append({
                        "id": "INS_PUMP_SCHEDULE",
                        "title": "Water Pump Rescheduling Opportunity",
                        "severity": "Medium",
                        "category": "Smart Scheduling",
                        "evidence": (
                            f"Water pump currently operates during evening peak (7:00 PM) at {format_inr(opt_res['baseline_cost_inr'])} per cycle. "
                            f"Shifting to {decimal_hour_to_hhmm(opt_res['recommended_start_hour'])} costs only {format_inr(opt_res['recommended_cost_inr'])}."
                        ),
                        "explanation": (
                            f"Moving daily pumping cycle yields {format_inr(opt_res['savings_inr'])} direct savings per run "
                            f"({opt_res['savings_percentage']}% savings) with zero lifestyle inconvenience."
                        ),
                        "suggested_action": (
                            f"Approve automated schedule to run water pump at {decimal_hour_to_hhmm(opt_res['recommended_start_hour'])} daily."
                        ),
                        "calculated_impact_inr": round(monthly_pump_save, 2),
                        "impact_label": f"Saves {format_inr(monthly_pump_save)}/month on pumping",
                        "limitations": "Requires overhead tank level to suffice until morning cycle."
                    })

        # Fallback informative insight if list is small
        if len(insights) < 3:
            insights.append({
                "id": "INS_TARIFF_AWARENESS",
                "title": "Time-of-Day Tariff Strategy Overview",
                "severity": "Info",
                "category": "Tariff Optimization",
                "evidence": "Kerala residential ToD tariffs reward consumption shifting to 10:00 PM – 06:00 AM.",
                "explanation": (
                    "Running laundry, electric water heaters, and water pumps during off-peak periods drops the unit rate from "
                    f"{format_inr(self.tariff_engine.get_rate_for_hour(19))}/kWh to {format_inr(self.tariff_engine.get_rate_for_hour(23))}/kWh."
                ),
                "suggested_action": "Utilize appliance delay timers for morning or late night operation.",
                "calculated_impact_inr": None,
                "impact_label": "Lowers effective blended rate per kWh",
                "limitations": "Noise from washing machines during late night hours may be a consideration in compact apartments."
            })

        return insights
