"""
Tariff Engine for Cliff Guard.
Provides Time-of-Day (ToD) electricity billing calculations in Indian Rupees (₹ / INR).
Handles tariff period boundaries, midnight-crossing windows, multi-period event splitting,
and baseline versus recommended schedule cost comparisons.
"""

import json
import os
from typing import Dict, List, Any, Optional

class TariffEngine:
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize TariffEngine with either a configuration file path or a dictionary.
        """
        if config_dict is not None:
            self.config = config_dict
        elif config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = json.load(f)
        else:
            # Fallback default illustrative Kerala ToD structure
            self.config = {
                "currency": "INR",
                "currency_symbol": "₹",
                "periods": [
                    {
                        "id": "off_peak",
                        "name": "Off-Peak (Night)",
                        "rate_per_kwh": 4.0,
                        "time_windows": [
                            {"start_hour": 22, "end_hour": 24},
                            {"start_hour": 0, "end_hour": 6}
                        ],
                        "color": "#10b981"
                    },
                    {
                        "id": "normal",
                        "name": "Normal (Daytime)",
                        "rate_per_kwh": 6.0,
                        "time_windows": [
                            {"start_hour": 6, "end_hour": 18}
                        ],
                        "color": "#3b82f6"
                    },
                    {
                        "id": "peak",
                        "name": "Peak (Evening)",
                        "rate_per_kwh": 8.0,
                        "time_windows": [
                            {"start_hour": 18, "end_hour": 22}
                        ],
                        "color": "#ef4444"
                    }
                ]
            }

        self._validate_and_build_lookup()

    def _validate_and_build_lookup(self):
        """Build 24-hour rate and period lookup tables for O(1) hourly queries."""
        self.hourly_rates: Dict[int, float] = {}
        self.hourly_periods: Dict[int, str] = {}
        self.hourly_names: Dict[int, str] = {}

        periods = self.config.get("periods", [])
        for period in periods:
            rate = float(period.get("rate_per_kwh", 0.0))
            if rate < 0:
                raise ValueError(f"Tariff rate cannot be negative: {rate}")
            pid = period.get("id")
            pname = period.get("name", pid)

            for window in period.get("time_windows", []):
                start = int(window.get("start_hour", 0))
                end = int(window.get("end_hour", 0))
                for h in range(start, end):
                    mod_h = h % 24
                    self.hourly_rates[mod_h] = rate
                    self.hourly_periods[mod_h] = pid
                    self.hourly_names[mod_h] = pname

        # Check coverage
        for h in range(24):
            if h not in self.hourly_rates:
                # Default fallback if any hour missed
                self.hourly_rates[h] = 6.0
                self.hourly_periods[h] = "normal"
                self.hourly_names[h] = "Normal (Daytime)"

    def update_rates(self, new_rates: Dict[str, float]):
        """Update period rates dynamically."""
        for period in self.config.get("periods", []):
            pid = period.get("id")
            if pid in new_rates:
                val = float(new_rates[pid])
                if val < 0:
                    raise ValueError(f"Rate for '{pid}' cannot be negative.")
                period["rate_per_kwh"] = val
        self._validate_and_build_lookup()

    def get_rate_for_hour(self, hour: int) -> float:
        """Get ₹/kWh rate for a specific integer hour (0-23)."""
        mod_h = int(hour) % 24
        return self.hourly_rates.get(mod_h, 6.0)

    def get_period_for_hour(self, hour: int) -> str:
        """Get period id ('off_peak', 'normal', 'peak') for an hour (0-23)."""
        mod_h = int(hour) % 24
        return self.hourly_periods.get(mod_h, "normal")

    def get_period_name_for_hour(self, hour: int) -> str:
        """Get human-readable period name for an hour (0-23)."""
        mod_h = int(hour) % 24
        return self.hourly_names.get(mod_h, "Normal (Daytime)")

    def calculate_hourly_cost(self, energy_kwh: float, hour: int) -> float:
        """Calculate cost in INR for a 1-hour energy block."""
        if energy_kwh <= 0:
            return 0.0
        rate = self.get_rate_for_hour(hour)
        return round(float(energy_kwh) * rate, 4)

    def calculate_event_cost(
        self,
        power_watts: float,
        start_hour: float,
        duration_minutes: float
    ) -> Dict[str, Any]:
        """
        Calculate total electricity cost in INR (₹) for an appliance event.
        Splits duration across tariff boundaries with minute-level precision
        under a constant power assumption.

        Handles events crossing midnight and multi-period transitions.
        """
        if power_watts <= 0 or duration_minutes <= 0:
            return {
                "total_cost_inr": 0.0,
                "total_kwh": 0.0,
                "power_watts": float(power_watts),
                "duration_minutes": float(duration_minutes),
                "breakdown": []
            }

        p_watts = float(power_watts)
        dur_mins = float(duration_minutes)
        total_kwh = (p_watts * dur_mins) / 60000.0

        # Start time in total minutes from 00:00 (0 to 1440)
        start_min_of_day = (float(start_hour) * 60.0) % 1440.0

        # Minute-by-minute proportional allocation
        # We aggregate contiguous segments of identical tariff period/rate
        current_min = start_min_of_day
        remaining_mins = dur_mins

        breakdown_dict: Dict[str, Dict[str, Any]] = {}

        # Iterate in discrete 1-minute steps (or bounded intervals)
        step_min = 1.0
        while remaining_mins > 0:
            step = min(step_min, remaining_mins)
            hour_index = int(current_min // 60) % 24
            period_id = self.get_period_for_hour(hour_index)
            period_name = self.get_period_name_for_hour(hour_index)
            rate = self.get_rate_for_hour(hour_index)

            if period_id not in breakdown_dict:
                breakdown_dict[period_id] = {
                    "period_id": period_id,
                    "period_name": period_name,
                    "rate_per_kwh": rate,
                    "minutes": 0.0,
                    "energy_kwh": 0.0,
                    "cost_inr": 0.0
                }

            seg_kwh = (p_watts * step) / 60000.0
            seg_cost = seg_kwh * rate

            breakdown_dict[period_id]["minutes"] += step
            breakdown_dict[period_id]["energy_kwh"] += seg_kwh
            breakdown_dict[period_id]["cost_inr"] += seg_cost

            current_min = (current_min + step) % 1440.0
            remaining_mins -= step

        breakdown_list = []
        total_cost = 0.0
        for pid, data in breakdown_dict.items():
            cost = round(data["cost_inr"], 4)
            kwh = round(data["energy_kwh"], 4)
            mins = int(round(data["minutes"]))
            total_cost += cost
            breakdown_list.append({
                "period_id": pid,
                "period_name": data["period_name"],
                "rate_per_kwh": data["rate_per_kwh"],
                "duration_minutes": mins,
                "energy_kwh": kwh,
                "cost_inr": cost
            })

        return {
            "total_cost_inr": round(total_cost, 2),
            "total_kwh": round(total_kwh, 4),
            "power_watts": p_watts,
            "duration_minutes": dur_mins,
            "start_hour": round(start_hour, 2),
            "finish_hour": round((start_hour + dur_mins / 60.0) % 24.0, 2),
            "breakdown": breakdown_list
        }

    def compare_schedules(
        self,
        power_watts: float,
        duration_minutes: float,
        baseline_start_hour: float,
        recommended_start_hour: float
    ) -> Dict[str, Any]:
        """
        Compare baseline and recommended schedule costs.
        Calculates genuine monetary savings in INR (₹).
        Never fabricates positive savings if costs are identical or higher.
        """
        base_res = self.calculate_event_cost(power_watts, baseline_start_hour, duration_minutes)
        rec_res = self.calculate_event_cost(power_watts, recommended_start_hour, duration_minutes)

        baseline_cost = base_res["total_cost_inr"]
        rec_cost = rec_res["total_cost_inr"]

        savings = round(baseline_cost - rec_cost, 2)
        pct_savings = round((savings / baseline_cost * 100.0), 1) if baseline_cost > 0 else 0.0

        return {
            "baseline_start_hour": baseline_start_hour,
            "recommended_start_hour": recommended_start_hour,
            "baseline_cost_inr": baseline_cost,
            "recommended_cost_inr": rec_cost,
            "savings_inr": savings,
            "savings_percentage": pct_savings,
            "is_saving": savings > 0.0,
            "baseline_breakdown": base_res["breakdown"],
            "recommended_breakdown": rec_res["breakdown"]
        }
