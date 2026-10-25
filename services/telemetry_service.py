"""
Real-time Synthetic Telemetry & Live Cost Savings Simulation Service for Cliff Guard.
Generates streaming second-by-second smart meter electrical metrics (power, voltage, current, power factor)
and calculates instantaneous burn rate (₹/hr), cumulative cost, and live savings.
"""

import time
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from services.tariff_engine import TariffEngine
from models.anomaly_detector import ApplianceAnomalyDetector

class TelemetrySimulator:
    def __init__(self, tariff_engine: TariffEngine, random_seed: int = 42):
        self.tariff_engine = tariff_engine
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        # Baseline appliance operating modes
        self.scenarios = {
            "standby": {
                "name": "Standby Baseload",
                "base_power": 180.0,
                "power_noise": 15.0,
                "device_type": "standby",
                "is_anomaly": 0,
                "power_factor": 0.92,
                "description": "Idling refrigerator, router, standby electronics."
            },
            "pump_normal": {
                "name": "Water Pump (Normal)",
                "base_power": 750.0,
                "power_noise": 35.0,
                "device_type": "water_pump",
                "is_anomaly": 0,
                "power_factor": 0.88,
                "description": "1HP domestic open-well water pump running normally."
            },
            "pump_anomaly": {
                "name": "Water Pump (Severe Bearing Drag Fault)",
                "base_power": 2450.0,
                "power_noise": 120.0,
                "device_type": "water_pump",
                "is_anomaly": 1,
                "power_factor": 0.72,
                "description": "Mechanical bearing friction / impeller obstruction causing severe current surge."
            },
            "heater_peak": {
                "name": "Water Heater (Peak Tariff Window)",
                "base_power": 2000.0,
                "power_noise": 40.0,
                "device_type": "water_heater",
                "is_anomaly": 0,
                "power_factor": 0.99,
                "description": "2kW geyser operating during evening peak tariff (₹8.00/kWh)."
            },
            "heater_optimized": {
                "name": "Water Heater (Cliff Guard Off-Peak Shift)",
                "base_power": 2000.0,
                "power_noise": 40.0,
                "device_type": "water_heater",
                "is_anomaly": 0,
                "power_factor": 0.99,
                "description": "2kW geyser scheduled during off-peak morning window (₹4.00/kWh)."
            },
            "washing_normal": {
                "name": "Washing Machine (Normal Wash)",
                "base_power": 520.0,
                "power_noise": 60.0,
                "device_type": "washing_machine",
                "is_anomaly": 0,
                "power_factor": 0.91,
                "description": "Drum motor agitation and heating cycle."
            }
        }

    def generate_point(
        self,
        scenario_key: str = "standby",
        hour_of_day: int = 19,
        step_seconds: float = 1.0,
        current_step: int = 0
    ) -> Dict[str, Any]:
        """
        Generate a single real-time electrical telemetry telemetry packet.
        """
        scen = self.scenarios.get(scenario_key, self.scenarios["standby"])
        base_p = scen["base_power"]
        noise_p = scen["power_noise"]
        dev_type = scen["device_type"]
        is_anom_truth = scen["is_anomaly"]
        pf = scen["power_factor"]

        # Synthetic micro-fluctuations
        power_watts = max(50.0, float(base_p + np.random.normal(0, noise_p)))
        # Grid voltage in Kerala nominally ~230V (+/- 3V)
        voltage_volts = round(float(230.0 + np.random.normal(0, 2.5)), 1)
        # Current in Amperes: I = P / (V * PF)
        current_amps = round(power_watts / (voltage_volts * max(pf, 0.5)), 2)
        # Grid frequency: 50.0 Hz (+/- 0.08 Hz)
        frequency_hz = round(float(50.0 + np.random.normal(0, 0.04)), 2)

        # Cost calculations
        active_rate = self.tariff_engine.get_rate_for_hour(hour_of_day)
        # If scenario is heater_optimized, simulate off-peak rate (₹4/kWh)
        if scenario_key == "heater_optimized":
            effective_rate = self.tariff_engine.get_rate_for_hour(4) # 04:00 AM off-peak
            baseline_rate = self.tariff_engine.get_rate_for_hour(19) # 19:00 PM peak
        elif scenario_key == "pump_normal" and hour_of_day in [18, 19, 20, 21]:
            # Baseline is peak, but if optimized it would be off-peak
            effective_rate = active_rate
            baseline_rate = active_rate
        else:
            effective_rate = active_rate
            baseline_rate = active_rate

        # Instantaneous burn rate in ₹/hr
        kw = power_watts / 1000.0
        burn_rate_inr_hr = round(kw * effective_rate, 3)
        baseline_burn_rate_inr_hr = round(kw * baseline_rate, 3)

        # Step energy & cost
        step_hours = step_seconds / 3600.0
        step_kwh = kw * step_hours
        step_cost_inr = step_kwh * effective_rate
        baseline_step_cost_inr = step_kwh * baseline_rate
        step_savings_inr = max(0.0, baseline_step_cost_inr - step_cost_inr)

        return {
            "step": current_step,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "scenario_key": scenario_key,
            "scenario_name": scen["name"],
            "device_type": dev_type,
            "power_watts": round(power_watts, 1),
            "voltage_volts": voltage_volts,
            "current_amps": current_amps,
            "power_factor": round(pf, 2),
            "frequency_hz": frequency_hz,
            "tariff_rate_inr_kwh": effective_rate,
            "burn_rate_inr_hr": burn_rate_inr_hr,
            "step_energy_kwh": round(step_kwh, 6),
            "step_cost_inr": round(step_cost_inr, 6),
            "step_savings_inr": round(step_savings_inr, 6),
            "is_anomaly_label": is_anom_truth
        }

    def generate_initial_stream(
        self,
        n_points: int = 30,
        scenario_key: str = "standby",
        hour_of_day: int = 19
    ) -> List[Dict[str, Any]]:
        """Pre-populate a rolling buffer of telemetry points."""
        stream = []
        for i in range(n_points):
            stream.append(self.generate_point(
                scenario_key=scenario_key,
                hour_of_day=hour_of_day,
                current_step=i
            ))
        return stream
