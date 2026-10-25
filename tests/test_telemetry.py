"""
Unit tests for the Real-Time Telemetry Simulation Service.
"""

import pytest
from services.tariff_engine import TariffEngine
from services.telemetry_service import TelemetrySimulator

@pytest.fixture
def telemetry_sim():
    engine = TariffEngine(config_path="config/tariff_config.json")
    return TelemetrySimulator(tariff_engine=engine, random_seed=42)

def test_generate_single_telemetry_point(telemetry_sim):
    """Verify schema and physical consistency of a single telemetry packet."""
    pt = telemetry_sim.generate_point(scenario_key="standby", hour_of_day=19)
    
    assert "power_watts" in pt
    assert "voltage_volts" in pt
    assert "current_amps" in pt
    assert "burn_rate_inr_hr" in pt
    assert "step_energy_kwh" in pt
    assert "step_cost_inr" in pt

    # Physical checks
    assert pt["power_watts"] > 0
    assert 220.0 <= pt["voltage_volts"] <= 245.0
    assert pt["current_amps"] >= 0.1
    assert pt["tariff_rate_inr_kwh"] == 8.0 # Hour 19 is peak

def test_anomaly_scenario_detection_magnitude(telemetry_sim):
    """Verify anomaly scenario simulates power surge (>2000W)."""
    pt = telemetry_sim.generate_point(scenario_key="pump_anomaly", hour_of_day=19)
    assert pt["power_watts"] > 2000.0
    assert pt["is_anomaly_label"] == 1

def test_cost_savings_in_optimized_scenario(telemetry_sim):
    """Verify that optimized scenario generates positive live savings vs peak baseline."""
    pt = telemetry_sim.generate_point(scenario_key="heater_optimized", hour_of_day=19)
    assert pt["step_savings_inr"] > 0
    assert pt["tariff_rate_inr_kwh"] == 4.0 # Off-peak rate applied

def test_generate_initial_stream_buffer(telemetry_sim):
    """Verify rolling stream pre-population."""
    stream = telemetry_sim.generate_initial_stream(n_points=25, scenario_key="pump_normal")
    assert len(stream) == 25
    assert all("power_watts" in p for p in stream)
