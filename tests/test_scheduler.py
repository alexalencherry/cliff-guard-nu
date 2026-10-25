"""
Unit tests for the Constraint-Aware Smart Scheduler.
"""

import pytest
from services.tariff_engine import TariffEngine
from services.scheduler import SmartScheduler

@pytest.fixture
def scheduler():
    engine = TariffEngine(config_path="config/tariff_config.json")
    return SmartScheduler(tariff_engine=engine, time_step_minutes=15)

def test_flexible_load_shifts_away_from_peak(scheduler):
    """
    Test that a water pump running at 19:00 (peak @ ₹8) is shifted to an off-peak or normal
    window when available.
    Power: 750W, Duration: 60 min.
    Permissible window: 05:00 to 20:00.
    Baseline at 19:00 -> ₹6.00 (750W * 1h * 8 = ₹6.00).
    Optimal slot should be 05:00 (off-peak @ ₹4) -> 750W * 1h * 4 = ₹3.00.
    Savings = ₹3.00.
    """
    res = scheduler.schedule_appliance(
        device_type="water_pump",
        power_watts=750,
        duration_minutes=60,
        earliest_start_hour=5.0,
        latest_finish_hour=20.0,
        preferred_start_hour=19.0,
        flexible_load=True
    )
    assert res["is_feasible"] is True
    assert res["recommended_start_hour"] == 5.0
    assert res["recommended_finish_hour"] == 6.0
    assert res["baseline_cost_inr"] == 6.00
    assert res["recommended_cost_inr"] == 3.00
    assert res["savings_inr"] == 3.00
    assert res["savings_percentage"] == 50.0

def test_inflexible_load_is_not_moved(scheduler):
    """Inflexible appliances must maintain their baseline schedule."""
    res = scheduler.schedule_appliance(
        device_type="washing_machine",
        power_watts=500,
        duration_minutes=60,
        earliest_start_hour=6.0,
        latest_finish_hour=22.0,
        preferred_start_hour=19.0,
        flexible_load=False
    )
    assert res["is_feasible"] is True
    assert res["flexible_load"] is False
    assert res["recommended_start_hour"] == 19.0
    assert res["savings_inr"] == 0.0

def test_infeasible_schedule_detected(scheduler):
    """If duration exceeds the permissible window, report infeasible."""
    # Window is 18:00 to 19:00 (60 mins), but appliance requires 90 mins.
    res = scheduler.schedule_appliance(
        device_type="water_heater",
        power_watts=2000,
        duration_minutes=90,
        earliest_start_hour=18.0,
        latest_finish_hour=19.0,
        preferred_start_hour=18.0,
        flexible_load=True
    )
    assert res["is_feasible"] is False
    assert res["recommended_start_hour"] is None
    assert "Infeasible" in res["status"]

def test_no_fabricated_savings_when_baseline_is_cheapest(scheduler):
    """If user already requested the cheapest slot, report 0.0 savings."""
    # Preferred start is 02:00 (already in off-peak)
    res = scheduler.schedule_appliance(
        device_type="washing_machine",
        power_watts=500,
        duration_minutes=60,
        earliest_start_hour=0.0,
        latest_finish_hour=6.0,
        preferred_start_hour=2.0,
        flexible_load=True
    )
    assert res["is_feasible"] is True
    assert res["savings_inr"] == 0.0
    assert res["savings_percentage"] == 0.0
    assert res["recommended_cost_inr"] == res["baseline_cost_inr"]

def test_tie_breaking_favors_preferred_hour(scheduler):
    """When multiple slots have the same rate, the one closest to preferred is chosen."""
    # Window 08:00 to 16:00 is all normal rate (₹6/kWh).
    # Preferred start is 14:00.
    res = scheduler.schedule_appliance(
        device_type="washing_machine",
        power_watts=500,
        duration_minutes=60,
        earliest_start_hour=8.0,
        latest_finish_hour=16.0,
        preferred_start_hour=14.0,
        flexible_load=True
    )
    assert res["is_feasible"] is True
    assert res["recommended_start_hour"] == 14.0
    assert res["savings_inr"] == 0.0
