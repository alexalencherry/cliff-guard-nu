"""
Unit tests for the Tariff Engine.
"""

import pytest
from services.tariff_engine import TariffEngine

@pytest.fixture
def tariff_engine():
    return TariffEngine(config_path="config/tariff_config.json")

def test_hourly_rates_and_periods(tariff_engine):
    """Test standard Kerala ToD rate assignments."""
    # Off-peak night: 22:00 to 06:00 -> ₹4.0/kWh
    assert tariff_engine.get_rate_for_hour(23) == 4.0
    assert tariff_engine.get_period_for_hour(23) == "off_peak"
    assert tariff_engine.get_rate_for_hour(0) == 4.0
    assert tariff_engine.get_rate_for_hour(3) == 4.0
    assert tariff_engine.get_rate_for_hour(5) == 4.0

    # Normal daytime: 06:00 to 18:00 -> ₹6.0/kWh
    assert tariff_engine.get_rate_for_hour(6) == 6.0
    assert tariff_engine.get_period_for_hour(6) == "normal"
    assert tariff_engine.get_rate_for_hour(12) == 6.0
    assert tariff_engine.get_rate_for_hour(17) == 6.0

    # Peak evening: 18:00 to 22:00 -> ₹8.0/kWh
    assert tariff_engine.get_rate_for_hour(18) == 8.0
    assert tariff_engine.get_period_for_hour(18) == "peak"
    assert tariff_engine.get_rate_for_hour(20) == 8.0
    assert tariff_engine.get_rate_for_hour(21) == 8.0

def test_event_cost_single_period(tariff_engine):
    """Test cost calculation inside a single tariff period."""
    # 1000W for 60 min in off-peak (02:00) = 1 kWh * ₹4 = ₹4.00
    res = tariff_engine.calculate_event_cost(power_watts=1000, start_hour=2.0, duration_minutes=60)
    assert res["total_kwh"] == 1.0
    assert res["total_cost_inr"] == 4.0
    assert len(res["breakdown"]) == 1
    assert res["breakdown"][0]["period_id"] == "off_peak"

    # 2000W for 30 min in peak (19:00) = 1 kWh * ₹8 = ₹8.00
    res_peak = tariff_engine.calculate_event_cost(power_watts=2000, start_hour=19.0, duration_minutes=30)
    assert res_peak["total_kwh"] == 1.0
    assert res_peak["total_cost_inr"] == 8.0
    assert res_peak["breakdown"][0]["period_id"] == "peak"

def test_event_cost_multi_period_split(tariff_engine):
    """
    Test event crossing a tariff boundary.
    Start at 17:30 for 60 mins:
    First 30 mins in normal (17:30 - 18:00 @ ₹6/kWh): 0.5 kWh * ₹6 = ₹3.00
    Next 30 mins in peak (18:00 - 18:30 @ ₹8/kWh): 0.5 kWh * ₹8 = ₹4.00
    Total cost: ₹7.00
    """
    res = tariff_engine.calculate_event_cost(power_watts=1000, start_hour=17.5, duration_minutes=60)
    assert res["total_kwh"] == 1.0
    assert res["total_cost_inr"] == 7.00
    assert len(res["breakdown"]) == 2

    periods = {b["period_id"]: b for b in res["breakdown"]}
    assert "normal" in periods
    assert "peak" in periods
    assert periods["normal"]["duration_minutes"] == 30
    assert periods["peak"]["duration_minutes"] == 30
    assert periods["normal"]["cost_inr"] == 3.00
    assert periods["peak"]["cost_inr"] == 4.00

def test_event_cost_crossing_midnight(tariff_engine):
    """
    Test event starting at 23:30 for 60 minutes (finishing at 00:30 next day).
    Both hours (23 and 0) are in off-peak @ ₹4/kWh.
    Total cost for 1000W = 1 kWh * ₹4 = ₹4.00.
    """
    res = tariff_engine.calculate_event_cost(power_watts=1000, start_hour=23.5, duration_minutes=60)
    assert res["total_kwh"] == 1.0
    assert res["total_cost_inr"] == 4.00
    assert res["finish_hour"] == 0.5

def test_negative_rates_rejected():
    """Verify that negative tariff rates trigger an exception."""
    with pytest.raises(ValueError):
        TariffEngine(config_dict={
            "periods": [
                {
                    "id": "bad",
                    "rate_per_kwh": -2.0,
                    "time_windows": [{"start_hour": 0, "end_hour": 24}]
                }
            ]
        })

def test_compare_schedules(tariff_engine):
    """Verify genuine baseline vs recommended comparison calculation."""
    # 1000W for 60 mins:
    # Baseline at 19:00 (peak @ ₹8) -> ₹8.00
    # Recommended at 04:00 (off-peak @ ₹4) -> ₹4.00
    # Savings = ₹4.00 (50.0%)
    comp = tariff_engine.compare_schedules(
        power_watts=1000,
        duration_minutes=60,
        baseline_start_hour=19.0,
        recommended_start_hour=4.0
    )
    assert comp["baseline_cost_inr"] == 8.00
    assert comp["recommended_cost_inr"] == 4.00
    assert comp["savings_inr"] == 4.00
    assert comp["savings_percentage"] == 50.0
    assert comp["is_saving"] is True
