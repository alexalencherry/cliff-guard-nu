"""
Integration tests for Cliff Guard core pipeline.
Verifies seamless interaction between DataService, TariffEngine,
AnomalyService, SmartScheduler, and InsightsService.
"""

import pytest
from services.data_service import DataService
from services.tariff_engine import TariffEngine
from services.anomaly_service import AnomalyService
from services.scheduler import SmartScheduler
from services.insights_service import InsightsService

@pytest.fixture
def integrated_pipeline():
    data_svc = DataService()
    tariff_eng = TariffEngine(config_path="config/tariff_config.json")
    anomaly_svc = AnomalyService(contamination=0.06)
    scheduler = SmartScheduler(tariff_engine=tariff_eng)
    insights_svc = InsightsService(tariff_engine=tariff_eng)

    return {
        "data_svc": data_svc,
        "tariff_eng": tariff_eng,
        "anomaly_svc": anomaly_svc,
        "scheduler": scheduler,
        "insights_svc": insights_svc
    }

def test_full_pipeline_flow(integrated_pipeline):
    """Test full integration sequence across services."""
    pipe = integrated_pipeline
    
    # 1. Data Service query
    cases = pipe["data_svc"].get_household_cases()
    assert len(cases) == 50
    hh_id = cases.iloc[0]["case_id"]
    hh_profile = pipe["data_svc"].get_household_by_id(hh_id)
    assert hh_profile is not None

    hourly_df = pipe["data_svc"].get_hourly_data_for_household(hh_id)
    assert len(hourly_df) == 24

    # 2. Hourly tariff calculations
    daily_cost = 0.0
    for _, row in hourly_df.iterrows():
        daily_cost += pipe["tariff_eng"].calculate_hourly_cost(row["energy_kwh"], int(row["hour"]))
    assert daily_cost > 0.0

    # 3. Anomaly detection on events
    events_df = pipe["data_svc"].get_device_consumption(case_id=hh_id)
    assert len(events_df) > 0
    annotated_events = pipe["anomaly_svc"].run_detection(events_df)
    assert "ml_predicted_anomaly" in annotated_events.columns
    assert "anomaly_score" in annotated_events.columns

    # 4. Smart Scheduling on first flexible event
    sample_event = events_df.iloc[0]
    sched_res = pipe["scheduler"].schedule_appliance(
        device_type=sample_event["device_type"],
        power_watts=float(sample_event["power_watts"]),
        duration_minutes=float(sample_event["duration_minutes"]),
        earliest_start_hour=float(sample_event["earliest_start_hour"]),
        latest_finish_hour=float(sample_event["latest_finish_hour"]),
        preferred_start_hour=float(sample_event["preferred_start_hour_simulated"]),
        flexible_load=bool(sample_event["flexible_load"])
    )
    assert sched_res["is_feasible"] is True
    assert sched_res["baseline_cost_inr"] >= 0.0
    assert sched_res["recommended_cost_inr"] >= 0.0

    # 5. Insights generation
    insights = pipe["insights_svc"].generate_household_insights(
        household_profile=hh_profile,
        hourly_df=hourly_df,
        events_df=annotated_events
    )
    assert len(insights) >= 2
    for ins in insights:
        assert "title" in ins
        assert "evidence" in ins
        assert "suggested_action" in ins
