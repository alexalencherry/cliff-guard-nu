# Cliff Guard — Build & Verification Status

**Date:** October 2026  
**Environment:** Python 3.11.2 | Streamlit 1.66+ | Debian Linux Container (AI Studio)  
**Overall Status:** ✅ **100% COMPLETE & VERIFIED**

---

## 1. Environment & Stack Verification

- [x] Environment check performed: Python 3.11, pip, and venv available.
- [x] Virtual environment `.venv` created and dependencies installed:
  - `streamlit>=1.30.0`
  - `pandas>=2.0.0`
  - `numpy>=1.24.0`
  - `plotly>=5.18.0`
  - `scikit-learn>=1.3.0`
  - `pytest>=7.4.0`
- [x] Streamlit configured on port 3000 via `.streamlit/config.toml` and `run_app.sh`.
- [x] Streamlit dev server tested and responding to HTTP requests on port 3000.

---

## 2. Synthetic Datasets Generation & Validation

All 4 datasets generated reproducibly via `scripts/generate_data.py` with seed 42 and marked `SYNTHETIC`:

| Dataset | Target File | Generated Rows | Validation Status |
|---|---|---|---|
| **A. Household Cases** | `data/household_cases.csv` | **50** | ✅ Passed (50 unique IDs, 8 Kerala districts, physical consistency) |
| **B. Hourly Consumption** | `data/household_hourly_data.csv` | **1,200** | ✅ Passed (Exactly 50 HH × 24h, power = energy × 1000) |
| **C. Device Consumption** | `data/device_consumption.csv` | **868** | ✅ Passed (800–1000 window, ~5.4% anomalies, energy formula verified) |
| **D. Appliance Profiles** | `data/appliance_profiles.csv` | **3** | ✅ Passed (washing_machine, water_pump, water_heater) |

---

## 3. Core Modules & Services Implemented

1. **Tariff Engine (`services/tariff_engine.py`):**
   - Configurable Kerala ToD rates (Off-peak ₹4, Normal ₹6, Peak ₹8).
   - Boundary split calculation under constant-power assumption.
   - Midnight-crossing window and event handling.
   - Genuine baseline vs recommended schedule cost comparison.
   - Negative rate rejection.

2. **ML Anomaly Detector (`models/anomaly_detector.py` & `services/anomaly_service.py`):**
   - Unsupervised `IsolationForest` (150 estimators, configurable contamination).
   - Defensible physical features: `power_watts`, `duration_minutes`, `energy_kwh`, `power_deviation_ratio`, `kwh_per_minute`, one-hot device indicators.
   - Strict anti-leakage: `is_synthetic_anomaly_label` and `scenario_type` never passed to model.
   - Diagnostic inspector with domain-specific mechanical checklists.
   - Evaluation tab with Confusion Matrix, Precision, Recall, and F1.

3. **Smart Scheduler (`services/scheduler.py`):**
   - Constraint-aware optimization within `[earliest_start_hour, latest_finish_hour]`.
   - Inflexible appliance preservation.
   - Infeasible window detection and explanatory messaging.
   - Preferred start tie-breaking when costs are identical.
   - Proportional ToD cost minimization.

4. **Evidence-Based Insights (`services/insights_service.py`):**
   - Ranked recommendation cards with severity badges (High, Medium, Info).
   - Quantified monthly rupee (₹) savings.
   - Technical action items and physical limitation disclaimers.

5. **Streamlit UI (`app.py`):**
   - Dark energy-tech visual theme.
   - 7 comprehensive navigation sections.
   - Interactive Plotly charts (diurnal curves, tariff zones, Gantt schedule timeline).
   - Simulated Automation Approval workflow with session state persistence.
   - Live tariff editor with instant recalculations.

---

## 4. Automated Test Results

Executed via `./.venv/bin/pytest -v`:

```text
============================= test session starts ==============================
platform linux -- Python 3.11.2, pytest-9.1.1, pluggy-1.7.0 -- /app/applet/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /app/applet
configfile: pytest.ini
testpaths: tests
plugins: anyio-4.15.1
collected 21 items

tests/test_anomaly.py::test_feature_extraction_strictly_excludes_labels PASSED [  4%]
tests/test_anomaly.py::test_anomaly_detector_training_and_prediction PASSED [  9%]
tests/test_anomaly.py::test_model_evaluation_metrics PASSED              [ 14%]
tests/test_anomaly.py::test_insufficient_samples_raises PASSED           [ 19%]
tests/test_data.py::test_datasets_exist PASSED                           [ 23%]
tests/test_data.py::test_household_cases_schema_and_values PASSED        [ 28%]
tests/test_data.py::test_household_hourly_data_schema_and_consistency PASSED [ 33%]
tests/test_data.py::test_device_consumption_events PASSED                [ 38%]
tests/test_data.py::test_appliance_profiles PASSED                       [ 42%]
tests/test_integration.py::test_full_pipeline_flow PASSED                [ 47%]
tests/test_scheduler.py::test_flexible_load_shifts_away_from_peak PASSED [ 52%]
tests/test_scheduler.py::test_inflexible_load_is_not_moved PASSED        [ 57%]
tests/test_scheduler.py::test_infeasible_schedule_detected PASSED        [ 61%]
tests/test_scheduler.py::test_no_fabricated_savings_when_baseline_is_cheapest PASSED [ 66%]
tests/test_scheduler.py::test_tie_breaking_favors_preferred_hour PASSED  [ 71%]
tests/test_tariff.py::test_hourly_rates_and_periods PASSED               [ 76%]
tests/test_tariff.py::test_event_cost_single_period PASSED               [ 80%]
tests/test_tariff.py::test_event_cost_multi_period_split PASSED          [ 85%]
tests/test_tariff.py::test_event_cost_crossing_midnight PASSED           [ 90%]
tests/test_tariff.py::test_negative_rates_rejected PASSED                [ 95%]
tests/test_tariff.py::test_compare_schedules PASSED                      [100%]

============================== 21 passed in 3.41s ==============================
```

---

## 5. Verification Commands

```bash
# Run test suite
./.venv/bin/pytest -v

# Run synthetic dataset generator
./.venv/bin/python scripts/generate_data.py

# Launch application
./run_app.sh
```
