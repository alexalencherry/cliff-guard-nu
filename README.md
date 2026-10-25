# Cliff Guard — Smarter Energy. Lower Bills. Safer Homes.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Isolation%20Forest-F7931E.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Build-Passing%2021%2F21%20Tests-success.svg)]()

> **Notice: Research & Software Prototype.**  
> All household profiles, hourly records, weather parameters, appliance event logs, and illustrative tariffs in this project are **SYNTHETIC** simulations. Automated schedule approval is simulated and does not trigger physical IoT actuation.

---

## 1. Problem Statement & Overview

Residential households in Kerala and across India face escalating electricity bills and power grid stress during peak evening hours (6:00 PM – 10:00 PM). At the same time, high-power flexible domestic appliances—such as **submersible water pumps**, **storage water heaters (geysers)**, and **washing machines**—are routinely operated during these high-cost periods. Furthermore, undetected mechanical drag (e.g. pump bearing friction, dry runs) or heating element scaling wastes energy and creates severe fire hazards.

**Cliff Guard** provides an intelligent, local-first energy intelligence platform that:
1. Visualizes 24-hour diurnal household energy patterns across Kerala districts.
2. Identifies multivariate physical anomalies using an unsupervised **Isolation Forest** machine learning model.
3. Optimizes flexible appliance start times under configurable **Time-of-Day (ToD)** Indian Rupee (₹) tariffs.
4. Generates evidence-based energy intelligence recommendations.
5. Provides interactive simulated automation approval for load shifting.

---

## 2. System Architecture & Project Structure

Cliff Guard is designed with clean separation of concerns:

```
cliff_guard/
├── app.py                      # Main Streamlit dashboard application
├── requirements.txt            # Python dependencies
├── pytest.ini                  # Pytest configuration
├── run_app.sh                  # Execution script for dev server on port 3000
├── README.md                   # Full documentation
├── BUILD_STATUS.md             # Development status and test reports
├── .gitignore                  # Git ignore rules
│
├── .streamlit/
│   └── config.toml             # Streamlit server & dark energy-tech theme config
│
├── config/
│   ├── tariff_config.json      # Illustrative Kerala ToD tariff parameters (₹/kWh)
│   └── app_config.json         # Global application metadata and device profiles
│
├── data/
│   ├── household_cases.csv     # 50 synthetic Kerala household profiles
│   ├── household_hourly_data.csv # 1,200 hourly consumption records (50 HH x 24h)
│   ├── device_consumption.csv  # 868 appliance cycle logs with physical parameters
│   └── appliance_profiles.csv  # Nominal engineering specs for supported devices
│
├── models/
│   ├── __init__.py
│   └── anomaly_detector.py     # Scikit-learn Isolation Forest model
│
├── services/
│   ├── __init__.py
│   ├── data_service.py         # Data caching, loading, and schema validation
│   ├── tariff_engine.py        # Indian Rupee (₹) multi-period ToD calculator
│   ├── scheduler.py            # Constraint-aware smart scheduling optimizer
│   ├── anomaly_service.py      # ML inference wrapper and diagnostic inspector
│   └── insights_service.py     # Deterministic evidence-based recommendations
│
├── utils/
│   ├── __init__.py
│   ├── validation.py           # Schema, physical consistency & range validation
│   └── formatting.py           # INR (₹), kWh, power, and time formatting helpers
│
├── scripts/
│   └── generate_data.py        # Reproducible synthetic dataset generator (Seed 42)
│
└── tests/
    ├── __init__.py
    ├── test_data.py            # Data schema, row count & physical formula tests
    ├── test_tariff.py          # Tariff rates, boundaries & midnight-crossing tests
    ├── test_scheduler.py       # Constraint, feasibility & tie-breaking tests
    ├── test_anomaly.py         # Isolation Forest training, features & metrics tests
    └── test_integration.py     # End-to-end pipeline integration tests
```

---

## 3. Technology Stack

- **Language:** Python 3.11
- **UI Framework:** Streamlit (v1.40+)
- **Data Engineering:** Pandas, NumPy
- **Interactive Visualizations:** Plotly Express & Plotly Graph Objects
- **Machine Learning:** Scikit-Learn (`sklearn.ensemble.IsolationForest`)
- **Automated Testing:** Pytest

---

## 4. Synthetic Datasets

All datasets are generated deterministically with **Random Seed 42**:

| Dataset | File Path | Records | Description |
|---|---|---|---|
| **A. Household Cases** | `data/household_cases.csv` | 50 rows | 50 synthetic Kerala households across Kottayam, Ernakulam, Alappuzha, Thrissur, Thiruvananthapuram, Palakkad, Kozhikode, and Kollam. |
| **B. Hourly Consumption** | `data/household_hourly_data.csv` | 1,200 rows | Exactly 50 households × 24 hours. Diurnal baseload, morning prep, daytime, and evening peak curves. |
| **C. Device Consumption** | `data/device_consumption.csv` | 868 rows | 7-day appliance event logs across washing machines, water pumps, and water heaters. Includes ~5.4% physical anomalies. |
| **D. Appliance Profiles** | `data/appliance_profiles.csv` | 3 rows | Nominal power ratings (W), typical durations (min), and flexibility windows. |

> **Anti-Leakage Guarantee:** `is_synthetic_anomaly_label` and `scenario_type` are strictly excluded from the Machine Learning training matrix and are used exclusively in post-hoc metric evaluation.

---

## 5. Indian Rupee (₹) Tariff Engine

The tariff engine implements realistic Time-of-Day (ToD) electricity billing:

- **Off-Peak (Night):** ₹4.00 / kWh (10:00 PM – 06:00 AM) — crosses midnight.
- **Normal (Daytime):** ₹6.00 / kWh (06:00 AM – 06:00 PM).
- **Peak (Evening):** ₹8.00 / kWh (06:00 PM – 10:00 PM).

### Features
1. **Accurate Tariff Boundary Handling:** Splits multi-period runs (e.g. 17:30 to 18:30) with minute-level precision under constant-power assumptions.
2. **Midnight-Crossing Periods:** Accurately accounts for tasks spanning across midnight (e.g. 23:30 to 00:30).
3. **Genuine Comparison:** Reports zero savings when baseline is already optimal; never fabricates positive numbers.
4. **Editable in UI:** Allows user to test custom tariffs and observe instant recalculation.

---

## 6. Machine Learning Anomaly Detection

- **Algorithm:** Unsupervised `IsolationForest` (150 estimators, configurable contamination 2%–15%).
- **Physical Features Used:**
  - `power_watts`: Active power draw.
  - `duration_minutes`: Cycle operating duration.
  - `energy_kwh`: Physical energy consumed.
  - `power_deviation_ratio`: Fractional deviation from nominal appliance baseline.
  - `kwh_per_minute`: Energy consumption rate.
  - One-hot device indicators (`is_washing_machine`, `is_water_pump`, `is_water_heater`).
- **Explainability:** Each flagged anomaly generates domain-specific mechanical explanations and a technician inspection checklist.

---

## 7. Constraint-Aware Smart Scheduler

The scheduler solves for the minimum-cost start time $\tau^*$ within permissible bounds:

$$\tau^* = \arg\min_{\tau \in \mathcal{C}} \text{Cost}(\tau, P, D)$$

subject to:
1. $\tau \ge \tau_{\text{earliest}}$ (Earliest permitted start).
2. $\tau + D \le \tau_{\text{latest}}$ (Deadline constraint).
3. Inflexible loads ($\text{flexible} = \text{False}$) remain fixed at baseline.
4. Infeasible requests are clearly reported rather than producing invalid timings.
5. In case of identical minimum cost, proximity to the user's preferred start time serves as a tie-breaker.

---

## 8. Installation & Launch Instructions

### Linux / macOS / Bash
```bash
# 1. Clone repository & navigate to directory
cd cliff_guard

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate synthetic datasets
python scripts/generate_data.py

# 5. Run test suite (All 21 tests)
pytest -v

# 6. Launch Streamlit Dashboard
streamlit run app.py
```

### Windows PowerShell
```powershell
# 1. Create virtual environment
python -m venv .venv

# 2. Activate virtual environment
.\.venv\Scripts\Activate.ps1
# If PowerShell execution policy blocks activation, run:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 3. Install dependencies
python -m pip install -r requirements.txt

# 4. Generate synthetic datasets
python scripts/generate_data.py

# 5. Run automated test suite
python -m pytest -q

# 6. Launch Streamlit dashboard
python -m streamlit run app.py
```

---

## 9. Hackathon Demonstration Sequence

Follow this step-by-step flow for judging:

1. **Overview Dashboard:**
   - Select Kerala household **HH_001 (Kottayam — Two-Storey Villa)**.
   - Observe 24h consumption profile overlaid with shaded ToD tariff zones.
   - Note evening peak demand at 19:00 (₹8/kWh peak rate).
2. **Appliance Analytics:**
   - Filter by **Water Pump**.
   - Inspect power vs duration scatter plot and operating hour distribution.
3. **AI Anomaly Detection:**
   - Navigate to **AI Anomaly Detection**.
   - Review flagged outlier cycles (e.g., pump running at 2,450 W or 180 min).
   - Open **Diagnostic Inspector** to view the physical evidence and the technician inspection checklist.
   - Check the **Model Performance** tab to review real Confusion Matrix and F1 score against evaluation labels.
4. **Smart Scheduling:**
   - Select **Water Pump** (750W, 45 min).
   - Notice baseline start at 19:00 (Evening peak: ₹4.50).
   - Set Earliest Start to 05:00 and Latest Finish to 20:00.
   - Click to observe optimal start at **05:00 AM** (Off-peak: ₹2.25, 50% savings).
   - Review the Gantt schedule comparison chart.
5. **Simulated Automation Approval:**
   - Click **Approve Schedule (Simulated)**.
   - Observe badge updating to `STATUS: APPROVED (SIMULATED)` with timestamp.
   - Note prototype transparency disclaimer.
6. **Recommendations & Insights:**
   - Inspect ranked insights with quantified monthly rupee savings and actionable checkboxes.

---

## 10. Limitations & Future Scope

- **Simulation Only:** Schedule approval is simulated and does not send MQTT/Zigbee signals to physical smart plugs or relays.
- **Tariff Simplification:** Real utility bills often feature bimonthly telescopic slab rates in addition to ToD surcharges.
- **Hardware Integration:** Production deployment would require smart energy meters (e.g. CT clamp monitors), local home-assistant bridges, and safety interlocks.
