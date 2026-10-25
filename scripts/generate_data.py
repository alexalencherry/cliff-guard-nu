#!/usr/bin/env python3
"""
Cliff Guard - Synthetic Data Generator
Generates reproducible synthetic datasets for household energy modeling,
device consumption analytics, anomaly detection, and smart scheduling.

All data produced is explicitly marked SYNTHETIC.
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RANDOM_SEED = 42

def set_seeds(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)

def generate_appliance_profiles(output_dir="data"):
    """
    Dataset D: Appliance Profiles
    Default illustrative physical ratings and typical operating windows.
    """
    profiles = [
        {
            "device_type": "washing_machine",
            "nominal_power_watts": 500.0,
            "typical_duration_minutes": 60,
            "typical_energy_kwh": 0.500,
            "flexible_load": 1,
            "earliest_start_hour": 6.0,
            "latest_finish_hour": 22.0,
            "illustrative_flag": 1,
            "notes": "Illustrative 7kg front/top load washer. Normal wash cycle ~60 min."
        },
        {
            "device_type": "water_pump",
            "nominal_power_watts": 750.0,
            "typical_duration_minutes": 45,
            "typical_energy_kwh": 0.5625,
            "flexible_load": 1,
            "earliest_start_hour": 5.0,
            "latest_finish_hour": 20.0,
            "illustrative_flag": 1,
            "notes": "Illustrative 1HP domestic open-well/submersible pump filling overhead tank."
        },
        {
            "device_type": "water_heater",
            "nominal_power_watts": 2000.0,
            "typical_duration_minutes": 30,
            "typical_energy_kwh": 1.000,
            "flexible_load": 1,
            "earliest_start_hour": 5.0,
            "latest_finish_hour": 22.0,
            "illustrative_flag": 1,
            "notes": "Illustrative 15-25L domestic storage geyser. Heating element ~2kW."
        }
    ]
    df = pd.DataFrame(profiles)
    path = os.path.join(output_dir, "appliance_profiles.csv")
    df.to_csv(path, index=False)
    print(f"Generated {len(df)} records in {path}")
    return df

def generate_household_cases(output_dir="data", n_cases=50):
    """
    Dataset A: Household Profiles
    50 synthetic households across Kerala districts.
    """
    districts = [
        "Kottayam", "Ernakulam", "Alappuzha", "Thrissur",
        "Thiruvananthapuram", "Palakkad", "Kozhikode", "Kollam"
    ]
    household_types = [
        "Independent House", "Two-Storey Villa", "Suburban House",
        "Apartment 3BHK", "Apartment 2BHK", "Traditional Nalukettu"
    ]
    scenario_types = [
        "Standard Suburban", "High Daytime Solar/AC", "Compact Low-Use",
        "Heavy Evening Peak", "Agricultural Well Pump Active"
    ]

    records = []
    for i in range(1, n_cases + 1):
        case_id = f"HH_{i:03d}"
        h_type = random.choice(household_types)
        district = random.choice(districts)
        scenario = random.choice(scenario_types)
        
        # Household size 2 to 6
        if "Apartment 2BHK" in h_type or "Compact" in scenario:
            size = random.randint(2, 3)
            base_kwh = random.uniform(5.5, 9.5)
            baseline_power = random.uniform(120, 200)
        elif "Villa" in h_type or "Nalukettu" in h_type:
            size = random.randint(4, 6)
            base_kwh = random.uniform(16.0, 26.0)
            baseline_power = random.uniform(280, 480)
        else:
            size = random.randint(3, 5)
            base_kwh = random.uniform(10.0, 18.0)
            baseline_power = random.uniform(180, 320)

        # Peak hour 18 to 22 (Kerala standard residential peak)
        if scenario == "High Daytime Solar/AC":
            peak_hour = random.choice([14, 15, 19, 20])
        else:
            peak_hour = random.choice([19, 20, 21])

        daily_kwh = round(base_kwh, 2)
        monthly_kwh = round(daily_kwh * 30.0, 1)

        records.append({
            "case_id": case_id,
            "household_type": h_type,
            "household_size": size,
            "district": district,
            "daily_energy_kwh": daily_kwh,
            "monthly_energy_kwh": monthly_kwh,
            "peak_usage_hour": peak_hour,
            "baseline_power_watts": round(baseline_power, 1),
            "scenario_type": scenario,
            "data_source": "SYNTHETIC"
        })

    df = pd.DataFrame(records)
    path = os.path.join(output_dir, "household_cases.csv")
    df.to_csv(path, index=False)
    print(f"Generated {len(df)} records in {path}")
    return df

def generate_household_hourly(household_cases_df, output_dir="data"):
    """
    Dataset B: Hourly Household Consumption
    Exactly 1,200 records: 50 households * 24 hours.
    """
    records = []
    base_date = datetime(2026, 10, 24, 0, 0, 0)

    # Typical 24-hour diurnal factor curve for tropical residential loads
    # 0 to 5: Night sleep (baseload)
    # 6 to 9: Morning routine (cooking, pump, geyser)
    # 10 to 17: Daytime (fans, refrigeration, light loads or daytime AC)
    # 18 to 22: Evening peak (lighting, entertainment, kitchen, dinner)
    # 23: Late night wind-down
    diurnal_curve = [
        0.022, 0.020, 0.019, 0.019, 0.021, 0.025,  # 00:00 - 05:00
        0.045, 0.065, 0.060, 0.048,                 # 06:00 - 09:00
        0.040, 0.038, 0.042, 0.045, 0.046, 0.042, 0.040, 0.045, # 10:00 - 17:00
        0.080, 0.095, 0.090, 0.075, 0.055,          # 18:00 - 22:00
        0.030                                       # 23:00
    ]
    # Normalize curve sum to 1.0
    diurnal_weights = np.array(diurnal_curve) / sum(diurnal_curve)

    for _, hh in household_cases_df.iterrows():
        case_id = hh["case_id"]
        daily_kwh = float(hh["daily_energy_kwh"])
        peak_hour = int(hh["peak_usage_hour"])

        # Temperature variation (25°C night to 33°C day)
        # Synthetic weather
        for hour in range(24):
            dt = base_date + timedelta(hours=hour)
            timestamp = dt.strftime("%Y-%m-%d %H:00:00")

            # Weight for this hour with individual household noise
            w = diurnal_weights[hour]
            if hour == peak_hour:
                w *= 1.25
            noise = np.random.uniform(0.92, 1.08)
            hour_kwh = max(0.05, round(daily_kwh * w * noise, 3))
            
            # Consistent power in watts: energy (kWh) in 1 hour = power (kW)
            power_watts = round(hour_kwh * 1000.0, 1)

            # Synthetic temperature
            if 0 <= hour <= 5:
                temp_c = round(24.5 + np.random.uniform(-0.5, 0.5), 1)
            elif 6 <= hour <= 11:
                temp_c = round(25.0 + (hour - 5) * 1.3 + np.random.uniform(-0.3, 0.3), 1)
            elif 12 <= hour <= 16:
                temp_c = round(32.5 + np.random.uniform(-0.8, 0.8), 1)
            else:
                temp_c = round(30.0 - (hour - 16) * 0.7 + np.random.uniform(-0.4, 0.4), 1)

            rainfall_mm = 0.0 if np.random.random() > 0.15 else round(np.random.uniform(1.0, 14.0), 1)
            day_type = "Weekend" if (dt.weekday() >= 5) else "Weekday"

            records.append({
                "case_id": case_id,
                "hour": hour,
                "timestamp": timestamp,
                "energy_kwh": hour_kwh,
                "power_watts": power_watts,
                "temperature_c": temp_c,
                "rainfall_mm": rainfall_mm,
                "day_type": day_type,
                "data_source": "SYNTHETIC"
            })

    df = pd.DataFrame(records)
    path = os.path.join(output_dir, "household_hourly_data.csv")
    df.to_csv(path, index=False)
    print(f"Generated {len(df)} records in {path} (Expected exactly 1200: {len(df) == 1200})")
    return df

def generate_device_consumption(household_cases_df, output_dir="data", target_events=900):
    """
    Dataset C: Appliance Events (7 simulated days)
    Supported devices: washing_machine, water_pump, water_heater.
    Target ~850-950 events across 50 households over 7 days.
    Deliberately introduces ~5-7% subtle & pronounced synthetic anomalies.
    """
    devices = [
        {
            "type": "washing_machine",
            "base_power": 500.0,
            "power_std": 45.0,
            "base_duration": 60,
            "duration_std": 10,
            "earliest_hour": 6.0,
            "latest_hour": 22.0,
            "flexible": 1,
            "preferred_hours": [7.0, 8.0, 18.0, 19.0, 20.0]
        },
        {
            "type": "water_pump",
            "base_power": 750.0,
            "power_std": 50.0,
            "base_duration": 45,
            "duration_std": 8,
            "earliest_hour": 5.0,
            "latest_hour": 20.0,
            "flexible": 1,
            "preferred_hours": [6.0, 6.5, 7.0, 18.0, 18.5, 19.0]
        },
        {
            "type": "water_heater",
            "base_power": 2000.0,
            "power_std": 120.0,
            "base_duration": 30,
            "duration_std": 6,
            "earliest_hour": 5.0,
            "latest_hour": 22.0,
            "flexible": 1,
            "preferred_hours": [6.0, 6.5, 7.0, 19.0, 19.5, 20.0]
        }
    ]

    records = []
    start_date = datetime(2026, 10, 18, 0, 0, 0)
    event_counter = 1

    case_ids = household_cases_df["case_id"].tolist()

    # Generate roughly 2 to 3 events per household across 7 days
    for day_offset in range(7):
        current_day = start_date + timedelta(days=day_offset)
        for case_id in case_ids:
            # Each household runs 2 or 3 appliance cycles per day
            n_events_today = random.choice([2, 3]) if random.random() < 0.92 else 2
            chosen_devices = random.sample(devices, k=min(n_events_today, len(devices)))

            for dev in chosen_devices:
                device_type = dev["type"]
                device_id = f"DEV_{case_id}_{device_type[:3].upper()}"

                # Check if this event should be a synthetic anomaly (~6% chance)
                is_anomaly = 1 if (random.random() < 0.065) else 0

                pref_hour = random.choice(dev["preferred_hours"])
                # Add random minute offset (0, 15, 30, 45)
                start_minute = random.choice([0, 15, 30, 45])
                event_time = current_day.replace(
                    hour=int(pref_hour),
                    minute=start_minute,
                    second=0
                )

                if is_anomaly:
                    anomaly_type = random.choice(["high_power", "long_duration", "both_high"])
                    if anomaly_type == "high_power":
                        # Power surge/bearing failure/element degradation
                        power_watts = dev["base_power"] * random.uniform(2.2, 3.2)
                        duration = dev["base_duration"] + random.randint(-5, 10)
                    elif anomaly_type == "long_duration":
                        # Stuck float switch / unmonitored heater / continuous cycle
                        power_watts = dev["base_power"] * random.uniform(0.95, 1.1)
                        duration = dev["base_duration"] * random.randint(3, 5)
                    else:
                        # High power & long duration
                        power_watts = dev["base_power"] * random.uniform(1.8, 2.5)
                        duration = dev["base_duration"] * random.randint(2, 4)
                else:
                    power_watts = max(200.0, np.random.normal(dev["base_power"], dev["power_std"]))
                    duration = max(15, int(np.random.normal(dev["base_duration"], dev["duration_std"])))

                power_watts = round(float(power_watts), 1)
                duration = int(duration)
                
                # Physical formula: energy_kwh = (power_watts * duration_minutes) / 60 / 1000
                energy_kwh = round((power_watts * duration) / 60.0 / 1000.0, 4)

                records.append({
                    "case_id": case_id,
                    "timestamp": event_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "device_id": device_id,
                    "device_type": device_type,
                    "power_watts": power_watts,
                    "duration_minutes": duration,
                    "energy_kwh": energy_kwh,
                    "flexible_load": dev["flexible"],
                    "earliest_start_hour": dev["earliest_hour"],
                    "latest_finish_hour": dev["latest_hour"],
                    "preferred_start_hour_simulated": round(pref_hour + start_minute / 60.0, 2),
                    "is_synthetic_anomaly_label": is_anomaly,
                    "data_source": "SYNTHETIC"
                })
                event_counter += 1

    df = pd.DataFrame(records)
    # Sort chronologically
    df = df.sort_values(by=["timestamp", "case_id"]).reset_index(drop=True)
    path = os.path.join(output_dir, "device_consumption.csv")
    df.to_csv(path, index=False)
    anom_count = int(df["is_synthetic_anomaly_label"].sum())
    print(f"Generated {len(df)} records in {path} (Synthetic anomalies: {anom_count}, {round(anom_count/len(df)*100, 1)}%)")
    return df

def generate_all():
    set_seeds(RANDOM_SEED)
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    print("=== Generating Cliff Guard Synthetic Datasets ===")
    df_profiles = generate_appliance_profiles(output_dir)
    df_cases = generate_household_cases(output_dir, n_cases=50)
    df_hourly = generate_household_hourly(df_cases, output_dir)
    df_events = generate_device_consumption(df_cases, output_dir)
    
    print("\nDataset Summary:")
    print(f"1. Household Profiles:     {len(df_cases)} rows (Expected 50)")
    print(f"2. Hourly Household Data:  {len(df_hourly)} rows (Expected 1200)")
    print(f"3. Appliance Events:       {len(df_events)} rows (Expected ~800-1000)")
    print(f"4. Appliance Profiles:     {len(df_profiles)} rows (Expected 3)")
    print("All datasets generated reproducibly with seed 42 and marked SYNTHETIC.")

if __name__ == "__main__":
    generate_all()
