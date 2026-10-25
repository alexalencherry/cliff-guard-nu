"""
Data and configuration validation utilities for Cliff Guard.
Provides clear error reporting, physical consistency checks, and schema validation.
"""

import os
from typing import Dict, List, Tuple, Any
import pandas as pd

class ValidationError(Exception):
    """Custom exception raised when data or config fails validation."""
    pass

def validate_dataframe_schema(
    df: pd.DataFrame,
    required_columns: List[str],
    dataset_name: str
) -> Tuple[bool, List[str]]:
    """
    Validate that required columns exist and there are no unexpected null values.
    """
    issues = []
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        issues.append(f"[{dataset_name}] Missing required columns: {missing_cols}")

    for col in required_columns:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                issues.append(f"[{dataset_name}] Column '{col}' contains {null_count} null values.")

    return (len(issues) == 0, issues)

def validate_household_cases(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate household_cases.csv."""
    required = [
        "case_id", "household_type", "household_size", "district",
        "daily_energy_kwh", "monthly_energy_kwh", "peak_usage_hour",
        "baseline_power_watts", "scenario_type", "data_source"
    ]
    valid, issues = validate_dataframe_schema(df, required, "household_cases")

    if len(df) != 50:
        issues.append(f"[household_cases] Expected 50 records, got {len(df)}")

    if df["case_id"].nunique() != len(df):
        issues.append("[household_cases] Found duplicate case_ids")

    if (df["daily_energy_kwh"] <= 0).any():
        issues.append("[household_cases] Found non-positive daily_energy_kwh")

    if (df["baseline_power_watts"] <= 0).any():
        issues.append("[household_cases] Found non-positive baseline_power_watts")

    if ((df["peak_usage_hour"] < 0) | (df["peak_usage_hour"] > 23)).any():
        issues.append("[household_cases] Found invalid peak_usage_hour (must be 0-23)")

    if not (df["data_source"] == "SYNTHETIC").all():
        issues.append("[household_cases] Non-synthetic data_source detected")

    return (len(issues) == 0, issues)

def validate_household_hourly(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate household_hourly_data.csv (exactly 1200 rows: 50 HH * 24 hours)."""
    required = [
        "case_id", "hour", "timestamp", "energy_kwh", "power_watts",
        "temperature_c", "rainfall_mm", "day_type", "data_source"
    ]
    valid, issues = validate_dataframe_schema(df, required, "household_hourly_data")

    if len(df) != 1200:
        issues.append(f"[household_hourly_data] Expected exactly 1,200 records, got {len(df)}")

    if df["case_id"].nunique() != 50:
        issues.append(f"[household_hourly_data] Expected 50 unique households, got {df['case_id'].nunique()}")

    # Check for duplicate (case_id, hour) pairs
    dups = df.duplicated(subset=["case_id", "hour"]).sum()
    if dups > 0:
        issues.append(f"[household_hourly_data] Found {dups} duplicate (case_id, hour) entries")

    if (df["energy_kwh"] < 0).any():
        issues.append("[household_hourly_data] Found negative energy_kwh")

    if (df["power_watts"] < 0).any():
        issues.append("[household_hourly_data] Found negative power_watts")

    if ((df["hour"] < 0) | (df["hour"] > 23)).any():
        issues.append("[household_hourly_data] Found invalid hour (must be 0-23)")

    # Energy vs Power consistency check within 1.0W tolerance
    # energy_kwh in 1 hour * 1000 = power_watts
    diff = (df["energy_kwh"] * 1000.0 - df["power_watts"]).abs()
    if (diff > 1.0).any():
        max_diff = diff.max()
        issues.append(f"[household_hourly_data] Energy-power inconsistency detected (max diff: {max_diff:.2f}W)")

    return (len(issues) == 0, issues)

def validate_device_consumption(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate device_consumption.csv."""
    required = [
        "case_id", "timestamp", "device_id", "device_type", "power_watts",
        "duration_minutes", "energy_kwh", "flexible_load", "earliest_start_hour",
        "latest_finish_hour", "preferred_start_hour_simulated",
        "is_synthetic_anomaly_label", "data_source"
    ]
    valid, issues = validate_dataframe_schema(df, required, "device_consumption")

    if not (800 <= len(df) <= 1000):
        issues.append(f"[device_consumption] Expected 800-1000 records, got {len(df)}")

    if (df["power_watts"] <= 0).any():
        issues.append("[device_consumption] Found non-positive power_watts")

    if (df["duration_minutes"] <= 0).any():
        issues.append("[device_consumption] Found non-positive duration_minutes")

    if (df["energy_kwh"] <= 0).any():
        issues.append("[device_consumption] Found non-positive energy_kwh")

    # Formula check: energy_kwh = power * duration / 60 / 1000 (tolerance 0.01 kWh)
    expected_kwh = (df["power_watts"] * df["duration_minutes"]) / 60000.0
    formula_diff = (expected_kwh - df["energy_kwh"]).abs()
    if (formula_diff > 0.01).any():
        issues.append(f"[device_consumption] Energy formula mismatch detected (max diff: {formula_diff.max():.4f} kWh)")

    if not df["is_synthetic_anomaly_label"].isin([0, 1]).all():
        issues.append("[device_consumption] is_synthetic_anomaly_label must be binary 0 or 1")

    return (len(issues) == 0, issues)

def validate_appliance_profiles(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate appliance_profiles.csv."""
    required = [
        "device_type", "nominal_power_watts", "typical_duration_minutes",
        "typical_energy_kwh", "flexible_load", "earliest_start_hour",
        "latest_finish_hour", "illustrative_flag", "notes"
    ]
    valid, issues = validate_dataframe_schema(df, required, "appliance_profiles")

    if len(df) < 3:
        issues.append(f"[appliance_profiles] Expected at least 3 profiles, got {len(df)}")

    expected_types = {"washing_machine", "water_pump", "water_heater"}
    present_types = set(df["device_type"].unique())
    missing_types = expected_types - present_types
    if missing_types:
        issues.append(f"[appliance_profiles] Missing device profiles: {missing_types}")

    return (len(issues) == 0, issues)

def validate_tariff_config(tariff_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate tariff configuration dictionary."""
    issues = []
    if "periods" not in tariff_config or not isinstance(tariff_config["periods"], list):
        issues.append("Tariff config must have a 'periods' list.")
        return (False, issues)

    periods = tariff_config["periods"]
    if len(periods) == 0:
        issues.append("Tariff config must contain at least one period.")

    covered_hours = set()
    for p in periods:
        rate = p.get("rate_per_kwh", -1)
        if rate < 0:
            issues.append(f"Period '{p.get('id')}' has negative rate ({rate}). Rates must be >= 0.")
        
        windows = p.get("time_windows", [])
        for w in windows:
            start = w.get("start_hour", -1)
            end = w.get("end_hour", -1)
            if not (0 <= start <= 24 and 0 <= end <= 24):
                issues.append(f"Window in {p.get('id')} has invalid hours: {start} - {end}")
            elif start == end:
                issues.append(f"Window in {p.get('id')} has identical start and end: {start}")
            else:
                for h in range(start, end):
                    if h in covered_hours:
                        issues.append(f"Hour {h} is duplicated across tariff periods.")
                    covered_hours.add(h)

    # Check 24 hour coverage (0 to 23)
    missing_hours = set(range(24)) - covered_hours
    if missing_hours:
        issues.append(f"Tariff windows do not cover full 24 hours. Missing hours: {sorted(list(missing_hours))}")

    return (len(issues) == 0, issues)
