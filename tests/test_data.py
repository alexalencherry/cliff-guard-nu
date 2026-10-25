"""
Unit tests for Cliff Guard data generation and schemas.
"""

import os
import pandas as pd
import pytest

from utils.validation import (
    validate_household_cases,
    validate_household_hourly,
    validate_device_consumption,
    validate_appliance_profiles
)

def test_datasets_exist():
    """Verify that all four required CSV files exist."""
    assert os.path.exists("data/household_cases.csv")
    assert os.path.exists("data/household_hourly_data.csv")
    assert os.path.exists("data/device_consumption.csv")
    assert os.path.exists("data/appliance_profiles.csv")

def test_household_cases_schema_and_values():
    """Verify household_cases.csv row count, columns, and physical sanity."""
    df = pd.read_csv("data/household_cases.csv")
    valid, issues = validate_household_cases(df)
    assert valid, f"Validation issues: {issues}"
    assert len(df) == 50
    assert df["case_id"].nunique() == 50
    assert (df["data_source"] == "SYNTHETIC").all()
    assert (df["daily_energy_kwh"] > 0).all()
    assert (df["baseline_power_watts"] > 0).all()

def test_household_hourly_data_schema_and_consistency():
    """Verify household_hourly_data.csv has exactly 1,200 records and consistent energy-power."""
    df = pd.read_csv("data/household_hourly_data.csv")
    valid, issues = validate_household_hourly(df)
    assert valid, f"Validation issues: {issues}"
    assert len(df) == 1200
    assert df["case_id"].nunique() == 50
    # Exactly 24 records per household
    hh_counts = df["case_id"].value_counts()
    assert (hh_counts == 24).all()
    # Physical consistency: power in watts ~= energy_kwh * 1000
    diff = (df["energy_kwh"] * 1000.0 - df["power_watts"]).abs()
    assert (diff <= 1.0).all()
    assert (df["data_source"] == "SYNTHETIC").all()

def test_device_consumption_events():
    """Verify device_consumption.csv schema, event counts, and energy formula."""
    df = pd.read_csv("data/device_consumption.csv")
    valid, issues = validate_device_consumption(df)
    assert valid, f"Validation issues: {issues}"
    assert 800 <= len(df) <= 1000
    # Energy formula: energy_kwh = (power_watts * duration_minutes) / 60 / 1000
    calc_kwh = (df["power_watts"] * df["duration_minutes"]) / 60000.0
    diff = (calc_kwh - df["energy_kwh"]).abs()
    assert (diff < 0.01).all()
    assert set(df["is_synthetic_anomaly_label"].unique()).issubset({0, 1})
    assert (df["data_source"] == "SYNTHETIC").all()

def test_appliance_profiles():
    """Verify default appliance profiles."""
    df = pd.read_csv("data/appliance_profiles.csv")
    valid, issues = validate_appliance_profiles(df)
    assert valid, f"Validation issues: {issues}"
    assert len(df) >= 3
    types = set(df["device_type"].values)
    assert "washing_machine" in types
    assert "water_pump" in types
    assert "water_heater" in types
