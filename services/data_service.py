"""
Data access and caching service for Cliff Guard.
Loads, validates, and queries the synthetic datasets.
"""

import os
from typing import Dict, List, Optional, Tuple
import pandas as pd

from utils.validation import (
    validate_household_cases,
    validate_household_hourly,
    validate_device_consumption,
    validate_appliance_profiles,
    ValidationError
)

class DataService:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.household_cases: Optional[pd.DataFrame] = None
        self.household_hourly: Optional[pd.DataFrame] = None
        self.device_consumption: Optional[pd.DataFrame] = None
        self.appliance_profiles: Optional[pd.DataFrame] = None
        self.load_all()

    def _ensure_data_files_exist(self):
        """Ensure all required CSV files exist; generate if missing."""
        required = [
            "household_cases.csv",
            "household_hourly_data.csv",
            "device_consumption.csv",
            "appliance_profiles.csv"
        ]
        missing = [f for f in required if not os.path.exists(os.path.join(self.data_dir, f))]
        if missing:
            print(f"Data files missing: {missing}. Automatically generating synthetic data...")
            from scripts.generate_data import generate_all
            generate_all()

    def load_all(self):
        """Load and validate all four synthetic datasets."""
        self._ensure_data_files_exist()

        cases_path = os.path.join(self.data_dir, "household_cases.csv")
        hourly_path = os.path.join(self.data_dir, "household_hourly_data.csv")
        device_path = os.path.join(self.data_dir, "device_consumption.csv")
        profiles_path = os.path.join(self.data_dir, "appliance_profiles.csv")

        self.household_cases = pd.read_csv(cases_path)
        self.household_hourly = pd.read_csv(hourly_path)
        self.device_consumption = pd.read_csv(device_path)
        self.appliance_profiles = pd.read_csv(profiles_path)

        # Validate
        v1, err1 = validate_household_cases(self.household_cases)
        v2, err2 = validate_household_hourly(self.household_hourly)
        v3, err3 = validate_device_consumption(self.device_consumption)
        v4, err4 = validate_appliance_profiles(self.appliance_profiles)

        all_errs = err1 + err2 + err3 + err4
        if all_errs:
            print(f"Warning: validation issues encountered: {all_errs}")

    def get_household_cases(self) -> pd.DataFrame:
        """Return all 50 household profiles."""
        if self.household_cases is None:
            self.load_all()
        return self.household_cases.copy()

    def get_household_by_id(self, case_id: str) -> Optional[Dict]:
        """Get profile dict for a specific household ID (e.g. 'HH_001')."""
        cases = self.get_household_cases()
        row = cases[cases["case_id"] == case_id]
        if len(row) > 0:
            return row.iloc[0].to_dict()
        return None

    def get_hourly_data_for_household(self, case_id: str) -> pd.DataFrame:
        """Get 24-hour consumption data for a specific household."""
        if self.household_hourly is None:
            self.load_all()
        df = self.household_hourly[self.household_hourly["case_id"] == case_id].copy()
        df = df.sort_values(by="hour").reset_index(drop=True)
        return df

    def get_device_consumption(
        self,
        case_id: Optional[str] = None,
        device_type: Optional[str] = None
    ) -> pd.DataFrame:
        """Get appliance event logs with optional filtering."""
        if self.device_consumption is None:
            self.load_all()
        df = self.device_consumption.copy()
        if case_id and case_id != "All Households":
            df = df[df["case_id"] == case_id]
        if device_type and device_type != "All Appliances":
            df = df[df["device_type"] == device_type]
        return df.reset_index(drop=True)

    def get_appliance_profiles(self) -> pd.DataFrame:
        """Get default appliance engineering profiles."""
        if self.appliance_profiles is None:
            self.load_all()
        return self.appliance_profiles.copy()

    def get_profile_for_device(self, device_type: str) -> Optional[Dict]:
        """Get nominal specifications for a device type."""
        profiles = self.get_appliance_profiles()
        row = profiles[profiles["device_type"] == device_type]
        if len(row) > 0:
            return row.iloc[0].to_dict()
        return None
