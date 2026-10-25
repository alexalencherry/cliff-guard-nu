"""
Formatting utilities for Cliff Guard (INR currency, energy, power, and time representations).
"""

def format_inr(amount: float, decimals: int = 2) -> str:
    """Format numeric values as Indian Rupee (₹)."""
    if amount is None:
        return "₹0.00"
    try:
        val = float(amount)
        return f"₹{val:,.{decimals}f}"
    except (ValueError, TypeError):
        return "₹0.00"

def format_kwh(kwh: float, decimals: int = 2) -> str:
    """Format energy in kilowatt-hours."""
    if kwh is None:
        return "0.00 kWh"
    try:
        val = float(kwh)
        return f"{val:.{decimals}f} kWh"
    except (ValueError, TypeError):
        return "0.00 kWh"

def format_power(watts: float) -> str:
    """Format power in watts (W) or kilowatts (kW) depending on magnitude."""
    if watts is None:
        return "0 W"
    try:
        w = float(watts)
        if abs(w) >= 1000:
            return f"{w / 1000.0:.2f} kW"
        return f"{w:.0f} W"
    except (ValueError, TypeError):
        return "0 W"

def decimal_hour_to_time_str(decimal_hour: float) -> str:
    """
    Convert decimal hour (e.g. 18.5) to '18:30' (24-hour) and '06:30 PM' string.
    """
    if decimal_hour is None:
        return "00:00"
    try:
        total_minutes = int(round(float(decimal_hour) * 60)) % (24 * 60)
        hours = total_minutes // 60
        minutes = total_minutes % 60
        period = "AM" if hours < 12 else "PM"
        h12 = hours % 12
        if h12 == 0:
            h12 = 12
        return f"{hours:02d}:{minutes:02d} ({h12:02d}:{minutes:02d} {period})"
    except (ValueError, TypeError):
        return "00:00"

def decimal_hour_to_hhmm(decimal_hour: float) -> str:
    """Convert decimal hour (e.g. 18.5) to clean 24-hr format '18:30'."""
    if decimal_hour is None:
        return "00:00"
    try:
        total_minutes = int(round(float(decimal_hour) * 60)) % (24 * 60)
        hours = total_minutes // 60
        minutes = total_minutes % 60
        return f"{hours:02d}:{minutes:02d}"
    except (ValueError, TypeError):
        return "00:00"

def format_minutes_to_duration(minutes: int) -> str:
    """Format duration in minutes as readable string (e.g. '1h 15m' or '45m')."""
    if not minutes:
        return "0m"
    m = int(minutes)
    if m >= 60:
        hrs = m // 60
        rem_m = m % 60
        return f"{hrs}h {rem_m}m" if rem_m > 0 else f"{hrs}h"
    return f"{m}m"
