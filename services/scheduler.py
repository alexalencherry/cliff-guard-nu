"""
Constraint-Aware Smart Scheduler for Cliff Guard.
Optimizes flexible appliance start times within permissible operating windows
to minimize electricity costs under Time-of-Day (ToD) tariffs while respecting
user deadlines and preferences.
"""

from typing import Dict, List, Any, Optional
from services.tariff_engine import TariffEngine
from utils.formatting import decimal_hour_to_hhmm, decimal_hour_to_time_str, format_inr

class SmartScheduler:
    def __init__(self, tariff_engine: TariffEngine, time_step_minutes: int = 15):
        """
        Initialize the scheduler with a TariffEngine instance.
        time_step_minutes: search resolution in minutes (default 15 minutes = 0.25h).
        """
        self.tariff_engine = tariff_engine
        self.time_step_minutes = time_step_minutes
        self.time_step_hours = time_step_minutes / 60.0

    def schedule_appliance(
        self,
        device_type: str,
        power_watts: float,
        duration_minutes: float,
        earliest_start_hour: float,
        latest_finish_hour: float,
        preferred_start_hour: float,
        flexible_load: bool = True,
        priority: str = "Normal"
    ) -> Dict[str, Any]:
        """
        Calculate the optimal start time for an appliance event.
        Returns a detailed result dictionary with feasibility, schedule recommendation,
        cost breakdown, and human-readable explanation.
        """
        dur_hours = duration_minutes / 60.0
        earliest = float(earliest_start_hour)
        latest = float(latest_finish_hour)
        preferred = float(preferred_start_hour)

        # Baseline evaluation
        baseline_cost_dict = self.tariff_engine.calculate_event_cost(
            power_watts=power_watts,
            start_hour=preferred,
            duration_minutes=duration_minutes
        )
        baseline_cost = baseline_cost_dict["total_cost_inr"]

        # Inflexible load check
        if not flexible_load:
            return {
                "is_feasible": True,
                "flexible_load": False,
                "recommended_start_hour": preferred,
                "recommended_finish_hour": (preferred + dur_hours) % 24.0,
                "baseline_start_hour": preferred,
                "baseline_finish_hour": (preferred + dur_hours) % 24.0,
                "baseline_cost_inr": baseline_cost,
                "recommended_cost_inr": baseline_cost,
                "savings_inr": 0.0,
                "savings_percentage": 0.0,
                "status": "Inflexible Load (Fixed)",
                "explanation": (
                    f"{device_type.replace('_', ' ').title()} is marked as a non-flexible load. "
                    "Its schedule cannot be automatically shifted without compromising user convenience."
                ),
                "breakdown": baseline_cost_dict["breakdown"],
                "all_candidates": []
            }

        # Check feasibility of time window
        # For simplicity, if latest <= earliest, assumes overnight window (e.g. 22:00 to 07:00 = 9h)
        if latest <= earliest:
            available_window_hours = (24.0 - earliest) + latest
        else:
            available_window_hours = latest - earliest

        if dur_hours > available_window_hours + 1e-5:
            return {
                "is_feasible": False,
                "flexible_load": True,
                "recommended_start_hour": None,
                "recommended_finish_hour": None,
                "baseline_start_hour": preferred,
                "baseline_finish_hour": (preferred + dur_hours) % 24.0,
                "baseline_cost_inr": baseline_cost,
                "recommended_cost_inr": None,
                "savings_inr": 0.0,
                "savings_percentage": 0.0,
                "status": "Infeasible Constraints",
                "explanation": (
                    f"Infeasible schedule: Appliance duration ({duration_minutes:.0f} mins / {dur_hours:.2f} hrs) "
                    f"exceeds the permitted operating window ({available_window_hours:.2f} hrs between "
                    f"{decimal_hour_to_hhmm(earliest)} and {decimal_hour_to_hhmm(latest)})."
                ),
                "breakdown": [],
                "all_candidates": []
            }

        # Generate discrete candidate start times
        candidates = []
        # Support both normal windows (earliest < latest) and overnight windows (earliest > latest)
        if earliest < latest:
            # Last valid start time is latest_finish_hour - dur_hours
            max_start = latest - dur_hours
            curr = earliest
            while curr <= max_start + 1e-4:
                candidates.append(curr)
                curr += self.time_step_hours
        else:
            # Window wraps past midnight
            # Segment 1: from earliest up to 24.0
            curr = earliest
            while curr <= 24.0:
                if (curr + dur_hours) <= (24.0 + latest):
                    candidates.append(curr % 24.0)
                curr += self.time_step_hours
            # Segment 2: from 0.0 up to latest - dur_hours
            max_start = latest - dur_hours
            curr = 0.0
            while curr <= max_start + 1e-4:
                candidates.append(curr)
                curr += self.time_step_hours

        if not candidates:
            candidates = [earliest]

        evaluated_candidates = []
        best_cand = None
        min_cost = float("inf")
        min_pref_diff = float("inf")

        for cand_start in candidates:
            cost_info = self.tariff_engine.calculate_event_cost(
                power_watts=power_watts,
                start_hour=cand_start,
                duration_minutes=duration_minutes
            )
            cost = cost_info["total_cost_inr"]
            finish_hr = (cand_start + dur_hours) % 24.0
            
            # Distance from preferred start for tie-breaking
            pref_diff = abs(cand_start - preferred)
            # circular distance on 24h clock
            pref_diff = min(pref_diff, 24.0 - pref_diff)

            evaluated_candidates.append({
                "start_hour": round(cand_start, 2),
                "start_time_str": decimal_hour_to_hhmm(cand_start),
                "finish_hour": round(finish_hr, 2),
                "finish_time_str": decimal_hour_to_hhmm(finish_hr),
                "cost_inr": cost,
                "pref_distance_hours": round(pref_diff, 2)
            })

            # Check if this candidate is better
            # 1. Lower cost
            # 2. Or if cost is identical, closer to preferred start
            is_better = False
            if cost < min_cost - 1e-4:
                is_better = True
            elif abs(cost - min_cost) <= 1e-4 and pref_diff < min_pref_diff:
                is_better = True

            if is_better:
                min_cost = cost
                min_pref_diff = pref_diff
                best_cand = (cand_start, finish_hr, cost_info)

        best_start, best_finish, best_cost_info = best_cand
        rec_cost = best_cost_info["total_cost_inr"]
        savings = round(baseline_cost - rec_cost, 2)
        pct_savings = round((savings / baseline_cost * 100.0), 1) if baseline_cost > 0 else 0.0

        # Construct explanation
        if savings > 0.0:
            rec_period = self.tariff_engine.get_period_name_for_hour(int(best_start))
            base_period = self.tariff_engine.get_period_name_for_hour(int(preferred))
            explanation = (
                f"Shifted operation from {decimal_hour_to_hhmm(preferred)} ({base_period}) to "
                f"{decimal_hour_to_hhmm(best_start)} ({rec_period}), saving {format_inr(savings)} "
                f"({pct_savings}% reduction) by taking advantage of lower ToD tariff rates."
            )
        elif abs(savings) <= 1e-4:
            explanation = (
                f"Baseline schedule at {decimal_hour_to_hhmm(preferred)} already utilizes the most economical "
                "available tariff window within your permissible constraints. No schedule change required."
            )
        else:
            explanation = (
                f"Selected constraint-compliant window at {decimal_hour_to_hhmm(best_start)}. "
                "Cost remains comparable to baseline."
            )

        return {
            "is_feasible": True,
            "flexible_load": True,
            "recommended_start_hour": best_start,
            "recommended_finish_hour": best_finish,
            "baseline_start_hour": preferred,
            "baseline_finish_hour": (preferred + dur_hours) % 24.0,
            "baseline_cost_inr": baseline_cost,
            "recommended_cost_inr": rec_cost,
            "savings_inr": savings,
            "savings_percentage": pct_savings,
            "status": "Schedule Optimized" if savings > 0 else "Baseline Optimal",
            "explanation": explanation,
            "breakdown": best_cost_info["breakdown"],
            "all_candidates": evaluated_candidates
        }
