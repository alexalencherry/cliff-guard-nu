"""
Cliff Guard — Smarter Energy. Lower Bills. Safer Homes.
Premium Energy Intelligence Platform featuring Unsupervised Machine Learning,
Constraint-Aware Tariff Optimization, and Evidence-Based Safety Diagnostics.

Designed with modern Lovable / Linear dark glassmorphic aesthetics.
All data, tariffs, and approval actions are SYNTHETIC simulations.
"""

import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from services.data_service import DataService
from services.tariff_engine import TariffEngine
from services.anomaly_service import AnomalyService
from services.scheduler import SmartScheduler
from services.insights_service import InsightsService
from utils.formatting import (
    format_inr,
    format_kwh,
    format_power,
    decimal_hour_to_hhmm,
    decimal_hour_to_time_str,
    format_minutes_to_duration
)

# ---------------------------------------------------------
# Page Configuration & Styling (Lovable / Linear High-End Dark)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Cliff Guard — Energy Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject modern styling: Inter font, glassmorphism, glowing micro-borders, refined cards
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap');

    /* Global typography */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    code, pre, .mono-val {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main background */
    .stApp {
        background-color: #0b0f19;
        color: #f8fafc;
    }

    /* Modern Glass Card */
    .glass-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.4rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(12px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .glass-card:hover {
        border-color: rgba(56, 189, 248, 0.25);
    }

    /* Hero Banner */
    .hero-banner {
        background: radial-gradient(120% 120% at 50% 0%, rgba(14, 165, 233, 0.15) 0%, rgba(15, 23, 42, 0.7) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 16px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .hero-title {
        font-size: 1.9rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
    }
    .hero-sub {
        font-size: 0.95rem;
        color: #94a3b8;
    }

    /* Pulsing Live Dot */
    .pulse-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 10px #10b981;
        margin-right: 6px;
        animation: pulse-glow 2s infinite;
    }
    @keyframes pulse-glow {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.2); }
    }

    /* Modern Badges */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.04em;
    }
    .badge-emerald {
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-amber {
        background: rgba(245, 158, 11, 0.12);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .badge-rose {
        background: rgba(244, 63, 94, 0.12);
        color: #fb7185;
        border: 1px solid rgba(244, 63, 94, 0.3);
    }
    .badge-sky {
        background: rgba(14, 165, 233, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(14, 165, 233, 0.3);
    }

    /* Custom KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 1.1rem;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .kpi-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94a3b8;
        margin-bottom: 0.4rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #ffffff;
        font-family: 'JetBrains Mono', monospace;
    }
    .kpi-sub {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 0.3rem;
    }

    /* Approved Stamp */
    .approved-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(6, 78, 59, 0.2) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 12px;
        padding: 1.2rem;
        margin-top: 1rem;
    }

    /* Sidebar refinement */
    section[data-testid="stSidebar"] {
        background-color: #080c14;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# State Initialization & Service Singletons
# ---------------------------------------------------------
@st.cache_resource
def get_data_service():
    return DataService(data_dir="data")

@st.cache_resource
def get_tariff_engine():
    return TariffEngine(config_path="config/tariff_config.json")

data_svc = get_data_service()

if "tariff_engine" not in st.session_state:
    st.session_state.tariff_engine = get_tariff_engine()

if "anomaly_contamination" not in st.session_state:
    st.session_state.anomaly_contamination = 0.06

if "approved_schedules" not in st.session_state:
    st.session_state.approved_schedules = []

tariff_engine: TariffEngine = st.session_state.tariff_engine
scheduler = SmartScheduler(tariff_engine=tariff_engine, time_step_minutes=15)
anomaly_svc = AnomalyService(contamination=st.session_state.anomaly_contamination)
insights_svc = InsightsService(tariff_engine=tariff_engine)

# ---------------------------------------------------------
# Sidebar Navigation & Household Selector
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1rem;">
            <div style="background: linear-gradient(135deg, #0284c7, #0369a1); width: 38px; height: 38px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.3rem; box-shadow: 0 0 15px rgba(2, 132, 199, 0.5);">⚡</div>
            <div>
                <div style="font-size: 1.25rem; font-weight: 800; letter-spacing: -0.02em; color: #ffffff;">CLIFF GUARD</div>
                <div style="font-size: 0.72rem; color: #38bdf8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">AI Energy Intelligence</div>
            </div>
        </div>
        <div style="margin-bottom: 1.2rem;">
            <span class="badge-pill badge-emerald"><span class="pulse-dot"></span>ENGINE ACTIVE</span>
            <span class="badge-pill badge-amber" style="margin-left: 4px;">SYNTHETIC</span>
        </div>
    """, unsafe_allow_html=True)

    nav_choice = st.radio(
        "NAVIGATION",
        [
            "1. Overview Dashboard",
            "2. Household & Device Analytics",
            "3. AI Anomaly Detection",
            "4. Smart Scheduling",
            "5. Recommendations & Insights",
            "6. 🧠 AI Architecture & Explainability",
            "7. Tariff Configuration",
            "8. About & Assumptions"
        ],
        index=0
    )

    st.markdown("<hr style='border-color: rgba(255,255,255,0.06);'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.75rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.4rem;'>Select Household</div>", unsafe_allow_html=True)
    
    cases_df = data_svc.get_household_cases()
    hh_options = [
        f"{row['case_id']} — {row['district']} ({row['household_type']})"
        for _, row in cases_df.iterrows()
    ]
    selected_hh_label = st.selectbox("Household", hh_options, index=0, label_visibility="collapsed")
    selected_case_id = selected_hh_label.split(" — ")[0]
    selected_hh = data_svc.get_household_by_id(selected_case_id)

    st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 10px; margin-top: 10px; font-size: 0.8rem; color: #94a3b8;">
            <div><b>District:</b> <span style="color: #f8fafc;">{selected_hh['district']}</span></div>
            <div><b>Type:</b> <span style="color: #f8fafc;">{selected_hh['household_type']}</span></div>
            <div><b>Occupants:</b> <span style="color: #f8fafc;">{selected_hh['household_size']} persons</span></div>
            <div><b>Baseload:</b> <span style="color: #38bdf8;">{format_power(selected_hh['baseline_power_watts'])}</span></div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color: rgba(255,255,255,0.06);'>", unsafe_allow_html=True)
    st.caption("Cliff Guard v1.0 • Lovable Energy-Tech Edition")

# ---------------------------------------------------------
# Plotly Luxury Dark Theme Helper
# ---------------------------------------------------------
def apply_luxury_chart_theme(fig, height=360):
    fig.update_layout(
        plot_bgcolor="#0d1424",
        paper_bgcolor="#090d16",
        font=dict(family="Plus Jakarta Sans", color="#f8fafc", size=11),
        margin=dict(l=35, r=25, t=40, b=35),
        height=height,
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.05)",
            zerolinecolor="rgba(255, 255, 255, 0.08)",
            tickfont=dict(color="#94a3b8")
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.05)",
            zerolinecolor="rgba(255, 255, 255, 0.08)",
            tickfont=dict(color="#94a3b8")
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#94a3b8")
        )
    )
    return fig

# ---------------------------------------------------------
# SECTION 1: OVERVIEW DASHBOARD
# ---------------------------------------------------------
if nav_choice == "1. Overview Dashboard":
    st.markdown(f"""
        <div class="hero-banner">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px;">
                <div>
                    <div class="hero-title">Household Energy Intelligence</div>
                    <div class="hero-sub">
                        Active Simulation Profile: <b style="color: #f8fafc;">{selected_hh['case_id']}</b> &nbsp;•&nbsp; 
                        <b>{selected_hh['district']}</b>, Kerala &nbsp;•&nbsp; 
                        {selected_hh['household_type']} ({selected_hh['household_size']} occupants)
                    </div>
                </div>
                <div style="display: flex; gap: 8px;">
                    <span class="badge-pill badge-sky">⚡ 24H DIURNAL MODEL</span>
                    <span class="badge-pill badge-amber">SCENARIO: {selected_hh['scenario_type'].upper()}</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    hourly_df = data_svc.get_hourly_data_for_household(selected_case_id)
    events_df = data_svc.get_device_consumption(case_id=selected_case_id)
    annotated_events = anomaly_svc.run_detection(events_df)

    daily_energy = hourly_df["energy_kwh"].sum()
    daily_cost_inr = sum(
        tariff_engine.calculate_hourly_cost(row["energy_kwh"], int(row["hour"]))
        for _, row in hourly_df.iterrows()
    )
    peak_hour = int(selected_hh["peak_usage_hour"])
    anomaly_count = int(annotated_events["ml_predicted_anomaly"].sum())
    flexible_events = int((events_df["flexible_load"] == 1).sum())

    potential_savings = 0.0
    for _, ev in events_df[events_df["flexible_load"] == 1].iterrows():
        opt = scheduler.schedule_appliance(
            device_type=ev["device_type"],
            power_watts=float(ev["power_watts"]),
            duration_minutes=float(ev["duration_minutes"]),
            earliest_start_hour=float(ev["earliest_start_hour"]),
            latest_finish_hour=float(ev["latest_finish_hour"]),
            preferred_start_hour=float(ev["preferred_start_hour_simulated"]),
            flexible_load=True
        )
        if opt["savings_inr"] > 0:
            potential_savings += opt["savings_inr"]

    # Refined 5 KPI Cards Row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label"><span>Daily Energy</span> <span>⚡</span></div>
                <div class="kpi-value">{daily_energy:.2f} <span style="font-size: 0.9rem; font-weight: 500; color: #94a3b8;">kWh</span></div>
                <div class="kpi-sub">Simulated 24-hr consumption</div>
            </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label"><span>Est. Daily Cost</span> <span>₹</span></div>
                <div class="kpi-value" style="color: #38bdf8;">{format_inr(daily_cost_inr)}</div>
                <div class="kpi-sub">Kerala ToD tariff model</div>
            </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label"><span>Peak Demand</span> <span>⏱️</span></div>
                <div class="kpi-value" style="color: #f59e0b;">{peak_hour:02d}:00</div>
                <div class="kpi-sub">Highest residential load window</div>
            </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label"><span>AI Anomalies</span> <span>🛡️</span></div>
                <div class="kpi-value" style="color: {'#fb7185' if anomaly_count > 0 else '#34d399'};">{anomaly_count}</div>
                <div class="kpi-sub">{'Outlier cycles flagged' if anomaly_count > 0 else 'All cycles optimal'}</div>
            </div>
        """, unsafe_allow_html=True)
    with k5:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label"><span>Sched. Savings</span> <span>💰</span></div>
                <div class="kpi-value" style="color: #34d399;">{format_inr(potential_savings)}</div>
                <div class="kpi-sub">Daily flexible load shift potential</div>
            </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Main Chart: 24h consumption profile with tariff overlays
    st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.5rem; margin-bottom: 0.2rem;">
            <div style="font-weight: 700; font-size: 1.1rem; color: #f8fafc;">24-Hour Diurnal Load Profile & Time-of-Day Tariff Bands</div>
            <div style="font-size: 0.8rem; color: #94a3b8;">
                <span style="color: #10b981;">■ Off-Peak (₹4)</span> &nbsp;
                <span style="color: #3b82f6;">■ Normal (₹6)</span> &nbsp;
                <span style="color: #ef4444;">■ Peak (₹8)</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    fig_hourly = go.Figure()
    # Tariff zone shaded backgrounds
    fig_hourly.add_vrect(x0=-0.5, x1=5.5, fillcolor="rgba(16, 185, 129, 0.08)", line_width=0)
    fig_hourly.add_vrect(x0=5.5, x1=17.5, fillcolor="rgba(59, 130, 246, 0.07)", line_width=0)
    fig_hourly.add_vrect(x0=17.5, x1=21.5, fillcolor="rgba(239, 68, 68, 0.12)", line_width=0)
    fig_hourly.add_vrect(x0=21.5, x1=23.5, fillcolor="rgba(16, 185, 129, 0.08)", line_width=0)

    # Gradient-filled area trace
    fig_hourly.add_trace(go.Scatter(
        x=hourly_df["hour"],
        y=hourly_df["energy_kwh"],
        mode="lines+markers",
        name="Household Energy (kWh)",
        line=dict(color="#38bdf8", width=3, shape="spline"),
        fill="tozeroy",
        fillcolor="rgba(56, 189, 248, 0.08)",
        marker=dict(size=6, color="#0284c7"),
        hovertemplate="<b>%{x}:00</b><br>Energy: <b>%{y:.2f} kWh</b><br>Rate: ₹" + hourly_df["hour"].apply(tariff_engine.get_rate_for_hour).astype(str) + "/kWh<extra></extra>"
    ))

    # Highlight peak hour
    peak_val = hourly_df[hourly_df["hour"] == peak_hour]["energy_kwh"].iloc[0]
    fig_hourly.add_trace(go.Scatter(
        x=[peak_hour],
        y=[peak_val],
        mode="markers+text",
        name="Peak Hour",
        marker=dict(size=14, color="#ef4444", symbol="diamond"),
        text=[f"Peak: {peak_val:.2f} kWh"],
        textposition="top center",
        hoverinfo="skip"
    ))

    apply_luxury_chart_theme(fig_hourly, height=350)
    fig_hourly.update_xaxes(title="Hour of Day (00:00 to 23:00)", dtick=2)
    fig_hourly.update_yaxes(title="Hourly Energy (kWh)")
    st.plotly_chart(fig_hourly, use_container_width=True)

    # Sub-visualizations: Appliance share & Monthly savings impact
    c_left, c_right = st.columns([1, 1])
    with c_left:
        st.markdown("<div style='font-weight: 700; font-size: 1rem; color: #f8fafc; margin-bottom: 0.5rem;'>Appliance Energy Consumption Distribution</div>", unsafe_allow_html=True)
        if len(events_df) > 0:
            dev_agg = events_df.groupby("device_type")["energy_kwh"].sum().reset_index()
            dev_agg["display"] = dev_agg["device_type"].str.replace("_", " ").str.title()
            fig_pie = px.pie(
                dev_agg,
                values="energy_kwh",
                names="display",
                hole=0.6,
                color_discrete_sequence=["#38bdf8", "#10b981", "#f59e0b"]
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            apply_luxury_chart_theme(fig_pie, height=300)
            st.plotly_chart(fig_pie, use_container_width=True)

    with c_right:
        st.markdown("<div style='font-weight: 700; font-size: 1rem; color: #f8fafc; margin-bottom: 0.5rem;'>Monthly Rupee Impact: Baseline vs. Optimized</div>", unsafe_allow_html=True)
        monthly_base_kwh = float(selected_hh["monthly_energy_kwh"])
        avg_rate_inr = daily_cost_inr / daily_energy if daily_energy > 0 else 6.0
        monthly_base_cost = monthly_base_kwh * avg_rate_inr
        monthly_opt_cost = max(0.0, monthly_base_cost - (potential_savings * 30.0))

        bar_df = pd.DataFrame({
            "Scenario": ["Current Baseline", "Cliff Guard Optimized"],
            "Monthly Cost (₹)": [round(monthly_base_cost, 2), round(monthly_opt_cost, 2)]
        })
        fig_bar = px.bar(
            bar_df,
            x="Scenario",
            y="Monthly Cost (₹)",
            text="Monthly Cost (₹)",
            color="Scenario",
            color_discrete_map={"Current Baseline": "#ef4444", "Cliff Guard Optimized": "#10b981"}
        )
        fig_bar.update_traces(texttemplate='₹%{y:,.0f}', textposition='inside')
        apply_luxury_chart_theme(fig_bar, height=300)
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------------
# SECTION 2: HOUSEHOLD & DEVICE ANALYTICS
# ---------------------------------------------------------
elif nav_choice == "2. Household & Device Analytics":
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">Appliance Analytics & Load Signatures</div>
            <div class="hero-sub">Inspect power curves, operating durations, and energy profiles across synthetic Kerala households.</div>
        </div>
    """, unsafe_allow_html=True)

    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        appliance_filter = st.selectbox(
            "Filter Appliance Type",
            ["All Appliances", "water_pump", "washing_machine", "water_heater"],
            format_func=lambda x: "All Appliances" if x == "All Appliances" else x.replace("_", " ").title()
        )
    with f_col2:
        all_cases = ["All Households"] + data_svc.get_household_cases()["case_id"].tolist()
        case_filter = st.selectbox("Filter Household ID", all_cases, index=all_cases.index(selected_case_id) if selected_case_id in all_cases else 0)
    with f_col3:
        st.write("")
        st.markdown("<span class='badge-pill badge-sky' style='margin-top: 15px;'>868 MULTI-DAY EVENTS</span>", unsafe_allow_html=True)

    events = data_svc.get_device_consumption(
        case_id=case_filter if case_filter != "All Households" else None,
        device_type=appliance_filter if appliance_filter != "All Appliances" else None
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Events Logged</div>
                <div class="kpi-value">{len(events)}</div>
            </div>
        """, unsafe_allow_html=True)
    with k2:
        tot_kwh = events["energy_kwh"].sum()
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Total Energy</div>
                <div class="kpi-value">{tot_kwh:.2f} <span style="font-size: 0.8rem; color: #94a3b8;">kWh</span></div>
            </div>
        """, unsafe_allow_html=True)
    with k3:
        avg_dur = events["duration_minutes"].mean() if len(events) > 0 else 0
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Avg. Duration</div>
                <div class="kpi-value">{avg_dur:.0f} <span style="font-size: 0.8rem; color: #94a3b8;">min</span></div>
            </div>
        """, unsafe_allow_html=True)
    with k4:
        avg_p = events["power_watts"].mean() if len(events) > 0 else 0
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Avg. Power</div>
                <div class="kpi-value">{format_power(avg_p)}</div>
            </div>
        """, unsafe_allow_html=True)
    with k5:
        max_p = events["power_watts"].max() if len(events) > 0 else 0
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Peak Power</div>
                <div class="kpi-value" style="color: #f59e0b;">{format_power(max_p)}</div>
            </div>
        """, unsafe_allow_html=True)

    st.write("")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("<div style='font-weight: 700; font-size: 1rem; color: #f8fafc; margin-bottom: 0.4rem;'>Active Power (Watts) vs. Operating Duration (Minutes)</div>", unsafe_allow_html=True)
        fig_scatter = px.scatter(
            events,
            x="duration_minutes",
            y="power_watts",
            color="device_type",
            size="energy_kwh",
            hover_data=["case_id", "timestamp"],
            color_discrete_map={"washing_machine": "#38bdf8", "water_pump": "#10b981", "water_heater": "#f59e0b"}
        )
        apply_luxury_chart_theme(fig_scatter, height=320)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_b:
        st.markdown("<div style='font-weight: 700; font-size: 1rem; color: #f8fafc; margin-bottom: 0.4rem;'>Appliance Start Time Distribution Across 24h</div>", unsafe_allow_html=True)
        fig_hist = px.histogram(
            events,
            x="preferred_start_hour_simulated",
            color="device_type",
            nbins=24,
            color_discrete_map={"washing_machine": "#38bdf8", "water_pump": "#10b981", "water_heater": "#f59e0b"}
        )
        apply_luxury_chart_theme(fig_hist, height=320)
        fig_hist.update_xaxes(title="Start Hour (0 to 23)")
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("<div style='font-weight: 700; font-size: 1rem; color: #f8fafc; margin-top: 1rem; margin-bottom: 0.5rem;'>Appliance Event Registry Table</div>", unsafe_allow_html=True)
    display_cols = ["timestamp", "case_id", "device_type", "power_watts", "duration_minutes", "energy_kwh", "flexible_load"]
    st.dataframe(events[display_cols].head(50), use_container_width=True)

# ---------------------------------------------------------
# SECTION 3: AI ANOMALY DETECTION
# ---------------------------------------------------------
elif nav_choice == "3. AI Anomaly Detection":
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">AI Anomaly Detection Hub</div>
            <div class="hero-sub">Unsupervised Isolation Forest algorithm isolating power surges, pump dry runs, and runtime anomalies in 7D physical feature space.</div>
        </div>
    """, unsafe_allow_html=True)

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])
    with ctrl_col1:
        new_contamination = st.slider(
            "Isolation Forest Contamination Rate (Expected Anomaly Threshold)",
            min_value=0.02,
            max_value=0.15,
            value=st.session_state.anomaly_contamination,
            step=0.01,
            help="Defines model sensitivity. Default 0.06 (6% anomaly proportion)."
        )
        if new_contamination != st.session_state.anomaly_contamination:
            st.session_state.anomaly_contamination = new_contamination
            anomaly_svc = AnomalyService(contamination=new_contamination)

    with ctrl_col2:
        device_filter_anom = st.selectbox(
            "Device Scope",
            ["All Devices", "water_pump", "washing_machine", "water_heater"],
            format_func=lambda x: "All Devices" if x == "All Devices" else x.replace("_", " ").title()
        )

    with ctrl_col3:
        st.write("")
        st.write("")
        retrain_clicked = st.button("⚡ Recompute ML Predictions", use_container_width=True, type="primary")

    raw_events = data_svc.get_device_consumption(
        device_type=device_filter_anom if device_filter_anom != "All Devices" else None
    )
    annotated = anomaly_svc.run_detection(raw_events, force_retrain=retrain_clicked)
    anom_events = annotated[annotated["ml_predicted_anomaly"] == 1]
    anom_pct = len(anom_events) / len(annotated) * 100.0 if len(annotated) > 0 else 0

    tab_detections, tab_inspection, tab_eval = st.tabs([
        "🚨 Outlier Events Visualizer",
        "🛠️ Deep Diagnostic Inspector",
        "📊 ML Validation & Confusion Matrix"
    ])

    with tab_detections:
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Events Analyzed</div>
                    <div class="kpi-value">{len(annotated)}</div>
                    <div class="kpi-sub">Total 7-day appliance cycles</div>
                </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Flagged Anomalies</div>
                    <div class="kpi-value" style="color: #fb7185;">{len(anom_events)} <span style="font-size: 0.8rem; color: #94a3b8;">({anom_pct:.1f}%)</span></div>
                    <div class="kpi-sub">Multivariate physical outliers</div>
                </div>
            """, unsafe_allow_html=True)
        with k3:
            crit_count = (anom_events["severity"] == "Critical").sum() if len(anom_events) > 0 else 0
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Critical Severity</div>
                    <div class="kpi-value" style="color: #ef4444;">{crit_count}</div>
                    <div class="kpi-sub">Score ≥ 0.85 (immediate danger)</div>
                </div>
            """, unsafe_allow_html=True)
        with k4:
            avg_score = anom_events["anomaly_score"].mean() if len(anom_events) > 0 else 0.0
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Mean Anomaly Score</div>
                    <div class="kpi-value" style="color: #f59e0b;">{avg_score:.2f}</div>
                    <div class="kpi-sub">Normalized decision function</div>
                </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.markdown("<div style='font-weight: 700; font-size: 1rem; color: #f8fafc; margin-bottom: 0.4rem;'>Isolation Forest Feature Space: Active Power vs. Runtime</div>", unsafe_allow_html=True)

        annotated["Detection_Label"] = annotated["ml_predicted_anomaly"].map({1: "⚠️ Outlier Anomaly", 0: "✓ Normal Baseline"})
        fig_anom = px.scatter(
            annotated,
            x="duration_minutes",
            y="power_watts",
            color="Detection_Label",
            symbol="device_type",
            size="anomaly_score",
            hover_data=["case_id", "timestamp", "energy_kwh", "evidence_summary"],
            color_discrete_map={"✓ Normal Baseline": "#38bdf8", "⚠️ Outlier Anomaly": "#ef4444"}
        )
        apply_luxury_chart_theme(fig_anom, height=360)
        st.plotly_chart(fig_anom, use_container_width=True)

        st.markdown("<div style='font-weight: 700; font-size: 1rem; color: #f8fafc; margin-top: 1rem; margin-bottom: 0.4rem;'>Flagged Anomaly Registry</div>", unsafe_allow_html=True)
        display_anom = anom_events[["timestamp", "case_id", "device_type", "power_watts", "duration_minutes", "energy_kwh", "anomaly_score", "severity", "evidence_summary"]]
        st.dataframe(display_anom.sort_values(by="anomaly_score", ascending=False), use_container_width=True)

    with tab_inspection:
        st.markdown("<div style='font-weight: 700; font-size: 1.1rem; color: #f8fafc; margin-bottom: 0.5rem;'>Select Flagged Event for Root-Cause Diagnostic Analysis</div>", unsafe_allow_html=True)
        if len(anom_events) > 0:
            event_keys = [
                f"{row['timestamp']} | {row['case_id']} | {row['device_type']} | {row['power_watts']:.0f}W | {row['duration_minutes']}m | Score {row['anomaly_score']:.2f}"
                for _, row in anom_events.head(25).iterrows()
            ]
            selected_anom_key = st.selectbox("Select Anomaly Cycle", event_keys)
            sel_ts = selected_anom_key.split(" | ")[0]
            sel_event = anom_events[anom_events["timestamp"] == sel_ts].iloc[0]

            ic1, ic2 = st.columns([1, 1])
            with ic1:
                st.markdown(f"""
                <div class="glass-card" style="border-left: 4px solid #ef4444;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                        <span style="font-weight: 800; font-size: 1.15rem; color: #f8fafc;">{sel_event['device_type'].replace('_', ' ').title()}</span>
                        <span class="badge-pill badge-rose">SEVERITY: {sel_event['severity'].upper()}</span>
                    </div>
                    <div style="font-size: 0.88rem; line-height: 1.6; color: #cbd5e1;">
                        <div><b>Household:</b> <span class="mono-val">{sel_event['case_id']}</span></div>
                        <div><b>Event Timestamp:</b> <span class="mono-val">{sel_event['timestamp']}</span></div>
                        <div><b>Active Power Draw:</b> <span class="mono-val" style="color: #ef4444; font-weight: bold;">{format_power(sel_event['power_watts'])}</span></div>
                        <div><b>Operating Duration:</b> <span class="mono-val">{format_minutes_to_duration(sel_event['duration_minutes'])}</span></div>
                        <div><b>Energy Drawn:</b> <span class="mono-val">{format_kwh(sel_event['energy_kwh'])}</span></div>
                        <div><b>ML Anomaly Score:</b> <span class="mono-val" style="color: #f59e0b; font-weight: bold;">{sel_event['anomaly_score']:.3f} / 1.000</span></div>
                    </div>
                    <div style="margin-top: 1rem; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 8px; font-size: 0.85rem; border: 1px solid rgba(255,255,255,0.06);">
                        <b style="color: #38bdf8;">Root-Cause Physical Evidence:</b><br>{sel_event['evidence_summary']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with ic2:
                checklist = anomaly_svc.get_inspection_checklist(sel_event["device_type"], sel_event["evidence_summary"])
                st.markdown(f"""
                <div class="glass-card" style="border-left: 4px solid #38bdf8;">
                    <div style="font-weight: 800; font-size: 1.1rem; color: #f8fafc; margin-bottom: 0.4rem;">Technician Safety Protocol Checklist</div>
                    <div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 0.8rem;">Perform on-site safety and electrical checks to mitigate motor burn-out or fire hazard:</div>
                </div>
                """, unsafe_allow_html=True)
                for step in checklist:
                    st.checkbox(step, key=f"chk_v2_{sel_ts}_{step[:10]}")

            st.caption("Notice: Unsupervised Isolation Forest flags statistical physical anomalies. Physical technician inspection is recommended.")
        else:
            st.success("No anomalies detected under current sensitivity setting.")

    with tab_eval:
        st.markdown("<div style='font-weight: 700; font-size: 1.1rem; color: #f8fafc; margin-bottom: 0.3rem;'>Quantitative Evaluation vs. Synthetic Ground Truth</div>", unsafe_allow_html=True)
        st.info("Anti-Leakage Architecture: Ground truth synthetic labels are strictly quarantined during training and only evaluated post-inference.")

        eval_metrics = anomaly_svc.get_model_evaluation(raw_events)
        e1, e2, e3, e4 = st.columns(4)
        with e1:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Precision</div>
                    <div class="kpi-value" style="color: #38bdf8;">{eval_metrics['precision']:.3f}</div>
                    <div class="kpi-sub">TP / (TP + FP)</div>
                </div>
            """, unsafe_allow_html=True)
        with e2:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Recall</div>
                    <div class="kpi-value" style="color: #34d399;">{eval_metrics['recall']:.3f}</div>
                    <div class="kpi-sub">TP / (TP + FN)</div>
                </div>
            """, unsafe_allow_html=True)
        with e3:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">F1-Score</div>
                    <div class="kpi-value" style="color: #f59e0b;">{eval_metrics['f1_score']:.3f}</div>
                    <div class="kpi-sub">Harmonic mean P & R</div>
                </div>
            """, unsafe_allow_html=True)
        with e4:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Accuracy</div>
                    <div class="kpi-value">{eval_metrics['accuracy']:.3f}</div>
                    <div class="kpi-sub">Overall classification</div>
                </div>
            """, unsafe_allow_html=True)

        st.write("")
        cm_df = pd.DataFrame({
            "Predicted Normal (0)": [eval_metrics["true_negatives"], eval_metrics["false_negatives"]],
            "Predicted Anomaly (1)": [eval_metrics["false_positives"], eval_metrics["true_positives"]]
        }, index=["Actual Normal (0)", "Actual Anomaly (1)"])
        st.markdown("<div style='font-weight: 700; font-size: 0.95rem; color: #f8fafc; margin-bottom: 0.3rem;'>Confusion Matrix:</div>", unsafe_allow_html=True)
        st.table(cm_df)

# ---------------------------------------------------------
# SECTION 4: SMART SCHEDULING
# ---------------------------------------------------------
elif nav_choice == "4. Smart Scheduling":
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">Constraint-Aware Smart Scheduler</div>
            <div class="hero-sub">Automatically shifts flexible heavy appliances away from costly evening peak tariffs to off-peak periods while respecting deadlines and user preferences.</div>
        </div>
    """, unsafe_allow_html=True)

    s_col1, s_col2 = st.columns([1, 1])
    with s_col1:
        st.markdown("<div style='font-weight: 700; font-size: 1.05rem; color: #f8fafc; margin-bottom: 0.5rem;'>1. Appliance Operating Specifications</div>", unsafe_allow_html=True)
        device_choice = st.selectbox(
            "Select Appliance",
            ["water_pump", "washing_machine", "water_heater"],
            format_func=lambda x: x.replace("_", " ").title()
        )
        profile = data_svc.get_profile_for_device(device_choice) or {}
        default_power = float(profile.get("nominal_power_watts", 750.0))
        default_dur = int(profile.get("typical_duration_minutes", 45))

        sched_power = st.number_input("Nominal Power Draw (Watts)", min_value=100.0, max_value=5000.0, value=default_power, step=50.0)
        sched_dur = st.number_input("Required Cycle Duration (Minutes)", min_value=10, max_value=360, value=default_dur, step=5)
        is_flexible = st.checkbox("Flexible Load (Permit Automated Schedule Shifting)", value=True, help="Disable if this task must run strictly at preferred time.")

    with s_col2:
        st.markdown("<div style='font-weight: 700; font-size: 1.05rem; color: #f8fafc; margin-bottom: 0.5rem;'>2. Permissible Time Constraints & User Preference</div>", unsafe_allow_html=True)
        earliest_start = st.slider("Earliest Permissible Start Hour", min_value=0.0, max_value=23.0, value=5.0, step=0.5, format="%.1fh")
        latest_finish = st.slider("Latest Permissible Finish Deadline", min_value=1.0, max_value=24.0, value=22.0, step=0.5, format="%.1fh")
        preferred_start = st.slider("Preferred Start Time (Baseline Request)", min_value=0.0, max_value=23.5, value=19.0, step=0.5, format="%.1fh")

        st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 10px; font-size: 0.85rem; color: #94a3b8; margin-top: 10px;">
                Permitted Window: <b style="color: #38bdf8;">{decimal_hour_to_hhmm(earliest_start)}</b> to <b style="color: #38bdf8;">{decimal_hour_to_hhmm(latest_finish)}</b> &nbsp;|&nbsp; 
                Baseline Request: <b style="color: #ef4444;">{decimal_hour_to_hhmm(preferred_start)}</b>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)

    sched_result = scheduler.schedule_appliance(
        device_type=device_choice,
        power_watts=sched_power,
        duration_minutes=sched_dur,
        earliest_start_hour=earliest_start,
        latest_finish_hour=latest_finish,
        preferred_start_hour=preferred_start,
        flexible_load=is_flexible
    )

    if not sched_result["is_feasible"]:
        st.error(f"❌ {sched_result['explanation']}")
    else:
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Recommended Start</div>
                    <div class="kpi-value" style="color: #38bdf8;">{decimal_hour_to_time_str(sched_result['recommended_start_hour'])}</div>
                    <div class="kpi-sub">Constraint-compliant slot</div>
                </div>
            """, unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Baseline Cost</div>
                    <div class="kpi-value" style="color: #ef4444;">{format_inr(sched_result['baseline_cost_inr'])}</div>
                    <div class="kpi-sub">At {decimal_hour_to_hhmm(preferred_start)}</div>
                </div>
            """, unsafe_allow_html=True)
        with r3:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Optimized Cost</div>
                    <div class="kpi-value" style="color: #10b981;">{format_inr(sched_result['recommended_cost_inr'])}</div>
                    <div class="kpi-sub">At {decimal_hour_to_hhmm(sched_result['recommended_start_hour'])}</div>
                </div>
            """, unsafe_allow_html=True)
        with r4:
            sav = sched_result["savings_inr"]
            pct = sched_result["savings_percentage"]
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Cost Savings</div>
                    <div class="kpi-value" style="color: #34d399;">{format_inr(sav)}</div>
                    <div class="kpi-sub">{pct}% reduction per cycle</div>
                </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.markdown(f"""
            <div class="glass-card" style="border-left: 4px solid #10b981;">
                <b style="color: #10b981;">Optimizer Decision Rationale:</b> {sched_result['explanation']}
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='font-weight: 700; font-size: 1rem; color: #f8fafc; margin-bottom: 0.4rem;'>24-Hour Visual Schedule Gantt Timeline</div>", unsafe_allow_html=True)
        gantt_df = pd.DataFrame([
            {
                "Schedule": "Baseline Schedule",
                "Start": sched_result["baseline_start_hour"],
                "Finish": sched_result["baseline_finish_hour"],
                "Duration": sched_dur / 60.0,
                "Cost": f"₹{sched_result['baseline_cost_inr']:.2f}"
            },
            {
                "Schedule": "Cliff Guard Recommended",
                "Start": sched_result["recommended_start_hour"],
                "Finish": sched_result["recommended_finish_hour"],
                "Duration": sched_dur / 60.0,
                "Cost": f"₹{sched_result['recommended_cost_inr']:.2f}"
            }
        ])

        fig_gantt = px.bar(
            gantt_df,
            x="Duration",
            y="Schedule",
            base="Start",
            orientation="h",
            color="Schedule",
            color_discrete_map={"Baseline Schedule": "#ef4444", "Cliff Guard Recommended": "#10b981"},
            text="Cost"
        )
        apply_luxury_chart_theme(fig_gantt, height=220)
        fig_gantt.update_xaxes(title="Hour of Day (00:00 to 24:00)", range=[0, 24], dtick=2)
        st.plotly_chart(fig_gantt, use_container_width=True)

        # Interactive Simulated Automation Approval Card
        st.markdown("<div style='font-weight: 700; font-size: 1.05rem; color: #f8fafc; margin-top: 1rem; margin-bottom: 0.4rem;'>Automated Schedule Approval & Actuation Simulation</div>", unsafe_allow_html=True)
        
        current_app = f"{device_choice.replace('_', ' ').title()} @ {decimal_hour_to_hhmm(sched_result['recommended_start_hour'])}"
        is_already_approved = any(a["key"] == current_app for a in st.session_state.approved_schedules)

        if is_already_approved:
            st.markdown(f"""
                <div class="approved-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="badge-pill badge-emerald">✓ STATUS: APPROVED (SIMULATED)</span>
                        <span style="font-size: 0.8rem; color: #94a3b8;">Active Session Dispatch</span>
                    </div>
                    <div style="margin-top: 0.7rem; font-size: 0.95rem; color: #f8fafc;">
                        Appliance: <b>{device_choice.replace('_', ' ').title()}</b> shifted to <b>{decimal_hour_to_hhmm(sched_result['recommended_start_hour'])}</b> &nbsp;|&nbsp; 
                        Direct Savings: <b style="color: #34d399;">{format_inr(sched_result['savings_inr'])} / cycle</b>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Cancel & Revoke Approval", type="secondary"):
                st.session_state.approved_schedules = [a for a in st.session_state.approved_schedules if a["key"] != current_app]
                st.rerun()
        else:
            if st.button("✅ Approve Schedule (Simulated Automation)", type="primary"):
                st.session_state.approved_schedules.append({
                    "key": current_app,
                    "device": device_choice,
                    "time": decimal_hour_to_hhmm(sched_result["recommended_start_hour"]),
                    "savings": sched_result["savings_inr"],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                st.rerun()

        st.caption("""
            <b>Safety Notice:</b> Cliff Guard is a software demonstration prototype. Schedule approval is simulated and does not transmit commands to physical smart plugs or relays.
        """)

# ---------------------------------------------------------
# SECTION 5: RECOMMENDATIONS & INSIGHTS
# ---------------------------------------------------------
elif nav_choice == "5. Recommendations & Insights":
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">Evidence-Based Energy Intelligence</div>
            <div class="hero-sub">Ranked, deterministic recommendations synthesized from household diurnal curves, ToD tariff structures, and machine learning anomaly alerts.</div>
        </div>
    """, unsafe_allow_html=True)

    hourly_df = data_svc.get_hourly_data_for_household(selected_case_id)
    events_df = data_svc.get_device_consumption(case_id=selected_case_id)
    annotated_events = anomaly_svc.run_detection(events_df)

    insights = insights_svc.generate_household_insights(
        household_profile=selected_hh,
        hourly_df=hourly_df,
        events_df=annotated_events
    )

    sev_border = {"High": "#ef4444", "Medium": "#f59e0b", "Info": "#38bdf8"}
    sev_badge = {"High": "badge-rose", "Medium": "badge-amber", "Info": "badge-sky"}

    for ins in insights:
        col = sev_border.get(ins["severity"], "#38bdf8")
        bdg = sev_badge.get(ins["severity"], "badge-sky")

        st.markdown(f"""
            <div class="glass-card" style="border-left: 4px solid {col};">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem;">
                    <div style="font-size: 1.15rem; font-weight: 700; color: #f8fafc;">{ins['title']}</div>
                    <span class="badge-pill {bdg}">{ins['severity'].upper()} PRIORITY</span>
                </div>
                <div style="font-size: 0.78rem; color: #94a3b8; margin-bottom: 0.75rem;">Category: <b style="color: #cbd5e1;">{ins['category']}</b></div>
                <div style="font-size: 0.88rem; line-height: 1.6; color: #cbd5e1;">
                    <p><b>📊 Measured Evidence:</b> {ins['evidence']}</p>
                    <p><b>🔍 Context & Explanation:</b> {ins['explanation']}</p>
                    <p><b>⚡ Recommended Action:</b> {ins['suggested_action']}</p>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.8rem; padding-top: 0.8rem; border-top: 1px solid rgba(255,255,255,0.06);">
                    <span style="color: #38bdf8; font-weight: 700; font-family: 'JetBrains Mono', monospace;">{ins.get('impact_label', '')}</span>
                    <span style="font-size: 0.75rem; color: #64748b;">Boundary: {ins['limitations']}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# SECTION 6: AI ARCHITECTURE & EXPLAINABILITY DEEP-DIVE
# ---------------------------------------------------------
elif nav_choice == "6. 🧠 AI Architecture & Explainability":
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">AI Engine Architecture & Methodology</div>
            <div class="hero-sub">How Cliff Guard applies Machine Learning to household energy: Model internals, mathematical formulations, and strict anti-leakage guarantees.</div>
        </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="glass-card" style="border-left: 4px solid #38bdf8;">
            <div style="font-size: 1.15rem; font-weight: 800; color: #38bdf8; margin-bottom: 0.5rem;">1. Unsupervised Isolation Forest (iForest)</div>
            <p style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.6;">
                Instead of modeling normal points, Isolation Forest explicitly isolates anomalies. Because physical faults (e.g. bearing drag, stuck float switch, heating element scaling) exhibit unusual combinations of power, duration, and energy rate, they require fewer random partition splits in decision trees.
            </p>
            <div style="background: rgba(0,0,0,0.3); border-radius: 8px; padding: 10px; font-size: 0.8rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace;">
                Anomaly Score: s(x, n) = 2^(- E(h(x)) / c(n))<br>
                Where E(h(x)) is mean path length across 150 trees.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="glass-card" style="border-left: 4px solid #10b981;">
            <div style="font-size: 1.15rem; font-weight: 800; color: #10b981; margin-bottom: 0.5rem;">2. 7D Defensible Physical Feature Vector</div>
            <p style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.6;">
                The model operates across seven strictly physical, domain-grounded dimensions:
            </p>
            <ul style="font-size: 0.84rem; color: #cbd5e1; margin-left: -15px;">
                <li><code>power_watts</code>: Instantaneous continuous electrical draw.</li>
                <li><code>duration_minutes</code>: Cycle run length.</li>
                <li><code>energy_kwh</code>: Cumulative cycle energy.</li>
                <li><code>power_deviation_ratio</code>: Fractional delta vs. nominal motor rating.</li>
                <li><code>kwh_per_minute</code>: Physical energy consumption velocity.</li>
                <li>One-hot indicators: Appliance class identification.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div class="glass-card" style="border-left: 4px solid #f59e0b;">
            <div style="font-size: 1.15rem; font-weight: 800; color: #f59e0b; margin-bottom: 0.5rem;">3. Strict Anti-Leakage & Explainable AI (XAI) Pipeline</div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; font-size: 0.88rem; color: #cbd5e1; line-height: 1.6;">
                <div>
                    <b style="color: #ffffff;">Zero Data Leakage Guarantee:</b><br>
                    The synthetic evaluation label (<code>is_synthetic_anomaly_label</code>) and scenario descriptors are strictly segregated from the training pipeline. The model makes independent predictions based purely on unsupervised feature distributions.
                </div>
                <div>
                    <b style="color: #ffffff;">From Anomaly Score to Root Cause:</b><br>
                    Raw scores are translated deterministically into domain-specific mechanical failure modes (e.g. dry run cavitation, pipe obstruction, thermostat calibration drift) with actionable technician safety protocols.
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# SECTION 7: TARIFF CONFIGURATION
# ---------------------------------------------------------
elif nav_choice == "7. Tariff Configuration":
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">Time-of-Day (ToD) Tariff Configuration</div>
            <div class="hero-sub">Edit illustrative Kerala electricity rates (₹/kWh) and time windows. All calculations across the platform adapt instantly.</div>
        </div>
    """, unsafe_allow_html=True)

    with st.form("tariff_edit_form_v2"):
        st.markdown("<div style='font-weight: 700; font-size: 1.05rem; color: #f8fafc; margin-bottom: 0.5rem;'>Edit Illustrative Time-of-Day Rates (₹/kWh)</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            off_peak_rate = st.number_input("Off-Peak (Night: 10 PM – 6 AM)", min_value=1.0, max_value=30.0, value=float(tariff_engine.get_rate_for_hour(23)), step=0.5)
        with c2:
            normal_rate = st.number_input("Normal (Daytime: 6 AM – 6 PM)", min_value=1.0, max_value=30.0, value=float(tariff_engine.get_rate_for_hour(10)), step=0.5)
        with c3:
            peak_rate = st.number_input("Peak (Evening: 6 PM – 10 PM)", min_value=1.0, max_value=30.0, value=float(tariff_engine.get_rate_for_hour(19)), step=0.5)

        apply_btn = st.form_submit_button("💾 Save & Recalculate Tariffs", type="primary")
        if apply_btn:
            tariff_engine.update_rates({
                "off_peak": off_peak_rate,
                "normal": normal_rate,
                "peak": peak_rate
            })
            st.session_state.tariff_engine = tariff_engine
            st.success("Tariff rates successfully updated across all modules.")
            st.rerun()

    st.write("")
    st.markdown("<div style='font-weight: 700; font-size: 1rem; color: #f8fafc; margin-bottom: 0.4rem;'>24-Hour Active Rate Schedule</div>", unsafe_allow_html=True)
    tariff_table = []
    for h in range(24):
        tariff_table.append({
            "Hour Window": f"{h:02d}:00 – {(h+1)%24:02d}:00",
            "Tariff Period": tariff_engine.get_period_name_for_hour(h),
            "Rate (₹/kWh)": tariff_engine.get_rate_for_hour(h)
        })
    st.dataframe(pd.DataFrame(tariff_table), use_container_width=True)

# ---------------------------------------------------------
# SECTION 8: ABOUT & ASSUMPTIONS
# ---------------------------------------------------------
elif nav_choice == "8. About & Assumptions":
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">About Cliff Guard & Hackathon Documentation</div>
            <div class="hero-sub">Smarter Energy. Lower Bills. Safer Homes.</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <div style="font-weight: 800; font-size: 1.2rem; color: #38bdf8; margin-bottom: 0.8rem;">Project Pillars & Mission</div>
        <p style="font-size: 0.9rem; color: #cbd5e1; line-height: 1.7;">
            <b>Cliff Guard</b> addresses the dual challenges of peak electricity grid stress and high consumer bills in Indian households. By combining unsupervised machine learning anomaly detection with constraint-aware load scheduling, it enables households to operate flexible appliances at the lowest feasible tariff while preempting electrical hazards.
        </p>
        <div style="margin-top: 1rem; font-size: 0.85rem; color: #94a3b8;">
            <b>Notice: Research Prototype & Synthetic Data Statement</b><br>
            All 50 household profiles, 1,200 hourly measurements, and 868 appliance cycles are <b>SYNTHETIC</b> data generated via <code>scripts/generate_data.py</code> using fixed random seed 42. Illustrative tariffs (₹4, ₹6, ₹8/kWh) do not represent official KSEB tariffs.
        </div>
    </div>
    """, unsafe_allow_html=True)
