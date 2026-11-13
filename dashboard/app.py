"""Minimal Streamlit front end for the actual live SUMO-GUI simulation."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))
from live_simulation import LiveSimulation
from data import load_evaluation_bundle
from evaluation_views import render_section

st.set_page_config(page_title="Urban Mobility Research Dashboard", page_icon="🚦", layout="wide")
st.markdown(
    """
    <style>
      :root, .stApp {
        color-scheme: dark;
        --primary-color: #14b8a6;
        --background-color: #0b1220;
        --secondary-background-color: #111827;
        --text-color: #e2e8f0;
      }
      .stApp, [data-testid="stAppViewContainer"] {
        background-color: #0b1220;
        color: #e2e8f0;
      }
      .block-container {padding-top: .8rem; padding-bottom: .6rem; max-width: 1800px;}
      h3 {font-size: 1.05rem !important; margin: 0 !important;}
      h1, h2, h3, h4, h5, h6 {color: #f8fafc !important;}
      p, label, li {color: #e2e8f0;}
      section[data-testid="stSidebar"] {
        min-width: 245px;
        background-color: #111827;
        border-right: 1px solid #334155;
      }
      section[data-testid="stSidebar"] > div:first-child {background-color: #111827;}

      /* Explicitly pair dark cards with light text, independent of the active theme. */
      div[data-testid="stMetric"] {
        box-sizing: border-box;
        min-height: 88px;
        padding: .75rem .9rem;
        background: #1e293b !important;
        border: 1px solid #475569;
        border-radius: .75rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, .18);
      }
      div[data-testid="stMetric"] [data-testid="stMetricLabel"],
      div[data-testid="stMetric"] [data-testid="stMetricLabel"] p,
      div[data-testid="stMetric"] [data-testid="stMetricLabel"] label {
        color: #cbd5e1 !important;
      }
      div[data-testid="stMetric"] [data-testid="stMetricValue"],
      div[data-testid="stMetric"] [data-testid="stMetricValue"] div {
        color: #f8fafc !important;
      }
      div[data-testid="stMetric"] [data-testid="stMetricDelta"] {font-weight: 600;}
      div[data-testid="stMetric"] [data-testid="stMetricHelp"] {color: #cbd5e1 !important;}

      div[data-testid="stCaptionContainer"],
      div[data-testid="stCaptionContainer"] p {color: #aebccc !important;}
      div[data-baseweb="select"] > div {
        color: #f8fafc;
        background-color: #111827;
        border-color: #475569;
      }
      div[role="listbox"], ul[role="listbox"] {background-color: #111827;}
      li[role="option"] {color: #e2e8f0;}
      li[role="option"][aria-selected="true"] {background-color: #334155;}
      input, textarea {color: #f8fafc !important; background-color: #111827 !important;}
      .stButton > button {
        color: #f8fafc !important;
        background-color: #1e293b !important;
        border: 1px solid #475569 !important;
        border-radius: .6rem;
      }
      .stButton > button:hover {background-color: #334155 !important; border-color: #64748b !important;}
      .stButton > button[kind="primary"] {
        color: #ffffff !important;
        background-color: #0f766e !important;
        border-color: #14b8a6 !important;
      }
      .stButton > button:disabled {color: #94a3b8 !important; opacity: .75;}
      div[data-testid="stDataFrame"], div[data-testid="stTable"] {
        border: 1px solid #334155;
        border-radius: .65rem;
        overflow: hidden;
      }
      div[data-testid="stTable"] table {background: #111827; color: #e2e8f0;}
      div[data-testid="stImage"] img {border-radius: .65rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

PAGES = [
    "Project Overview",
    "Traffic Prediction (ST-GNN)",
    "Cost Prediction (S1)",
    "QUBO Optimization (S2)",
    "Solver Comparison (S3)",
    "Traffic Performance",
    "Limitations",
    "Live SUMO",
]
evaluation_bundle = load_evaluation_bundle(ROOT / "results")
with st.sidebar:
    st.markdown("## HSR Mobility Lab")
    selected_page = st.radio("Dashboard sections", PAGES, key="dashboard_page")
    st.caption("Research prototype · 4 junctions · SUMO / ST-GNN / QUBO")

if selected_page != "Live SUMO":
    render_section(selected_page, evaluation_bundle)
    st.stop()

st.title("Live SUMO Simulation")
st.caption("Live TraCI monitoring with an optional per-junction adaptive controller. Automatic control uses the existing ST-GNN queue forecasts when available and otherwise falls back to observed traffic pressure; the research QUBO and solver modules remain separate.")


def stop_current_simulation() -> None:
    simulation = st.session_state.pop("live_simulation", None)
    if simulation is not None:
        try:
            simulation.close()
        finally:
            st.session_state.is_playing = False


simulation = st.session_state.get("live_simulation")
status = "STOPPED" if simulation is None else ("LIVE" if st.session_state.get("is_playing", True) else "PAUSED")
head, state = st.columns([8, 2], vertical_alignment="center")
with head:
    st.markdown("### AI-Driven Quantum Optimization for Dynamic Urban Mobility Management")
with state:
    ai_status = simulation.auto_control_enabled if simulation is not None else st.session_state.get("auto_control_enabled", True)
    st.markdown(f"**SUMO · {status} · AI control {'ON' if ai_status else 'OFF'}**")

view_col, controls_col = st.columns([7, 3], gap="medium", vertical_alignment="top")

with controls_col:
    st.markdown("#### Simulation controls")
    scenario_options = ["low", "medium", "high", "peak", "incident"]
    scenario_names = {
        "low": "Low demand",
        "medium": "Medium demand",
        "high": "High demand",
        "peak": "Peak demand",
        "incident": "Incident",
    }
    scenario = st.selectbox(
        "Scenario",
        scenario_options,
        index=scenario_options.index("high"),
        format_func=lambda value: scenario_names[value],
        key="scenario_choice",
    )
    seed = st.selectbox("Route seed", list(range(5)), index=0, key="seed_choice")

    auto_control_enabled = st.toggle(
        "Automatic AI Control",
        value=True,
        key="auto_control_enabled",
        help="The controller observes live lane demand, uses ST-GNN queue forecasts when available, and selects only SUMO-defined phases after clearance.",
    )

    btn_start, btn_step = st.columns(2)
    with btn_start:
        start_clicked = st.button("Start / restart", type="primary", use_container_width=True)
    with btn_step:
        step_clicked = st.button("Step 1 second", use_container_width=True, disabled=simulation is None)

    if start_clicked:
        stop_current_simulation()
        try:
            new_simulation = LiveSimulation.start(scenario, int(seed), auto_control=auto_control_enabled)
            # A real SUMO warm-up makes the first embedded GUI frame visibly populated.
            new_simulation.advance(60)
            st.session_state.live_simulation = new_simulation
            st.session_state.is_playing = True
            st.session_state.live_error = None
        except Exception as exc:
            st.session_state.live_error = f"Could not start SUMO-GUI: {exc}"

    simulation = st.session_state.get("live_simulation")
    if simulation is not None:
        simulation.set_auto_control(auto_control_enabled)
        play_pause = "Pause" if st.session_state.get("is_playing", True) else "Resume"
        if st.button(play_pause, use_container_width=True):
            st.session_state.is_playing = not st.session_state.get("is_playing", True)
        if step_clicked:
            st.session_state.is_playing = False
            try:
                simulation.advance(1)
            except Exception as exc:
                st.session_state.live_error = str(exc)
                stop_current_simulation()
                st.rerun()
        if simulation.controller_error:
            st.warning(f"Automatic controller unavailable; SUMO native timing remains active. {simulation.controller_error}")
        elif simulation.controller and simulation.controller.error:
            st.warning(simulation.controller.status)
        st.caption(f"Controller: {simulation.controller_status}")
        st.caption(f"Prediction: {simulation.model_status}")
        if st.button("Stop", use_container_width=True):
            stop_current_simulation()
            st.rerun()
        if simulation.last_signal_error:
            st.warning(simulation.last_signal_error)
        elif any(simulation.target_plans[j] != simulation.actual_plans[j] for j in ("J0", "J1", "J2", "J3")):
            st.caption("A requested plan is waiting for a safe signal-cycle boundary.")

    if st.session_state.get("live_error"):
        st.error(st.session_state.live_error)

    simulation = st.session_state.get("live_simulation")
    if simulation is not None:
        st.caption(f"Active: {scenario_names[simulation.scenario]} · seed {simulation.seed}")
    else:
        st.caption("Choose a scenario and seed, then start. The live view uses the same SUMO-GUI/TraCI session as the controls.")

with view_col:
    if simulation is None:
        st.info("The live SUMO-GUI view will appear here when the simulation starts.")
    else:
        @st.fragment(run_every="1s")
        def show_live_sumo() -> None:
            current = st.session_state.get("live_simulation")
            if current is None:
                st.info("SUMO is stopped. Start a simulation to display the live view.")
                return
            if st.session_state.get("is_playing", True) and not current.finished:
                try:
                    current.advance(1)
                except Exception as exc:
                    st.session_state.live_error = str(exc)
                    try:
                        current.close()
                    finally:
                        st.session_state.pop("live_simulation", None)
                        st.session_state.is_playing = False
                    st.error(f"SUMO-GUI stopped: {exc}")
                    return
            if current.finished:
                st.session_state.is_playing = False
            junction_rows = current.signal_status
            metric_cols = st.columns(6)
            metric_cols[0].metric("Time", f"{current.time_s:.0f} s")
            metric_cols[1].metric("Vehicles", current.vehicle_count)
            metric_cols[2].metric("Stopped", current.stopped_count)
            metric_cols[3].metric("Queue length", current.total_queue)
            metric_cols[4].metric("Waiting", f"{sum(float(r.get('waiting_time_vehicle_s', 0) or 0) for r in junction_rows):.1f} veh-s")
            metric_cols[5].metric("Throughput", sum(int(r.get("throughput", 0) or 0) for r in junction_rows))
            if current.frame_bytes:
                st.image(current.frame_bytes, width="stretch")
            else:
                st.warning("Waiting for the first SUMO-GUI frame…")
            if current.finished:
                st.info("All vehicles have completed this route set. Restart to run it again.")
            st.caption(f"Controller: {current.controller_status} · Prediction: {current.model_status}")
            if current.controller_error:
                st.warning(f"Automatic controller unavailable; native SUMO timing is retained. {current.controller_error}")
            st.subheader("Junction telemetry and next green")
            st.dataframe(pd.DataFrame(junction_rows), hide_index=True, use_container_width=True)
            if current.decision_history:
                st.subheader("Recent AI decision history")
                st.dataframe(pd.DataFrame(current.decision_history[-20:]), hide_index=True, use_container_width=True)

        show_live_sumo()

