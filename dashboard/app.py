"""Minimal Streamlit front end for the actual live SUMO-GUI simulation."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))
from live_simulation import LiveSimulation

st.set_page_config(page_title="HSR SUMO Simulation", page_icon="🚦", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: .8rem; padding-bottom: .6rem; max-width: 1800px;}
      h3 {font-size: 1.05rem !important; margin: 0 !important;}
      div[data-testid="stMetric"] {padding: .25rem .4rem;}
      section[data-testid="stSidebar"] {display:none;}
    </style>
    """,
    unsafe_allow_html=True,
)


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
    st.markdown(f"**SUMO · {status}**")

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

    plan_labels = {0: "Plan 0", 1: "Plan 1", 2: "Plan 2", 3: "Plan 3"}
    st.markdown("**Signals**")
    plan_choices: dict[str, int] = {}
    plan_cols = st.columns(2)
    for idx, junction in enumerate(("J0", "J1", "J2", "J3")):
        with plan_cols[idx % 2]:
            plan_choices[junction] = st.selectbox(
                junction,
                [0, 1, 2, 3],
                format_func=lambda plan: plan_labels[plan],
                key=f"live_plan_{junction}",
            )

    btn_start, btn_step = st.columns(2)
    with btn_start:
        start_clicked = st.button("Start / restart", type="primary", use_container_width=True)
    with btn_step:
        step_clicked = st.button("Step 1 second", use_container_width=True, disabled=simulation is None)

    if start_clicked:
        stop_current_simulation()
        try:
            new_simulation = LiveSimulation.start(scenario, int(seed))
            new_simulation.request_plans(plan_choices)
            # A real SUMO warm-up makes the first embedded GUI frame visibly populated.
            new_simulation.advance(60)
            st.session_state.live_simulation = new_simulation
            st.session_state.is_playing = True
            st.session_state.live_error = None
        except Exception as exc:
            st.session_state.live_error = f"Could not start SUMO-GUI: {exc}"

    simulation = st.session_state.get("live_simulation")
    if simulation is not None:
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
        if plan_choices != simulation.target_plans:
            try:
                simulation.request_plans(plan_choices)
            except Exception as exc:
                st.session_state.live_error = f"Signal plan update failed: {exc}"
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
            metric_cols = st.columns(4)
            metric_cols[0].metric("Time", f"{current.time_s:.0f} s")
            metric_cols[1].metric("Vehicles", current.vehicle_count)
            metric_cols[2].metric("Stopped", current.stopped_count)
            metric_cols[3].metric("Queue", current.total_queue)
            if current.frame_bytes:
                st.image(current.frame_bytes, width="stretch")
            else:
                st.warning("Waiting for the first SUMO-GUI frame…")
            if current.finished:
                st.info("All vehicles have completed this route set. Restart to run it again.")
            with st.expander("Junction details"):
                st.dataframe(pd.DataFrame(current.signal_status), hide_index=True, use_container_width=True)

        show_live_sumo()

