"""Small interactive dashboard for displaying and manually controlling SUMO."""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))
from live_simulation import LiveSimulation, load_map_geometry, render_simulation

st.set_page_config(
    page_title="HSR Layout SUMO Simulation",
    page_icon="🚦",
    layout="wide",
)
st.title("HSR Layout Traffic Simulation")
st.caption("Live SUMO simulation · Bengaluru · Manually controlled signal plans")


def stop_current_simulation() -> None:
    simulation = st.session_state.pop("live_simulation", None)
    if simulation is not None:
        try:
            simulation.close()
        except Exception:
            pass


scenarios = ["low", "medium", "high", "peak", "incident"]
scenario_names = {
    "low": "Low demand",
    "medium": "Medium demand",
    "high": "High demand",
    "peak": "Peak demand",
    "incident": "Incident",
}

with st.sidebar:
    st.header("Simulation controls")
    scenario = st.selectbox(
        "Traffic scenario",
        scenarios,
        index=scenarios.index("high"),
        format_func=lambda value: scenario_names[value],
        key="scenario_choice",
    )
    seed = st.selectbox("Route seed", list(range(5)), index=0, key="seed_choice")
    st.markdown("**Signal plan by junction**")
    plan_choices: dict[str, int] = {}
    plan_cols = st.columns(2)
    plan_labels = {0: "Baseline", 1: "Main heavy", 2: "Side heavy", 3: "Offset variant"}
    for idx, junction in enumerate(("J0", "J1", "J2", "J3")):
        with plan_cols[idx % 2]:
            plan_choices[junction] = st.selectbox(
                junction,
                [0, 1, 2, 3],
                format_func=lambda plan: f"{plan} · {plan_labels[plan]}",
                key=f"manual_plan_{junction}",
            )

    start_clicked = st.button("Start / restart", type="primary", use_container_width=True)
    if start_clicked:
        stop_current_simulation()
        try:
            st.session_state.live_simulation = LiveSimulation.start(scenario, int(seed))
            st.session_state.live_simulation.request_plans(plan_choices)
            # Fill the network before the first rendered frame so it does not look
            # like a blank or disconnected simulation while vehicles are departing.
            st.session_state.live_simulation.advance(60)
            st.session_state.is_playing = True
            st.session_state.live_notice = None
        except Exception as exc:
            st.session_state.live_notice = f"Could not start SUMO: {exc}"

    simulation = st.session_state.get("live_simulation")
    if simulation is not None:
        if "is_playing" not in st.session_state:
            st.session_state.is_playing = True
        play_label = "Pause" if st.session_state.is_playing else "Play"
        if st.button(play_label, use_container_width=True):
            st.session_state.is_playing = not st.session_state.is_playing
        if st.button("Step 1 second", use_container_width=True):
            st.session_state.is_playing = False
            simulation.advance(1)
        if st.button("Apply selected signal plans", use_container_width=True):
            simulation.request_plans(plan_choices)
        if st.button("Stop simulation", use_container_width=True):
            stop_current_simulation()
            st.session_state.is_playing = False
            st.rerun()
    st.caption("Simulation plays automatically after start. Choose and apply signal plans manually; they take effect at the next safe cycle boundary.")

notice = st.session_state.pop("live_notice", None)
if notice:
    st.error(notice)

simulation = st.session_state.get("live_simulation")
if simulation is None:
    st.info("Choose a traffic scenario and route seed, then select **Start / restart** to display the live SUMO simulation.")
    st.markdown("The dashboard controls the SUMO simulation directly through TraCI. It does not change or publish any saved experiment results.")
else:
    @st.fragment(run_every="1s")
    def live_simulation_view() -> None:
        current = st.session_state.get("live_simulation")
        if current is None:
            st.info("Choose a traffic scenario and route seed, then select **Start / restart** to display the live SUMO simulation.")
            return
        if st.session_state.get("is_playing", True) and not current.finished:
            try:
                current.advance(5)
            except Exception as exc:
                st.session_state.is_playing = False
                st.error(f"SUMO stopped: {exc}")

        top = st.columns(4)
        top[0].metric("Simulation time", f"{current.time_s:.0f} s")
        top[1].metric("Vehicles on network", str(current.vehicle_count))
        top[2].metric("Stopped vehicles", str(current.stopped_count))
        top[3].metric("Queued at junctions", str(current.total_queue))

        geometry = load_map_geometry(ROOT / "sumo_files" / "hsr.net.xml", ROOT / "sumo_files" / "cluster_config.json")
        fig = render_simulation(geometry, current.vehicle_positions, current.signal_status)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.subheader("Junction signal status")
        signal_cols = st.columns(4)
        for col, row in zip(signal_cols, current.signal_status):
            with col:
                st.metric(row["Junction"], f"Plan {row['Applied plan']}", f"Requested {row['Requested plan']}")
                st.caption(f"Phase {row['Phase']} · {row['Signal state']}")
        st.caption("Vehicles and signals are read from the active SUMO/TraCI session. Solver-based control is omitted because S2 is not scientifically validated.")

    live_simulation_view()

