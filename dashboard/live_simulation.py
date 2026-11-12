"""Live SUMO/TraCI session and lightweight network renderer for the dashboard."""
from __future__ import annotations

import json
import os
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.collections import LineCollection

ROOT = Path(__file__).resolve().parents[1]
JUNCTIONS = ("J0", "J1", "J2", "J3")


@st.cache_data(show_spinner="Loading HSR road network")
def load_map_geometry(net_path: Path, config_path: Path) -> dict[str, Any]:
    """Read real lane geometries and selected junction positions from project files."""
    net_path, config_path = Path(net_path), Path(config_path)
    root = ET.parse(net_path).getroot()
    segments: list[list[tuple[float, float]]] = []
    for edge in root.findall("edge"):
        if edge.get("function") == "internal" or edge.get("id", "").startswith(":"):
            continue
        for lane in edge.findall("lane"):
            raw = lane.get("shape", "")
            if not raw:
                continue
            points = []
            for point in raw.split():
                x, y = point.split(",")[:2]
                points.append((float(x), float(y)))
            if len(points) >= 2:
                segments.append(points)

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    junctions = []
    for _tls_id, meta in cfg["tls_meta"].items():
        short_id = meta["short_id"]
        if short_id in JUNCTIONS:
            x, y = meta["approx_coords"]
            junctions.append({"junction": short_id, "x": float(x), "y": float(y), "name": meta.get("name", short_id)})
    return {"segments": segments, "junctions": junctions}


class LiveSimulation:
    """A single user-controlled SUMO simulation instance."""

    def __init__(self, traci_module, connection, label: str, scenario: str, seed: int, incoming: dict[str, list[str]], tls_by_j: dict[str, str], incident: dict[str, Any]):
        self.traci = traci_module
        self.conn = connection
        self.label = label
        self.scenario = scenario
        self.seed = int(seed)
        self.incoming = incoming
        self.tls_by_j = tls_by_j
        self.incident = incident
        self.time_s = float(connection.simulation.getTime())
        self.target_plans = {j: 0 for j in JUNCTIONS}
        self.actual_plans = {}
        self.closed = False
        self.finished = False
        self._old_incident_speed = None
        self._incident_applied = False
        self._incident_restored = False
        for junction in JUNCTIONS:
            tid = tls_by_j[junction]
            program = str(connection.trafficlight.getProgram(tid))
            self.actual_plans[junction] = int(program) if program.isdigit() else 0

    @classmethod
    def start(cls, scenario: str, seed: int) -> "LiveSimulation":
        import traci

        cfg = json.loads((ROOT / "sumo_files" / "cluster_config.json").read_text(encoding="utf-8"))
        tls_by_j = {meta["short_id"]: tls_id for tls_id, meta in cfg["tls_meta"].items()}
        net_root = ET.parse(ROOT / "sumo_files" / "hsr.net.xml").getroot()
        edge_lanes = {edge.get("id"): [lane.get("id") for lane in edge.findall("lane")] for edge in net_root.findall("edge")}
        incoming = {junction: [] for junction in JUNCTIONS}
        for connection in net_root.findall(".//connection"):
            tid = connection.get("tl")
            if tid not in cfg["cluster_tls_ids"]:
                continue
            junction = cfg["tls_meta"][tid]["short_id"]
            for lane in edge_lanes.get(connection.get("from"), []):
                if lane not in incoming[junction]:
                    incoming[junction].append(lane)
        if any(not incoming[j] for j in JUNCTIONS):
            raise RuntimeError("The selected network is missing controlled incoming lanes")

        route = ROOT / "sumo_files" / "routes" / f"{scenario}_s{seed}.rou.xml"
        if not route.exists():
            raise FileNotFoundError(f"Route file not found: {route}")
        sumo_home = Path(os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\SUMO"))
        sumo = sumo_home / "bin" / "sumo.exe"
        if not sumo.exists():
            raise FileNotFoundError(f"SUMO executable not found: {sumo}. Set SUMO_HOME to your SUMO installation.")
        label = f"dashboard_{uuid.uuid4().hex[:10]}"
        command = [
            str(sumo), "-n", str(ROOT / "sumo_files" / "hsr.net.xml"), "-r", str(route),
            "--seed", str(int(seed) + 100), "--no-step-log", "true", "--no-warnings", "true",
            "--time-to-teleport", "300",
        ]
        traci.start(command, label=label)
        connection = traci.getConnection(label)
        for junction in JUNCTIONS:
            connection.trafficlight.setProgram(tls_by_j[junction], "0")
        scenario_cfg = __import__("yaml").safe_load((ROOT / "configs" / "scenarios.yaml").read_text(encoding="utf-8"))["scenarios"][scenario]
        return cls(traci, connection, label, scenario, seed, incoming, tls_by_j, scenario_cfg)

    @property
    def vehicle_count(self) -> int:
        return len(self.conn.vehicle.getIDList()) if not self.closed and not self.finished else 0

    @property
    def stopped_count(self) -> int:
        if self.closed or self.finished:
            return 0
        return sum(1 for vehicle in self.conn.vehicle.getIDList() if self.conn.vehicle.getSpeed(vehicle) < 0.1)

    @property
    def total_queue(self) -> int:
        if self.closed or self.finished:
            return 0
        return sum(self.conn.lane.getLastStepHaltingNumber(lane) for lanes in self.incoming.values() for lane in lanes)

    @property
    def vehicle_positions(self) -> list[tuple[float, float, float]]:
        if self.closed or self.finished:
            return []
        positions = []
        for vehicle in self.conn.vehicle.getIDList():
            x, y = self.conn.vehicle.getPosition(vehicle)
            positions.append((float(x), float(y), float(self.conn.vehicle.getSpeed(vehicle))))
        return positions

    @property
    def signal_status(self) -> list[dict[str, Any]]:
        rows = []
        for junction in JUNCTIONS:
            tid = self.tls_by_j[junction]
            if self.closed:
                phase, state = "—", "Stopped"
            else:
                phase = self.conn.trafficlight.getPhase(tid)
                state = self.conn.trafficlight.getRedYellowGreenState(tid)
            rows.append({
                "Junction": junction,
                "Applied plan": self.actual_plans[junction],
                "Requested plan": self.target_plans[junction],
                "Phase": phase,
                "Signal state": state,
            })
        return rows

    def request_plans(self, plans: dict[str, int]) -> None:
        for junction in JUNCTIONS:
            self.target_plans[junction] = int(plans[junction])
        self._apply_safe_plans()

    def _apply_safe_plans(self, cycle_wraps: set[str] | None = None) -> None:
        cycle_wraps = cycle_wraps or set()
        for junction in JUNCTIONS:
            desired = self.target_plans[junction]
            if desired == self.actual_plans[junction]:
                continue
            tid = self.tls_by_j[junction]
            phase = self.conn.trafficlight.getPhase(tid)
            if phase != 0 and junction not in cycle_wraps:
                continue
            old_program = self.actual_plans[junction]
            old_state = self.conn.trafficlight.getRedYellowGreenState(tid)
            try:
                self.conn.trafficlight.setProgram(tid, str(desired))
                new_state = self.conn.trafficlight.getRedYellowGreenState(tid)
                if new_state != old_state:
                    self.conn.trafficlight.setProgram(tid, str(old_program))
                    raise RuntimeError(f"{junction}: requested plan changes the active signal state")
                self.actual_plans[junction] = desired
            except Exception:
                # Keep the actual program unchanged and leave the request pending.
                self.target_plans[junction] = old_program

    def advance(self, seconds: int, wall_delay_s: float = 0.0) -> None:
        if self.closed or self.finished:
            return
        for _ in range(int(seconds)):
            current_t = int(self.conn.simulation.getTime())
            blocked_lane = self.incident.get("blocked_lane")
            if self.incident.get("incident") and blocked_lane:
                start = int(self.incident.get("incident_start_s", 1200))
                end = int(self.incident.get("incident_end_s", 1800))
                if current_t >= start and not self._incident_applied:
                    self._old_incident_speed = self.conn.lane.getMaxSpeed(blocked_lane)
                    self.conn.lane.setMaxSpeed(blocked_lane, 0.1)
                    self._incident_applied = True
                if current_t >= end and self._incident_applied and not self._incident_restored:
                    self.conn.lane.setMaxSpeed(blocked_lane, self._old_incident_speed)
                    self._incident_restored = True
            previous_phases = {j: self.conn.trafficlight.getPhase(self.tls_by_j[j]) for j in JUNCTIONS}
            try:
                self.conn.simulationStep()
            except Exception as exc:
                self.finished = True
                raise RuntimeError(f"SUMO stopped before the next simulation step: {exc}") from exc
            self.time_s = float(self.conn.simulation.getTime())
            wrapped = {
                j for j in JUNCTIONS
                if previous_phases[j] != 0 and self.conn.trafficlight.getPhase(self.tls_by_j[j]) == 0
            }
            self._apply_safe_plans(wrapped)
            if self.conn.simulation.getMinExpectedNumber() == 0:
                self.finished = True
                break
            if wall_delay_s:
                time.sleep(wall_delay_s)

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.conn.close()
        finally:
            self.closed = True


def render_simulation(geometry: dict[str, Any], vehicles: list[tuple[float, float, float]], signal_status: list[dict[str, Any]]):
    """Render the actual SUMO road geometry, vehicle coordinates, and junctions."""
    fig, ax = plt.subplots(figsize=(13, 7), facecolor="#111827")
    ax.set_facecolor("#111827")
    roads = LineCollection(geometry["segments"], colors="#607080", linewidths=0.48, alpha=0.66, zorder=1)
    ax.add_collection(roads)

    if vehicles:
        x, y, speed = np.asarray(vehicles, dtype=float).T
        points = ax.scatter(x, y, c=speed, cmap="RdYlGn", vmin=0, vmax=14, s=12, alpha=0.95, linewidths=0, zorder=3)
        colorbar = fig.colorbar(points, ax=ax, fraction=0.024, pad=0.012)
        colorbar.set_label("Vehicle speed (m/s)", color="#e5e7eb")
        colorbar.ax.yaxis.set_tick_params(color="#e5e7eb")
        plt.setp(colorbar.ax.get_yticklabels(), color="#e5e7eb")

    for junction in geometry["junctions"]:
        ax.scatter([junction["x"]], [junction["y"]], marker="s", s=60, color="#38bdf8", edgecolor="white", linewidth=0.7, zorder=4)
        ax.annotate(junction["junction"], (junction["x"], junction["y"]), xytext=(5, 6), textcoords="offset points", color="white", fontsize=9, weight="bold", zorder=5)

    if geometry["segments"]:
        all_x = [p[0] for segment in geometry["segments"] for p in segment]
        all_y = [p[1] for segment in geometry["segments"] for p in segment]
        ax.set_xlim(min(all_x), max(all_x))
        ax.set_ylim(min(all_y), max(all_y))
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title("Live SUMO network · moving vehicles · HSR Layout", color="white", loc="left", pad=10, fontsize=13)
    fig.tight_layout(pad=0.5)
    return fig

