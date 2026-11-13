"""One live SUMO-GUI process controlled by the dashboard through TraCI."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
JUNCTIONS = ("J0", "J1", "J2", "J3")
FRAME_WIDTH = 1400
FRAME_HEIGHT = 900


def _write_gui_settings(path: Path) -> None:
    """Make vehicles visible at the dashboard's zoom while retaining SUMO drawing."""
    settings = """<?xml version="1.0" encoding="UTF-8"?>
<viewsettings>
  <scheme name="capstone-live">
    <background backgroundColor="18,24,33" showGrid="0"/>
    <edges laneEdgeMode="0" laneShowBorders="1" widthExaggeration="1.1">
      <colorScheme name="uniform">
        <entry color="114,130,149" name="road"/>
        <entry color="67,78,91" name="sidewalk"/>
        <entry color="192,112,77" name="bike lane"/>
        <entry color="18,24,33" name="green verge"/>
        <entry color="52,66,80" name="waterway"/>
      </colorScheme>
    </edges>
    <vehicles vehicleMode="8" vehicleQuality="2" vehicle_minSize="2.0"
      vehicle_exaggeration="1.5" vehicle_constantSize="1">
      <colorScheme name="given vehicle/type/route color"><entry color="yellow"/></colorScheme>
      <colorScheme name="uniform"><entry color="yellow"/></colorScheme>
      <colorScheme name="given/assigned vehicle color"><entry color="yellow"/></colorScheme>
      <colorScheme name="given/assigned type color"><entry color="yellow"/></colorScheme>
      <colorScheme name="given/assigned route color"><entry color="yellow"/></colorScheme>
      <colorScheme name="depart position as HSV"><entry color="yellow"/></colorScheme>
      <colorScheme name="arrival position as HSV"><entry color="yellow"/></colorScheme>
      <colorScheme name="direction/distance as HSV"><entry color="yellow"/></colorScheme>
      <colorScheme name="by speed" interpolated="1">
        <entry color="red" threshold="0.0"/>
        <entry color="yellow" threshold="8.33"/>
        <entry color="green" threshold="15.28"/>
        <entry color="cyan" threshold="22.22"/>
      </colorScheme>
    </vehicles>
  </scheme>
</viewsettings>
"""
    path.write_text(settings, encoding="utf-8")


class LiveSimulation:
    """The GUI screenshot, traffic metrics, signal state and controls share one TraCI session."""

    def __init__(
        self,
        traci_module,
        connection,
        label: str,
        scenario: str,
        seed: int,
        incoming: dict[str, list[str]],
        tls_by_j: dict[str, str],
        incident: dict[str, Any],
        temp_dir: Path,
    ):
        self.traci = traci_module
        self.conn = connection
        self.label = label
        self.scenario = scenario
        self.seed = int(seed)
        self.incoming = incoming
        self.tls_by_j = tls_by_j
        self.incident = incident
        self.temp_dir = temp_dir
        self.view_id = connection.gui.getIDList()[0]
        self.time_s = float(connection.simulation.getTime())
        self.target_plans = {junction: 0 for junction in JUNCTIONS}
        self.actual_plans: dict[str, int] = {}
        self.closed = False
        self.finished = False
        self.frame_bytes: bytes | None = None
        self._frame_number = 0
        self.last_signal_error: str | None = None
        self._old_incident_speed: float | None = None
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
        net_path = ROOT / "sumo_files" / "hsr.net.xml"
        net_root = ET.parse(net_path).getroot()
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
        if any(not incoming[junction] for junction in JUNCTIONS):
            raise RuntimeError("The HSR network is missing controlled incoming lanes")

        plans = cfg["plans"]
        for junction, tid in tls_by_j.items():
            plan_list = plans[tid]
            if [plan["plan_id"] for plan in plan_list] != [0, 1, 2, 3]:
                raise RuntimeError(f"{junction}: expected the four configured signal plans")
            phase_sequences = [[phase["state"] for phase in plan["phases"]] for plan in plan_list]
            if any(sequence != phase_sequences[0] for sequence in phase_sequences[1:]):
                raise RuntimeError(f"{junction}: signal plans do not share a legal phase sequence")

        route = ROOT / "sumo_files" / "routes" / f"{scenario}_s{seed}.rou.xml"
        if not route.exists():
            raise FileNotFoundError(f"Route file not found: {route}")
        sumo_home = Path(os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\SUMO"))
        sumo_gui = sumo_home / "bin" / "sumo-gui.exe"
        if not sumo_gui.exists():
            raise FileNotFoundError(f"SUMO-GUI not found: {sumo_gui}. Set SUMO_HOME to the SUMO installation.")

        temp_dir = Path(tempfile.mkdtemp(prefix="hsr_sumo_dashboard_"))
        settings_path = temp_dir / "viewsettings.xml"
        _write_gui_settings(settings_path)
        label = f"dashboard_gui_{uuid.uuid4().hex[:10]}"
        command = [
            str(sumo_gui), "-n", str(net_path), "-r", str(route),
            "--seed", str(int(seed) + 100), "--start",
            "--window-size", "1400,900", "--gui-settings-file", str(settings_path),
            "--no-step-log", "true", "--no-warnings", "true", "--time-to-teleport", "300",
        ]
        try:
            traci.start(command, label=label, numRetries=60)
            connection = traci.getConnection(label)
            simulation = cls(
                traci, connection, label, scenario, seed, incoming, tls_by_j,
                __import__("yaml").safe_load((ROOT / "configs" / "scenarios.yaml").read_text(encoding="utf-8"))["scenarios"][scenario],
                temp_dir,
            )
            # Keep all four selected junctions in frame; this is SUMO-GUI's own view.
            connection.gui.setBoundary(simulation.view_id, 350, 300, 1650, 1950)
            for junction in JUNCTIONS:
                connection.trafficlight.setProgram(tls_by_j[junction], "0")
            return simulation
        except Exception:
            try:
                if traci.isLoaded():
                    traci.close(wait=True)
            except Exception:
                pass
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

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
        return sum(
            self.conn.lane.getLastStepHaltingNumber(lane)
            for lanes in self.incoming.values()
            for lane in lanes
        )

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
                "Queue": sum(self.conn.lane.getLastStepHaltingNumber(lane) for lane in self.incoming[junction]) if not self.closed else 0,
            })
        return rows

    def _switch_if_safe(self, cycle_wraps: set[str] | None = None) -> None:
        cycle_wraps = cycle_wraps or set()
        for junction in JUNCTIONS:
            requested = self.target_plans[junction]
            if requested == self.actual_plans[junction]:
                continue
            tid = self.tls_by_j[junction]
            if self.conn.trafficlight.getPhase(tid) != 0 and junction not in cycle_wraps:
                continue
            old_plan = self.actual_plans[junction]
            old_state = self.conn.trafficlight.getRedYellowGreenState(tid)
            try:
                self.conn.trafficlight.setProgram(tid, str(requested))
                if self.conn.trafficlight.getRedYellowGreenState(tid) != old_state:
                    self.conn.trafficlight.setProgram(tid, str(old_plan))
                    raise RuntimeError("requested program would change the active signal state")
                self.actual_plans[junction] = requested
                self.last_signal_error = None
            except Exception as exc:
                self.last_signal_error = f"{junction}: plan {requested} remains pending: {exc}"

    def request_plans(self, plans: dict[str, int]) -> None:
        for junction in JUNCTIONS:
            plan = int(plans[junction])
            if plan not in (0, 1, 2, 3):
                raise ValueError(f"{junction}: plan must be 0, 1, 2, or 3")
            self.target_plans[junction] = plan
        self._switch_if_safe()

    def _request_gui_frame(self, path: Path) -> None:
        self.conn.gui.screenshot(self.view_id, str(path), FRAME_WIDTH, FRAME_HEIGHT)

    def _read_gui_frame(self, path: Path, timeout_s: float = 3.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if path.exists() and path.stat().st_size > 1000:
                previous = self.temp_dir / f"frame_{self._frame_number}.png"
                self.frame_bytes = path.read_bytes()
                self._frame_number += 1
                if previous.exists() and previous != path:
                    previous.unlink(missing_ok=True)
                return
            time.sleep(0.02)
        raise TimeoutError("SUMO-GUI did not produce a frame; the GUI view may have closed")

    def advance(self, seconds: int = 1) -> None:
        if self.closed or self.finished:
            return
        count = max(1, int(seconds))
        captured_path = self.temp_dir / f"frame_{self._frame_number + 1}.png"
        for offset in range(count):
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
            if offset == count - 1:
                captured_path = self.temp_dir / f"frame_{self._frame_number + 1}.png"
                self._request_gui_frame(captured_path)
            try:
                self.conn.simulationStep()
            except Exception as exc:
                self.finished = True
                raise RuntimeError(f"SUMO-GUI/TraCI connection failed: {exc}") from exc
            self.time_s = float(self.conn.simulation.getTime())
            wrapped = {
                j for j in JUNCTIONS
                if previous_phases[j] != 0 and self.conn.trafficlight.getPhase(self.tls_by_j[j]) == 0
            }
            self._switch_if_safe(wrapped)
            if self.conn.simulation.getMinExpectedNumber() == 0:
                self.finished = True
                break
        # The GUI writes asynchronously at the simulation step boundary. Always
        # poll for the file; checking existence immediately races the renderer.
        self._read_gui_frame(captured_path, timeout_s=5.0)

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.conn.close(wait=True)
        except Exception:
            process = getattr(self.conn, "_process", None)
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except Exception:
                    process.kill()
                    process.wait(timeout=5)
        finally:
            self.closed = True
            shutil.rmtree(self.temp_dir, ignore_errors=True)

