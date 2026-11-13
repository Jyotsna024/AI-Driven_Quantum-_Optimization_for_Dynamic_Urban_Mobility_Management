"""Safety-constrained adaptive signal control for SUMO.
All actuated states are copied from the active SUMO program; control decisions are
made only after an existing yellow/all-red clearance has completed.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable
import math
import numpy as np

JUNCTIONS = ("J0", "J1", "J2", "J3")
FEATURES = ("queue_length", "vehicle_count", "mean_speed_mps", "mean_delay_s", "throughput", "arrivals")

def _greens(state: str) -> frozenset[int]:
    return frozenset(i for i, c in enumerate(state) if c in "Gg")

def _yellows(state: str) -> frozenset[int]:
    return frozenset(i for i, c in enumerate(state) if c in "Yy")

def analyze_signal_program(states: Iterable[str]) -> dict[str, Any]:
    """Classify existing service/clearance phases and safe decision boundaries."""
    states = tuple(map(str, states))
    if not states or any(not s for s in states) or len({len(s) for s in states}) != 1:
        raise ValueError("SUMO program has missing or inconsistent signal states")
    greens = tuple(_greens(s) for s in states)
    persistent = set.intersection(*(set(s) for s in greens))
    service = frozenset(i for i, s in enumerate(states) if greens[i] and not _yellows(s))
    if not service:
        raise ValueError("SUMO program contains no green service phases")
    clearances = {}
    for i, state in enumerate(states):
        if greens[i] - persistent:
            continue
        prior = greens[(i - 1) % len(states)] - persistent
        if prior and prior.issubset(_yellows(state)):
            clearances[i] = tuple(sorted(service))
    if not clearances:
        raise ValueError("No verifiable yellow/all-red clearance leads to a green phase")
    return {"states": states, "green_positions": greens, "persistent_green_positions": frozenset(persistent),
            "service_phases": service, "clearance_to_service": clearances}

def adaptive_green_seconds(pressure: float, min_green_s: float = 10.0, max_green_s: float = 50.0,
                           saturation_pressure: float = 12.0, default_s: float = 30.0) -> float:
    """Map demand pressure monotonically into a bounded green duration."""
    if not (0 < min_green_s <= max_green_s):
        raise ValueError("green bounds must satisfy 0 < min <= max")
    if not math.isfinite(float(pressure)):
        return float(min(max(default_s, min_green_s), max_green_s))
    demand = max(0.0, float(pressure))
    if demand < .05:
        return float(min_green_s)
    return float(min_green_s + (max_green_s - min_green_s) * demand / (demand + max(saturation_pressure, .1)))

@dataclass(frozen=True)
class ControllerConfig:
    sample_interval_s: int = 60
    min_green_s: float = 10.0
    max_green_s: float = 50.0
    saturation_pressure: float = 12.0
    downstream_weight: float = .45
    waiting_weight: float = .08
    arrival_weight: float = .25
    forecast_weight: float = .45
    fairness_weight: float = .10
    fairness_cap_s: float = 90.0
    switch_margin: float = 1.0

class AdaptiveSignalController:
    """Per-junction pressure, fairness and ST-GNN controller for legal SUMO phases."""
    def __init__(self, connection, tls_by_junction, incoming_lanes, predictor=None, enabled=True,
                 config=None, predictor_error=None):
        self.conn = connection
        self.tls_by_junction = dict(tls_by_junction)
        self.incoming_lanes = {j: list(incoming_lanes[j]) for j in JUNCTIONS}
        self.predictor = predictor
        self.config = config or ControllerConfig()
        self.enabled = bool(enabled)
        self.error = None
        self.model_status = "ST-GNN warming up" if predictor is not None else f"Pressure-only fallback: {predictor_error or 'ST-GNN unavailable'}"
        self.status = "Automatic AI control active" if enabled else "Automatic AI control off; SUMO native program active"
        self._phase_info, self._links = {}, {}
        self._previous_lane_ids, self._interval_arrivals, self._interval_throughput = {}, {}, {}
        self._previous_junction_assignments, self._crossing_targets = {}, {}
        self._junction_interval_arrivals, self._junction_interval_throughput = {}, {}
        self._history = deque(maxlen=3)
        self.predicted_queue = {j: None for j in JUNCTIONS}
        self.telemetry = {j: {} for j in JUNCTIONS}
        self._last_served = {j: {} for j in JUNCTIONS}
        self._last_applied_phase = {j: None for j in JUNCTIONS}
        self._active_duration = {j: None for j in JUNCTIONS}
        self._next_decision = {j: None for j in JUNCTIONS}
        self.decision_history = []
        self.green_duration_history = {j: [] for j in JUNCTIONS}
        self.cumulative_stopped_vehicle_seconds = 0.0
        self._total_throughput = {j: 0 for j in JUNCTIONS}
        for j in JUNCTIONS:
            tid = self.tls_by_junction[j]
            program_id = str(connection.trafficlight.getProgram(tid))
            logics = connection.trafficlight.getAllProgramLogics(tid)
            logic = next((x for x in logics if str(x.programID) == program_id), None)
            if logic is None:
                raise ValueError(f"{j}: active SUMO program {program_id} is unavailable")
            states = [str(p.state) for p in logic.phases]
            topology = analyze_signal_program(states)
            links = connection.trafficlight.getControlledLinks(tid)
            if len(links) != len(states[0]):
                raise ValueError(f"{j}: controlled-link count differs from SUMO state width")
            self._phase_info[j] = {**topology, "durations": [float(p.duration) for p in logic.phases], "program_id": program_id}
            self._links[j] = [list(group) for group in links]
            crossing_targets = {}
            for group in self._links[j]:
                for incoming, outgoing, via in group:
                    crossing_targets.setdefault(str(incoming), set()).update(
                        str(lane) for lane in (via, outgoing) if lane
                    )
            self._crossing_targets[j] = crossing_targets
            for lane in self.incoming_lanes[j]:
                self._previous_lane_ids[lane] = set(connection.lane.getLastStepVehicleIDs(lane))
                self._interval_arrivals[lane] = self._interval_throughput[lane] = 0
            self._previous_junction_assignments[j] = {
                vehicle: lane for lane in self.incoming_lanes[j]
                for vehicle in self._previous_lane_ids[lane]
            }
            self._junction_interval_arrivals[j] = self._junction_interval_throughput[j] = 0
            self._last_served[j] = {int(i): 0.0 for i in topology["service_phases"]}
        self._collect_telemetry(0.0)

    def set_enabled(self, enabled):
        self.enabled = bool(enabled) and self.error is None
        self.status = (f"Fallback: native SUMO progression retained ({self.error})" if self.error else
                       "Automatic AI control active" if self.enabled else "Automatic AI control off; SUMO native program active")

    def _lane_snapshot(self, lanes):
        lanes = sorted(set(lanes)); ids_all = set(); queue = 0; length = 0.0; speed = wait = 0.0
        for lane in lanes:
            ids = set(self.conn.lane.getLastStepVehicleIDs(lane)); prev = self._previous_lane_ids.get(lane, set())
            self._interval_arrivals[lane] = self._interval_arrivals.get(lane, 0) + len(ids - prev)
            self._interval_throughput[lane] = self._interval_throughput.get(lane, 0) + len(prev - ids)
            self._previous_lane_ids[lane] = ids
            queue += int(self.conn.lane.getLastStepHaltingNumber(lane)); length += float(self.conn.lane.getLength(lane)); ids_all.update(ids)
        for vehicle in ids_all:
            try:
                speed += float(self.conn.vehicle.getSpeed(vehicle)); wait += float(self.conn.vehicle.getWaitingTime(vehicle))
            except Exception: pass
        n = len(ids_all)
        return {"queue_length": queue, "vehicle_count": n, "mean_speed_mps": speed/n if n else 0.0,
                "mean_delay_s": wait/n if n else 0.0, "waiting_time_vehicle_s": wait,
                "density_veh_per_km_lane": n/length*1000.0 if length else 0.0,
                "arrivals": sum(self._interval_arrivals.get(x, 0) for x in lanes),
                "throughput": sum(self._interval_throughput.get(x, 0) for x in lanes)}

    def _collect_telemetry(self, now):
        sample = {}; active_vehicle_ids = set(self.conn.vehicle.getIDList())
        for j in JUNCTIONS:
            assignments = {
                vehicle: lane for lane in self.incoming_lanes[j]
                for vehicle in self.conn.lane.getLastStepVehicleIDs(lane)
            }
            previous = self._previous_junction_assignments[j]
            self._junction_interval_arrivals[j] += len(assignments.keys() - previous.keys())
            for vehicle in previous.keys() - assignments.keys():
                if vehicle not in active_vehicle_ids:
                    continue
                try:
                    lane_now = str(self.conn.vehicle.getLaneID(vehicle))
                    if lane_now in self._crossing_targets[j].get(previous[vehicle], set()):
                        self._junction_interval_throughput[j] += 1
                except Exception:
                    pass
            self._previous_junction_assignments[j] = assignments
            stat = self._lane_snapshot(self.incoming_lanes[j])
            # Count unique arrivals to controlled approaches and departures into a
            # SUMO-defined via/outgoing lane; lane changes within the approach set
            # are not treated as completed junction movements.
            stat["arrivals"] = self._junction_interval_arrivals[j]
            stat["throughput"] = self._junction_interval_throughput[j]
            stat["time_s"] = float(now); self.telemetry[j] = stat
            sample[j] = {key: float(stat[key]) for key in FEATURES}
            for lane in self.incoming_lanes[j]:
                for vehicle in self.conn.lane.getLastStepVehicleIDs(lane):
                    try:
                        if self.conn.vehicle.getSpeed(vehicle) < .1: self.cumulative_stopped_vehicle_seconds += 1.0
                    except Exception: pass
        if int(round(now)) > 0 and int(round(now)) % self.config.sample_interval_s == 0:
            self._history.append(sample); self._update_forecasts()
            for j in JUNCTIONS: self._total_throughput[j] += int(self.telemetry[j]["throughput"])
            self._interval_arrivals = {x: 0 for x in self._interval_arrivals}
            self._interval_throughput = {x: 0 for x in self._interval_throughput}
            self._junction_interval_arrivals = {j: 0 for j in JUNCTIONS}
            self._junction_interval_throughput = {j: 0 for j in JUNCTIONS}

    def _update_forecasts(self):
        if self.predictor is None: return
        if len(self._history) < 3:
            self.model_status = f"ST-GNN warming up: {len(self._history)}/3 traffic samples"; return
        try:
            raw = np.asarray([[[sample[j][f] for f in FEATURES] for sample in self._history] for j in JUNCTIONS], dtype=np.float32)
            values = np.asarray(self.predictor.predict(raw), dtype=float)
            if values.shape != (4, 3) or not np.isfinite(values).all(): raise ValueError(f"unexpected ST-GNN output {values.shape}")
            self.predicted_queue = {j: max(0.0, float(values[i, 0])) for i, j in enumerate(JUNCTIONS)}
            self.model_status = "ST-GNN predictions active (60/120/180 s)"
        except Exception as exc:
            self.model_status = f"Pressure-only fallback: ST-GNN inference failed: {exc}"; self.predictor = None

    def _movement_scores(self, j, now):
        info = self._phase_info[j]; link_scores = {}; details = {}; observed = {}
        for idx, group in enumerate(self._links[j]):
            inc = sorted({str(x[0]) for x in group if x and x[0]}); out = sorted({str(x[1]) for x in group if len(x)>1 and x[1]})
            lane = self._lane_snapshot(inc); downstream = sum(int(self.conn.lane.getLastStepVehicleNumber(x)) for x in out)
            link_scores[idx] = (lane["queue_length"] + self.config.waiting_weight*lane["waiting_time_vehicle_s"]
                                + self.config.arrival_weight*lane["arrivals"] - self.config.downstream_weight*downstream)
            observed[idx] = max(0.0, lane["queue_length"] + .25*lane["arrivals"])
            details[idx] = {"queue": int(lane["queue_length"]), "arrivals": int(lane["arrivals"]), "downstream": downstream,
                            "movement": ", ".join(f"{a}→{b}" for a,b,_ in group[:3]) or f"signal link {idx}"}
        total = sum(observed.values()); scores = {}
        for phase in info["service_phases"]:
            score = 0.0
            for idx in info["green_positions"][phase]:
                score += link_scores.get(idx, 0.0)
                forecast = self.predicted_queue.get(j)
                if forecast is not None:
                    share = observed.get(idx, 0.0)/total if total else 1.0/max(len(info["green_positions"][phase]),1)
                    score += self.config.forecast_weight*forecast*share
            age = min(max(0.0, now-self._last_served[j].get(phase,0.0)),self.config.fairness_cap_s)
            scores[phase] = float(score + self.config.fairness_weight*age)
        return scores, details

    def _choose_phase(self, j, clearance, default, now):
        scores, details = self._movement_scores(j, now)
        candidates = [p for p in self._phase_info[j]["clearance_to_service"].get(clearance, ()) if p in scores]
        if default in scores and default not in candidates: candidates.append(default)
        if not candidates: raise RuntimeError(f"{j}: no validated target after clearance {clearance}")
        selected = max(candidates, key=lambda p:(scores[p],-p))
        if default in candidates and selected != default:
            age = now-self._last_served[j].get(selected,0.0)
            if scores[selected] < scores[default]+self.config.switch_margin and age < self.config.fairness_cap_s: selected=default
        reason = "highest movement pressure/fairness" if selected != default else "native next phase retained by pressure/fairness"
        return selected, scores, details, reason

    def _movement_label(self, j, phase):
        labels=[]
        for idx in sorted(self._phase_info[j]["green_positions"][phase]):
            labels.extend(f"{a} → {b}" for a,b,_ in self._links[j][idx][:2])
        return "; ".join(labels[:6]) or f"Phase {phase}"

    def _apply_green(self,j,phase,score,now,clearance,reason):
        tid=self.tls_by_junction[j]
        if phase not in self._phase_info[j]["service_phases"]: raise RuntimeError(f"{j}: rejected non-service phase {phase}")
        if int(self.conn.trafficlight.getPhase(tid)) != phase:
            self.conn.trafficlight.setPhase(tid,phase)
            if int(self.conn.trafficlight.getPhase(tid)) != phase: raise RuntimeError(f"{j}: SUMO did not enter phase {phase}")
        default=self._phase_info[j]["durations"][phase]
        duration=adaptive_green_seconds(score,self.config.min_green_s,self.config.max_green_s,self.config.saturation_pressure,default)
        self.conn.trafficlight.setPhaseDuration(tid,duration)
        self._last_applied_phase[j]=phase; self._active_duration[j]=duration; self._last_served[j][phase]=now
        self.green_duration_history[j].append(duration)
        row={"time_s":round(now,2),"junction":j,"clearance_phase":clearance,"green_phase":phase,
             "movement":self._movement_label(j,phase),"pressure_score":round(float(score),4),
             "predicted_queue_60s":self.predicted_queue.get(j),"green_duration_s":round(duration,2),"reason":reason}
        self._next_decision[j]=row; self.decision_history.append(row)
        if len(self.decision_history)>500: del self.decision_history[:-500]

    def after_step(self, previous_phases):
        """Observe a completed TraCI step; only change phases at verified clearances."""
        try:
            now=float(self.conn.simulation.getTime()); self._collect_telemetry(now)
            for j in JUNCTIONS:
                tid=self.tls_by_junction[j]; phase=int(self.conn.trafficlight.getPhase(tid)); info=self._phase_info[j]
                if phase not in info["service_phases"] or self._last_applied_phase[j]==phase: continue
                if not self.enabled:
                    self._last_applied_phase[j]=phase; self._active_duration[j]=info["durations"][phase]; self._last_served[j][phase]=now; continue
                prior=int(previous_phases[j])
                if prior in info["clearance_to_service"]:
                    default=(prior+1)%len(info["states"]); selected,scores,_details,reason=self._choose_phase(j,prior,default,now)
                    self._apply_green(j,selected,scores.get(selected,0.0),now,prior,reason)
                else:
                    scores,_details=self._movement_scores(j,now)
                    self._apply_green(j,phase,scores.get(phase,0.0),now,None,"native SUMO phase retained; duration adapted")
        except Exception as exc:
            self.error=f"{type(exc).__name__}: {exc}"; self.enabled=False
            self.status=f"Fallback: native SUMO phase progression retained ({self.error})"

    def junction_status(self,j):
        tid=self.tls_by_junction[j]; phase=int(self.conn.trafficlight.getPhase(tid))
        state=str(self.conn.trafficlight.getRedYellowGreenState(tid)); now=float(self.conn.simulation.getTime())
        try: duration=float(self.conn.trafficlight.getPhaseDuration(tid)); remaining=max(0.0,float(self.conn.trafficlight.getNextSwitch(tid))-now)
        except Exception: duration=self._active_duration[j] or self._phase_info[j]["durations"][phase]; remaining=None
        stat=dict(self.telemetry[j]); green=bool(_greens(state)); nxt=self._next_decision[j]
        stat.update({"Junction":j,"Phase":phase,"Signal state":state,
          "Active movement":self._movement_label(j,phase) if green else "Clearance",
          "Green duration (s)":round(duration,1) if green else None,
          "Green remaining (s)":round(remaining,1) if green and remaining is not None else None,
          "Next AI phase":nxt.get("green_phase") if nxt else None,
          "Next AI movement":nxt.get("movement") if nxt else None,
          "Next AI duration (s)":nxt.get("green_duration_s") if nxt else None,
          "Predicted queue (60 s)":self.predicted_queue.get(j),"Controller":self.status})
        return stat
