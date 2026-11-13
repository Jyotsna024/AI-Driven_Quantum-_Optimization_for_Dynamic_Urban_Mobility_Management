"""Safety, fallback, feature and decision tests for the adaptive controller."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.adaptive_signal_controller import (
    AdaptiveSignalController, ControllerConfig, JUNCTIONS,
    adaptive_green_seconds, analyze_signal_program,
)

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/"sumo_files/cluster_config.json").read_text(encoding="utf-8"))
TLS_BY_J={m["short_id"]:tid for tid,m in CFG["tls_meta"].items()}

class FakeLane:
    def __init__(self): self.queues={}; self.vehicles={}
    def getLastStepVehicleIDs(self,lane): return list(self.vehicles.get(lane,()))
    def getLastStepHaltingNumber(self,lane): return self.queues.get(lane,0)
    def getLastStepVehicleNumber(self,lane): return len(self.vehicles.get(lane,()))
    def getLength(self,lane): return 100.0

class FakeVehicle:
    def __init__(self): self.lane_ids={}
    def getIDList(self): return list(self.lane_ids)
    def getLaneID(self,vehicle): return self.lane_ids[vehicle]
    def getSpeed(self,vehicle): return 0.0
    def getWaitingTime(self,vehicle): return 10.0

class FakeTrafficLight:
    def __init__(self):
        self.program={}; self.phase={}; self.duration={}; self.logics={}; self.links={}; self.phase_writes=[]; self.fail_duration=False
        for tid,meta in CFG["tls_meta"].items():
            phases=CFG["plans"][tid][0]["phases"]
            self.program[tid]="0"; self.phase[tid]=0
            self.logics[tid]=[SimpleNamespace(programID="0",phases=[SimpleNamespace(state=p["state"],duration=p["duration"]) for p in phases])]
            n=len(phases[0]["state"])
            self.links[tid]=[[(f"in_{meta['short_id']}_{i}",f"out_{meta['short_id']}_{i}","")] for i in range(n)]
            self.duration[tid]=float(phases[0]["duration"])
    def getProgram(self,tid): return self.program[tid]
    def getAllProgramLogics(self,tid): return self.logics[tid]
    def getControlledLinks(self,tid): return self.links[tid]
    def getPhase(self,tid): return self.phase[tid]
    def setPhase(self,tid,phase): self.phase_writes.append((tid,int(phase))); self.phase[tid]=int(phase)
    def getRedYellowGreenState(self,tid): return self.logics[tid][0].phases[self.phase[tid]].state
    def setPhaseDuration(self,tid,duration):
        if self.fail_duration: raise RuntimeError("simulated TraCI failure")
        self.duration[tid]=float(duration)
    def getPhaseDuration(self,tid): return self.duration[tid]
    def getNextSwitch(self,tid): return self.conn.simulation.time+self.duration[tid]

class FakeSimulation:
    time=1.0
    def getTime(self): return self.time

class FakeConnection:
    def __init__(self):
        self.lane=FakeLane(); self.vehicle=FakeVehicle(); self.trafficlight=FakeTrafficLight(); self.simulation=FakeSimulation()
        self.trafficlight.conn=self

def make_controller(conn=None,**kwargs):
    conn=conn or FakeConnection()
    incoming={j:[f"in_{j}_{i}" for i in range(len(conn.trafficlight.links[TLS_BY_J[j]]))] for j in JUNCTIONS}
    ctl=AdaptiveSignalController(conn,TLS_BY_J,incoming,**kwargs)
    return ctl,conn

def test_signal_program_phases_are_existing_and_clearance_bounded():
    for tid in CFG["cluster_tls_ids"]:
        phases=CFG["plans"][tid][0]["phases"]
        topology=analyze_signal_program(p["state"] for p in phases)
        assert topology["service_phases"]
        assert topology["clearance_to_service"]
        for target in topology["service_phases"]:
            assert topology["states"][target] in {p["state"] for p in phases}

def test_signal_topology_rejects_missing_clearance():
    with pytest.raises(ValueError,match="clearance"):
        analyze_signal_program(["GG", "GG"])

def test_green_duration_is_monotonic_and_bounded():
    values=[adaptive_green_seconds(x) for x in (0,1,5,20,100)]
    assert values==sorted(values)
    assert all(10 <= x <= 50 for x in values)
    assert adaptive_green_seconds(float("nan"))==30

def test_throughput_excludes_lane_changes_and_counts_declared_movement_departures():
    conn=FakeConnection(); lane="in_J0_0"
    conn.lane.vehicles[lane]=["veh"]; conn.vehicle.lane_ids["veh"]=lane
    ctl,_=make_controller(conn)
    conn.lane.vehicles[lane]=[]; conn.lane.vehicles["in_J0_1"]=["veh"]
    conn.vehicle.lane_ids["veh"]="in_J0_1"
    ctl._collect_telemetry(1)
    assert ctl.telemetry["J0"]["throughput"]==0
    conn.lane.vehicles["in_J0_1"]=[]; conn.vehicle.lane_ids["veh"]="out_J0_1"
    ctl._collect_telemetry(2)
    assert ctl.telemetry["J0"]["throughput"]==1

def test_automatic_decisions_use_only_existing_phases_after_clearance_for_all_junctions():
    ctl,conn=make_controller()
    for j in JUNCTIONS:
        tid=TLS_BY_J[j]
        phases=ctl._phase_info[j]
        default=(1+1)%len(phases["states"])
        conn.trafficlight.phase[tid]=default
        # Heavily load the existing phase-0 movement groups.
        for idx in phases["green_positions"][0]:
            conn.lane.queues[f"in_{j}_{idx}"]=20
    previous={j:1 for j in JUNCTIONS}
    ctl.after_step(previous)
    decisions={r["junction"]:r for r in ctl.decision_history}
    assert set(decisions)==set(JUNCTIONS)
    for j in JUNCTIONS:
        row=decisions[j]
        assert row["green_phase"] in ctl._phase_info[j]["service_phases"]
        assert row["green_phase"]==0
        assert row["clearance_phase"]==1
        assert 10 <= row["green_duration_s"] <= 50
    assert len(conn.trafficlight.phase_writes)==4

def test_controller_never_selects_phase_without_preceding_clearance():
    ctl,conn=make_controller()
    for j in JUNCTIONS:
        tid=TLS_BY_J[j]
        conn.trafficlight.phase[tid]=2
    ctl.after_step({j:0 for j in JUNCTIONS})
    assert conn.trafficlight.phase_writes==[]
    assert all(ctl._last_applied_phase[j]==2 for j in JUNCTIONS)

def test_predictor_failure_falls_back_to_live_pressure():
    class BrokenPredictor:
        def predict(self,_): raise RuntimeError("inference unavailable")
    ctl,_=make_controller(predictor=BrokenPredictor())
    ctl._history.extend([{j:{f:1.0 for f in ("queue_length","vehicle_count","mean_speed_mps","mean_delay_s","throughput","arrivals")} for j in JUNCTIONS} for _ in range(3)])
    ctl._update_forecasts()
    assert ctl.predictor is None
    assert "Pressure-only fallback" in ctl.model_status
    assert ctl.enabled

def test_actuation_failure_disables_ai_and_retains_native_progression():
    ctl,conn=make_controller()
    conn.trafficlight.fail_duration=True
    for j in JUNCTIONS:
        conn.trafficlight.phase[TLS_BY_J[j]]=2
    ctl.after_step({j:1 for j in JUNCTIONS})
    assert not ctl.enabled
    assert "Fallback" in ctl.status
    assert "simulated TraCI failure" in ctl.status
