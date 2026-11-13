"""Checks that dashboard views read saved project evidence without hardcoded metrics."""
import sys
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"dashboard"))
from data import load_json,load_network,load_phase8

def test_dashboard_loads_real_network_and_episode_outputs():
    cfg=load_network(ROOT/"sumo_files/cluster_config.json")
    assert set(m["short_id"] for m in cfg["tls_meta"].values())=={"J0","J1","J2","J3"}
    episodes=load_phase8(ROOT/"results/phase8_eval_v1")
    assert len(episodes)==40
    assert set(episodes.controller)=={"fixed_time","max_pressure","sa_qubo","qaoa_aer_sampled"}
    assert set(episodes.scenario)=={"low","medium","high","peak","incident"}
    assert episodes.total_waiting_time_vehicle_s.notna().all()

def test_dashboard_reads_forecasts_and_controller_logs():
    d=ROOT/"results/phase8_eval_v1"
    f=load_phase8(d,suffix="_forecasts.csv")
    logs=load_phase8(d,suffix="_decisions.csv")
    assert len(f)>0 and {"predicted_queue","actual_queue","horizon"}.issubset(f.columns)
    assert len(logs)==400*4
    assert set(logs.objective_status)=={"SCIENTIFICALLY_NOT_VALIDATED"}

def test_dashboard_reads_s2_invalid_status_and_s3_solver_results():
    diagnosis=load_json(ROOT/"results/s2_diagnostic_v3.json")
    solvers=load_json(ROOT/"results/solver_metrics_s3_v3.json")
    assert diagnosis["status"]=="SCIENTIFIC_NOT_VALIDATED"
    assert set(solvers["solvers"])=={"ExactSolver","SimulatedAnnealing","QAOA"}
