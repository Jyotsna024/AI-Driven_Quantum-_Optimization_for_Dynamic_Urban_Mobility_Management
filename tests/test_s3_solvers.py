"""tests/test_s3_solvers.py - S3 Solver Benchmark Test Suite"""
import json
import math
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[1]

def test_solver_metrics_exist():
    p = PROJECT / "results" / "solver_metrics.json"
    assert p.exists()
    d = json.loads(p.read_text())
    assert "ExactSolver" in d and "SimulatedAnnealing" in d and "QAOA_Aer" in d

def test_identical_qubo_consumption():
    d = json.loads((PROJECT / "results" / "solver_metrics.json").read_text())
    assert d["summary"]["identical_qubo_verified"] is True
    assert d["summary"]["qubo_variables"] == 16

def test_exact_solver_optimality():
    d = json.loads((PROJECT / "results" / "solver_metrics.json").read_text())
    qm = json.loads((PROJECT / "models" / "qubo_meta.json").read_text())
    es = d["ExactSolver"]
    assert es["feasible"] is True
    assert abs(es["best_energy"] - qm["best_energy"]) < 1e-3
    assert es["best_plans"] == qm["best_plans"]

def test_sa_quality_and_feasibility():
    d = json.loads((PROJECT / "results" / "solver_metrics.json").read_text())
    sa = d["SimulatedAnnealing"]
    assert sa["feasible_rate"] >= 0.80
    assert sa["approx_ratio"] >= 0.95

def test_qaoa_aer_execution():
    d = json.loads((PROJECT / "results" / "solver_metrics.json").read_text())
    qaoa = d["QAOA_Aer"]
    assert "Aer simulator" in qaoa["hardware_label"]
    assert qaoa["feasible"] is True
    assert qaoa["approx_ratio"] >= 0.90
    assert math.isfinite(qaoa["best_energy"])
