"""tests/test_s2_qubo.py - S2 QUBO Formulation Test Suite"""
import json
import numpy as np
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[1]

def test_qubo_npz():
    p = PROJECT / "models" / "qubo.npz"
    assert p.exists()
    d = np.load(p)
    assert "Q" in d
    assert d["Q"].shape == (16, 16)

def test_qubo_meta():
    p = PROJECT / "models" / "qubo_meta.json"
    assert p.exists()
    d = json.loads(p.read_text())
    assert d["N"] == 16
    assert d["K"] == 4
    assert d["penalty_gap_ok"] is True

def test_qubo_brute_force():
    p = PROJECT / "results" / "qubo_metrics.json"
    assert p.exists()
    d = json.loads(p.read_text())
    bf = d["brute_force"]
    assert bf["n_feasible"] == 256
    assert bf["n_infeasible"] == 65280
    assert bf["penalty_gap_ok"] is True

def test_energy_delay_consistency():
    mq = json.loads((PROJECT / "models" / "qubo_meta.json").read_text())
    d = np.load(PROJECT / "models" / "qubo.npz")
    Q = d["Q"]
    A = float(d["A"][0])
    x = np.array(mq["best_x"], dtype=float)
    computed_e = float(x @ Q @ x)
    stored_e = mq["best_energy"]
    assert abs(computed_e - stored_e) < 1e-3
    c = np.array(mq["cost_matrix"])
    bp = mq["best_plans"]
    manual_delay = sum(c[i, bp[i]] for i in range(4))
    qubo_delay = computed_e + 4.0 * A
    assert abs(manual_delay - qubo_delay) < 1e-3
