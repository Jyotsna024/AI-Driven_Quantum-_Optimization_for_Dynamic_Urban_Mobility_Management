"""tests/test_s2_qubo.py"""
import json, numpy as np
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[1]

def test_qubo_npz():
    p = PROJECT / "models" / "qubo.npz"
    assert p.exists()
    d = np.load(p)
    assert "Q" in d
    Q = d["Q"]
    assert Q.shape == (16, 16)

def test_qubo_meta():
    p = PROJECT / "models" / "qubo_meta.json"
    assert p.exists()
    d = json.loads(p.read_text())
    assert d["N"] == 16
    assert d["K"] == 4
    assert d["penalty_gap_ok"] is True

def test_qubo_metrics():
    p = PROJECT / "results" / "qubo_metrics.json"
    assert p.exists()
    d = json.loads(p.read_text())
    bf = d["brute_force"]
    assert bf["n_feasible"] == 256, f"Expected 256 feasible, got {bf['n_feasible']}"
    assert bf["penalty_gap_ok"] is True

def test_manual_energy():
    """Verify best solution is consistent: x^T Q x equals stored best_energy."""
    import numpy as np
    mq = json.loads((PROJECT / "models" / "qubo_meta.json").read_text())
    d  = np.load(PROJECT / "models" / "qubo.npz")
    Q  = d["Q"]
    x  = np.array(mq["best_x"], dtype=float)
    computed_e = float(x @ Q @ x)
    stored_e   = mq["best_energy"]
    assert abs(computed_e - stored_e) < 0.1, f"Energy mismatch: computed={computed_e} stored={stored_e}"
    # Also verify feasibility
    K = mq["K"]
    for i in range(4):
        s = int(sum(x[i*K:(i+1)*K]))
        assert s == 1, f"Junction {i} not one-hot: sum={s}"
