"""tests/test_s1_cost_model.py - S1 Cost Model Test Suite (Audited & Leakage-Free)"""
import json, math, pickle
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[1]

def test_params_exist():
    p = PROJECT / "models" / "cost_model_params.json"
    assert p.exists()
    d = json.loads(p.read_text())
    assert d["leakage_audit"] == "PASSED (Zero post-rollout queue/throughput/delay used as inputs)"
    assert "coef" in d

def test_pkl_exists():
    p = PROJECT / "models" / "cost_model.pkl"
    assert p.exists()
    with open(p, "rb") as f:
        d = pickle.load(f)
    assert "model" in d and "scaler" in d and "feat_cols" in d

def test_rollouts_csv():
    p = PROJECT / "results" / "rollouts.csv"
    assert p.exists()
    import pandas as pd
    df = pd.read_csv(p)
    assert len(df) >= 20
    assert "mean_delay_s" in df.columns

def test_cost_model_metrics():
    p = PROJECT / "results" / "cost_model_metrics.json"
    assert p.exists()
    d = json.loads(p.read_text())
    assert d["leakage_free"] is True
    tm = d["test_metrics"]
    assert tm["n"] >= 10
    assert tm["spearman"] is not None and math.isfinite(tm["spearman"])

def test_spearman_gate():
    p = PROJECT / "results" / "cost_model_metrics.json"
    d = json.loads(p.read_text())
    assert d["test_metrics"]["spearman"] >= 0.50, f"Spearman {d['test_metrics']['spearman']} < 0.50"
