"""tests/test_s1_cost_model.py"""
import json, math, pickle
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[1]

def test_params_exist():
    p = PROJECT / "models" / "cost_model_params.json"
    assert p.exists()
    d = json.loads(p.read_text())
    assert "model_type" in d
    assert "coef" in d

def test_pkl_exists():
    p = PROJECT / "models" / "cost_model.pkl"
    assert p.exists()
    import pickle
    with open(p,"rb") as f:
        d = pickle.load(f)
    assert "model" in d and "scaler" in d

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
    assert "test_metrics" in d
    tm = d["test_metrics"]
    assert tm["n"] >= 10
    assert tm["spearman"] is not None and math.isfinite(tm["spearman"])

def test_spearman_gate():
    p = PROJECT / "results" / "cost_model_metrics.json"
    d = json.loads(p.read_text())
    # Gate: Spearman >= 0.5 on test
    assert d["test_metrics"]["spearman"] >= 0.5, f"Spearman {d['test_metrics']['spearman']} < 0.5"
