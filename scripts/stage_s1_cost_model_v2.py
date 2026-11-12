"""
scripts/stage_s1_cost_model_v2.py  -  S1 (Part 2 only): Ridge regression cost model.
Reads existing results/rollouts.csv; outputs:
  results/cost_model_metrics.json  (overwrite)
  models/cost_model_params.json    (overwrite with Ridge params)
  tests/test_s1_cost_model.py      (overwrite)
Does NOT re-run TraCI.
"""
import os, sys, json, pickle, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"
MODELS  = PROJECT / "models"
RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

JUNCTION_ORDER = ["J0","J1","J2","J3"]
FEAT_COLS = ["plan_0","plan_1","plan_2","plan_3",
             "junc_J0","junc_J1","junc_J2","junc_J3",
             "mean_queue_m","throughput"]

# =========================================================================
def build_features(df):
    df = df.copy()
    for pid in range(4):
        df[f"plan_{pid}"] = (df["plan_id"] == pid).astype(float)
    for jid in JUNCTION_ORDER:
        df[f"junc_{jid}"] = (df["junction"] == jid).astype(float)
    df["mean_queue_m"] = df["mean_queue_m"].fillna(df["mean_queue_m"].median())
    df["throughput"]   = df["throughput"].fillna(0)
    return df

def evaluate_split(model, scaler, df, split_name):
    sub = df[df["split"] == split_name].dropna(subset=["mean_delay_s"])
    if sub.empty:
        return {"n": 0, "mae": None, "rmse": None, "spearman": None, "top1_pct": None, "per_junction": {}}
    X = scaler.transform(sub[FEAT_COLS])
    preds = model.predict(X)
    trues = sub["mean_delay_s"].values
    mae   = float(np.mean(np.abs(preds - trues)))
    rmse  = float(np.sqrt(np.mean((preds - trues)**2)))
    rho, _ = spearmanr(preds, trues)

    # Top-1
    sub2 = sub.copy(); sub2["pred"] = preds
    correct, total = 0, 0
    for (sn, jn), grp in sub2.groupby(["snapshot_path","junction"]):
        gv = grp.dropna(subset=["pred","mean_delay_s"])
        if len(gv) >= 2:
            pb = gv.loc[gv["pred"].idxmin(), "plan_id"]
            tb = gv.loc[gv["mean_delay_s"].idxmin(), "plan_id"]
            correct += int(pb == tb); total += 1
    top1 = correct / total * 100 if total > 0 else None

    # Per-junction
    per_junc = {}
    for jid in JUNCTION_ORDER:
        jdf = sub2[sub2["junction"] == jid]
        if jdf.empty: continue
        j_mae = float(np.mean(np.abs(jdf["pred"] - jdf["mean_delay_s"])))
        j_rho, _ = spearmanr(jdf["pred"], jdf["mean_delay_s"]) if len(jdf) > 2 else (float("nan"), None)
        per_junc[jid] = {"n": len(jdf), "mae": round(j_mae, 3), "spearman": round(float(j_rho), 3)}

    return {"n": int(len(sub)), "mae": round(mae, 3), "rmse": round(rmse, 3),
            "spearman": round(float(rho), 4),
            "top1_pct": round(top1, 1) if top1 is not None else None,
            "per_junction": per_junc}

# =========================================================================
# Gate A and Gate B (re-computed on rollouts, same logic as before)
def check_gates(df):
    gates = {}
    # Gate A: best-vs-worst plan delay diff >= 10% at >= 3/4 junctions
    ga_pass = 0; ga_detail = {}
    for jid in JUNCTION_ORDER:
        sub = df[(df["junction"]==jid) & df["split"].isin(["train","val"])].dropna(subset=["mean_delay_s"])
        if sub.empty: ga_detail[jid]={"verdict":"NO_DATA"}; continue
        plan_delays = sub.groupby("plan_id")["mean_delay_s"].mean()
        if len(plan_delays)<2: ga_detail[jid]={"verdict":"INSUFFICIENT_PLANS"}; continue
        best=float(plan_delays.min()); worst=float(plan_delays.max())
        diff_pct = (worst-best)/best*100 if best>0 else 0.0
        passed = diff_pct >= 10.0
        if passed: ga_pass += 1
        ga_detail[jid]={"best_delay":round(best,2),"worst_delay":round(worst,2),
                        "diff_pct":round(diff_pct,1),"pass":passed}
    gates["gate_A"]={"junctions_pass":ga_pass,"threshold_junctions":3,
                     "verdict":"PASS" if ga_pass>=3 else "FAIL","detail":ga_detail}

    # Gate B: no single plan best in >70% at >=3 junctions
    gb_pass=0; gb_detail={}
    from collections import Counter
    for jid in JUNCTION_ORDER:
        sub = df[(df["junction"]==jid) & df["split"].isin(["train","val"])].dropna(subset=["mean_delay_s"])
        if sub.empty: gb_detail[jid]={"verdict":"NO_DATA"}; continue
        best_plans=[]
        for snap, grp in sub.groupby("snapshot_path"):
            gv = grp.dropna(subset=["mean_delay_s"])
            if len(gv["plan_id"].unique())>=2:
                best_plans.append(int(gv.loc[gv["mean_delay_s"].idxmin(),"plan_id"]))
        if not best_plans: gb_detail[jid]={"verdict":"NO_DATA"}; continue
        cnt=Counter(best_plans); total=len(best_plans)
        dominant_pct=max(cnt.values())/total*100
        passed=dominant_pct<=70.0
        if passed: gb_pass+=1
        gb_detail[jid]={"plan_freq":dict(cnt),"dominant_pct":round(dominant_pct,1),"pass":passed}
    gates["gate_B"]={"junctions_pass":gb_pass,"threshold_junctions":3,
                     "verdict":"PASS" if gb_pass>=3 else "FAIL","detail":gb_detail}
    return gates

# =========================================================================
if __name__ == "__main__":
    print("[1/4] Loading rollouts...")
    df_raw = pd.read_csv(RESULTS / "rollouts.csv")
    print(f"  {len(df_raw)} rows loaded")

    print("[2/4] Checking gates A and B...")
    gates = check_gates(df_raw)
    print(f"  Gate A: {gates['gate_A']['verdict']}  Gate B: {gates['gate_B']['verdict']}")

    print("[3/4] Training Ridge regression cost model...")
    df = build_features(df_raw)
    train = df[df["split"]=="train"].dropna(subset=["mean_delay_s"])
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train[FEAT_COLS])
    model = Ridge(alpha=1.0)
    model.fit(X_tr, train["mean_delay_s"])

    val_metrics  = evaluate_split(model, scaler, df, "val")
    test_metrics = evaluate_split(model, scaler, df, "test")
    print(f"  Val  Spearman={val_metrics['spearman']}  MAE={val_metrics['mae']}")
    print(f"  Test Spearman={test_metrics['spearman']}  MAE={test_metrics['mae']}  top1={test_metrics['top1_pct']}%")

    spearman_gate = "PASS" if (test_metrics["spearman"] is not None and test_metrics["spearman"] >= 0.5) else "FAIL"
    print(f"  Spearman gate (>=0.5): {spearman_gate}")

    # Save model artifacts
    params = {
        "model_type": "Ridge",
        "alpha": 1.0,
        "feature_cols": FEAT_COLS,
        "coef": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    (MODELS / "cost_model_params.json").write_text(json.dumps(params, indent=2))
    # Also save sklearn objects for use in downstream stages
    with open(MODELS / "cost_model.pkl", "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "feat_cols": FEAT_COLS}, f)
    print(f"  Saved models/cost_model_params.json + models/cost_model.pkl")

    print("[4/4] Writing results and tests...")
    OUT = {
        "gates": gates,
        "model_type": "Ridge regression",
        "calibrated_params": {"alpha": 1.0},
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "spearman_gate": spearman_gate,
    }
    (RESULTS / "cost_model_metrics.json").write_text(json.dumps(OUT, indent=2))
    print(f"  results/cost_model_metrics.json written")

    # Write tests
    test_code = '''"""tests/test_s1_cost_model.py"""
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
'''
    (PROJECT / "tests" / "test_s1_cost_model.py").write_text(test_code)
    print("  tests/test_s1_cost_model.py written")
    print(f"\nS1-v2 DONE. Gates: A={gates['gate_A']['verdict']} B={gates['gate_B']['verdict']} Spearman={spearman_gate}")
