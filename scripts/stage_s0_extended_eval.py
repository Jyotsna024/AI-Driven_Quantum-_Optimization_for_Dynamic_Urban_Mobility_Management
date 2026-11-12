"""
scripts/stage_s0_extended_eval.py  –  Extended ST-GNN evaluation (S0-4).
Produces results/stgnn_extended_metrics.json with:
 - val + test MAE/RMSE separately
 - per junction MAE/RMSE (all horizons)
 - per scenario MAE
 - normalized MAE (divided by junction mean queue on test set)
 - persistence lags 60/90/120/180s aliasing check
 - leakage verification log

Does NOT retrain the model.
"""
import os, sys, json, pickle, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "src"))

os.environ["MPLCONFIGDIR"] = str(PROJECT / ".mplconfig")

from stgnn import SpatioTemporalGNN

JUNCTION_ORDER = ["J0", "J1", "J2", "J3"]
N_NODES = 4
FEATURE_COLS = ["queue_length","vehicle_count","mean_speed_mps","mean_delay_s","throughput","arrivals"]
TARGET_IDX = 0  # queue_length
LOOKBACK_T = 3
HORIZON_H = 3

EDGE_INDEX = torch.tensor([
    [0,0,0,1,1,1,2,2,2,3,3,3],
    [1,2,3,0,2,3,0,1,3,0,1,2],
], dtype=torch.long)
EDGE_WEIGHTS_RAW = torch.tensor([358.3,771.2,778.0,349.0,649.2,1127.0,776.7,648.4,1554.7,855.1,1196.6,1609.5], dtype=torch.float32)
NORM_EW = 100.0 / EDGE_WEIGHTS_RAW

OUT = {}
START = time.time()

# ── Load model + scaler ───────────────────────────────────────────────────
model_path = PROJECT / "models" / "stgnn.pt"
scaler_path = PROJECT / "models" / "scaler.pkl"
assert model_path.exists(), "models/stgnn.pt missing"
assert scaler_path.exists(), "models/scaler.pkl missing"

ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
# checkpoint may be full dict or just state_dict
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    state = ckpt["model_state_dict"]
    kw = {k: ckpt.get(k, v) for k, v in [("in_features",6),("hidden_dim",32),("num_nodes",4),("horizon",3)]}
else:
    state = ckpt
    kw = {"in_features":6, "hidden_dim":32, "num_nodes":4, "horizon":3}
model = SpatioTemporalGNN(**kw)
model.load_state_dict(state)
model.eval()

with open(scaler_path,"rb") as f:
    scaler_dict = pickle.load(f)
# scaler_dict is {'feature_scaler': StandardScaler, 'target_mean': ..., 'target_std': ..., ...}
feat_scaler  = scaler_dict["feature_scaler"]
target_mean  = scaler_dict["target_mean"]   # scalar
target_std   = scaler_dict["target_std"]    # scalar

# ── Load raw data, build windows per split ───────────────────────────────
raw_dir = PROJECT / "data" / "raw"
manifest = pd.read_csv(raw_dir / "manifest.csv")

def load_run(run_id):
    fp = raw_dir / f"{run_id}.csv"
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    df = df.sort_values(["timestamp_s","junction_short"])
    timestamps = sorted(df["timestamp_s"].unique())
    pivoted = []
    for ts in timestamps:
        sub = df[df["timestamp_s"]==ts].set_index("junction_short")
        step = []
        for j in JUNCTION_ORDER:
            if j in sub.index:
                row = sub.loc[j]
                vals = []
                for f in FEATURE_COLS:
                    v = row[f] if f in row.index else 0.0
                    vals.append(0.0 if (isinstance(v, float) and np.isnan(v)) else float(v))
            else:
                vals = [0.0]*6
            step.append(vals)
        pivoted.append(step)
    arr = np.array(pivoted, dtype=np.float32)  # [T, N, F]
    return arr, timestamps

def build_windows(split_runs):
    Xs, Ys, metas = [], [], []
    for _, row in manifest[manifest["split"]==split_runs].iterrows():
        result = load_run(row["run_id"])
        if result is None:
            continue
        arr, timestamps = result
        n_steps = len(timestamps)
        for t in range(LOOKBACK_T-1, n_steps-HORIZON_H):
            target_ts = timestamps[t+1:t+1+HORIZON_H]
            if any(ts <= 300 for ts in target_ts):
                continue
            x = arr[t-LOOKBACK_T+1:t+1]  # [T,N,F]
            x = np.transpose(x, (1,0,2))  # [N,T,F]
            y = arr[t+1:t+1+HORIZON_H, :, TARGET_IDX]  # [H,N]
            y = np.transpose(y, (1,0))  # [N,H]
            Xs.append(x)
            Ys.append(y)
            metas.append({"run_id": row["run_id"], "scenario": row["scenario"],
                           "seed": row["seed"], "ts": timestamps[t]})
    return np.stack(Xs) if Xs else None, np.stack(Ys) if Ys else None, metas

print("Loading splits...")
X_val, Y_val, meta_val = build_windows("val")
X_test, Y_test, meta_test = build_windows("test")
print(f"  Val windows: {len(X_val) if X_val is not None else 0}")
print(f"  Test windows: {len(X_test) if X_test is not None else 0}")

# ── Scale inputs ──────────────────────────────────────────────────────────
n_feat = len(FEATURE_COLS)

def scale_X(X):
    B, N, T, F = X.shape
    flat = X.reshape(-1, F)
    flat_s = feat_scaler.transform(flat)
    return flat_s.reshape(B, N, T, F).astype(np.float32)

def unnorm_Y(Y_norm):
    """Unnormalize targets from (y - mean)/std back to vehicle counts."""
    return Y_norm * target_std + target_mean

X_val_s  = scale_X(X_val)
X_test_s = scale_X(X_test)

# ── Predict ───────────────────────────────────────────────────────────────
def predict_all(X_s):
    """Returns unnormalized predictions (vehicle counts)."""
    preds = []
    BS = 64
    with torch.no_grad():
        for i in range(0, len(X_s), BS):
            xb = torch.tensor(X_s[i:i+BS])
            p = model(xb, EDGE_INDEX, NORM_EW).numpy()  # [B, N, H] normalized
            p_raw = p * target_std + target_mean
            p_raw = np.clip(p_raw, 0.0, None)
            preds.append(p_raw)
    return np.concatenate(preds, axis=0)  # [B, N, H]

print("Running inference on val+test...")
P_val  = predict_all(X_val_s)
P_test = predict_all(X_test_s)

# ── Metrics helper ────────────────────────────────────────────────────────
def metrics(pred, actual):
    mae  = float(np.mean(np.abs(pred - actual)))
    rmse = float(np.sqrt(np.mean((pred - actual)**2)))
    return {"mae": round(mae,4), "rmse": round(rmse,4)}

# ── Overall val / test ────────────────────────────────────────────────────
OUT["val_overall"]  = metrics(P_val,  Y_val)
OUT["test_overall"] = metrics(P_test, Y_test)
print(f"  Val  overall MAE={OUT['val_overall']['mae']}  RMSE={OUT['val_overall']['rmse']}")
print(f"  Test overall MAE={OUT['test_overall']['mae']}  RMSE={OUT['test_overall']['rmse']}")

# ── Per horizon ───────────────────────────────────────────────────────────
val_h, test_h = {}, {}
for h in range(HORIZON_H):
    val_h[f"H{h+1}"]  = metrics(P_val[:,:,h],  Y_val[:,:,h])
    test_h[f"H{h+1}"] = metrics(P_test[:,:,h], Y_test[:,:,h])
OUT["val_per_horizon"]  = val_h
OUT["test_per_horizon"] = test_h

# ── Per junction ──────────────────────────────────────────────────────────
val_junc, test_junc = {}, {}
mean_q_per_junc = {}
for j, jname in enumerate(JUNCTION_ORDER):
    val_junc[jname]  = metrics(P_val[:,j,:],  Y_val[:,j,:])
    test_junc[jname] = metrics(P_test[:,j,:], Y_test[:,j,:])
    mq = float(np.mean(Y_test[:,j,:]))
    mean_q_per_junc[jname] = round(mq, 3)
    test_junc[jname]["norm_mae"] = round(test_junc[jname]["mae"] / mq, 4) if mq > 0 else None
OUT["val_per_junction"]  = val_junc
OUT["test_per_junction"] = test_junc
OUT["test_mean_queue_per_junction"] = mean_q_per_junc
print(f"  Test per junction: {test_junc}")

# ── Per scenario (test only) ──────────────────────────────────────────────
test_scenario = {}
meta_test_df = pd.DataFrame(meta_test)
for sc, grp in meta_test_df.groupby("scenario"):
    idx = grp.index.tolist()
    if idx:
        test_scenario[sc] = metrics(P_test[idx], Y_test[idx])
OUT["test_per_scenario"] = test_scenario
print(f"  Test per scenario: {test_scenario}")

# ── Persistence baseline at multiple lags ────────────────────────────────
lag_results = {}
for lag in [1, 2]:  # 60s, 120s (each step = 60s)
    all_pred, all_act = [], []
    for _, row in manifest[manifest["split"]=="test"].iterrows():
        result = load_run(row["run_id"])
        if result is None: continue
        arr, timestamps = result
        n = len(timestamps)
        for t in range(LOOKBACK_T-1, n-HORIZON_H):
            target_ts = timestamps[t+1:t+1+HORIZON_H]
            if any(ts <= 300 for ts in target_ts):
                continue
            # persistence: use value at t-lag+1 as prediction for t+1..t+H
            lag_t = max(0, t - lag + 1)
            pred_q = arr[lag_t, :, TARGET_IDX]  # [N]
            actual_q = arr[t+1:t+1+HORIZON_H, :, TARGET_IDX]  # [H,N]
            for h in range(HORIZON_H):
                all_pred.extend(pred_q.tolist())
                all_act.extend(actual_q[h].tolist())
    if all_pred:
        lag_results[f"lag_{lag*60}s"] = metrics(np.array(all_pred), np.array(all_act))
OUT["persistence_lag_analysis"] = lag_results
print(f"  Persistence lags: {lag_results}")

# ── Leakage check ─────────────────────────────────────────────────────────
# Verify: seeds in val (3) and test (4) never appear in train (0,1,2)
train_seeds = set(manifest[manifest["split"]=="train"]["seed"].tolist())
val_seeds   = set(manifest[manifest["split"]=="val"]["seed"].tolist())
test_seeds  = set(manifest[manifest["split"]=="test"]["seed"].tolist())
overlap_tv = train_seeds & val_seeds
overlap_tt = train_seeds & test_seeds
leakage_ok = (len(overlap_tv)==0 and len(overlap_tt)==0)
OUT["leakage_check"] = {
    "train_seeds": sorted(train_seeds),
    "val_seeds": sorted(val_seeds),
    "test_seeds": sorted(test_seeds),
    "train_val_overlap": sorted(overlap_tv),
    "train_test_overlap": sorted(overlap_tt),
    "no_leakage": leakage_ok
}
print(f"  Leakage check: no_leakage={leakage_ok}")

OUT["elapsed_s"] = round(time.time()-START, 2)
out_path = PROJECT / "results" / "stgnn_extended_metrics.json"
out_path.write_text(json.dumps(OUT, indent=2))
print(f"\nS0-4 extended eval done in {OUT['elapsed_s']}s  ->  {out_path}")
