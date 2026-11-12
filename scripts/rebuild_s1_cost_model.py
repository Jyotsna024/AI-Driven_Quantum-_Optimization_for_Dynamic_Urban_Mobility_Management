"""Rebuild S1 with target-junction plan interventions and time-safe features.

Writes versioned artifacts only. Existing S1 outputs are preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stgnn import STGNNPredictor, JUNCTION_ORDER

RAW = ROOT / "data" / "raw"
STATES = ROOT / "data" / "states"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
NETWORK = ROOT / "sumo_files" / "hsr.net.xml"
SNAP_INDEX = STATES / "snapshot_index.csv"
ROLLOUTS_OUT = ROOT / "data" / "rollouts_s1_v2.csv"
FEATURES_OUT = ROOT / "data" / "features_s1_v2.csv"
MODEL_OUT = MODELS / "cost_model_s1_v2.pkl"
PARAMS_OUT = MODELS / "cost_model_s1_v2.json"
METRICS_OUT = RESULTS / "cost_model_metrics_s1_v2.json"
CHECKPOINT_OUT = ROOT / "data" / "rollouts_s1_v2_checkpoint.csv"
JUNCTIONS = ["J0", "J1", "J2", "J3"]
SCENARIOS = ["low", "medium", "high", "peak", "incident"]
TIMES = [900, 2100, 3300]
PLAN_IDS = [0, 1, 2, 3]
HORIZON_S = 180
FEATURES = [
    "pre_queue_veh", "pre_queue_m", "pre_delay_s", "pre_arrivals",
    "pre_throughput", "pre_speed_mps", "pre_vehicle_count",
    "stgnn_queue_h1", "stgnn_queue_h2", "stgnn_queue_h3",
    "scenario_low", "scenario_medium", "scenario_high", "scenario_peak",
    "scenario_incident",
    "plan_0", "plan_1", "plan_2", "plan_3",
    "junction_J0", "junction_J1", "junction_J2", "junction_J3",
    *[f"plan{p}_junction_{j}" for p in PLAN_IDS for j in JUNCTIONS],
    *[f"plan{p}_queue_h{h}" for p in PLAN_IDS for h in (1, 2, 3)],
    *[f"plan{p}_pre_queue_m" for p in PLAN_IDS],
    *[f"plan{p}_pre_arrivals" for p in PLAN_IDS],
]


def _tls_map() -> dict[str, dict]:
    cfg = json.loads((ROOT / "sumo_files" / "cluster_config.json").read_text(encoding="utf-8"))
    mapping = {}
    for tls_id, meta in cfg["tls_meta"].items():
        short = meta["short_id"]
        if short in JUNCTIONS:
            mapping[short] = {"tls_id": tls_id, "incoming_lanes": []}
    import xml.etree.ElementTree as ET
    net = ET.parse(NETWORK).getroot()
    controlled = {tid: set() for tid in cfg["cluster_tls_ids"]}
    for conn in net.findall(".//connection"):
        tid = conn.get("tl")
        if tid in controlled:
            controlled[tid].add(conn.get("from"))
    edge_lanes = {edge.get("id"): [lane.get("id") for lane in edge.findall("lane")]
                  for edge in net.findall(".//edge")}
    for short, info in mapping.items():
        info["incoming_lanes"] = [lane for edge in controlled[info["tls_id"]]
                                   for lane in edge_lanes.get(edge, [])]
        if not info["incoming_lanes"]:
            raise RuntimeError(f"No incoming lanes resolved for {short}")
    return mapping


def _raw_run(scenario: str, seed: int) -> pd.DataFrame:
    path = RAW / f"{scenario}_s{seed}_varied.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _feature_rows(snapshot_rows: list[dict], predictor: STGNNPredictor) -> pd.DataFrame:
    cache: dict[tuple[str, int], pd.DataFrame] = {}
    rows = []
    for snap in snapshot_rows:
        key = (snap["scenario"], snap["seed"])
        cache.setdefault(key, _raw_run(*key))
        df = cache[key]
        t = snap["snapshot_time_s"]
        history = []
        for ts in (t - 120, t - 60, t):
            part = df[df.timestamp_s == ts].set_index("junction_short")
            if not set(JUNCTIONS).issubset(part.index):
                raise ValueError(f"Missing full junction history at {snap['snapshot_id']} t={ts}")
            arr = []
            for jid in JUNCTIONS:
                r = part.loc[jid]
                v = []
                for col in predictor.feature_cols:
                    x = r[col]
                    if pd.isna(x):
                        x = 14.0 if col == "mean_speed_mps" else 0.0
                    v.append(float(x))
                arr.append(v)
            history.append(arr)
        pred = predictor.predict(np.asarray(history, dtype=np.float32).transpose(1, 0, 2))
        current = df[df.timestamp_s == t].set_index("junction_short")
        for target in JUNCTIONS:
            pre = current.loc[target]
            pre_map = {
                "pre_queue_veh": float(pre["queue_length"]),
                "pre_queue_m": float(pre["queue_length_m"]),
                "pre_delay_s": float(pre["mean_delay_s"]) if pd.notna(pre["mean_delay_s"]) else 0.0,
                "pre_arrivals": float(pre["arrivals"]),
                "pre_throughput": float(pre["throughput"]),
                "pre_speed_mps": float(pre["mean_speed_mps"]) if pd.notna(pre["mean_speed_mps"]) else 0.0,
                "pre_vehicle_count": float(pre["vehicle_count"]),
            }
            for plan in PLAN_IDS:
                r = {
                    "snapshot_id": snap["snapshot_id"], "snapshot_path": snap["snapshot_path"],
                    "scenario": snap["scenario"], "seed": snap["seed"], "split": snap["split"],
                    "snapshot_time_s": t, "target_junction": target,
                    "applied_plan_id": plan, **pre_map,
                    "stgnn_queue_h1": float(pred[JUNCTIONS.index(target), 0]),
                    "stgnn_queue_h2": float(pred[JUNCTIONS.index(target), 1]),
                    "stgnn_queue_h3": float(pred[JUNCTIONS.index(target), 2]),
                }
                for s in SCENARIOS:
                    r[f"scenario_{s}"] = float(snap["scenario"] == s)
                for p in PLAN_IDS:
                    r[f"plan_{p}"] = float(plan == p)
                for j in JUNCTIONS:
                    r[f"junction_{j}"] = float(target == j)
                for p in PLAN_IDS:
                    for j in JUNCTIONS:
                        r[f"plan{p}_junction_{j}"] = float(plan == p and target == j)
                    for h in (1, 2, 3):
                        r[f"plan{p}_queue_h{h}"] = float(plan == p) * r[f"stgnn_queue_h{h}"]
                    r[f"plan{p}_pre_queue_m"] = float(plan == p) * pre_map["pre_queue_m"]
                    r[f"plan{p}_pre_arrivals"] = float(plan == p) * pre_map["pre_arrivals"]
                rows.append(r)
    return pd.DataFrame(rows)


def _select_snapshots() -> list[dict]:
    idx = pd.read_csv(SNAP_INDEX)
    selected = []
    train_time = {0: 900, 1: 2100, 2: 3300}
    for scenario in SCENARIOS:
        for seed, t in train_time.items():
            selected.extend(_select_one(idx, scenario, seed, t, "train"))
        for seed, split in ((3, "val"), (4, "test")):
            for t in TIMES:
                selected.extend(_select_one(idx, scenario, seed, t, split))
    if len(selected) != 45:
        raise RuntimeError(f"Expected 45 stratified snapshots, found {len(selected)}")
    return selected


def _select_one(idx, scenario, seed, t, split) -> list[dict]:
    rows = idx[(idx.scenario == scenario) & (idx.seed == seed) & (idx.sim_time_s == t)]
    if len(rows) != 1:
        raise ValueError(f"Expected one snapshot: {scenario} seed={seed} time={t}; got {len(rows)}")
    path = Path(rows.iloc[0]["path"]).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return [{"snapshot_id": f"{scenario}_s{seed}_t{t}", "snapshot_path": str(path),
             "scenario": scenario, "seed": int(seed), "split": split,
             "snapshot_time_s": int(t)}]


def _run_one(task: dict) -> list[dict]:
    """Process worker: clone one state, change exactly one TLS, measure all TLS."""
    import traci
    mapping = _tls_map()
    sumo_home = Path(os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\SUMO"))
    sumo = sumo_home / "bin" / "sumo.exe"
    route = ROOT / "sumo_files" / "routes" / f"{task['scenario']}_s{task['seed']}.rou.xml"
    port = 24000 + (os.getpid() % 8000)
    label = f"s1v2_{os.getpid()}_{task['target_junction']}_{task['applied_plan_id']}"
    cmd = [str(sumo), "-n", str(NETWORK), "-r", str(route), "--load-state", task["snapshot_path"],
           "--seed", str(task["seed"] + 100), "--no-step-log", "true", "--no-warnings", "true",
           "--time-to-teleport", "300"]
    conn = None
    try:
        traci.start(cmd, port=port, label=label)
        conn = traci.getConnection(label)
        for jid, info in mapping.items():
            pid = task["applied_plan_id"] if jid == task["target_junction"] else 0
            conn.trafficlight.setProgram(info["tls_id"], str(pid))
            actual = conn.trafficlight.getProgram(info["tls_id"])
            if str(actual) != str(pid):
                raise RuntimeError(f"TLS {jid} program mismatch {actual} != {pid}")

        metrics = {j: {"delay": [], "queue_m": [], "queue_veh": [],
                       "throughput": 0, "arrivals": 0} for j in JUNCTIONS}
        previous = {j: set() for j in JUNCTIONS}
        for step in range(HORIZON_S):
            conn.simulationStep()
            if (step + 1) % 60:
                continue
            for jid, info in mapping.items():
                curr, waits = set(), []
                queue_m = 0.0
                queue_veh = 0
                for lane in info["incoming_lanes"]:
                    vids = conn.lane.getLastStepVehicleIDs(lane)
                    curr.update(vids)
                    for vid in vids:
                        waits.append(float(conn.vehicle.getWaitingTime(vid)))
                        if conn.vehicle.getSpeed(vid) < 0.1:
                            queue_veh += 1
                            queue_m += float(conn.vehicle.getLength(vid)) + 2.5
                metrics[jid]["arrivals"] += len(curr - previous[jid])
                metrics[jid]["throughput"] += len(previous[jid] - curr)
                previous[jid] = curr
                metrics[jid]["delay"].append(float(np.mean(waits)) if waits else 0.0)
                metrics[jid]["queue_m"].append(queue_m)
                metrics[jid]["queue_veh"].append(queue_veh)
        out = []
        for measured in JUNCTIONS:
            m = metrics[measured]
            out.append({
                **{k: task[k] for k in ("snapshot_id", "snapshot_path", "scenario", "seed", "split",
                                        "snapshot_time_s", "target_junction", "applied_plan_id")},
                "measured_junction": measured, "sumo_seed": task["seed"] + 100,
                "other_junctions_baseline": True, "rollout_duration_s": HORIZON_S,
                "status": "COMPLETED", "mean_delay_s": float(np.mean(m["delay"])),
                "mean_queue_m": float(np.mean(m["queue_m"])),
                "mean_queue_veh": float(np.mean(m["queue_veh"])),
                "throughput": int(m["throughput"]), "arrivals": int(m["arrivals"]),
            })
        return out
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _evaluate(df: pd.DataFrame, pred: np.ndarray) -> dict:
    d = df.copy()
    d["prediction"] = pred
    y = d["mean_delay_s"].to_numpy()
    metrics = {
        "n": int(len(d)),
        "mae": float(np.mean(np.abs(pred - y))),
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "spearman": _safe_spearman(pred, y),
    }
    top1_total = top1_correct = 0
    snap_rhos = []
    per_snapshot = []
    for (sid, jid), g in d.groupby(["snapshot_id", "target_junction"], sort=True):
        if set(g.applied_plan_id) != set(PLAN_IDS) or len(g) != 4:
            continue
        true_plan = int(g.loc[g.mean_delay_s.idxmin(), "applied_plan_id"])
        pred_plan = int(g.loc[g.prediction.idxmin(), "applied_plan_id"])
        rho = spearmanr(g.mean_delay_s, g.prediction).statistic
        snap_rhos.append(float(rho))
        top1_correct += int(pred_plan == true_plan)
        top1_total += 1
        per_snapshot.append({"snapshot_id": sid, "target_junction": jid,
                             "true_best_plan": true_plan, "predicted_best_plan": pred_plan,
                             "top1_correct": pred_plan == true_plan,
                             "spearman": float(rho)})
    metrics.update(_ranking_metrics(d))
    metrics["per_snapshot"] = per_snapshot
    per_junction = {}
    for jid, g in d.groupby("target_junction", sort=True):
        m = _evaluate_basic(g, g.prediction.to_numpy())
        m["ranking"] = _ranking_metrics(g)
        per_junction[jid] = m
    metrics["per_junction"] = per_junction
    return metrics


def _evaluate_basic(g, pred) -> dict:
    y = g.mean_delay_s.to_numpy()
    return {"n": int(len(g)), "mae": float(np.mean(np.abs(pred-y))),
            "rmse": float(np.sqrt(np.mean((pred-y)**2))),
            "spearman": _safe_spearman(pred, y)}


def _safe_spearman(a, b):
    value = float(spearmanr(a, b).statistic)
    return value if np.isfinite(value) else None


def _ranking_metrics(df):
    correct = total = 0
    rhos = []
    per_snapshot = []
    for (sid, jid), g in df.groupby(["snapshot_id", "target_junction"], sort=True):
        if set(g.applied_plan_id) != set(PLAN_IDS) or len(g) != 4:
            continue
        true_plan = int(g.loc[g.mean_delay_s.idxmin(), "applied_plan_id"])
        pred_plan = int(g.loc[g.prediction.idxmin(), "applied_plan_id"])
        rho = _safe_spearman(g.mean_delay_s, g.prediction)
        if rho is not None:
            rhos.append(rho)
        correct += int(pred_plan == true_plan)
        total += 1
        per_snapshot.append({"snapshot_id": sid, "target_junction": jid,
                             "true_best_plan": true_plan, "predicted_best_plan": pred_plan,
                             "top1_correct": pred_plan == true_plan, "spearman": rho})
    return {"per_snapshot_mean_spearman": float(np.mean(rhos)) if rhos else None,
            "top1_correct": correct, "top1_total": total,
            "top1_agreement": correct / total if total else None,
            "per_snapshot": per_snapshot}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(6, max(1, (os.cpu_count() or 4) - 2)))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for path in (ROLLOUTS_OUT, FEATURES_OUT, MODEL_OUT, PARAMS_OUT, METRICS_OUT):
        if path.exists() and not args.resume:
            raise FileExistsError(f"Refusing to overwrite versioned artifact {path}; pass --resume or archive it first")
    MODELS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    start = time.time()
    snapshots = _select_snapshots()
    predictor = STGNNPredictor()
    features = _feature_rows(snapshots, predictor)
    tasks = []
    for snap in snapshots:
        for target in JUNCTIONS:
            for plan in PLAN_IDS:
                tasks.append({**snap, "target_junction": target, "applied_plan_id": plan})
    completed = set()
    existing = []
    if args.resume and CHECKPOINT_OUT.exists():
        prior = pd.read_csv(CHECKPOINT_OUT)
        existing = prior[prior.status == "COMPLETED"].to_dict("records")
        completed = {(r["snapshot_id"], r["target_junction"], int(r["applied_plan_id"]))
                     for r in existing if r.get("status") == "COMPLETED"}
    tasks = [t for t in tasks if (t["snapshot_id"], t["target_junction"], t["applied_plan_id"]) not in completed]
    print(f"S1 v2: {len(snapshots)} snapshots, 16 interventions each, {len(tasks)} remaining; workers={args.workers}")
    records = list(existing)
    done = len(completed)
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_one, t): t for t in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                records.extend(future.result())
            except Exception as e:
                failures.append({**task, "status": "FAILED", "error": repr(e)})
            done += 1
            if done % 20 == 0 or done == 1:
                pd.DataFrame(records).to_csv(CHECKPOINT_OUT, index=False)
                print(f"  completed {done}/{45*16} intervention rollouts")
    rollouts = pd.DataFrame(records)
    rollouts.to_csv(CHECKPOINT_OUT, index=False)
    if failures:
        pd.DataFrame(failures).to_csv(ROOT / "data" / "rollouts_s1_v2_failures.csv", index=False)
        raise RuntimeError(f"{len(failures)} intervention rollouts failed; inspect data/rollouts_s1_v2_failures.csv and rerun with --resume")
    rollouts.to_csv(ROLLOUTS_OUT, index=False)

    outcomes = rollouts[rollouts.measured_junction == rollouts.target_junction].copy()
    if len(outcomes) != len(features):
        raise RuntimeError(f"Expected {len(features)} target outcomes, found {len(outcomes)}")
    feature_key = ["snapshot_id", "target_junction", "applied_plan_id"]
    model_df = features.merge(outcomes[feature_key + ["measured_junction", "mean_delay_s", "mean_queue_m", "throughput"]],
                              on=feature_key, validate="one_to_one")
    model_df.to_csv(FEATURES_OUT, index=False)
    train = model_df[model_df.split == "train"].copy()
    val = model_df[model_df.split == "val"].copy()
    test = model_df[model_df.split == "test"].copy()
    if set(train.seed.unique()) != {0, 1, 2} or set(val.seed.unique()) != {3} or set(test.seed.unique()) != {4}:
        raise RuntimeError("Unexpected seed split in rebuilt S1 data")

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[FEATURES])
    x_val = scaler.transform(val[FEATURES])
    x_test = scaler.transform(test[FEATURES])
    candidates = []
    for alpha in (0.1, 1.0, 10.0, 50.0, 100.0):
        model = Ridge(alpha=alpha).fit(x_train, train.mean_delay_s)
        val_metrics = _evaluate(val, np.clip(model.predict(x_val), 0.0, None))
        candidates.append((val_metrics["top1_agreement"], -val_metrics["mae"], alpha, model, val_metrics))
    _, _, alpha, model, val_metrics = max(candidates, key=lambda q: (q[0], q[1], -q[2]))
    test_metrics = _evaluate(test, np.clip(model.predict(x_test), 0.0, None))
    model_payload = {"model": model, "scaler": scaler, "feature_cols": FEATURES,
                     "alpha": alpha, "train_seeds": [0, 1, 2], "val_seed": 3, "test_seed": 4,
                     "target": "mean_delay_s at measured_junction == target_junction"}
    with MODEL_OUT.open("wb") as f:
        pickle.dump(model_payload, f)
    params = {"model_type": "Ridge", "alpha": alpha, "feature_cols": FEATURES,
              "coef": model.coef_.tolist(), "intercept": float(model.intercept_),
              "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
              "seed_partition": {"train": [0, 1, 2], "validation": [3], "test": [4]},
              "target_definition": model_payload["target"],
              "feature_leakage_audit": "Predictors are pre-decision telemetry and ST-GNN forecasts only; rollout outcomes excluded."}
    PARAMS_OUT.write_text(json.dumps(params, indent=2), encoding="utf-8")
    metrics = {"stage": "S1 v2", "status": "COMPLETE", "n_snapshots": len(snapshots),
               "n_interventions": len(tasks) + len(completed), "n_outcome_rows": len(rollouts),
               "train_seeds": sorted(train.seed.unique().tolist()), "validation_seeds": sorted(val.seed.unique().tolist()),
               "test_seeds": sorted(test.seed.unique().tolist()), "selected_alpha": alpha,
               "selected_on": "seed 3 top-1 agreement, then MAE", "feature_columns": FEATURES,
               "feature_leakage_audit": "Predictors use only telemetry through snapshot time and ST-GNN forecasts from the preceding three intervals.",
               "train": _evaluate(train, np.clip(model.predict(x_train), 0.0, None)), "validation": val_metrics,
               "test": test_metrics, "elapsed_s": round(time.time()-start, 2),
               "artifacts": {"rollouts": str(ROLLOUTS_OUT.relative_to(ROOT)),
                             "features": str(FEATURES_OUT.relative_to(ROOT)),
                             "model": str(MODEL_OUT.relative_to(ROOT)),
                             "params": str(PARAMS_OUT.relative_to(ROOT))}}
    METRICS_OUT.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    print(f"S1 v2 complete: val top1={val_metrics['top1_agreement']}, test top1={test_metrics['top1_agreement']}, elapsed={metrics['elapsed_s']}s")


if __name__ == "__main__":
    main()
