"""Build a train-only QUBO with rollout-supported pair interactions.

Versioned S2 artifacts only; does not replace legacy QUBO outputs.
"""
from __future__ import annotations

import argparse
import itertools
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from rebuild_s1_cost_model import JUNCTIONS, PLAN_IDS, HORIZON_S, _tls_map

STATES = ROOT / "data" / "states"
S1_ROLLOUTS = ROOT / "data" / "rollouts_s1_v2.csv"
S1_FEATURES = ROOT / "data" / "features_s1_v2.csv"
S1_MODEL = ROOT / "models" / "cost_model_s1_v2.pkl"
SNAP_INDEX = STATES / "snapshot_index.csv"
NETWORK = ROOT / "sumo_files" / "hsr.net.xml"
JOINTS_OUT = ROOT / "data" / "joint_rollouts_s2_v3.csv"
CHECKPOINT_OUT = ROOT / "data" / "joint_rollouts_s2_v3_checkpoint.csv"
FAIL_OUT = ROOT / "data" / "joint_rollouts_s2_v3_failures.csv"
QUBO_OUT = ROOT / "models" / "qubo_s2_v3.npz"
META_OUT = ROOT / "models" / "qubo_s2_v3.json"
METRICS_OUT = ROOT / "results" / "qubo_metrics_s2_v3.json"
TRAIN_PAIR_SNAPSHOTS = sorted(
    pd.read_csv(S1_FEATURES, usecols=["snapshot_id", "split"])
      .query("split == 'train'")["snapshot_id"].unique().tolist()
)
VALIDATION_SNAPSHOTS = ["high_s3_t2100", "incident_s3_t2100"]
TEST_SNAPSHOTS = ["peak_s4_t2100", "incident_s4_t2100"]
PAIRS = list(itertools.combinations(range(4), 2))
ALL_PLANS = list(itertools.product(PLAN_IDS, repeat=4))
N = 16


def _snapshot_lookup() -> dict[str, dict]:
    idx = pd.read_csv(SNAP_INDEX)
    result = {}
    for _, row in idx.iterrows():
        sid = f"{row.scenario}_s{int(row.seed)}_t{int(row.sim_time_s)}"
        path = Path(row.path).resolve()
        result[sid] = {"snapshot_id": sid, "snapshot_path": str(path),
                       "scenario": str(row.scenario), "seed": int(row.seed),
                       "snapshot_time_s": int(row.sim_time_s)}
    return result


def _canonical_vector(value) -> str:
    parsed = json.loads(value) if isinstance(value, str) else value
    return json.dumps([int(x) for x in parsed], separators=(",", ":"))


def _make_tasks() -> list[dict]:
    snapshots = _snapshot_lookup()
    tasks = []
    for sid in TRAIN_PAIR_SNAPSHOTS:
        snap = snapshots[sid]
        for i, j in PAIRS:
            for pi in (1, 2, 3):
                for pj in (1, 2, 3):
                    vector = [0, 0, 0, 0]
                    vector[i], vector[j] = pi, pj
                    tasks.append({**snap, "split": "train", "experiment": "pair_train",
                                  "pair_i": JUNCTIONS[i], "pair_j": JUNCTIONS[j],
                                  "plan_vector": vector})
    for sid, split in [(s, "val") for s in VALIDATION_SNAPSHOTS] + [(s, "test") for s in TEST_SNAPSHOTS]:
        snap = snapshots[sid]
        for vector in ALL_PLANS:
            tasks.append({**snap, "split": split, "experiment": "full_joint_validation",
                          "pair_i": "", "pair_j": "", "plan_vector": list(vector)})
    ids = [(t["snapshot_id"], tuple(t["plan_vector"])) for t in tasks]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate S2 intervention tasks")
    return tasks


def _run_vector(task: dict) -> list[dict]:
    import traci
    mapping = _tls_map()
    sumo_home = Path(os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\SUMO"))
    sumo = sumo_home / "bin" / "sumo.exe"
    route = ROOT / "sumo_files" / "routes" / f"{task['scenario']}_s{task['seed']}.rou.xml"
    port = 32000 + (os.getpid() % 7000)
    vector = task["plan_vector"]
    run_id = task["experiment"] + "__" + task["snapshot_id"] + "__" + "".join(map(str, vector))
    label = f"s2v3_{os.getpid()}_{run_id[-20:]}"
    cmd = [str(sumo), "-n", str(NETWORK), "-r", str(route), "--load-state", task["snapshot_path"],
           "--seed", str(task["seed"] + 100), "--no-step-log", "true", "--no-warnings", "true",
           "--time-to-teleport", "300"]
    conn = None
    try:
        traci.start(cmd, port=port, label=label)
        conn = traci.getConnection(label)
        for jid, info in mapping.items():
            pid = int(vector[JUNCTIONS.index(jid)])
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
        result = []
        for measured in JUNCTIONS:
            m = metrics[measured]
            result.append({
                "run_id": run_id, "snapshot_id": task["snapshot_id"],
                "snapshot_path": task["snapshot_path"], "snapshot_time_s": task["snapshot_time_s"],
                "scenario": task["scenario"], "seed": task["seed"], "split": task["split"],
                "experiment": task["experiment"], "plan_vector": json.dumps(vector),
                **{f"plan_{j}": int(vector[n]) for n, j in enumerate(JUNCTIONS)},
                "pair_i": task["pair_i"], "pair_j": task["pair_j"],
                "measured_junction": measured, "sumo_seed": task["seed"] + 100,
                "rollout_duration_s": HORIZON_S, "status": "COMPLETED",
                "mean_delay_s": float(np.mean(m["delay"])),
                "mean_queue_m": float(np.mean(m["queue_m"])),
                "mean_queue_veh": float(np.mean(m["queue_veh"])),
                "throughput": int(m["throughput"]), "arrivals": int(m["arrivals"]),
            })
        return result
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _load_costs() -> tuple[np.ndarray, float, dict, pd.DataFrame]:
    with S1_MODEL.open("rb") as f:
        saved = pickle.load(f)
    feature_cols = saved["feature_cols"]
    model, scaler = saved["model"], saved["scaler"]
    feat = pd.read_csv(S1_FEATURES)
    train = feat[feat.split == "train"].copy()
    if set(train.seed.unique()) != {0, 1, 2}:
        raise RuntimeError("S2 cost inputs are not restricted to training seeds")
    pred = np.clip(model.predict(scaler.transform(train[feature_cols])), 0.0, None)
    train["predicted_delay_s"] = pred
    cost = np.zeros((4, 4), dtype=float)
    for i, jid in enumerate(JUNCTIONS):
        for p in PLAN_IDS:
            block = train[(train.target_junction == jid) & (train.applied_plan_id == p)]
            if block.empty:
                raise RuntimeError(f"No train cost rows for {jid} plan {p}")
            cost[i, p] = float(block.predicted_delay_s.mean())
    return cost, float(saved["alpha"]), saved, train


def _total_by_action(s1: pd.DataFrame, snapshot_id: str, target: str, plan: int) -> float:
    block = s1[(s1.snapshot_id == snapshot_id) & (s1.target_junction == target) &
               (s1.applied_plan_id == plan)]
    if len(block) != 4 or block.measured_junction.nunique() != 4:
        raise RuntimeError(f"Missing S1 single rollout for {snapshot_id} {target} plan {plan}")
    return float(block.mean_delay_s.sum())


def _pair_interactions(s1: pd.DataFrame, joint: pd.DataFrame) -> tuple[np.ndarray, dict]:
    pairwise = np.zeros((len(PAIRS), 4, 4), dtype=float)
    support = {}
    for pair_idx, (i, j) in enumerate(PAIRS):
        pair_i, pair_j = JUNCTIONS[i], JUNCTIONS[j]
        vals = {(pi, pj): [] for pi in PLAN_IDS for pj in PLAN_IDS}
        for sid in TRAIN_PAIR_SNAPSHOTS:
            d0 = np.mean([_total_by_action(s1, sid, jid, 0) for jid in JUNCTIONS])
            for pi in PLAN_IDS:
                di = _total_by_action(s1, sid, pair_i, pi)
                for pj in PLAN_IDS:
                    dj = _total_by_action(s1, sid, pair_j, pj)
                    if pi == 0 or pj == 0:
                        # A baseline plan is the reference intervention, so its pair correction is zero.
                        vals[(pi, pj)].append(0.0)
                        continue
                    vector = [0, 0, 0, 0]
                    vector[i], vector[j] = pi, pj
                    block = joint[(joint.snapshot_id == sid) &
                                  (joint.plan_vector == _canonical_vector(vector)) &
                                  (joint.measured_junction.isin(JUNCTIONS))]
                    if len(block) != 4:
                        raise RuntimeError(f"Missing pair rollout rows for {sid} {pair_i}:{pi} {pair_j}:{pj}")
                    dij = float(block.mean_delay_s.sum())
                    vals[(pi, pj)].append(dij - di - dj + d0)
        for (pi, pj), samples in vals.items():
            pairwise[pair_idx, pi, pj] = float(np.mean(samples))
            support[f"{pair_i}:{pi}|{pair_j}:{pj}"] = {
                "n_train_snapshots": len(samples), "mean_interaction_s": pairwise[pair_idx, pi, pj],
                "source": "matched joint SUMO rollouts" if pi and pj else "zero reference (one plan at baseline)"}
    return pairwise, support


def _base_objective(plan_matrix: np.ndarray, c: np.ndarray, pairwise: np.ndarray) -> np.ndarray:
    x = np.asarray(plan_matrix, dtype=int)
    out = np.zeros(len(x), dtype=float)
    for i in range(4):
        out += c[i, x[:, i]]
    for q, (i, j) in enumerate(PAIRS):
        out += pairwise[q, x[:, i], x[:, j]]
    return out


def _build_qubo(c: np.ndarray, pairwise: np.ndarray) -> tuple[np.ndarray, float, dict]:
    plans = np.array(list(itertools.product(PLAN_IDS, repeat=4)), dtype=int)
    base = _base_objective(plans, c, pairwise)
    # Bound infeasible bitstrings too: their selected linear/pair terms can
    # exceed the range seen among the 256 one-hot plan vectors.
    bits = ((np.arange(1 << N, dtype=np.uint32)[:, None] >> np.arange(N, dtype=np.uint32)) & 1).astype(float)
    grouped = bits.reshape(-1, 4, 4).sum(axis=2)
    feasible = (grouped == 1).all(axis=1)
    raw_base = np.zeros(len(bits), dtype=float)
    for i in range(4):
        for p in PLAN_IDS:
            raw_base += bits[:, i * 4 + p] * c[i, p]
    for qidx, (i, j) in enumerate(PAIRS):
        for pi in PLAN_IDS:
            for pj in PLAN_IDS:
                raw_base += bits[:, i * 4 + pi] * bits[:, j * 4 + pj] * pairwise[qidx, pi, pj]
    base_bounds = {
        "feasible_min": float(raw_base[feasible].min()),
        "feasible_max": float(raw_base[feasible].max()),
        "infeasible_min": float(raw_base[~feasible].min()),
        "infeasible_max": float(raw_base[~feasible].max()),
    }
    penalty = float(max(1.0, base_bounds["feasible_max"] - base_bounds["infeasible_min"] + 1.0))
    q = np.zeros((N, N), dtype=float)
    for i in range(4):
        for p in PLAN_IDS:
            v = i * 4 + p
            q[v, v] = c[i, p] - penalty
        for p in PLAN_IDS:
            for other in range(p + 1, 4):
                q[i * 4 + p, i * 4 + other] = 2.0 * penalty
    for pair_idx, (i, j) in enumerate(PAIRS):
        for pi in PLAN_IDS:
            for pj in PLAN_IDS:
                q[i * 4 + pi, j * 4 + pj] = pairwise[pair_idx, pi, pj]
    base_bounds["onehot_base_range"] = float(base.max() - base.min())
    return q, penalty, base_bounds


def _bruteforce(q: np.ndarray, c: np.ndarray, pairwise: np.ndarray,
                penalty: float, offset: float) -> dict:
    bitstrings = ((np.arange(1 << N, dtype=np.uint32)[:, None] >> np.arange(N, dtype=np.uint32)) & 1).astype(float)
    energies = np.einsum("bi,ij,bj->b", bitstrings, q, bitstrings, optimize=True)
    grouped = bitstrings.reshape(-1, 4, 4).sum(axis=2)
    feasible_mask = (grouped == 1).all(axis=1)
    feasible_idx = np.flatnonzero(feasible_mask)
    infeasible_idx = np.flatnonzero(~feasible_mask)
    best_idx = int(feasible_idx[np.argmin(energies[feasible_mask])])
    best_matrix = bitstrings[best_idx].reshape(4, 4)
    best_plans = best_matrix.argmax(axis=1).astype(int).tolist()
    best_base = float(_base_objective(np.array([best_plans]), c, pairwise)[0])
    exact_energy = float(energies[best_idx])
    min_infeasible = float(energies[infeasible_idx].min())
    max_feasible = float(energies[feasible_idx].max())
    return {
        "n_bitstrings": int(1 << N), "n_feasible": int(feasible_mask.sum()),
        "n_infeasible": int((~feasible_mask).sum()), "best_x": bitstrings[best_idx].astype(int).tolist(),
        "best_plans": best_plans, "best_energy": exact_energy,
        "best_base_objective": best_base, "best_predicted_delay_s": best_base + offset,
        "max_feasible_energy": max_feasible, "min_infeasible_energy": min_infeasible,
        "penalty_gap_s": min_infeasible - max_feasible,
        "penalty_gap_ok": bool(min_infeasible > max_feasible),
        "one_hot_verified_all_feasible": bool(np.all(grouped[feasible_mask] == 1)),
    }


def _safe_spearman(a, b):
    v = float(spearmanr(a, b).statistic)
    return v if np.isfinite(v) else None


def _validate_snapshots(joint: pd.DataFrame, q: np.ndarray, penalty: float,
                        offset: float, exact: dict, *, s1_saved: dict,
                        s1_features: pd.DataFrame, pairwise: np.ndarray) -> dict:
    out = {}
    for sid, block in joint[joint.experiment == "full_joint_validation"].groupby("snapshot_id", sort=True):
        state_rows = s1_features[(s1_features.snapshot_id == sid) &
                                 (s1_features.target_junction.isin(JUNCTIONS))]
        if len(state_rows) != 16:
            raise RuntimeError(f"Expected 16 pre-decision S1 feature rows for {sid}, got {len(state_rows)}")
        state_rows = state_rows.copy()
        cols = s1_saved["feature_cols"]
        pred = np.clip(s1_saved["model"].predict(s1_saved["scaler"].transform(state_rows[cols])), 0.0, None)
        c_state = np.zeros((4, 4), dtype=float)
        for i, jid in enumerate(JUNCTIONS):
            for p in PLAN_IDS:
                chosen = state_rows[(state_rows.target_junction == jid) & (state_rows.applied_plan_id == p)]
                if len(chosen) != 1:
                    raise RuntimeError(f"Expected one snapshot cost for {sid} {jid} plan {p}")
                c_state[i, p] = float(pred[state_rows.index.get_loc(chosen.index[0])])
        q_state, penalty_state, _ = _build_qubo(c_state, pairwise)
        exact_state = _bruteforce(q_state, c_state, pairwise, penalty_state, 0.0)
        actual = block.groupby("plan_vector").mean_delay_s.sum()
        if len(actual) != 256 or set(actual.index) != {_canonical_vector(list(p)) for p in ALL_PLANS}:
            raise RuntimeError(f"Expected all 256 plan vectors for {sid}, got {len(actual)}")
        ordered_plans = np.array(ALL_PLANS, dtype=int)
        x = np.zeros((256, N), dtype=float)
        for n, planvec in enumerate(ordered_plans):
            x[n, np.arange(4) * 4 + planvec] = 1.0
        predicted = np.einsum("bi,ij,bj->b", x, q_state, x) + 4.0 * penalty_state + offset
        y = np.array([actual.loc[_canonical_vector(list(p))] for p in ALL_PLANS], dtype=float)
        best_true = int(np.argmin(y))
        best_pred = int(np.argmin(predicted))
        out[sid] = {
            "split": str(block.iloc[0].split), "scenario": str(block.iloc[0].scenario),
            "seed": int(block.iloc[0].seed), "n_plans": int(len(y)),
            "mae_s": float(np.mean(np.abs(predicted - y))),
            "rmse_s": float(np.sqrt(np.mean((predicted - y) ** 2))),
            "spearman": _safe_spearman(predicted, y),
            "predicted_best_plan_vector": list(ALL_PLANS[best_pred]),
            "sumo_best_plan_vector": list(ALL_PLANS[best_true]),
            "top1_match": bool(best_pred == best_true),
            "predicted_best_sumo_delay_s": float(y[best_pred]),
            "sumo_optimal_delay_s": float(y[best_true]),
            "regret_s": float(y[best_pred] - y[best_true]),
            "qubo_optimum_is_exact": bool(list(ALL_PLANS[best_pred]) == exact_state["best_plans"]),
            "snapshot_conditioned_cost_matrix_s": c_state.tolist(),
            "snapshot_penalty_A": penalty_state,
            "snapshot_exact_plan_vector": exact_state["best_plans"],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(6, max(1, (os.cpu_count() or 4) - 2)))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for path in (JOINTS_OUT, QUBO_OUT, META_OUT, METRICS_OUT):
        if path.exists() and not args.resume:
            raise FileExistsError(f"Refusing to overwrite {path}; pass --resume only for rollout checkpoints")
    start = time.time()
    s1 = pd.read_csv(S1_ROLLOUTS)
    tasks = _make_tasks()
    allowed_task_keys = {(t["snapshot_id"], t["experiment"], _canonical_vector(t["plan_vector"]))
                         for t in tasks}
    completed = set()
    records = []
    if args.resume and CHECKPOINT_OUT.exists():
        old = pd.read_csv(CHECKPOINT_OUT)
        records = old[old.status == "COMPLETED"].to_dict("records")
        # A previous checkpoint may have been written before the last futures
        # finished, then resumed. Deduplicate any replayed completed runs.
        seen = set()
        unique_records = []
        for row in records:
            row["plan_vector"] = _canonical_vector(row["plan_vector"])
            if (row["snapshot_id"], row["experiment"], row["plan_vector"]) not in allowed_task_keys:
                continue
            key = (row["snapshot_id"], row["experiment"], row["plan_vector"], row["measured_junction"])
            if key not in seen:
                seen.add(key)
                unique_records.append(row)
        records = unique_records
        completed = {(r["snapshot_id"], r["experiment"], _canonical_vector(r["plan_vector"])) for r in records}
    remaining = [t for t in tasks if (t["snapshot_id"], t["experiment"], _canonical_vector(t["plan_vector"])) not in completed]
    print(f"S2 v3: {len(tasks)} matched/joint rollouts total, {len(remaining)} remaining; workers={args.workers}")
    failures = []
    done = len(completed)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_vector, t): t for t in remaining}
        for future in as_completed(futures):
            task = futures[future]
            try:
                records.extend(future.result())
            except Exception as e:
                failures.append({"snapshot_id": task["snapshot_id"], "scenario": task["scenario"],
                                 "seed": task["seed"], "experiment": task["experiment"],
                                 "plan_vector": json.dumps(task["plan_vector"]), "error": repr(e)})
            done += 1
            if done % 20 == 0 or done == 1:
                pd.DataFrame(records).to_csv(CHECKPOINT_OUT, index=False)
                print(f"  completed {done}/{len(tasks)} joint interventions")
    pd.DataFrame(records).to_csv(CHECKPOINT_OUT, index=False)
    if failures:
        pd.DataFrame(failures).to_csv(FAIL_OUT, index=False)
        raise RuntimeError(f"{len(failures)} SUMO interventions failed; inspect {FAIL_OUT}")
    joint = pd.DataFrame(records)
    joint["plan_vector"] = joint.plan_vector.map(_canonical_vector)
    joint.to_csv(JOINTS_OUT, index=False)
    if set(joint.status) != {"COMPLETED"}:
        raise RuntimeError("Joint rollout table contains non-completed records")

    # Linear coefficients are S1 model predictions evaluated only on train-seed snapshots.
    c, s1_alpha, s1_model, train_features = _load_costs()
    pair_rows = joint[(joint.experiment == "pair_train") & (joint.split == "train")]
    pairwise, support = _pair_interactions(s1, pair_rows)
    train_states = pair_rows.snapshot_id.unique().tolist()
    if set(pair_rows.seed.unique()) != {0, 1, 2}:
        raise RuntimeError("Pairwise terms include non-training seeds")

    baseline_totals = []
    for sid in TRAIN_PAIR_SNAPSHOTS:
        d0 = np.mean([_total_by_action(s1, sid, jid, 0) for jid in JUNCTIONS])
        baseline_totals.append(float(d0 - c[:, 0].sum()))
    offset = float(np.mean(baseline_totals))

    q, penalty, base_bounds = _build_qubo(c, pairwise)
    exact = _bruteforce(q, c, pairwise, penalty, offset)
    if exact["n_feasible"] != 256 or exact["n_infeasible"] != 65280 or not exact["penalty_gap_ok"]:
        raise RuntimeError("Independent exhaustive one-hot penalty validation failed")
    s1_features = pd.read_csv(S1_FEATURES)
    validation = _validate_snapshots(joint, q, penalty, offset, exact,
                                     s1_saved=s1_model, s1_features=s1_features,
                                     pairwise=pairwise)
    if set(v["split"] for v in validation.values()) != {"val", "test"}:
        raise RuntimeError("Expected held-out validation and test snapshots")

    validation_q = []
    validation_ids = sorted(validation)
    validation_penalties = []
    for sid in validation_ids:
        c_state = np.asarray(validation[sid]["snapshot_conditioned_cost_matrix_s"], dtype=float)
        q_state, a_state, _ = _build_qubo(c_state, pairwise)
        validation_q.append(q_state)
        validation_penalties.append(a_state)
    np.savez_compressed(QUBO_OUT, Q=q, c=c, pairwise=pairwise,
                        pair_indices=np.asarray(PAIRS, dtype=np.int8),
                        A=np.asarray([penalty]), offset=np.asarray([offset]),
                        validation_Q=np.asarray(validation_q),
                        validation_A=np.asarray(validation_penalties),
                        validation_snapshot_ids=np.asarray(validation_ids))
    meta = {
        "schema_version": 2, "junction_order": JUNCTIONS, "plans_per_junction": 4,
        "variables": N, "variable_map": {f"x_{JUNCTIONS[i]}_p{p}": i * 4 + p
                                             for i in range(4) for p in PLAN_IDS},
        "training_seed_only": True, "training_seeds": [0, 1, 2],
        "validation_seeds": [3], "test_seeds": [4],
        "s1_model_alpha": s1_alpha, "training_snapshots_for_pairs": TRAIN_PAIR_SNAPSHOTS,
        "pair_interaction_support": support, "cost_matrix_s": c.tolist(),
        "pairwise_array_s": pairwise.tolist(), "pair_indices": [[JUNCTIONS[i], JUNCTIONS[j]] for i, j in PAIRS],
        "offset_s": offset, "penalty_A": penalty, "base_objective_bounds": base_bounds,
        "brute_force": exact,
        "input_rollout_file": str(S1_ROLLOUTS.relative_to(ROOT)),
        "validation_snapshots": VALIDATION_SNAPSHOTS, "test_snapshots": TEST_SNAPSHOTS,
    }
    META_OUT.write_text(json.dumps(meta, indent=2, allow_nan=False), encoding="utf-8")
    metrics = {
        "stage": "S2 v3", "status": "VALIDATION_LIMITED", "implementation_status": "EXECUTED",
        "scientific_validation_pass": False,
        "validation_note": "Exhaustive SUMO grid and QUBO mechanics pass, but held-out ranking/regret remain weak; do not use for closed-loop control.",
        "n_training_pair_rollouts": int(len(TRAIN_PAIR_SNAPSHOTS) * 6 * 9),
        "n_full_joint_validation_rollouts": int(4 * 256), "n_joint_intervention_runs": len(tasks),
        "train_seeds_used_for_cost_and_interactions": [0, 1, 2],
        "validation_snapshots": VALIDATION_SNAPSHOTS, "test_snapshots": TEST_SNAPSHOTS,
        "all_256_plan_vectors_evaluated_per_snapshot": True,
        "cost_inputs_use_actual_stgnn_predictions": True,
        "pairwise_terms_use_joint_sumo_rollouts": True,
        "cost_matrix_s": c.tolist(), "pair_interaction_support": support,
        "penalty_A": penalty, "base_objective_bounds": base_bounds,
        "independent_brute_force": exact, "validation_by_snapshot": validation,
        "mean_validation_spearman": float(np.mean([v["spearman"] for v in validation.values()
                                                      if v["split"] == "val" and v["spearman"] is not None])),
        "mean_test_spearman": float(np.mean([v["spearman"] for v in validation.values()
                                                if v["split"] == "test" and v["spearman"] is not None])),
        "mean_validation_regret_s": float(np.mean([v["regret_s"] for v in validation.values() if v["split"] == "val"])),
        "mean_test_regret_s": float(np.mean([v["regret_s"] for v in validation.values() if v["split"] == "test"])),
        "elapsed_s": round(time.time() - start, 2),
    }
    METRICS_OUT.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    print(f"S2 v3 complete: A={penalty:.4f}, exact plans={exact['best_plans']}, "
          f"val rho={metrics['mean_validation_spearman']:.4f}, test rho={metrics['mean_test_spearman']:.4f}")


if __name__ == "__main__":
    main()
