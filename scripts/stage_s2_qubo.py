"""
scripts/stage_s2_qubo.py - S2: Build and Validate QUBO Matrix (Audited & Leakage-Free).

QUBO formulation:
- Variables: x[i, k] in {0, 1}, where i in {0..3} (junction), k in {0..3} (signal plan). N = 16 binary variables.
- Objective: Minimize total corridor delay: sum_{i=0}^3 sum_{k=0}^3 c[i, k] * x[i, k]
- One-hot Constraint: For each junction i, sum_{k=0}^3 x[i, k] = 1
- Penalty formulation: A * sum_{i=0}^3 (sum_{k=0}^3 x[i, k] - 1)^2
  Expanded for binary x: (sum x - 1)^2 = sum x^2 + 2 sum_{k<l} x_k x_l - 2 sum x + 1
                        = - sum x + 2 sum_{k<l} x_k x_l + 1
  QUBO matrix (upper-triangular):
    Q[i*K+k, i*K+k] = c[i, k] - A
    Q[i*K+k, i*K+l] = 2 * A  (for k < l)
  Feasible constant shift: sum_{i=0}^3 (-A) = -4A.
  For any feasible x: x^T Q x + 4A = sum_{i=0}^3 c[i, k_i*] (exact traffic delay in seconds).

Outputs:
  models/qubo.npz
  models/qubo_meta.json
  results/qubo_metrics.json
  tests/test_s2_qubo.py
"""

import os
import sys
import json
import pickle
import math
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
TESTS_DIR = PROJECT_ROOT / "tests"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TESTS_DIR.mkdir(parents=True, exist_ok=True)

JUNCTION_ORDER = ["J0", "J1", "J2", "J3"]
K = 4
N = 16


def build_cost_matrix():
    """Evaluate clean leakage-free cost model on representative corridor conditions."""
    with open(MODELS_DIR / "cost_model.pkl", "rb") as f:
        cm = pickle.load(f)
    model = cm["model"]
    scaler = cm["scaler"]
    feat_cols = cm["feat_cols"]

    # Load rollouts and extract median pre-decision conditions per junction
    rollouts = pd.read_csv(RESULTS_DIR / "rollouts.csv")
    raw_cache = {}
    def get_raw_df(sc, sd):
        k = (sc, sd)
        if k not in raw_cache:
            fp = RAW_DIR / f"{sc}_s{sd}_varied.csv"
            if not fp.exists():
                fp = RAW_DIR / f"{sc}_s{sd}_fixed.csv"
            df = pd.read_csv(fp)
            df["mean_speed_mps"] = df["mean_speed_mps"].fillna(0.0)
            df["mean_delay_s"] = df["mean_delay_s"].fillna(0.0)
            df["waiting_time_s"] = df["waiting_time_s"].fillna(0.0)
            raw_cache[k] = df
        return raw_cache[k]

    conditions = []
    for _, r in rollouts[rollouts["scenario"].isin(["high", "peak"])].iterrows():
        raw_df = get_raw_df(r["scenario"], int(r["seed"]))
        sub = raw_df[(raw_df["timestamp_s"] == int(r["snapshot_time"])) & (raw_df["junction_short"] == r["junction"])]
        if not sub.empty:
            raw_r = sub.iloc[0]
            conditions.append({
                "junction": r["junction"],
                "pre_queue_m": float(raw_r["queue_length_m"]),
                "pre_delay": float(raw_r["mean_delay_s"]),
                "pre_arrivals": float(raw_r["arrivals"]),
                "pre_throughput": float(raw_r["throughput"]),
                "pre_speed": float(raw_r["mean_speed_mps"]),
                "pre_vehs": float(raw_r["vehicle_count"]),
                "stgnn_q_h1": float(raw_r["queue_length"]),
                "stgnn_q_h2": float(raw_r["queue_length"]),
                "stgnn_q_h3": float(raw_r["queue_length"]),
                "stgnn_q_mean": float(raw_r["queue_length"]),
            })
    df_cond = pd.DataFrame(conditions)
    med_stats = df_cond.groupby("junction").median()

    c = np.zeros((4, K))
    for ji, jid in enumerate(JUNCTION_ORDER):
        st = med_stats.loc[jid]
        for k in range(K):
            row_dict = {
                "pre_queue_m": float(st["pre_queue_m"]),
                "pre_delay": float(st["pre_delay"]),
                "pre_arrivals": float(st["pre_arrivals"]),
                "pre_throughput": float(st["pre_throughput"]),
                "pre_speed": float(st["pre_speed"]),
                "pre_vehs": float(st["pre_vehs"]),
                "stgnn_q_h1": float(st["stgnn_q_h1"]),
                "stgnn_q_h2": float(st["stgnn_q_h2"]),
                "stgnn_q_h3": float(st["stgnn_q_h3"]),
                "stgnn_q_mean": float(st["stgnn_q_mean"]),
            }
            for p in range(K):
                row_dict[f"plan_{p}"] = float(k == p)
            for j in JUNCTION_ORDER:
                row_dict[f"junc_{j}"] = float(jid == j)
            for p in range(K):
                for j in JUNCTION_ORDER:
                    row_dict[f"p{p}_j_{j}"] = float(k == p and jid == j)
                row_dict[f"p{p}_q"] = float(k == p) * row_dict["pre_queue_m"]
                row_dict[f"p{p}_stgnn"] = float(k == p) * row_dict["stgnn_q_mean"]
                row_dict[f"p{p}_arr"] = float(k == p) * row_dict["pre_arrivals"]

            feat_vector = np.array([[row_dict[col] for col in feat_cols]])
            scaled_feat = scaler.transform(feat_vector)
            pred_delay = float(model.predict(scaled_feat)[0])
            c[ji, k] = pred_delay

    return c


def build_qubo(c, A=50.0):
    """Build upper-triangular 16x16 QUBO matrix."""
    Q = np.zeros((N, N))
    for i in range(4):
        for k in range(K):
            vi = i * K + k
            Q[vi, vi] += c[i, k] - A
        for k in range(K):
            for l in range(k + 1, K):
                vi = i * K + k
                vj = i * K + l
                Q[vi, vj] += 2.0 * A
    return Q


def is_feasible(x):
    for i in range(4):
        if sum(x[i * K:(i + 1) * K]) != 1:
            return False
    return True


def brute_force_solve(Q, c, A):
    """Exhaustive check across all 2^16 = 65,536 bitstrings."""
    best_feasible_e = math.inf
    best_feasible_x = None
    min_infeasible_e = math.inf
    max_feasible_e = -math.inf
    n_feasible = 0
    n_infeasible = 0

    for bits in itertools.product(range(2), repeat=N):
        x = np.array(bits, dtype=float)
        e = float(x @ Q @ x)
        feas = is_feasible(x)
        if feas:
            n_feasible += 1
            if e < best_feasible_e:
                best_feasible_e = e
                best_feasible_x = list(bits)
            if e > max_feasible_e:
                max_feasible_e = e
        else:
            n_infeasible += 1
            if e < min_infeasible_e:
                min_infeasible_e = e

    best_plans = [int(np.argmax(best_feasible_x[i * K:(i + 1) * K])) for i in range(4)]
    manual_delay = float(sum(c[i, best_plans[i]] for i in range(4)))
    qubo_delay = float(best_feasible_e + 4.0 * A)
    penalty_gap_ok = bool(min_infeasible_e > max_feasible_e)

    return {
        "n_feasible": n_feasible,
        "n_infeasible": n_infeasible,
        "best_energy": round(best_feasible_e, 4),
        "best_x": best_feasible_x,
        "best_plans": best_plans,
        "best_plans_dict": dict(zip(JUNCTION_ORDER, best_plans)),
        "manual_delay_s": round(manual_delay, 4),
        "qubo_delay_s": round(qubo_delay, 4),
        "diff_s": round(abs(manual_delay - qubo_delay), 8),
        "max_feasible_energy": round(max_feasible_e, 4),
        "min_infeasible_energy": round(min_infeasible_e, 4),
        "penalty_gap_ok": penalty_gap_ok,
    }


def main():
    print("[1/4] Building cost matrix from clean leakage-free cost model...")
    c = build_cost_matrix()
    print("  Cost matrix c[junction, plan] (delay seconds):")
    for ji, jid in enumerate(JUNCTION_ORDER):
        print(f"    {jid}: {[round(c[ji, k], 2) for k in range(K)]} -> min plan={int(np.argmin(c[ji]))}")

    print("[2/4] Constructing upper-triangular QUBO with penalty A=50.0...")
    A = 50.0
    Q = build_qubo(c, A)
    print(f"  Q shape: {Q.shape}, non-zero entries: {np.count_nonzero(Q)}")

    print("[3/4] Exhaustive brute-force check over all 65,536 combinations...")
    bf = brute_force_solve(Q, c, A)
    print(f"  Feasible count: {bf['n_feasible']} (expected 256), Infeasible count: {bf['n_infeasible']} (expected 65280)")
    print(f"  Optimal plans: {bf['best_plans_dict']}")
    print(f"  Optimal QUBO energy: {bf['best_energy']}")
    print(f"  Optimal corridor delay: {bf['manual_delay_s']}s (QUBO delay: {bf['qubo_delay_s']}s, diff: {bf['diff_s']}s)")
    print(f"  Feasible energy range: [{bf['best_energy']}, {bf['max_feasible_energy']}]")
    print(f"  Minimum infeasible energy: {bf['min_infeasible_energy']}")
    print(f"  Penalty gap valid (infeas_min > feas_max): {bf['penalty_gap_ok']}")

    print("[4/4] Serializing QUBO matrix and metadata...")
    var_map = {f"x_{jid}_p{k}": i * K + k for i, jid in enumerate(JUNCTION_ORDER) for k in range(K)}
    np.savez(MODELS_DIR / "qubo.npz", Q=Q, c=c, A=np.array([A]))

    qubo_meta = {
        "var_map": var_map,
        "junction_order": JUNCTION_ORDER,
        "K": K,
        "N": N,
        "A": A,
        "cost_matrix": c.tolist(),
        "best_x": bf["best_x"],
        "best_energy": bf["best_energy"],
        "best_plans": bf["best_plans"],
        "best_plans_dict": bf["best_plans_dict"],
        "delay_objective_s": bf["manual_delay_s"],
        "penalty_gap_ok": bf["penalty_gap_ok"],
    }
    with open(MODELS_DIR / "qubo_meta.json", "w") as f:
        json.dump(qubo_meta, f, indent=2)

    qubo_metrics = {
        "A": A,
        "Q_shape": list(Q.shape),
        "Q_nonzero": int(np.count_nonzero(Q)),
        "brute_force": bf,
    }
    with open(RESULTS_DIR / "qubo_metrics.json", "w") as f:
        json.dump(qubo_metrics, f, indent=2)

    test_code = '''"""tests/test_s2_qubo.py - S2 QUBO Formulation Test Suite"""
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
'''
    with open(TESTS_DIR / "test_s2_qubo.py", "w") as f:
        f.write(test_code)

    print("Stage S2 QUBO formulation complete!")


if __name__ == "__main__":
    main()
