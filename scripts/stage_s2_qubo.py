"""
scripts/stage_s2_qubo.py  -  S2: QUBO formulation from cost model.
Outputs:
  models/qubo.npz              - Q matrix, variable map
  results/qubo_metrics.json    - energy checks, feasibility, brute-force verify
  tests/test_s2_qubo.py        - unit tests
"""
import os, sys, json, math, pickle, itertools
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
RESULTS = PROJECT / "results"
MODELS  = PROJECT / "models"
RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

JUNCTION_ORDER = ["J0","J1","J2","J3"]
K = 4   # plans per junction
N = len(JUNCTION_ORDER) * K   # 16 binary variables

# =========================================================================
# 1. Build cost vector c[i,k] from Ridge model on rollouts
#    c[i,k] = predicted mean delay at junction i under plan k
#    Use median conditions (queue/throughput) from high+peak rollouts
# =========================================================================
def build_cost_vector():
    """Return c: (4 junctions x 4 plans) numpy array."""
    with open(MODELS / "cost_model.pkl", "rb") as f:
        cm = pickle.load(f)
    model   = cm["model"]
    scaler  = cm["scaler"]
    feat_cols = cm["feat_cols"]

    # Get median queue/throughput per junction from high+peak rollouts
    df = pd.read_csv(RESULTS / "rollouts.csv")
    df_hp = df[df["scenario"].isin(["high","peak"])].copy()
    junc_stats = df_hp.groupby("junction")[["mean_queue_m","throughput"]].median()

    rows = []
    for ji, jid in enumerate(JUNCTION_ORDER):
        q_med = float(junc_stats.loc[jid, "mean_queue_m"]) if jid in junc_stats.index else 20.0
        thr_med = float(junc_stats.loc[jid, "throughput"]) if jid in junc_stats.index else 5.0
        for k in range(K):
            feat = {f"plan_{p}": float(p==k) for p in range(K)}
            feat.update({f"junc_{jj}": float(jj==jid) for jj in JUNCTION_ORDER})
            feat["mean_queue_m"] = q_med
            feat["throughput"]   = thr_med
            X = np.array([[feat[fc] for fc in feat_cols]])
            pred_delay = float(model.predict(scaler.transform(X))[0])
            rows.append({"junction": jid, "ji": ji, "plan": k, "cost": pred_delay})

    df_costs = pd.DataFrame(rows)
    c = np.zeros((4, K))
    for _, r in df_costs.iterrows():
        c[int(r["ji"]), int(r["plan"])] = r["cost"]
    return c, df_costs

# =========================================================================
# 2. Build QUBO Q matrix
#    Variables: x[i,k] = 1 if junction i uses plan k
#    Objective: minimize Σ_{i,k} c[i,k] * x[i,k]
#    Constraint: Σ_k x[i,k] = 1  for each junction i
#      Penalty: A * Σ_i (Σ_k x[i,k] - 1)^2
#
# QUBO convention: minimize x^T Q x  (upper-triangular)
# Variable index: var(i,k) = i*K + k
# =========================================================================
def build_qubo(c, A):
    """Build N×N upper-triangular QUBO matrix."""
    N = 4 * K
    Q = np.zeros((N, N))

    # Linear terms: c[i,k] on diagonal + constraint diagonal
    # (Σ_k x[i,k] - 1)^2 = Σ_k x[i,k]^2 + 2 Σ_{k<l} x[i,k]x[i,l] - 2 Σ_k x[i,k] + 1
    # For binary vars: x^2 = x, so diagonal += -2A per var + A*1 (const dropped)
    # Cross terms within junction: +2A
    for i in range(4):
        for k in range(K):
            vi = i*K + k
            # Objective linear
            Q[vi, vi] += c[i, k]
            # Constraint diagonal  (-2A from expansion of -(Σ=1) part + A from x^2=x)
            Q[vi, vi] += A * (1 - 2)  # = -A
        # Constraint cross-terms within junction
        for k in range(K):
            for l in range(k+1, K):
                vi = i*K + k
                vj = i*K + l
                Q[vi, vj] += 2 * A

    return Q

def qubo_energy(Q, x):
    """Compute QUBO energy x^T Q x."""
    return float(x @ Q @ x)

def is_feasible(x, K=4):
    """Check one-hot constraint: each junction has exactly one plan."""
    x = np.array(x)
    for i in range(4):
        if x[i*K:(i+1)*K].sum() != 1:
            return False
    return True

# =========================================================================
# 3. Tune A: find minimum A such that all feasible solutions beat all infeasible
# =========================================================================
def tune_penalty(c):
    """Binary search for A: feasible solutions must have lower energy than infeasible."""
    # Max cost difference between any two feasible assignments
    max_obj_range = float(c.max() - c.min()) * 4  # rough bound

    # Brute-force 256 combinations with A=candidate, check feasibility ordering
    def check_A(A):
        Q = build_qubo(c, A)
        energies = []
        for bits in itertools.product(range(2), repeat=N):
            x = np.array(bits, dtype=float)
            e = qubo_energy(Q, x)
            feas = is_feasible(x)
            energies.append((e, feas))
        min_infeas = min(e for e, f in energies if not f)
        max_feas   = max(e for e, f in energies if f)
        min_feas   = min(e for e, f in energies if f)
        return min_infeas > max_feas, min_feas, min_infeas

    for A_cand in [5, 10, 20, 50, 100, 200, 500]:
        ok, min_f, min_inf = check_A(A_cand)
        print(f"  A={A_cand}: feasible_min={min_f:.2f} infeasible_min={min_inf:.2f} -> {'OK' if ok else 'FAIL'}")
        if ok:
            return A_cand, min_f, min_inf
    return 500, None, None

# =========================================================================
# 4. Brute-force verify: ExactSolver equivalent (all 256 combos)
# =========================================================================
def brute_force(Q):
    """Return (best_x, best_energy, all_energies) over all 256 binary strings."""
    best_x, best_e = None, math.inf
    energies = []
    for bits in itertools.product(range(2), repeat=N):
        x = np.array(bits, dtype=float)
        e = qubo_energy(Q, x)
        feas = is_feasible(x)
        energies.append({"bits": list(bits), "energy": e, "feasible": feas})
        if feas and e < best_e:
            best_e = e
            best_x = list(bits)
    return best_x, best_e, energies

# =========================================================================
# MAIN
# =========================================================================
if __name__ == "__main__":
    print("[1/5] Building cost vector from Ridge model...")
    c, df_costs = build_cost_vector()
    print("  Cost matrix c[junction, plan] (delay in seconds):")
    for i, jid in enumerate(JUNCTION_ORDER):
        print(f"    {jid}: {[round(c[i,k],2) for k in range(K)]}")

    print("[2/5] Tuning penalty coefficient A...")
    A, min_feas, min_infeas = tune_penalty(c)
    print(f"  Selected A={A}")

    print("[3/5] Building final QUBO matrix...")
    Q = build_qubo(c, A)
    print(f"  Q shape: {Q.shape}, non-zero entries: {np.count_nonzero(Q)}")

    print("[4/5] Brute-force solving all 256 combinations...")
    best_x, best_e, all_energies = brute_force(Q)
    feas_energies = [e["energy"] for e in all_energies if e["feasible"]]
    infeas_energies = [e["energy"] for e in all_energies if not e["feasible"]]
    print(f"  Best feasible energy: {best_e:.4f}")
    print(f"  Best solution x: {best_x}")
    best_plans = [int(np.argmax(np.array(best_x[i*K:(i+1)*K]))) for i in range(4)]
    print(f"  Best plans per junction: {dict(zip(JUNCTION_ORDER, best_plans))}")
    print(f"  Feasible solutions: {len(feas_energies)}/256 (expected 4^4={4**4})")
    print(f"  Infeasible min energy: {min(infeas_energies):.4f}")
    print(f"  Feasible min energy:   {min(feas_energies):.4f}")
    gap_ok = min(infeas_energies) > min(feas_energies)
    print(f"  Penalty gap correct: {gap_ok}")

    # Manual energy verification: compute best_x energy manually
    manual_e = sum(c[i, best_plans[i]] for i in range(4))
    print(f"  Manual objective (sum of best plan delays): {manual_e:.4f}s")

    print("[5/5] Saving artifacts...")
    # Save Q matrix
    var_map = {f"x_{jid}_p{k}": i*K+k for i, jid in enumerate(JUNCTION_ORDER) for k in range(K)}
    np.savez(MODELS / "qubo.npz", Q=Q, c=c, A=np.array([A]))
    # Save var_map separately (json-serializable)
    qubo_meta = {
        "var_map": var_map,
        "junction_order": JUNCTION_ORDER,
        "K": K,
        "N": N,
        "A": A,
        "cost_matrix": c.tolist(),
        "best_x": best_x,
        "best_energy": round(best_e, 4),
        "best_plans": best_plans,
        "penalty_gap_ok": gap_ok,
    }
    (MODELS / "qubo_meta.json").write_text(json.dumps(qubo_meta, indent=2))

    metrics = {
        "A": A,
        "Q_shape": list(Q.shape),
        "Q_nonzero": int(np.count_nonzero(Q)),
        "brute_force": {
            "n_feasible": len(feas_energies),
            "n_infeasible": len(infeas_energies),
            "best_energy": round(best_e, 4),
            "best_plans": best_plans,
            "best_plans_dict": dict(zip(JUNCTION_ORDER, best_plans)),
            "feasible_min_energy": round(min(feas_energies), 4),
            "infeasible_min_energy": round(min(infeas_energies), 4),
            "penalty_gap_ok": gap_ok,
            "manual_objective_s": round(manual_e, 4),
        },
    }
    (RESULTS / "qubo_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"  models/qubo.npz + models/qubo_meta.json + results/qubo_metrics.json written")

    # Write tests
    test_code = '''"""tests/test_s2_qubo.py"""
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
    """Best energy from QUBO == manual sum of best-plan costs."""
    import pickle
    mq = json.loads((PROJECT / "models" / "qubo_meta.json").read_text())
    c  = np.array(mq["cost_matrix"])
    bp = mq["best_plans"]
    manual = sum(c[i, bp[i]] for i in range(4))
    assert abs(manual - mq["best_energy"]) < 1e-2, f"Energy mismatch: manual={manual} qubo={mq['best_energy']}"
'''
    (PROJECT / "tests" / "test_s2_qubo.py").write_text(test_code)
    print("  tests/test_s2_qubo.py written")
    print(f"\nS2 DONE. A={A} gap_ok={gap_ok} best_plans={dict(zip(JUNCTION_ORDER, best_plans))}")
