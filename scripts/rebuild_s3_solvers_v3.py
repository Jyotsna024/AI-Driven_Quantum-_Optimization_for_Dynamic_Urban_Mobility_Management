"""Run reproducible solvers on the versioned S2 QUBO input."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import dimod
import neal
import numpy as np
from scipy.optimize import minimize
from qiskit import transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

ROOT = Path(__file__).resolve().parents[1]
QUBO_PATH = ROOT / "models" / "qubo_s2_v3.npz"
META_PATH = ROOT / "models" / "qubo_s2_v3.json"
OUTPUT = ROOT / "results" / "solver_metrics_s3_v3.json"
N, K, JUNCTIONS = 16, 4, ["J0", "J1", "J2", "J3"]
N_SEEDS, SA_READS, QAOA_SHOTS, OPT_MAXITER = 10, 250, 1500, 15


def load_input():
    with np.load(QUBO_PATH) as data:
        q = data["Q"].astype(float)
        a = float(data["A"][0])
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return q, a, meta


def energy(q, sample):
    x = np.asarray(sample, dtype=float)
    return float(x @ q @ x)


def feasible(sample):
    x = np.asarray(sample, dtype=int)
    return x.size == N and bool(np.all(x.reshape(4, 4).sum(axis=1) == 1))


def plans_from_sample(sample):
    if not feasible(sample):
        return None
    return np.asarray(sample, dtype=int).reshape(4, 4).argmax(axis=1).astype(int).tolist()


def bqm_from_q(q):
    linear = {i: float(q[i, i]) for i in range(N) if q[i, i] != 0}
    quadratic = {(i, j): float(q[i, j]) for i in range(N) for j in range(i + 1, N)
                 if q[i, j] != 0}
    return dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)


def positive_delay_ratio(delay, exact_delay):
    if delay is None or exact_delay is None or not math.isfinite(delay) or delay <= 0:
        return None
    return float(exact_delay / delay)


def run_exact(q, a):
    start = time.perf_counter()
    samples = dimod.ExactSolver().sample(bqm_from_q(q))
    elapsed = time.perf_counter() - start
    all_rows = [(np.asarray([row.sample[i] for i in range(N)], dtype=int), float(row.energy))
                for row in samples.data(fields=["sample", "energy"])]
    feasible_rows = [(x, e) for x, e in all_rows if feasible(x)]
    x, e = min(feasible_rows, key=lambda item: item[1])
    return {"solver": "dimod ExactSolver", "best_x": x.tolist(), "best_plans": plans_from_sample(x),
            "best_energy": e, "traffic_delay_s": e + 4 * a,
            "n_raw_samples": len(all_rows), "n_feasible_samples": len(feasible_rows),
            "feasible_rate": len(feasible_rows) / len(all_rows), "runtime_s": elapsed}


def run_sa(q, a, exact_delay):
    bqm = bqm_from_q(q)
    raw_n = feasible_n = 0
    best = None
    per_seed = []
    start = time.perf_counter()
    for seed in range(N_SEEDS):
        run_start = time.perf_counter()
        result = neal.SimulatedAnnealingSampler().sample(bqm, num_reads=SA_READS, seed=seed)
        seed_best = None
        for row in result.data(fields=["sample", "energy", "num_occurrences"]):
            x = np.asarray([row.sample[i] for i in range(N)], dtype=int)
            count, e = int(row.num_occurrences), float(row.energy)
            raw_n += count
            if feasible(x):
                feasible_n += count
                if seed_best is None or e < seed_best[1]:
                    seed_best = (x, e)
                if best is None or e < best[1]:
                    best = (x, e)
        per_seed.append({"seed": seed, "runtime_s": time.perf_counter() - run_start,
                         "best_feasible_energy": None if seed_best is None else seed_best[1],
                         "best_feasible_x": None if seed_best is None else seed_best[0].tolist()})
    elapsed = time.perf_counter() - start
    x, e = best if best is not None else (None, None)
    delay = None if e is None else e + 4 * a
    return {"solver": "dwave-neal simulated annealing", "best_x": None if x is None else x.tolist(),
            "best_plans": None if x is None else plans_from_sample(x), "best_energy": e,
            "traffic_delay_s": delay, "approx_ratio": positive_delay_ratio(delay, exact_delay),
            "n_raw_samples": raw_n, "n_feasible_samples": feasible_n,
            "feasible_rate": feasible_n / raw_n, "runtime_s": elapsed, "n_seeds": N_SEEDS,
            "reads_per_seed": SA_READS, "per_seed": per_seed}


def qubo_to_ising(q):
    """Return SparsePauliOp and scalar offset for x=(1-Z)/2, upper-triangular Q."""
    offset = float(np.trace(q) / 2.0)
    h = -np.diag(q).astype(float) / 2.0
    terms = []
    for i in range(N):
        for j in range(i + 1, N):
            coeff = float(q[i, j])
            if coeff:
                offset += coeff / 4.0
                h[i] -= coeff / 4.0
                h[j] -= coeff / 4.0
                terms.append(("ZZ", [i, j], coeff / 4.0))
    terms.extend(("Z", [i], float(h[i])) for i in range(N) if h[i] != 0)
    jmat = np.triu(q, 1).astype(float) / 4.0
    return SparsePauliOp.from_sparse_list(terms, num_qubits=N), offset, h, jmat


def _counts_to_samples(counts):
    for bitstring, count in counts.items():
        x = np.asarray([int(bitstring[N - 1 - i]) for i in range(N)], dtype=int)
        yield x, int(count)


def run_qaoa(q, a, exact_delay):
    h_op, ising_offset, _h, _j = qubo_to_ising(q)
    n_raw = feasible_n = 0
    best = None
    per_seed = []
    start = time.perf_counter()
    for seed in range(N_SEEDS):
        sim = AerSimulator()
        circuit = QAOAAnsatz(cost_operator=h_op, reps=1)
        circuit.measure_all()
        eval_index = 0

        def objective(params):
            nonlocal eval_index
            eval_index += 1
            tc = transpile(circuit.assign_parameters(params), sim, seed_transpiler=seed)
            job = sim.run(tc, shots=QAOA_SHOTS,
                          seed_simulator=seed * 100_000 + eval_index)
            counts = job.result().get_counts()
            total, total_energy = 0, 0.0
            for x, count in _counts_to_samples(counts):
                total += count
                total_energy += energy(q, x) * count
            return total_energy / total

        opt = minimize(objective, [0.35 + seed * 0.01, 0.2 + seed * 0.01],
                       method="COBYLA", options={"maxiter": OPT_MAXITER, "rhobeg": 0.2})
        run = AerSimulator()
        tc = transpile(circuit.assign_parameters(opt.x), run, seed_transpiler=seed)
        counts = run.run(tc, shots=QAOA_SHOTS, seed_simulator=seed + 50_000).result().get_counts()
        seed_best = None
        for x, count in _counts_to_samples(counts):
            n_raw += count
            if feasible(x):
                feasible_n += count
                e = energy(q, x)
                if seed_best is None or e < seed_best[1]:
                    seed_best = (x, e)
                if best is None or e < best[1]:
                    best = (x, e)
        per_seed.append({"seed": seed, "optimizer_success": bool(opt.success),
                         "optimizer_message": str(opt.message), "objective_evaluations": eval_index,
                         "optimized_parameters_in_circuit_order": [float(v) for v in opt.x],
                         "best_feasible_energy": None if seed_best is None else seed_best[1],
                         "best_feasible_x": None if seed_best is None else seed_best[0].tolist()})
    elapsed = time.perf_counter() - start
    x, e = best if best is not None else (None, None)
    delay = None if e is None else e + 4 * a
    return {"solver": "QAOA p=1 on Qiskit Aer", "hardware_label": "Aer simulator; no quantum hardware",
            "best_x": None if x is None else x.tolist(),
            "best_plans": None if x is None else plans_from_sample(x), "best_energy": e,
            "traffic_delay_s": delay, "approx_ratio": positive_delay_ratio(delay, exact_delay),
            "n_raw_samples": n_raw, "n_feasible_samples": feasible_n,
            "feasible_rate": feasible_n / n_raw if n_raw else 0.0,
            "runtime_s": elapsed, "n_seeds": N_SEEDS, "shots_per_seed": QAOA_SHOTS,
            "ising_offset": ising_offset, "per_seed": per_seed}


def verify_results(q, a, results):
    """Independent recomputation from serialized Q and raw returned bitstrings."""
    verified = {}
    for name, result in results.items():
        x = result["best_x"]
        if x is None:
            assert result["best_plans"] is None and result["best_energy"] is None
            verified[name] = {"no_feasible_sample": True}
            continue
        assert feasible(x), f"Reported {name} optimum is infeasible"
        recomputed_e = energy(q, x)
        assert math.isclose(recomputed_e, result["best_energy"], rel_tol=0, abs_tol=1e-8)
        assert plans_from_sample(x) == result["best_plans"]
        delay = recomputed_e + 4 * a
        assert math.isclose(delay, result["traffic_delay_s"], rel_tol=0, abs_tol=1e-8)
        verified[name] = {"energy_recomputed": recomputed_e, "feasible_recomputed": True,
                          "plan_recomputed": plans_from_sample(x), "delay_recomputed_s": delay}
    return verified


def main():
    q, a, meta = load_input()
    if q.shape != (N, N) or not np.isfinite(q).all() or not np.allclose(np.tril(q, -1), 0):
        raise ValueError("S2 Q must be finite 16x16 upper triangular")
    brute = meta["brute_force"]
    exact_energy = float(brute["best_energy"])
    exact_delay = exact_energy + 4 * a
    if not brute["penalty_gap_ok"] or brute["n_feasible"] != 256:
        raise ValueError("S2 QUBO failed its saved exhaustive feasibility evidence")
    print(f"S3 v3: serialized input={QUBO_PATH.name}, A={a:.6f}, exact_energy={exact_energy:.8f}")
    exact = run_exact(q, a)
    if not math.isclose(exact["best_energy"], exact_energy, rel_tol=0, abs_tol=1e-7):
        raise RuntimeError("dimod ExactSolver disagrees with the S2 exhaustive optimum")
    exact["approx_ratio"] = 1.0
    exact["exact_reference_delay_s"] = exact_delay
    print("ExactSolver done", exact["best_plans"], exact["best_energy"])
    sa = run_sa(q, a, exact_delay)
    print("SA done", sa["best_plans"], sa["best_energy"], sa["feasible_rate"])
    qaoa = run_qaoa(q, a, exact_delay)
    print("QAOA done", qaoa["best_plans"], qaoa["best_energy"], qaoa["feasible_rate"])
    solver_results = {"ExactSolver": exact, "SimulatedAnnealing": sa, "QAOA": qaoa}
    verified = verify_results(q, a, solver_results)
    out = {
        "schema_version": 2, "status": "COMPLETE", "input_file": str(QUBO_PATH.relative_to(ROOT)),
        "input_sha256": __import__("hashlib").sha256(QUBO_PATH.read_bytes()).hexdigest(),
        "identical_serialized_input": True, "qubo_variables": N, "penalty_A": a,
        "exact_energy": exact_energy, "exact_delay_s": exact_delay,
        "seed_schedule": {"SA": list(range(N_SEEDS)), "QAOA": list(range(N_SEEDS))},
        "sample_policy": "Feasibility computed over raw sampler samples; no forced one-hot decoding.",
        "independent_verification": verified, "solvers": solver_results,
    }
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")
    print(f"S3 v3 complete; JSON={OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
