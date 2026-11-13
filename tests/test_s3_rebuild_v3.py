"""Independent S3 input, Ising conversion, and solver-output verification."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from rebuild_s3_solvers_v3 import (
    energy, feasible, load_input, plans_from_sample, positive_delay_ratio,
    qubo_to_ising,
)

def test_s3_ising_conversion_matches_qubo_for_full_feasible_space_and_samples():
    q, _a, _meta = load_input()
    _op, offset, h, jmat = qubo_to_ising(q)
    samples = []
    for n in range(4**4):
        plans = np.base_repr(n, base=4).zfill(4)
        x = np.zeros(16, dtype=int)
        for junction, plan in enumerate(plans):
            x[junction * 4 + int(plan)] = 1
        samples.append(x)
    rng = np.random.default_rng(74021)
    samples.extend(rng.integers(0, 2, size=(256, 16)))
    for x in samples:
        z = 1 - 2 * x
        converted = offset + float(h @ z) + float(np.sum(jmat * np.outer(z, z)))
        assert converted == pytest.approx(energy(q, x), abs=1e-8)


def test_s3_exactly_one_hot_feasibility_and_plan_decoding():
    x = np.zeros(16, dtype=int)
    x[3] = x[4] = x[10] = x[15] = 1
    assert feasible(x)
    assert plans_from_sample(x) == [3, 0, 2, 3]
    x[0] = 1
    assert not feasible(x)
    assert plans_from_sample(x) is None


def test_s3_positive_delay_approximation_ratio_definition():
    assert positive_delay_ratio(120.0, 90.0) == pytest.approx(0.75)
    assert positive_delay_ratio(80.0, 90.0) == pytest.approx(1.125)
    assert positive_delay_ratio(0.0, 90.0) is None


def test_s3_saved_solver_metrics_are_independently_consistent():
    path = ROOT / "results/solver_metrics_s3_v3.json"
    assert path.exists()
    result = json.loads(path.read_text())
    q, a, _meta = load_input()
    assert result["input_sha256"] == hashlib.sha256((ROOT / "models/qubo_s2_v3.npz").read_bytes()).hexdigest()
    assert result["identical_serialized_input"] is True
    for name, solver in result["solvers"].items():
        if solver["best_x"] is None:
            assert solver["best_energy"] is None
            assert solver["best_plans"] is None
            continue
        x = np.asarray(solver["best_x"], dtype=int)
        assert feasible(x)
        assert solver["best_energy"] == pytest.approx(energy(q, x), abs=1e-8)
        assert solver["best_plans"] == plans_from_sample(x)
        assert solver["traffic_delay_s"] == pytest.approx(energy(q, x) + 4 * a, abs=1e-8)
        if name != "ExactSolver":
            assert solver["n_raw_samples"] > 0
            assert solver["n_feasible_samples"] <= solver["n_raw_samples"]
            assert solver["feasible_rate"] == pytest.approx(
                solver["n_feasible_samples"] / solver["n_raw_samples"])
