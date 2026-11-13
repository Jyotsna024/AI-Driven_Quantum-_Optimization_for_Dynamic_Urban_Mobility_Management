"""Independent checks for the rebuilt S2 QUBO and held-out rollout grid."""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    meta = json.loads((ROOT / "models/qubo_s2_v2.json").read_text())
    metrics = json.loads((ROOT / "results/qubo_metrics_s2_v2.json").read_text())
    with np.load(ROOT / "models/qubo_s2_v2.npz") as data:
        q = data["Q"].copy()
        c = data["c"].copy()
        pairwise = data["pairwise"].copy()
        validation_q = data["validation_Q"].copy()
        validation_ids = data["validation_snapshot_ids"].tolist()
    rollouts = pd.read_csv(ROOT / "data/joint_rollouts_s2_v2.csv")
    return meta, metrics, q, c, pairwise, validation_q, validation_ids, rollouts


def _energy(q, x):
    x = np.asarray(x, dtype=float)
    return float(x @ q @ x)


def test_s2_rollout_grid_complete_and_unique():
    meta, metrics, *_rest, rollouts = _load()
    assert len(rollouts) == 1294 * 4
    key = ["snapshot_id", "experiment", "plan_vector", "measured_junction"]
    assert not rollouts.duplicated(key).any()
    assert set(rollouts.status) == {"COMPLETED"}
    full = rollouts[rollouts.experiment == "full_joint_validation"]
    assert full.groupby("snapshot_id").plan_vector.nunique().eq(256).all()
    assert full.groupby(["snapshot_id", "plan_vector"]).measured_junction.nunique().eq(4).all()
    assert metrics["n_joint_intervention_runs"] == 1294
    assert meta["training_seed_only"] is True


def test_s2_costs_and_interactions_are_source_supported():
    meta, metrics, q, c, pairwise, *_ = _load()
    assert c.shape == (4, 4)
    assert pairwise.shape == (6, 4, 4)
    assert metrics["cost_inputs_use_actual_stgnn_predictions"] is True
    assert metrics["pairwise_terms_use_joint_sumo_rollouts"] is True
    for key, evidence in meta["pair_interaction_support"].items():
        if "|" in key and ":0" not in key.split("|")[0] and ":0" not in key.split("|")[1]:
            assert evidence["source"] == "matched joint SUMO rollouts"
            assert evidence["n_train_snapshots"] == 5
    assert np.allclose(q, q.T) is False  # Q is upper triangular by convention.


def test_s2_exact_energy_and_penalty_exhaustive_verification():
    meta, metrics, q, c, pairwise, *_ = _load()
    brute = meta["brute_force"]
    best_x = np.asarray(brute["best_x"], dtype=int)
    assert _energy(q, best_x) == pytest.approx(brute["best_energy"], abs=1e-7)
    assert brute["n_bitstrings"] == 65536
    assert brute["n_feasible"] == 256
    assert brute["n_infeasible"] == 65280
    assert brute["penalty_gap_ok"] is True
    assert brute["one_hot_verified_all_feasible"] is True
    assert meta["penalty_A"] > (meta["base_objective_bounds"]["feasible_max"] -
                                meta["base_objective_bounds"]["infeasible_min"])
    assert metrics["all_256_plan_vectors_evaluated_per_snapshot"] is True


def test_s2_validation_qubos_match_all_feasible_plan_vectors():
    meta, metrics, _q, _c, _pairwise, qs, ids, rollouts = _load()
    assert set(ids) == set(metrics["validation_by_snapshot"])
    assert qs.shape == (4, 16, 16)
    all_plans = list(itertools.product(range(4), repeat=4))
    full = rollouts[rollouts.experiment == "full_joint_validation"]
    for sid, q in zip(ids, qs):
        rec = metrics["validation_by_snapshot"][sid]
        assert len(all_plans) == 256
        assert rec["n_plans"] == 256
        assert len(rec["snapshot_conditioned_cost_matrix_s"]) == 4
        assert np.isfinite(rec["spearman"])
        assert rec["snapshot_exact_plan_vector"] == rec["predicted_best_plan_vector"]
        block = full[full.snapshot_id == sid]
        assert block.plan_vector.nunique() == 256
