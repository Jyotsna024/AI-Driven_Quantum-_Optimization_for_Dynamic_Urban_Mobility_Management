"""Read-only tests for rebuilt, target-correct S1 artifacts."""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
JUNCTIONS = ["J0", "J1", "J2", "J3"]
PLANS = {0, 1, 2, 3}
FEATURES = [
    "pre_queue_veh", "pre_queue_m", "pre_delay_s", "pre_arrivals",
    "pre_throughput", "pre_speed_mps", "pre_vehicle_count",
    "stgnn_queue_h1", "stgnn_queue_h2", "stgnn_queue_h3",
    *[f"scenario_{s}" for s in ("low", "medium", "high", "peak", "incident")],
    *[f"plan_{p}" for p in range(4)],
    *[f"junction_{j}" for j in JUNCTIONS],
    *[f"plan{p}_junction_{j}" for p in range(4) for j in JUNCTIONS],
    *[f"plan{p}_queue_h{h}" for p in range(4) for h in (1, 2, 3)],
    *[f"plan{p}_pre_queue_m" for p in range(4)],
    *[f"plan{p}_pre_arrivals" for p in range(4)],
]


def _artifacts():
    return {
        "rollouts": ROOT / "data" / "rollouts_s1_v2.csv",
        "features": ROOT / "data" / "features_s1_v2.csv",
        "model": ROOT / "models" / "cost_model_s1_v2.pkl",
        "params": ROOT / "models" / "cost_model_s1_v2.json",
        "metrics": ROOT / "results" / "cost_model_metrics_s1_v2.json",
    }


def test_versioned_s1_artifacts_exist_and_are_complete():
    paths = _artifacts()
    assert all(p.exists() for p in paths.values())
    r = pd.read_csv(paths["rollouts"])
    assert len(r) == 45 * 4 * 4 * 4
    assert set(r.status) == {"COMPLETED"}
    assert {"snapshot_id", "target_junction", "applied_plan_id", "measured_junction",
            "seed", "scenario"}.issubset(r.columns)


def test_every_plan_was_run_from_each_snapshot_for_each_target():
    r = pd.read_csv(_artifacts()["rollouts"])
    counts = r.groupby(["snapshot_id", "target_junction", "applied_plan_id"]).measured_junction.nunique()
    assert len(counts) == 45 * 4 * 4
    assert set(counts) == {4}
    for _, g in r.groupby(["snapshot_id", "target_junction"]):
        assert set(g.applied_plan_id) == PLANS
        assert g.groupby("applied_plan_id").size().eq(4).all()
    assert r.other_junctions_baseline.astype(bool).all()


def test_model_training_rows_only_use_target_junction_outcomes():
    paths = _artifacts()
    d = pd.read_csv(paths["features"])
    assert len(d) == 45 * 4 * 4
    assert (d.target_junction == d.measured_junction).all()
    assert set(d.applied_plan_id) == PLANS
    assert set(d[d.split == "train"].seed) == {0, 1, 2}
    assert set(d[d.split == "val"].seed) == {3}
    assert set(d[d.split == "test"].seed) == {4}
    assert d[FEATURES].notna().all().all()
    assert not set(FEATURES) & {"mean_delay_s", "mean_queue_m", "throughput", "mean_queue_veh"}
    assert len(FEATURES) == len(set(FEATURES))


def test_saved_model_and_metrics_use_predeclared_feature_schema():
    paths = _artifacts()
    params = json.loads(paths["params"].read_text())
    metrics = json.loads(paths["metrics"].read_text())
    with paths["model"].open("rb") as f:
        model = pickle.load(f)
    assert params["feature_cols"] == FEATURES
    assert model["feature_cols"] == FEATURES
    assert params["feature_leakage_audit"].startswith("Predictors are pre-decision")
    assert metrics["feature_columns"] == FEATURES
    assert metrics["train_seeds"] == [0, 1, 2]
    assert metrics["validation_seeds"] == [3]
    assert metrics["test_seeds"] == [4]
    for split in ("train", "validation", "test"):
        block = metrics[split]
        assert block["n"] > 0
        assert np.isfinite(block["mae"])
        assert np.isfinite(block["rmse"])
        assert block["top1_total"] == (60 if split == "train" else 60)
        assert len(block["per_snapshot"]) == block["top1_total"]


def test_snapshot_level_ranking_is_reported_for_each_junction():
    metrics = json.loads(_artifacts()["metrics"].read_text())
    for split in ("validation", "test"):
        by_junction = metrics[split]["per_junction"]
        assert set(by_junction) == set(JUNCTIONS)
        for j in JUNCTIONS:
            assert by_junction[j]["ranking"]["top1_total"] == 15
            assert len(by_junction[j]["ranking"]["per_snapshot"]) == 15
            assert np.isfinite(by_junction[j]["mae"])
            assert np.isfinite(by_junction[j]["rmse"])
