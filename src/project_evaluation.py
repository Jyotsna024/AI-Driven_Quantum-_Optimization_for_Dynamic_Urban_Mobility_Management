"""Leakage-aware evaluation of the saved ST-GNN, S1, S2, S3, and Phase 8 data.

The evaluator only reads existing artifacts and performs ST-GNN inference with
the saved checkpoint. It does not train a model, invoke SUMO, or run a solver.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
GRAPHS = RESULTS / "graphs"
JUNCTIONS = ("J0", "J1", "J2", "J3")
FEATURES = ("queue_length", "vehicle_count", "mean_speed_mps", "mean_delay_s", "throughput", "arrivals")
HORIZONS = (1, 2, 3)
SCENARIOS = ("low", "medium", "high", "peak", "incident")
PLAN_VECTORS = tuple(itertools.product(range(4), repeat=4))
OUTPUTS = (
    RESULTS / "project_evaluation.json",
    RESULTS / "project_evaluation.csv",
    RESULTS / "evaluation_report.md",
    GRAPHS / "stgnn_prediction_comparison.png",
    GRAPHS / "waiting_time_comparison.png",
    GRAPHS / "queue_length_comparison.png",
    GRAPHS / "throughput_comparison.png",
    GRAPHS / "solver_comparison.png",
)


def regression_metrics(actual: Any, predicted: Any) -> dict[str, Any]:
    """Regression metrics over paired finite observations.

    RMAE is 100 * MAE / mean(abs(actual)); zero denominator is unavailable.
    R² is 1 - SSE/SST; it is unavailable for constant actuals.
    """
    y = np.asarray(actual, dtype=float).reshape(-1)
    p = np.asarray(predicted, dtype=float).reshape(-1)
    if y.shape != p.shape:
        raise ValueError("actual and predicted arrays must have equal shapes")
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    n = int(y.size)
    if n == 0:
        return {"n": 0, "mae": None, "rmae_pct": None, "rmse": None,
                "r2": None, "pearson_r": None, "spearman_r": None}
    error = p - y
    mae = float(np.mean(np.abs(error)))
    denominator = float(np.mean(np.abs(y)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    if n >= 2 and np.std(y) > 0 and np.std(p) > 0:
        pearson = float(np.corrcoef(y, p)[0, 1])
        from scipy.stats import spearmanr
        spearman = float(spearmanr(y, p).statistic)
        if not math.isfinite(spearman):
            spearman = None
    else:
        pearson, spearman = None, None
    return {
        "n": n,
        "mae": mae,
        "rmae_pct": None if denominator == 0 else 100.0 * mae / denominator,
        "rmse": rmse,
        "r2": None if n < 2 or sst == 0 else 1.0 - float(np.sum(error ** 2)) / sst,
        "pearson_r": pearson,
        "spearman_r": spearman,
    }


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _stgnn_test_windows() -> list[dict]:
    """Recreate the published seed-4 windows from raw inputs, without fitting."""
    raw_dir = ROOT / "data" / "raw"
    out: list[dict] = []
    # The checked-in pipeline split is seed-based. The current working-tree
    # manifest is user-modified and contains only seed 0, so resolve the held-
    # out seed-4 run files directly rather than changing or trusting that file.
    selected = [(scenario, 4, mode, f"{scenario}_s4_{mode}")
                for scenario in SCENARIOS for mode in ("fixed", "varied")]
    for scenario, seed, mode, run_id in selected:
        csv_path = raw_dir / f"{run_id}.csv"
        if not csv_path.is_file():
            continue
        frame = pd.read_csv(csv_path)
        timesteps = sorted(pd.to_numeric(frame["timestamp_s"]).unique())
        rows_by_time = []
        for ts in timesteps:
            sub = frame[frame["timestamp_s"] == ts].set_index("junction_short")
            vals = []
            for junction in JUNCTIONS:
                if junction not in sub.index:
                    raise ValueError(f"Missing {junction} at {run_id}, timestamp {ts}")
                row = sub.loc[junction]
                vals.append([float(row[f]) if pd.notna(row[f]) else (14.0 if f == "mean_speed_mps" else 0.0)
                             for f in FEATURES])
            rows_by_time.append(vals)
        series = np.asarray(rows_by_time, dtype=np.float32)
        for t in range(2, len(timesteps) - 3):
            target_indices = range(t + 1, t + 4)
            if any(timesteps[i] <= 300 for i in target_indices):
                continue
            x = np.transpose(series[t - 2:t + 1], (1, 0, 2))
            y = np.transpose(series[t + 1:t + 4, :, 0], (1, 0))
            out.append({"scenario": str(scenario), "seed": seed, "mode": mode,
                        "run_id": run_id, "forecast_start_s": int(timesteps[t]),
                        "x": x, "actual": y})
    if not out:
        raise FileNotFoundError("No seed-4 post-warm-up ST-GNN windows could be reconstructed")
    return out


def _stgnn_evaluation() -> tuple[dict, dict]:
    sys.path.insert(0, str(ROOT / "src"))
    from stgnn import STGNNPredictor, EDGE_INDEX, NORM_EDGE_WEIGHTS
    import torch

    predictor = STGNNPredictor(ROOT / "models" / "stgnn.pt", ROOT / "models" / "scaler.pkl")
    samples = _stgnn_test_windows()
    actual = np.stack([item["actual"] for item in samples])
    features = np.stack([item["x"] for item in samples])
    predictions = []
    with torch.no_grad():
        for start in range(0, len(features), 32):
            batch = features[start:start + 32]
            b, n, t, f = batch.shape
            normalized = predictor.scaler.transform(batch.reshape(-1, f)).reshape(b, n, t, f).astype(np.float32)
            raw = predictor.model(torch.from_numpy(normalized), EDGE_INDEX, NORM_EDGE_WEIGHTS).numpy()
            predictions.append(np.clip(raw * predictor.target_std + predictor.target_mean, 0.0, None))
    prediction = np.concatenate(predictions, axis=0)
    persistence = np.stack([np.repeat(item["x"][:, -1, 0][:, None], 3, axis=1) for item in samples])

    def summarize(y: np.ndarray, pred: np.ndarray) -> dict:
        result: dict[str, Any] = {"overall": regression_metrics(y, pred), "by_horizon": {},
                                  "by_junction": {}, "by_scenario": {},
                                  "by_scenario_horizon": {}, "by_scenario_junction_horizon": {}}
        for h in range(3):
            result["by_horizon"][f"H{h+1}"] = regression_metrics(y[:, :, h], pred[:, :, h])
        for j, junction in enumerate(JUNCTIONS):
            result["by_junction"][junction] = regression_metrics(y[:, j, :], pred[:, j, :])
        scenario_array = np.asarray([x["scenario"] for x in samples])
        for scenario in SCENARIOS:
            idx = np.flatnonzero(scenario_array == scenario)
            if not len(idx):
                continue
            result["by_scenario"][scenario] = regression_metrics(y[idx], pred[idx])
            result["by_scenario_horizon"][scenario] = {
                f"H{h+1}": regression_metrics(y[idx, :, h], pred[idx, :, h]) for h in range(3)}
            result["by_scenario_junction_horizon"][scenario] = {
                junction: {f"H{h+1}": regression_metrics(y[idx, j, h], pred[idx, j, h])
                           for h in range(3)} for j, junction in enumerate(JUNCTIONS)}
        return result

    eval_data = {
        "status": "AVAILABLE",
        "source": "saved models/stgnn.pt and scaler.pkl; raw seed-4 test runs only",
        "split": "test",
        "test_seed": 4,
        "n_forecast_windows": len(samples),
        "n_raw_test_runs": len({x["run_id"] for x in samples}),
        "prediction_horizon_seconds": [60, 120, 180],
        "stgnn": summarize(actual, prediction),
        "persistence": summarize(actual, persistence),
    }
    # Keep true/prediction pairs only in memory for a test-set plot; don't emit
    # duplicate raw datasets into the evaluation package.
    plot_data = {"actual": actual, "stgnn": prediction, "persistence": persistence}
    return eval_data, plot_data


def _s1_evaluation() -> dict:
    with (ROOT / "models" / "cost_model_s1_v2.pkl").open("rb") as handle:
        saved = pickle.load(handle)
    frame = pd.read_csv(ROOT / "data" / "features_s1_v2.csv")
    test = frame[(frame["split"] == "test") & (frame["seed"] == 4)].copy()
    if test.empty:
        return {"status": "NOT AVAILABLE", "reason": "No held-out seed-4 S1 rows."}
    cols = saved["feature_cols"]
    predicted = np.clip(saved["model"].predict(saved["scaler"].transform(test[cols])), 0.0, None)
    actual = pd.to_numeric(test["mean_delay_s"], errors="coerce").to_numpy(float)
    test["actual_delay_s"] = actual
    test["predicted_delay_s"] = predicted

    by_junction = {}
    by_scenario = {}
    top1_rows = []
    for junction, group in test.groupby("target_junction"):
        by_junction[str(junction)] = regression_metrics(group.actual_delay_s, group.predicted_delay_s)
    for scenario, group in test.groupby("scenario"):
        by_scenario[str(scenario)] = regression_metrics(group.actual_delay_s, group.predicted_delay_s)
    by_snapshot_junction = {}
    for (sid, junction), group in test.groupby(["snapshot_id", "target_junction"]):
        actual_min = group.loc[group.actual_delay_s.idxmin(), "applied_plan_id"]
        pred_min = group.loc[group.predicted_delay_s.idxmin(), "applied_plan_id"]
        correct = bool(actual_min == pred_min)
        top1_rows.append(correct)
        by_snapshot_junction[f"{sid}:{junction}"] = {
            "scenario": str(group.scenario.iloc[0]),
            "actual_best_plan": int(actual_min), "predicted_best_plan": int(pred_min),
            "top1_agreement": correct, "n_plans": int(len(group)),
            "spearman": regression_metrics(group.actual_delay_s, group.predicted_delay_s)["spearman_r"],
        }
    def top1_summary(rows: list[dict]) -> dict:
        correct = sum(bool(row["top1_agreement"]) for row in rows)
        return {"correct": correct, "total": len(rows),
                "rate": correct / len(rows) if rows else None,
                "denominator": "snapshot x target-junction groups with four plan outcomes"}
    top1_by_junction = {}
    top1_by_scenario = {}
    for junction in sorted(test.target_junction.unique()):
        group_rows = [v for key, v in by_snapshot_junction.items() if key.endswith(f":{junction}")]
        top1_by_junction[str(junction)] = top1_summary(group_rows)
    for scenario in sorted(test.scenario.unique()):
        group_rows = [v for v in by_snapshot_junction.values() if v["scenario"] == scenario]
        top1_by_scenario[str(scenario)] = top1_summary(group_rows)
    return {
        "status": "AVAILABLE", "version": "S1 v2", "split": "test", "test_seed": 4,
        "n_plan_outcomes": int(len(test)),
        "regression": regression_metrics(actual, predicted),
        "spearman_plan_outcomes": regression_metrics(actual, predicted)["spearman_r"],
        "top1_plan_agreement": {"correct": int(sum(top1_rows)), "total": len(top1_rows),
                                "rate": float(np.mean(top1_rows)) if top1_rows else None,
                                "denominator": "snapshot x target-junction groups with four plan outcomes"},
        "top1_by_junction": top1_by_junction, "top1_by_scenario": top1_by_scenario,
        "by_junction": by_junction, "by_scenario": by_scenario,
        "by_snapshot_junction": by_snapshot_junction,
    }


def _s2_evaluation() -> dict:
    path = ROOT / "models" / "qubo_s2_v3.npz"
    with np.load(path) as data:
        q_values = data["validation_Q"].copy()
        snapshot_ids = [str(x) for x in data["validation_snapshot_ids"].tolist()]
    rollouts = pd.read_csv(ROOT / "data" / "joint_rollouts_s2_v3.csv")
    full = rollouts[(rollouts["experiment"] == "full_joint_validation") &
                    (rollouts["split"] == "test")].copy()
    full["actual_total_delay_s"] = pd.to_numeric(full["mean_delay_s"], errors="coerce")
    results_by_snapshot = {}
    all_predicted, all_actual = [], []
    all_bitstrings = ((np.arange(1 << 16, dtype=np.uint32)[:, None] >> np.arange(16, dtype=np.uint32)) & 1).astype(float)
    for sid, q in zip(snapshot_ids, q_values):
        block = full[full.snapshot_id == sid]
        if block.empty:
            continue
        if block.plan_vector.nunique() != 256 or not block.groupby("plan_vector").measured_junction.nunique().eq(4).all():
            continue
        per_plan = block.groupby("plan_vector", sort=False).actual_total_delay_s.sum()
        # Keep only complete plan vectors with all four junction outcomes.
        grouped = block.groupby("plan_vector", sort=False).agg(
            actual_total_delay_s=("actual_total_delay_s", "sum"),
            junction_count=("measured_junction", "nunique"))
        grouped = grouped[grouped.junction_count == 4]
        vectors = [tuple(json.loads(text)) for text in grouped.index]
        xmat = np.zeros((len(vectors), 16), dtype=float)
        for row, vector in enumerate(vectors):
            for j, plan in enumerate(vector):
                xmat[row, j * 4 + plan] = 1.0
        pred_objectives = np.einsum("bi,ij,bj->b", xmat, q, xmat)
        actual_values = grouped.actual_total_delay_s.to_numpy(float)
        pred_idx = int(np.argmin(pred_objectives))
        actual_idx = int(np.argmin(actual_values))
        pred_vector, actual_vector = vectors[pred_idx], vectors[actual_idx]
        actual_min_mask = np.isclose(actual_values, actual_values[actual_idx], rtol=0, atol=1e-9)
        raw_bitstring_energy = np.einsum("bi,ij,bj->b", all_bitstrings, q, all_bitstrings)
        raw_best_mask = np.isclose(raw_bitstring_energy, raw_bitstring_energy.min(), rtol=0, atol=1e-9)
        raw_best_samples = all_bitstrings[raw_best_mask].astype(int).reshape(-1, 4, 4)
        raw_best_feasible = np.all(raw_best_samples.sum(axis=2) == 1, axis=1)
        predicted_actual_delay = float(actual_values[pred_idx])
        optimal_actual_delay = float(actual_values[actual_idx])
        results_by_snapshot[sid] = {
            "split": "test", "scenario": str(block.scenario.iloc[0]), "seed": int(block.seed.iloc[0]),
            "n_complete_plan_vectors": len(vectors), "predicted_optimum_plan_vector": list(pred_vector),
            "actual_sumo_optimum_plan_vector": list(actual_vector),
            "top1_plan_agreement": bool(actual_min_mask[pred_idx]),
            "actual_sumo_optimal_plan_tie_count": int(actual_min_mask.sum()),
            "predicted_qubo_optimal_energy_tie_count": int(np.isclose(pred_objectives, pred_objectives[pred_idx], rtol=0, atol=1e-9).sum()),
            "predicted_optimum_actual_sumo_delay_s": predicted_actual_delay,
            "actual_sumo_optimum_delay_s": optimal_actual_delay,
            "optimization_regret_s": predicted_actual_delay - optimal_actual_delay,
            "spearman_predicted_qubo_vs_actual_sumo": regression_metrics(actual_values, pred_objectives)["spearman_r"],
            "spearman_denominator_plan_vectors": len(vectors),
            "unconstrained_qubo_global_minimum_feasible": bool(np.all(raw_best_feasible)),
            "unconstrained_qubo_global_minimum_tie_count": int(raw_best_mask.sum()),
            "feasible_candidate_count": int(np.all(xmat.reshape(-1, 4, 4).sum(axis=2) == 1, axis=1).sum()),
            "candidate_count": len(vectors),
        }
        all_predicted.extend(pred_objectives.tolist())
        all_actual.extend(actual_values.tolist())
    if not results_by_snapshot:
        return {"status": "NOT AVAILABLE", "reason": "No complete held-out full-plan S2 v3 rollouts."}
    top_matches = sum(v["top1_plan_agreement"] for v in results_by_snapshot.values())
    candidate_n = sum(v["candidate_count"] for v in results_by_snapshot.values())
    feasible_n = sum(v["unconstrained_qubo_global_minimum_feasible"] for v in results_by_snapshot.values())
    regrets = [v["optimization_regret_s"] for v in results_by_snapshot.values()]
    snapshot_spearman = [v["spearman_predicted_qubo_vs_actual_sumo"]
                         for v in results_by_snapshot.values()
                         if v["spearman_predicted_qubo_vs_actual_sumo"] is not None]
    return {
        "status": "AVAILABLE", "version": "S2 v3", "split": "test", "test_seed": 4,
        "scientific_validation_pass": False,
        "prediction_actual_optimum_distinction": "Predicted optimum minimizes serialized QUBO energy; actual SUMO optimum minimizes measured total delay across the same 256 plan vectors.",
        "top1_plan_agreement": {"correct": top_matches, "total": len(results_by_snapshot),
                                "rate": top_matches / len(results_by_snapshot),
                                "denominator": "held-out test snapshots"},
        "spearman_predicted_vs_actual": {
            "label": "Pooled Spearman",
            "aggregation": "pooled",
            "value": regression_metrics(all_actual, all_predicted)["spearman_r"],
            "n_plan_vectors": len(all_actual),
            "denominator": "all held-out feasible plan vectors across complete snapshots",
        },
        "mean_snapshot_spearman": {
            "label": "Mean snapshot-level Spearman",
            "aggregation": "arithmetic_mean_of_within_snapshot_correlations",
            "value": float(np.mean(snapshot_spearman)) if snapshot_spearman else None,
            "n_snapshots": len(snapshot_spearman),
            "denominator": "complete held-out snapshots with defined within-snapshot Spearman",
        },
        "mean_optimization_regret_s": float(np.mean(regrets)),
        "feasible_solution_rate": {"feasible": feasible_n, "total": len(results_by_snapshot),
                                   "rate": feasible_n / len(results_by_snapshot),
                                   "denominator": "held-out snapshots whose unconstrained serialized-QUBO global minimum is exactly-one-hot"},
        "candidate_grid_feasibility_rate": {"feasible": candidate_n, "total": candidate_n,
                                            "rate": candidate_n / candidate_n,
                                            "denominator": "all candidate plan vectors in the complete held-out SUMO grid"},
        "by_snapshot": results_by_snapshot,
    }


def _s3_evaluation() -> dict:
    payload = _read_json(RESULTS / "solver_metrics_s3_v3.json")
    solvers = payload.get("solvers", {})
    exact = solvers.get("ExactSolver", {})
    exact_plan = exact.get("best_plans")
    exact_energy = exact.get("best_energy")
    out = {"status": "AVAILABLE", "version": "S3 v3", "qubo_input": payload.get("input_file"),
           "exact_reference": {"plan_vector": exact_plan, "energy": exact_energy,
                               "runtime_s": exact.get("runtime_s"),
                               "exhaustive_search_space_size": exact.get("n_raw_samples"),
                               "exhaustive_search_space_feasible_assignments": exact.get("n_feasible_samples"),
                               "exhaustive_search_space_feasible_fraction": exact.get("feasible_rate")},
           "solvers": {}}
    for name in ("ExactSolver", "SimulatedAnnealing", "QAOA"):
        item = solvers.get(name)
        if not item:
            out["solvers"][name] = {"status": "NOT AVAILABLE", "reason": "No saved solver result."}
            continue
        seed_records = item.get("per_seed", [])
        seed_plans = [record.get("best_feasible_x") for record in seed_records]
        # Convert 16-bit one-hot samples to plan vectors only when feasible.
        def decode(x):
            if x is None or len(x) != 16:
                return None
            a = np.asarray(x, dtype=int).reshape(4, 4)
            return a.argmax(axis=1).tolist() if np.all(a.sum(axis=1) == 1) else None
        per_seed_plans = [decode(x) for x in seed_plans]
        exact_matches = [p == exact_plan for p in per_seed_plans if p is not None]
        energies = [r.get("best_feasible_energy") for r in seed_records if r.get("best_feasible_energy") is not None]
        feasibility_metric = _s3_feasibility_metric(name, item)
        out["solvers"][name] = {
            "solution_plan_vector": item.get("best_plans"),
            "exact_plan_agreement": {
                "value": None if name == "ExactSolver" else (item.get("best_plans") == exact_plan if item.get("best_plans") is not None else None),
                "status": "REFERENCE_OPTIMUM" if name == "ExactSolver" else "AVAILABLE",
                "note": "ExactSolver defines the reference optimum; self-agreement is not an independent performance result." if name == "ExactSolver" else None,
            },
            "best_energy": item.get("best_energy"), "traffic_delay_s": item.get("traffic_delay_s"),
            "runtime_s_total": item.get("runtime_s"),
            "runtime_s_per_seed": [r.get("runtime_s") for r in seed_records] if all("runtime_s" in r for r in seed_records) else None,
            "runtime_per_seed_status": "AVAILABLE" if seed_records and all("runtime_s" in r for r in seed_records) else "NOT AVAILABLE",
            "runtime_per_seed_reason": (None if seed_records and all("runtime_s" in r for r in seed_records) else
                                         "The saved S3 artifact contains only one ExactSolver timing." if name == "ExactSolver" else
                                         "The saved QAOA artifact records aggregate runtime but no per-seed timings." if name == "QAOA" else
                                         "No repeated per-seed timing data are saved."),
            "feasibility_metric": feasibility_metric,
            "raw_samples": item.get("n_raw_samples") if name != "ExactSolver" else None,
            "feasible_raw_samples": item.get("n_feasible_samples") if name != "ExactSolver" else None,
            "best_solution_feasible": item.get("best_plans") is not None,
            "repeat_runs": len(seed_records),
            "per_seed_exact_plan_agreement": {"correct": int(sum(exact_matches)), "total": len(exact_matches),
                                               "rate": float(np.mean(exact_matches)) if exact_matches else None},
            "per_seed_unique_best_plan_count": len({tuple(p) for p in per_seed_plans if p is not None}) if seed_records else None,
            "per_seed_best_energy_mean": float(np.mean(energies)) if energies else None,
            "per_seed_best_energy_std": float(np.std(energies, ddof=1)) if len(energies) > 1 else (0.0 if energies else None),
        }
    return out


def _s3_feasibility_metric(name: str, item: dict) -> dict:
    """Return explicitly named feasibility metrics with their true denominator."""
    feasible = int(item.get("n_feasible_samples", 0))
    total = int(item.get("n_raw_samples", 0))
    if name == "ExactSolver":
        return {
            "label": "Exhaustive search-space feasible fraction",
            "kind": "exhaustive_search_space_feasible_fraction",
            "feasible": feasible,
            "total": total,
            "value": float(item.get("feasible_rate")) if item.get("feasible_rate") is not None else None,
            "denominator": "all 16-bit assignments enumerated by ExactSolver",
        }
    return {
        "label": "Sampled-output feasibility rate",
        "kind": "sampled_output_feasibility_rate",
        "feasible": feasible,
        "total": total,
        "value": float(item.get("feasible_rate")) if item.get("feasible_rate") is not None else None,
        "denominator": "all raw samples returned by this solver",
    }


def _phase8_evaluation() -> dict:
    path = RESULTS / "phase8_eval_v1" / "episode_metrics.csv"
    if not path.is_file():
        return {"status": "NOT AVAILABLE", "reason": "Saved Phase 8 episode_metrics.csv is missing."}
    frame = pd.read_csv(path)
    specs = {
        "waiting_time": ("total_waiting_time_vehicle_s", "vehicle-s"),
        "queue_length": ("average_queue_vehicles", "vehicles"),
        "throughput": ("throughput_total_junction_crossings", "junction crossings"),
    }
    result = {"status": "AVAILABLE", "source": str(path.relative_to(ROOT)), "episode_count": int(len(frame)),
              "metrics_by_controller": {}}
    for controller, group in frame.groupby("controller"):
        result["metrics_by_controller"][str(controller)] = {}
        for name, (column, unit) in specs.items():
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            result["metrics_by_controller"][str(controller)][name] = {
                "n": int(len(values)), "mean": float(values.mean()) if len(values) else None,
                "std": float(values.std(ddof=1)) if len(values) > 1 else (0.0 if len(values) == 1 else None),
                "unit": unit}
    return result


def build_evaluation() -> tuple[dict, dict]:
    stgnn, stgnn_plot_data = _stgnn_evaluation()
    evaluation = {
        "schema_version": 1,
        "project": "AI-Driven Quantum Optimization for Dynamic Urban Mobility Management",
        "data_policy": "Saved project artifacts only; held-out test data are scored separately from train and validation data.",
        "metric_definitions": {
            "MAE": "mean(|prediction - actual|) over paired finite values",
            "RMAE_percent": "100 × MAE / mean(|actual|); NOT AVAILABLE when mean(|actual|) is zero",
            "RMSE": "sqrt(mean((prediction - actual)^2)) over paired finite values",
            "R2": "1 - sum((prediction - actual)^2) / sum((actual - mean(actual))^2); NOT AVAILABLE for fewer than two values or constant actuals",
            "Pearson_r": "Pearson linear correlation on paired finite values; NOT AVAILABLE for fewer than two or constant inputs",
            "Spearman_r": "Spearman rank correlation on paired finite values; NOT AVAILABLE for fewer than two or constant inputs",
            "S2_pooled_Spearman": "rank all candidate-plan outcomes pooled across held-out snapshots and correlate predicted QUBO energy with measured SUMO delay; denominator is the complete plan outcomes (512 here)",
            "S2_mean_snapshot_Spearman": "arithmetic mean of within-snapshot Spearman correlations; denominator is held-out snapshots with defined correlations (2 here); this differs from pooled Spearman",
            "Top1_plan_agreement": "count(predicted minimum-cost plan equals measured minimum-cost plan) / eligible snapshot or snapshot-junction groups",
            "S2_regret_s": "measured SUMO delay for predicted-QUBO-optimum plan minus measured SUMO delay for the actual SUMO-optimum plan, same snapshot and plan grid",
            "S2_feasible_global_minimum_rate": "held-out snapshots whose unconstrained serialized-QUBO global minimum is one-hot feasible / all held-out snapshots",
            "S3_exhaustive_search_space_feasible_fraction": "feasible assignments / all bitstrings exhaustively enumerated by ExactSolver",
            "S3_sampled_output_feasibility_rate": "feasible returned samples / all raw sampled outputs from SA or QAOA; denominator differs from ExactSolver's exhaustive search space",
        },
        "stgnn": stgnn,
        "s1_cost_model": _s1_evaluation(),
        "s2_qubo": _s2_evaluation(),
        "s3_solvers": _s3_evaluation(),
        "phase8_controller_comparison": _phase8_evaluation(),
        "limitations": [
            "The modified medium_s0 raw files are training-seed data. The checkpoint and scaler are used as saved; no retraining occurs. Held-out ST-GNN scoring uses seed 4 only.",
            "Phase 8 controller episodes are simulation outcomes, not a regression target or proof of controller improvement.",
            "S2 v3 is marked scientifically unvalidated in its saved artifacts; plan-selection and regret results are diagnostic only.",
            "QAOA results are from Qiskit Aer simulation, not quantum hardware.",
        ],
    }
    return evaluation, stgnn_plot_data


def _metric_rows(evaluation: dict) -> list[dict]:
    rows: list[dict] = []
    def walk(system: str, obj: Any, scope: str = "overall") -> None:
        if isinstance(obj, dict):
            if {"label", "value", "denominator"}.issubset(obj):
                rows.append({
                    "system": system, "scope": scope, "metric": obj["label"],
                    "label": obj["label"], "value": obj.get("value"),
                    "n": obj.get("n_plan_vectors", obj.get("n_snapshots", obj.get("total"))),
                    "numerator": obj.get("feasible"), "denominator": obj.get("denominator"),
                    "aggregation": obj.get("aggregation"),
                    "unit": "dimensionless",
                })
                return
            if {"value", "status", "note"}.issubset(obj):
                rows.append({
                    "system": system, "scope": scope,
                    "metric": "Exact-plan agreement",
                    "label": "Exact-plan agreement",
                    "value": obj.get("value"), "n": None, "numerator": None,
                    "denominator": obj.get("note"), "aggregation": obj.get("status"),
                    "unit": "plan agreement",
                })
                return
            if "n" in obj and any(k in obj for k in ("mae", "rmse", "r2", "spearman_r")):
                for metric, value in obj.items():
                    if metric == "n":
                        continue
                    rows.append({"system": system, "scope": scope, "metric": metric,
                                 "label": metric, "value": value, "n": obj.get("n"),
                                 "numerator": None, "denominator": "paired finite observations", "aggregation": None,
                                 "unit": "dimensionless" if metric in ("r2", "pearson_r", "spearman_r") else "target units"})
            else:
                for key, value in obj.items():
                    walk(system, value, f"{scope}/{key}" if scope != "overall" else str(key))
        elif isinstance(obj, list):
            return
        elif isinstance(obj, (int, float)) or obj is None:
            rows.append({"system": system, "scope": scope.rsplit("/", 1)[0] if "/" in scope else "overall",
                         "metric": scope.rsplit("/", 1)[-1], "label": scope.rsplit("/", 1)[-1],
                         "value": obj, "n": None, "numerator": None, "denominator": None,
                         "aggregation": None, "unit": None})

    for name in ("stgnn", "s1_cost_model"):
        walk(name, evaluation.get(name, {}))
    for name in ("s2_qubo", "s3_solvers", "phase8_controller_comparison"):
        section = evaluation.get(name, {})
        # Flatten numeric summary fields while retaining their structured scope.
        walk(name, section)
    return rows


def _safe_write(path: Path, data: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing project data: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    if tmp.exists():
        raise FileExistsError(f"Refusing to overwrite existing temporary file: {tmp}")
    tmp.write_bytes(data)
    tmp.replace(path)


def _write_json(path: Path, data: dict) -> None:
    _safe_write(path, (json.dumps(data, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def _solver_comparison_graph(evaluation: dict) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from io import BytesIO

    solvers = evaluation["s3_solvers"].get("solvers", {})
    order = ("ExactSolver", "SimulatedAnnealing", "QAOA")
    labels = {"ExactSolver": "Exact (reference)", "SimulatedAnnealing": "SA", "QAOA": "QAOA Aer"}
    entries = [(name, solvers[name]) for name in order if name in solvers]
    fig, axs = plt.subplots(1, 3, figsize=(14, 5))
    colors = ["#4c78a8", "#f58518", "#e45756"]

    runtimes = [entry.get("runtime_s_total") for _, entry in entries]
    axs[0].bar([labels[name] for name, _ in entries], runtimes, color=colors[:len(entries)])
    axs[0].set_title("Total saved runtime")
    axs[0].set_ylabel("Seconds")
    axs[0].grid(axis="y", alpha=.2)

    energies = [entry.get("best_energy") for _, entry in entries]
    axs[1].bar([labels[name] for name, _ in entries], energies, color=colors[:len(entries)])
    axs[1].set_title("Best feasible QUBO energy")
    axs[1].set_ylabel("QUBO energy (lower is better)")
    axs[1].grid(axis="y", alpha=.2)

    feasibility = [entry.get("feasibility_metric", {}) for _, entry in entries]
    values = [metric.get("value") for metric in feasibility]
    bar_colors = ["#4c78a8" if name == "ExactSolver" else "#f58518" for name, _ in entries]
    bars = axs[2].bar([labels[name] for name, _ in entries], values, color=bar_colors)
    for bar, metric in zip(bars, feasibility):
        if metric.get("total"):
            bar_label = f"{metric['feasible']}/{metric['total']}"
            axs[2].annotate(bar_label, (bar.get_x() + bar.get_width()/2, bar.get_height()),
                            xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
    axs[2].set_title("Feasibility (different denominators)")
    axs[2].set_ylabel("Feasible fraction")
    axs[2].set_ylim(0, 1.12)
    axs[2].grid(axis="y", alpha=.2)
    axs[2].legend(handles=[
        Patch(facecolor="#4c78a8", label="Exhaustive search-space feasible fraction"),
        Patch(facecolor="#f58518", label="Sampled-output feasibility rate"),
    ], loc="upper left", fontsize=8)
    fig.suptitle("S3 solver comparison — feasibility bases differ", fontsize=13)
    fig.text(.5, .01, "ExactSolver defines the reference optimum; its self-match is not an independent performance result.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .92))
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=160)
    plt.close(fig)
    return buffer.getvalue()


def _generate_graphs(evaluation: dict, stgnn_data: dict) -> dict[str, bytes]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from io import BytesIO

    outputs: dict[str, bytes] = {}
    actual, model, persistence = (stgnn_data[k] for k in ("actual", "stgnn", "persistence"))
    means = [float(np.mean(x[:, :, h])) for x in (actual, model, persistence) for h in range(3)]
    fig, ax = plt.subplots(figsize=(8, 5))
    horizons = np.arange(3)
    ax.plot(horizons, [actual[:, :, h].mean() for h in range(3)], marker="o", label="Actual")
    ax.plot(horizons, [model[:, :, h].mean() for h in range(3)], marker="o", label="ST-GNN")
    ax.plot(horizons, [persistence[:, :, h].mean() for h in range(3)], marker="o", label="Persistence")
    ax.set(xticks=horizons, xticklabels=["60 s", "120 s", "180 s"], ylabel="Queue length (vehicles)",
           title="Held-out test queue prediction by horizon")
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    buf=BytesIO(); fig.savefig(buf, format="png", dpi=160); plt.close(fig); outputs["stgnn_prediction_comparison.png"]=buf.getvalue()

    phase8 = evaluation["phase8_controller_comparison"]
    names = {"fixed_time":"Fixed time", "max_pressure":"Max pressure", "sa_qubo":"SA + QUBO", "qaoa_aer_sampled":"QAOA Aer"}
    for key, title, ylabel in (("waiting_time", "waiting_time_comparison.png", "Total waiting time (vehicle-s)"),
                               ("queue_length", "queue_length_comparison.png", "Average queue (vehicles)"),
                               ("throughput", "throughput_comparison.png", "Junction crossings")):
        values = phase8.get("metrics_by_controller", {})
        controllers = [x for x in names if x in values]
        means = [values[c][key]["mean"] for c in controllers]
        stds = [values[c][key]["std"] or 0 for c in controllers]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar([names[c] for c in controllers], means, yerr=stds, capsize=4, color="#4c78a8")
        ax.set_ylabel(ylabel); ax.set_title(title.replace("_", " ").removesuffix(".png").title()+" across saved episodes")
        ax.tick_params(axis="x", rotation=15); ax.grid(axis="y", alpha=.2); fig.tight_layout()
        buf=BytesIO(); fig.savefig(buf, format="png", dpi=160); plt.close(fig); outputs[title]=buf.getvalue()

    outputs["solver_comparison.png"] = _solver_comparison_graph(evaluation)
    return outputs


def _report(evaluation: dict) -> str:
    st= evaluation["stgnn"]["stgnn"]["overall"]
    pers=evaluation["stgnn"]["persistence"]["overall"]
    s1=evaluation["s1_cost_model"]
    s2=evaluation["s2_qubo"]
    s3=evaluation["s3_solvers"]["solvers"]
    lines=["# Project Evaluation Report", "", "Metrics are calculated from saved project artifacts. No models were retrained, no SUMO simulations were run, and solver outputs were not regenerated.",
           "", "## ST-GNN held-out test (seed 4)", "", f"Forecast windows: {evaluation['stgnn']['n_forecast_windows']}; horizons: 60, 120, and 180 seconds.",
           "", "| Model | n | MAE | RMAE (%) | RMSE | R² | Pearson r | Spearman r |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, m in (("ST-GNN", st), ("Persistence", pers)):
        lines.append(f"| {label} | {m['n']} | {_fmt(m['mae'])} | {_fmt(m['rmae_pct'])} | {_fmt(m['rmse'])} | {_fmt(m['r2'])} | {_fmt(m['pearson_r'])} | {_fmt(m['spearman_r'])} |")
    lines += ["", "Detailed metrics are also grouped by horizon, junction, scenario, and scenario × junction × horizon in the JSON output.",
              "", "## S1 cost model (held-out test seed 4)", "", f"Outcomes: {s1.get('n_plan_outcomes', 'NOT AVAILABLE')}; top-1 plan agreement: {_format_rate(s1.get('top1_plan_agreement',{}).get('correct'),s1.get('top1_plan_agreement',{}).get('total'))}.",
              f"Regression: {_metrics_line(s1.get('regression',{}))}", "",
              "Top-1 compares the plan with minimum predicted delay against the minimum measured SUMO delay within each snapshot × target-junction group.",
              "", "## S2 QUBO (held-out test snapshots)", "", f"Top-1 plan agreement: {_format_rate(s2.get('top1_plan_agreement',{}).get('correct'),s2.get('top1_plan_agreement',{}).get('total'))}; mean regret: {_fmt(s2.get('mean_optimization_regret_s'))} s; feasible unconstrained QUBO global-minimum rate: {_format_rate(s2.get('feasible_solution_rate',{}).get('feasible'),s2.get('feasible_solution_rate',{}).get('total'))}.",
              f"Pooled Spearman across {s2.get('spearman_predicted_vs_actual',{}).get('n_plan_vectors')} plan outcomes: {_fmt(s2.get('spearman_predicted_vs_actual',{}).get('value'))}.",
              f"Mean snapshot-level Spearman across {s2.get('mean_snapshot_spearman',{}).get('n_snapshots')} held-out snapshots: {_fmt(s2.get('mean_snapshot_spearman',{}).get('value'))}.",
              "These summarize the same held-out evaluations with different aggregation methods: pooled Spearman ranks all plan outcomes together, while mean snapshot-level Spearman averages each snapshot's within-snapshot rank correlation. They are distinct and both are retained.",
              "", "For each held-out snapshot, the predicted optimum minimizes serialized QUBO energy over the feasible plan grid. The actual SUMO optimum minimizes the measured sum of four junction delays over the same 256 vectors. Regret is the actual delay at the predicted optimum minus the actual optimum delay.",
              "", "The saved S2 v3 result is explicitly marked scientifically not validated; these held-out metrics are diagnostic and do not justify closed-loop optimization.", "", "## S3 solver results", "",
              "| Solver | Exact-plan agreement | Best energy | Total runtime (s) | Feasibility metric | Feasible / denominator | Rate | Repeated best-plan agreement |", "|---|---|---:|---:|---|---:|---:|---:|"]
    for name in ("ExactSolver", "SimulatedAnnealing", "QAOA"):
        x=s3.get(name,{})
        match=x.get("per_seed_exact_plan_agreement",{})
        agreement="Reference optimum (n/a)" if name == "ExactSolver" else str(x.get("exact_plan_agreement", {}).get("value"))
        repeat_agreement="n/a" if match.get("total",0)==0 else _format_rate(match.get("correct"),match.get("total"))
        feasibility=x.get("feasibility_metric", {})
        feasible=f"{feasibility.get('feasible')} / {feasibility.get('total')}"
        rate=_fmt(feasibility.get("value"))
        lines.append(f"| {name} | {agreement} | {_fmt(x.get('best_energy'))} | {_fmt(x.get('runtime_s_total'))} | {feasibility.get('label', 'NOT AVAILABLE')} | {feasible} | {rate} | {repeat_agreement} |")
    lines += ["", "SA and QAOA have ten saved seeds each; QAOA is classical Qiskit Aer simulation. ExactSolver enumerates 65,536 bitstrings, of which 256 satisfy exactly-one-plan-per-junction.",
              "The ExactSolver 256/65,536 value is the exhaustive search-space feasible fraction. SA and QAOA values are sampled-output feasibility rates. Their denominators describe different sets and are not directly comparable. ExactSolver defines the reference optimum; its self-match is not an independent performance result.",
              "Per-seed SA runtime is saved. Per-seed QAOA runtime is NOT AVAILABLE because only its aggregate runtime was recorded; ExactSolver has one saved timing and no repeated-run estimate.",
              "", "## Phase 8 controller outcomes", "", "The three requested traffic comparison plots summarize existing identical-condition episode results. They are simulation outcomes and descriptive summaries; they are not prediction accuracy metrics.",
              "", "## Metric formulas and denominators", "", "- MAE = mean absolute prediction error over paired finite values.",
              "- RMAE (%) = 100 × MAE / mean absolute actual. It is NOT AVAILABLE when mean absolute actual is zero.",
              "- RMSE = square root of mean squared prediction error.",
              "- R² = 1 − SSE/SST; it is NOT AVAILABLE for fewer than two observations or constant actual values.",
              "- Pearson and Spearman are computed on the same paired finite observations; correlation is NOT AVAILABLE with fewer than two observations or a constant input.",
              "- ST-GNN regression denominators are forecast windows × four junctions × horizons represented by the relevant group. Persistence is the last observed queue repeated across horizons.",
              "- S1 top-1 denominator is the count of test snapshot x target-junction groups with all four plan outcomes.",
              "- S2 top-1 denominator is held-out test snapshots; regret is in seconds and paired within each snapshot. Pooled Spearman denominator is 512 plan outcomes ranked together. Mean snapshot-level Spearman denominator is two snapshots with defined within-snapshot correlations; it is the arithmetic mean of those correlations.",
              "- S3 ExactSolver exhaustive search-space feasible fraction denominator is all 65,536 bitstrings. SA/QAOA sampled-output feasibility-rate denominators are each solver's raw sample counts. These rates are not directly comparable because one covers the full bitstring space and the others cover sampled outputs. Repeated plan agreement denominator is saved runs with a feasible best plan.",
              "", "## Limitations", ""]
    lines += [f"- {item}" for item in evaluation["limitations"]]
    lines += ["", "A JSON field is null when a metric is not calculable from saved data; its scope and denominators are retained in the structured output.", ""]
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "NOT AVAILABLE"
    return f"{value:.4f}"


def _format_rate(numerator: Any, denominator: Any) -> str:
    if numerator is None or not denominator:
        return "NOT AVAILABLE"
    return f"{int(numerator)}/{int(denominator)} ({100*float(numerator)/float(denominator):.1f}%)"


def _metrics_line(metrics: dict) -> str:
    return ", ".join(f"{key}={_fmt(metrics.get(key))}" for key in ("mae", "rmae_pct", "rmse", "r2", "spearman_r"))


def main() -> None:
    global ROOT, RESULTS, GRAPHS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Project root (defaults to this checkout)")
    args = parser.parse_args()
    ROOT = args.root.resolve()
    RESULTS = ROOT / "results"
    GRAPHS = RESULTS / "graphs"
    destinations = (
        RESULTS / "project_evaluation.json", RESULTS / "project_evaluation.csv", RESULTS / "evaluation_report.md",
        GRAPHS / "stgnn_prediction_comparison.png", GRAPHS / "waiting_time_comparison.png",
        GRAPHS / "queue_length_comparison.png", GRAPHS / "throughput_comparison.png", GRAPHS / "solver_comparison.png")
    existing = [str(p) for p in destinations if p.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing evaluation output(s): " + ", ".join(existing))
    evaluation, plot_data = build_evaluation()
    graph_blobs = _generate_graphs(evaluation, plot_data)
    csv = pd.DataFrame(_metric_rows(evaluation)).to_csv(index=False, na_rep="NOT AVAILABLE")
    _write_json(RESULTS / "project_evaluation.json", evaluation)
    _safe_write(RESULTS / "project_evaluation.csv", csv.encode("utf-8"))
    _safe_write(RESULTS / "evaluation_report.md", _report(evaluation).encode("utf-8"))
    for name, blob in graph_blobs.items():
        _safe_write(GRAPHS / name, blob)
    print(f"Saved project evaluation to {RESULTS}; ST-GNN test windows={evaluation['stgnn']['n_forecast_windows']}; S2 held-out snapshots={len(evaluation['s2_qubo'].get('by_snapshot', {}))}.")


if __name__ == "__main__":
    main()
