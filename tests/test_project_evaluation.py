"""Unit tests for leakage-aware project evaluation metrics."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from project_evaluation import regression_metrics, _s3_feasibility_metric


def test_regression_metrics_perfect_predictions():
    result = regression_metrics([1, 2, 3], [1, 2, 3])
    assert result["n"] == 3
    assert result["mae"] == pytest.approx(0)
    assert result["rmae_pct"] == pytest.approx(0)
    assert result["rmse"] == pytest.approx(0)
    assert result["r2"] == pytest.approx(1)
    assert result["pearson_r"] == pytest.approx(1)
    assert result["spearman_r"] == pytest.approx(1)


def test_rmae_uses_mean_absolute_actual_and_handles_zero_observations():
    result = regression_metrics([0, 2], [1, 3])
    assert result["mae"] == pytest.approx(1)
    assert result["rmae_pct"] == pytest.approx(100)


def test_rmae_unavailable_when_all_actual_values_are_zero():
    result = regression_metrics([0, 0, 0], [1, 0, 2])
    assert result["mae"] == pytest.approx(1)
    assert result["rmae_pct"] is None


def test_r2_unavailable_for_constant_actual_values():
    result = regression_metrics([4, 4, 4], [3, 4, 5])
    assert result["rmse"] == pytest.approx(np.sqrt(2 / 3))
    assert result["r2"] is None


def test_nonfinite_pairs_are_excluded_and_counted():
    result = regression_metrics([1, np.nan, 3, np.inf], [2, 9, 3, 0])
    assert result["n"] == 2
    assert result["mae"] == pytest.approx(0.5)


def test_empty_inputs_return_unavailable_metrics():
    result = regression_metrics([], [])
    assert result["n"] == 0
    assert all(result[k] is None for k in ("mae", "rmae_pct", "rmse", "r2", "pearson_r", "spearman_r"))


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError, match="equal shapes"):
        regression_metrics([1, 2], [1])


def test_constant_prediction_has_unavailable_correlation_but_finite_errors():
    result = regression_metrics([1, 2, 3], [2, 2, 2])
    assert result["mae"] == pytest.approx(2 / 3)
    assert result["pearson_r"] is None
    assert result["spearman_r"] is None


def test_s3_feasibility_labels_preserve_distinct_denominators():
    exact = _s3_feasibility_metric("ExactSolver", {
        "n_feasible_samples": 256, "n_raw_samples": 65536, "feasible_rate": 256 / 65536,
    })
    sa = _s3_feasibility_metric("SimulatedAnnealing", {
        "n_feasible_samples": 2500, "n_raw_samples": 2500, "feasible_rate": 1.0,
    })
    qaoa = _s3_feasibility_metric("QAOA", {
        "n_feasible_samples": 150, "n_raw_samples": 15000, "feasible_rate": 0.01,
    })
    assert exact["label"] == "Exhaustive search-space feasible fraction"
    assert exact["kind"] == "exhaustive_search_space_feasible_fraction"
    assert (exact["feasible"], exact["total"]) == (256, 65536)
    for sampled in (sa, qaoa):
        assert sampled["label"] == "Sampled-output feasibility rate"
        assert sampled["kind"] == "sampled_output_feasibility_rate"


def test_saved_evaluation_keeps_pooled_and_snapshot_mean_spearman():
    payload = json.loads((ROOT / "results/project_evaluation.json").read_text(encoding="utf-8"))
    s2 = payload["s2_qubo"]
    pooled = s2["spearman_predicted_vs_actual"]
    snapshot_mean = s2["mean_snapshot_spearman"]
    assert pooled["value"] == pytest.approx(-0.1134133493, abs=1e-9)
    assert pooled["n_plan_vectors"] == 512
    assert snapshot_mean["value"] == pytest.approx(0.1597623169, abs=1e-9)
    assert snapshot_mean["n_snapshots"] == 2
    per_snapshot = [row["spearman_predicted_qubo_vs_actual_sumo"]
                    for row in s2["by_snapshot"].values()]
    assert snapshot_mean["value"] == pytest.approx(float(np.mean(per_snapshot)))
    assert pooled["aggregation"] != snapshot_mean["aggregation"]

    csv = pd.read_csv(ROOT / "results/project_evaluation.csv")
    s2_rows = csv[(csv.system == "s2_qubo") & csv.metric.isin([
        "Pooled Spearman", "Mean snapshot-level Spearman"])]
    assert set(s2_rows.metric) == {"Pooled Spearman", "Mean snapshot-level Spearman"}
    values = dict(zip(s2_rows.metric, pd.to_numeric(s2_rows.value, errors="raise")))
    assert values["Pooled Spearman"] == pytest.approx(pooled["value"])
    assert values["Mean snapshot-level Spearman"] == pytest.approx(snapshot_mean["value"])


def test_saved_solver_outputs_use_explicit_feasibility_labels_and_reference():
    payload = json.loads((ROOT / "results/project_evaluation.json").read_text(encoding="utf-8"))
    solvers = payload["s3_solvers"]["solvers"]
    exact = solvers["ExactSolver"]
    assert exact["feasibility_metric"]["label"] == "Exhaustive search-space feasible fraction"
    assert exact["exact_plan_agreement"]["value"] is None
    assert exact["exact_plan_agreement"]["status"] == "REFERENCE_OPTIMUM"
    assert "not an independent performance result" in exact["exact_plan_agreement"]["note"]
    for name in ("SimulatedAnnealing", "QAOA"):
        assert solvers[name]["feasibility_metric"]["label"] == "Sampled-output feasibility rate"

    csv = pd.read_csv(ROOT / "results/project_evaluation.csv")
    labels = set(csv[csv.system == "s3_solvers"].label.dropna())
    assert "Exhaustive search-space feasible fraction" in labels
    assert "Sampled-output feasibility rate" in labels
    feasibility_rows = csv[(csv.system == "s3_solvers") & csv.metric.isin([
        "Exhaustive search-space feasible fraction", "Sampled-output feasibility rate"])]
    exact_row = feasibility_rows[(feasibility_rows.scope.str.contains("ExactSolver")) &
                                 (feasibility_rows.metric == "Exhaustive search-space feasible fraction")].iloc[0]
    assert int(float(exact_row.n)) == 65536
    assert int(float(exact_row.numerator)) == 256
    sampled_sa = feasibility_rows[(feasibility_rows.scope.str.contains("SimulatedAnnealing")) &
                                  (feasibility_rows.metric == "Sampled-output feasibility rate")].iloc[0]
    assert int(float(sampled_sa.n)) == 2500
    assert int(float(sampled_sa.numerator)) == 2500
    exact_match_row = csv[(csv.system == "s3_solvers") & (csv.metric == "Exact-plan agreement") &
                          csv.scope.str.contains("ExactSolver")].iloc[0]
    assert exact_match_row.aggregation == "REFERENCE_OPTIMUM"
    assert "not an independent performance result" in exact_match_row.denominator
    report = (ROOT / "results/evaluation_report.md").read_text(encoding="utf-8")
    assert "These summarize the same held-out evaluations with different aggregation methods" in report
    assert "Their denominators describe different sets and are not directly comparable" in report
    assert "ExactSolver defines the reference optimum" in report


def test_solver_comparison_graph_exists_and_is_png():
    image = (ROOT / "results/graphs/solver_comparison.png").read_bytes()
    assert image.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(image) > 10_000
