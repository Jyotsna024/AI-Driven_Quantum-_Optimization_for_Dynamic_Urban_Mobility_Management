"""Read-only integrity check for the saved Phase 8 evaluation package.

This module intentionally does not launch SUMO or rewrite result artifacts.
Use ``scripts/run_phase8_evaluation.py`` when new simulations are explicitly
needed; this entry point audits the results already on disk.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "phase8_eval_v1"
SCENARIOS = ("low", "medium", "high", "peak", "incident")
SEEDS = (0, 1)
CONTROLLERS = ("fixed_time", "max_pressure", "sa_qubo", "qaoa_aer_sampled")
EPISODE_SUFFIXES = ("metrics.json", "telemetry.csv", "decisions.csv", "signal_updates.csv", "forecasts.csv")
AGGREGATE_FILES = (
    "episode_metrics.csv",
    "paired_controller_deltas.csv",
    "phase8_evaluation.json",
    "PHASE8_EVALUATION_REPORT.md",
    "average_queue_vehicles_by_controller.png",
    "controller_metrics_boxplots.png",
    "mean_completed_trip_travel_time_s_by_controller.png",
    "throughput_total_junction_crossings_by_controller.png",
    "total_time_loss_s_by_controller.png",
)


def expected_run_ids() -> tuple[str, ...]:
    return tuple(
        f"{controller}_{scenario}_s{seed}_3600"
        for scenario in SCENARIOS
        for seed in SEEDS
        for controller in CONTROLLERS
    )


def audit(directory: Path = RESULTS) -> dict:
    """Return a completeness audit without changing files or computing metrics."""
    expected = expected_run_ids()
    manifest_path = directory / "run_manifest.json"
    manifest = {}
    errors: list[str] = []
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"Invalid run manifest: {exc}")
    else:
        errors.append("Missing run_manifest.json")

    completed = set(manifest.get("completed", ()))
    missing_runs = [run_id for run_id in expected if run_id not in completed]
    missing_artifacts = [
        f"{run_id}_{suffix}"
        for run_id in expected
        for suffix in EPISODE_SUFFIXES
        if not (directory / f"{run_id}_{suffix}").is_file()
    ]
    missing_aggregates = [name for name in AGGREGATE_FILES if not (directory / name).is_file()]

    aggregate = {}
    aggregate_path = directory / "phase8_evaluation.json"
    if aggregate_path.is_file():
        try:
            aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"Invalid phase8_evaluation.json: {exc}")
    if aggregate:
        if aggregate.get("n_expected_episodes") != len(expected):
            errors.append("Aggregate expected-episode count does not match the configured matrix")
        if aggregate.get("n_completed_episodes") != len(expected):
            errors.append("Aggregate does not report all expected episodes complete")
        if aggregate.get("status") != "COMPLETE":
            errors.append(f"Aggregate status is {aggregate.get('status')!r}, expected 'COMPLETE'")

    return {
        "directory": str(directory),
        "expected_episode_count": len(expected),
        "manifest_completed_count": len(completed.intersection(expected)),
        "missing_runs": missing_runs,
        "missing_episode_artifacts": missing_artifacts,
        "missing_aggregate_outputs": missing_aggregates,
        "errors": errors,
    }


def main() -> None:
    result = audit()
    print(f"Phase 8 saved-result audit: {result['directory']}")
    print(
        f"Episodes in manifest: {result['manifest_completed_count']} / "
        f"{result['expected_episode_count']}"
    )
    for label, key in (
        ("Missing episode entries", "missing_runs"),
        ("Missing per-episode files", "missing_episode_artifacts"),
        ("Missing aggregate outputs", "missing_aggregate_outputs"),
        ("Audit errors", "errors"),
    ):
        values = result[key]
        print(f"{label}: {len(values)}")
        for value in values:
            print(f"  - {value}")
    if any(result[key] for key in ("missing_runs", "missing_episode_artifacts", "missing_aggregate_outputs", "errors")):
        raise SystemExit(1)
    print("Saved Phase 8 artifacts are complete. No simulations were run and no files were changed.")


if __name__ == "__main__":
    main()
