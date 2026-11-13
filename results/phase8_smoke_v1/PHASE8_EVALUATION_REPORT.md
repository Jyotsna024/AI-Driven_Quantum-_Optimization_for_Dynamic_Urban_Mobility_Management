# Phase 8 Controller Evaluation

**Status:** INCOMPLETE (4/40 episodes).

Evaluation is SUMO-only. Every controller uses the same scenario, seed, route file, duration, and SUMO seed within each paired condition. Optimization results are experimental because S2 v3 is scientifically not validated.
No claim of optimization improvement is made from these results.

## Controller metrics

| Controller | Travel time mean ± SD (s) | Waiting time mean ± SD (veh-s) | Avg queue | Max queue | Throughput crossings | Delay (time loss, s) | Solver runtime mean (s) | Raw feasibility | Fallbacks / decisions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed time | 50.79 ± 0.00 | 2867.00 ± 0.00 | 1.35 ± 0.00 | 7.00 ± 0.00 | 193.00 ± 0.00 | 3199.14 ± 0.00 | n/a | n/a | 0/6 |
| Max pressure | 49.85 ± 0.00 | 3183.00 ± 0.00 | 1.55 ± 0.00 | 6.00 ± 0.00 | 191.00 ± 0.00 | 3509.40 ± 0.00 | n/a | n/a | 0/6 |
| SA + QUBO | 48.41 ± 0.00 | 2454.00 ± 0.00 | 1.10 ± 0.00 | 6.00 ± 0.00 | 196.00 ± 0.00 | 2759.25 ± 0.00 | 0.01 ± 0.00 | 1.000 | 6/6 |
| QAOA Aer sampled | 50.79 ± 0.00 | 2867.00 ± 0.00 | 1.35 ± 0.00 | 7.00 ± 0.00 | 193.00 ± 0.00 | 3199.14 ± 0.00 | 1.07 ± 0.00 | 0.000 | 1/6 |

## Paired differences vs fixed time

Negative values mean lower measured outcome for the listed delta; they do not imply statistical or causal significance.

| Controller | Waiting-time delta (veh-s) | Travel-time delta (s) | Time-loss delta (s) | Queue delta (veh) | Throughput delta |
|---|---:|---:|---:|---:|---:|
| Max pressure | 316.00 | -0.94 | 310.26 | 0.20 | -2.0 |
| QAOA Aer sampled | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 |
| SA + QUBO | -413.00 | -2.38 | -439.89 | -0.25 | 3.0 |

## Artifacts

- `phase8_evaluation.json`: full run configuration, episode metrics, and summary.
- `episode_metrics.csv` and `paired_controller_deltas.csv`.
- Per-episode telemetry, decision logs, applied-signal updates, and forecast CSVs.
- Controller comparison graphs are saved as PNG files in this directory.

## Limitations

- Two seeds per scenario provide a small sample; means and standard deviations are descriptive.
- Waiting time is integrated stopped-vehicle seconds on the four selected junction approaches. Travel time is mean for completed route vehicles; incomplete trips are excluded.
- Delay is SUMO vehicle time loss for observed vehicles on the selected approaches. Junction throughput counts approach-lane exits and can count the same trip at multiple junctions.
- S2 v3 failed held-out ranking/regret diagnosis. SA/QAOA controller actions are exploratory; fallbacks and feasibility are reported.
- Plan 3 may be rejected by the safe-switch check when its offset would change the current signal state; baseline is queued and the rejection is logged.
- QAOA Aer is classical simulation; no quantum hardware or quantum advantage is claimed.
