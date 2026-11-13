# Phase 8 Controller Evaluation

**Status:** COMPLETE (40/40 episodes).

Evaluation is SUMO-only. Every controller uses the same scenario, seed, route file, duration, and SUMO seed within each paired condition. Optimization results are experimental because S2 v3 is scientifically not validated.
No claim of optimization improvement is made from these results.

## Controller metrics

| Controller | Travel time mean ± SD (s) | Waiting time mean ± SD (veh-s) | Avg queue | Max queue | Throughput crossings | Delay (time loss, s) | Solver runtime mean (s) | Raw feasibility | Fallbacks / decisions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed time | 72.29 ± 24.58 | 86313.00 ± 73474.41 | 6.96 ± 5.57 | 33.40 ± 30.54 | 3441.20 ± 1556.34 | 103311.06 ± 92565.52 | n/a | n/a | 0/400 |
| Max pressure | 73.80 ± 25.78 | 91458.50 ± 76703.29 | 7.09 ± 5.66 | 33.50 ± 31.06 | 3439.70 ± 1552.17 | 108839.15 ± 95647.46 | n/a | n/a | 0/400 |
| SA + QUBO | 72.18 ± 25.41 | 89528.30 ± 78506.27 | 6.88 ± 5.83 | 31.60 ± 30.96 | 3442.50 ± 1555.34 | 106688.60 ± 97952.66 | 0.01 ± 0.00 | 1.000 | 390/400 |
| QAOA Aer sampled | 72.36 ± 24.47 | 88431.00 ± 76142.19 | 7.12 ± 5.78 | 33.30 ± 30.39 | 3439.30 ± 1551.20 | 105617.31 ± 95597.72 | 147.93 ± 464.35 | 0.006 | 207/400 |

## Paired differences vs fixed time

Negative values mean lower measured outcome for the listed delta; they do not imply statistical or causal significance.

| Controller | Waiting-time delta (veh-s) | Travel-time delta (s) | Time-loss delta (s) | Queue delta (veh) | Throughput delta |
|---|---:|---:|---:|---:|---:|
| Max pressure | 5145.50 | 1.51 | 5528.09 | 0.12 | -1.5 |
| QAOA Aer sampled | 2118.00 | 0.06 | 2306.25 | 0.16 | -1.9 |
| SA + QUBO | 3215.30 | -0.11 | 3377.54 | -0.08 | 1.3 |

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
