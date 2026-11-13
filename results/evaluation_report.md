# Project Evaluation Report

Metrics are calculated from saved project artifacts. No models were retrained, no SUMO simulations were run, and solver outputs were not regenerated.

## ST-GNN held-out test (seed 4)

Forecast windows: 530; horizons: 60, 120, and 180 seconds.

| Model | n | MAE | RMAE (%) | RMSE | R² | Pearson r | Spearman r |
|---|---:|---:|---:|---:|---:|---:|---:|
| ST-GNN | 6360 | 2.4096 | 30.7817 | 3.6305 | 0.9119 | 0.9579 | 0.7799 |
| Persistence | 6360 | 3.5445 | 45.2798 | 5.3481 | 0.8087 | 0.9019 | 0.5702 |

Detailed metrics are also grouped by horizon, junction, scenario, and scenario × junction × horizon in the JSON output.

## S1 cost model (held-out test seed 4)

Outcomes: 240; top-1 plan agreement: 23/60 (38.3%).
Regression: mae=6.2112, rmae_pct=39.6035, rmse=9.0872, r2=0.2847, spearman_r=0.6775

Top-1 compares the plan with minimum predicted delay against the minimum measured SUMO delay within each snapshot × target-junction group.

## S2 QUBO (held-out test snapshots)

Top-1 plan agreement: 0/2 (0.0%); mean regret: 9.5212 s; feasible unconstrained QUBO global-minimum rate: 2/2 (100.0%).
Pooled Spearman across 512 plan outcomes: -0.1134.
Mean snapshot-level Spearman across 2 held-out snapshots: 0.1598.
These summarize the same held-out evaluations with different aggregation methods: pooled Spearman ranks all plan outcomes together, while mean snapshot-level Spearman averages each snapshot's within-snapshot rank correlation. They are distinct and both are retained.

For each held-out snapshot, the predicted optimum minimizes serialized QUBO energy over the feasible plan grid. The actual SUMO optimum minimizes the measured sum of four junction delays over the same 256 vectors. Regret is the actual delay at the predicted optimum minus the actual optimum delay.

The saved S2 v3 result is explicitly marked scientifically not validated; these held-out metrics are diagnostic and do not justify closed-loop optimization.

## S3 solver results

| Solver | Exact-plan agreement | Best energy | Total runtime (s) | Feasibility metric | Feasible / denominator | Rate | Repeated best-plan agreement |
|---|---|---:|---:|---|---:|---:|---:|
| ExactSolver | Reference optimum (n/a) | -291.7752 | 0.0528 | Exhaustive search-space feasible fraction | 256 / 65536 | 0.0039 | n/a |
| SimulatedAnnealing | True | -291.7752 | 0.2827 | Sampled-output feasibility rate | 2500 / 2500 | 1.0000 | 10/10 (100.0%) |
| QAOA | True | -291.7752 | 21.4871 | Sampled-output feasibility rate | 150 / 15000 | 0.0100 | 1/10 (10.0%) |

SA and QAOA have ten saved seeds each; QAOA is classical Qiskit Aer simulation. ExactSolver enumerates 65,536 bitstrings, of which 256 satisfy exactly-one-plan-per-junction.
The ExactSolver 256/65,536 value is the exhaustive search-space feasible fraction. SA and QAOA values are sampled-output feasibility rates. Their denominators describe different sets and are not directly comparable. ExactSolver defines the reference optimum; its self-match is not an independent performance result.
Per-seed SA runtime is saved. Per-seed QAOA runtime is NOT AVAILABLE because only its aggregate runtime was recorded; ExactSolver has one saved timing and no repeated-run estimate.

## Phase 8 controller outcomes

The three requested traffic comparison plots summarize existing identical-condition episode results. They are simulation outcomes and descriptive summaries; they are not prediction accuracy metrics.

## Metric formulas and denominators

- MAE = mean absolute prediction error over paired finite values.
- RMAE (%) = 100 × MAE / mean absolute actual. It is NOT AVAILABLE when mean absolute actual is zero.
- RMSE = square root of mean squared prediction error.
- R² = 1 − SSE/SST; it is NOT AVAILABLE for fewer than two observations or constant actual values.
- Pearson and Spearman are computed on the same paired finite observations; correlation is NOT AVAILABLE with fewer than two observations or a constant input.
- ST-GNN regression denominators are forecast windows × four junctions × horizons represented by the relevant group. Persistence is the last observed queue repeated across horizons.
- S1 top-1 denominator is the count of test snapshot x target-junction groups with all four plan outcomes.
- S2 top-1 denominator is held-out test snapshots; regret is in seconds and paired within each snapshot. Pooled Spearman denominator is 512 plan outcomes ranked together. Mean snapshot-level Spearman denominator is two snapshots with defined within-snapshot correlations; it is the arithmetic mean of those correlations.
- S3 ExactSolver exhaustive search-space feasible fraction denominator is all 65,536 bitstrings. SA/QAOA sampled-output feasibility-rate denominators are each solver's raw sample counts. These rates are not directly comparable because one covers the full bitstring space and the others cover sampled outputs. Repeated plan agreement denominator is saved runs with a feasible best plan.

## Limitations

- The modified medium_s0 raw files are training-seed data. The checkpoint and scaler are used as saved; no retraining occurs. Held-out ST-GNN scoring uses seed 4 only.
- Phase 8 controller episodes are simulation outcomes, not a regression target or proof of controller improvement.
- S2 v3 is marked scientifically unvalidated in its saved artifacts; plan-selection and regret results are diagnostic only.
- QAOA results are from Qiskit Aer simulation, not quantum hardware.

A JSON field is null when a metric is not calculable from saved data; its scope and denominators are retained in the structured output.
