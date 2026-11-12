# S2 v3 Held-out Diagnosis

**Status: SCIENTIFICALLY NOT VALIDATED.** Frozen model only; no tuning on validation or test outcomes.

The table decomposes QUBO error into baseline mismatch, per-junction main-effect error, train-learned pair error, and omitted higher-order residual. Pair effects and higher-order residuals are estimated from the complete held-out 256-plan SUMO grid.

| State | Additive Spearman | QUBO Spearman | QUBO regret (s) | Pair RMSE (s) | Higher-order RMSE (s) |
|---|---:|---:|---:|---:|---:|
| high_s3_t2100 | 0.860 | 0.849 | 0.761 | 4.306 | 2.712 |
| incident_s3_t2100 | -0.196 | -0.172 | 31.501 | 4.328 | 2.221 |
| incident_s4_t2100 | 0.010 | -0.001 | 15.759 | 0.809 | 0.307 |
| peak_s4_t2100 | 0.324 | 0.320 | 3.284 | 14.981 | 7.236 |

Mean validation QUBO Spearman: 0.339; mean validation regret: 16.131 s.
Mean test QUBO Spearman: 0.160; mean test regret: 9.521 s.

## Error sources

Individual S1 costs by junction: `{"J0": {"mae_s": 12.089026299144123, "n": 16, "rmse_s": 15.50144814409324, "spearman": -0.11176470588235293}, "J1": {"mae_s": 3.8965886150982407, "n": 16, "rmse_s": 4.89439269123507, "spearman": 0.4058823529411764}, "J2": {"mae_s": 6.355937418363317, "n": 16, "rmse_s": 7.378471904416976, "spearman": 0.5352941176470588}, "J3": {"mae_s": 4.6605865331672325, "n": 16, "rmse_s": 5.288633573669723, "spearman": 0.8088235294117646}}`.

Per-state component diagnostics and scenario/junction tables are in `s2_diagnostic_v3.json`; the 1,024 plan-level rows are in `s2_plan_diagnostics_v3.csv`.

| Scenario | Junction | Local cost MAE (s) | Predicted increment MAE (s) | Mean predicted / actual system increment (s) |
|---|---|---:|---:|---:|
| high | J0 | 5.418 | 3.905 | -2.542 / -6.446 |
| high | J1 | 5.088 | 1.975 | 1.848 / -0.128 |
| high | J2 | 9.400 | 5.144 | -0.081 / 3.887 |
| high | J3 | 5.967 | 1.256 | 0.492 / 0.900 |
| incident | J0 | 14.712 | 10.180 | -2.432 / 5.680 |
| incident | J1 | 1.510 | 1.432 | 1.889 / 3.013 |
| incident | J2 | 6.560 | 3.449 | -0.756 / -3.142 |
| incident | J3 | 2.792 | 0.967 | 0.490 / 1.406 |
| peak | J0 | 13.513 | 4.967 | -2.797 / 0.925 |
| peak | J1 | 7.478 | 1.754 | 1.645 / 2.655 |
| peak | J2 | 2.902 | 11.693 | -0.204 / 9.400 |
| peak | J3 | 7.091 | 0.653 | 0.286 / -0.367 |

The additive model is equal or better on the high and peak snapshots; pair terms only slightly change the incident rankings. Large J0 single-cost errors and J0/J2 main-effect increment errors are evident, especially for incident and peak demand. The residual decomposition shows pair-term mismatch is large at peak, while higher-order residual is also material there. On incident snapshots, poor ranking is mainly present even before pair terms; the tested higher-order residual is smaller than the main-effect error. These held-out findings are diagnostic only; no tuning followed.
