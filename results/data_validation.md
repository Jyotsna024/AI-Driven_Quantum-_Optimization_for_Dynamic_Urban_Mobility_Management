# Phase 2 Validation & Data Integrity Report (Corrected)

## 1. Summary of Acceptance Gates
| Gate | Description | Criteria | Observed Result | Status |
|---|---|---|---|---|
| **Gate 1** | Peak Congestion | Mean queue >= 3.0 & P90 >= 8.0 at EVERY junction | {'J0': (np.float64(4.87), np.float64(8.0)), 'J1': (np.float64(7.47), np.float64(13.0)), 'J2': (np.float64(50.33), np.float64(74.0)), 'J3': (np.float64(7.43), np.float64(14.0))} | **PASS** |
| **Gate 2** | Demand Hierarchy | low < medium < high < peak at EVERY junction | {'J0': np.True_, 'J1': np.True_, 'J2': np.True_, 'J3': np.True_} | **PASS** |
| **Gate 3** | Plan Sensitivity | Best vs worst plan delay diff >= 10% at >= 3 junctions | 4/4 junctions pass | **PASS** |
| **Gate 4** | Plan Non-Triviality | Best plan varies across sampled snapshots | [0, 1, 2, 3] | **PASS** |
| **Gate 5** | Throughput Reconciliation | Throughput reconciles with arrivals & <= capacity | Max throughput <= 60 veh/min | **PASS** |
| **Gate 6** | No Gridlock | Final 600s throughput > 0, ~0 teleports | Peak final throughput = 22.55 veh/min | **PASS** |
| **Gate 7** | Determinism & Reload | Bit-exact metrics on repeated rollout from state | r1 == r2 identical match (True) | **PASS** |
| **Gate 8** | Incident Effect | Measurable speed/flow impact during closure | Speed: 1.18 m/s active vs 2.20 m/s inactive | **PASS** |

## 2. Per-Junction Queue Analysis (Post Warm-Up)
| scenario | junction_short | mean_queue | median_queue | p90_queue | max_queue | mean_queue_m | p90_queue_m | mean_delay | mean_throughput |
| -------- | -------------- | ---------- | ------------ | --------- | --------- | ------------ | ----------- | ---------- | --------------- |
| high     | J0             | 3.92       | 4.0          | 7.0       | 10        | 23.0         | 39.8        | 28.6       | 16.95           |
| high     | J1             | 5.32       | 5.0          | 9.0       | 17        | 31.86        | 56.11       | 13.46      | 16.96           |
| high     | J2             | 23.97      | 22.0         | 44.0      | 68        | 142.03       | 262.9       | 23.3       | 16.43           |
| high     | J3             | 5.15       | 4.0          | 11.0      | 20        | 30.49        | 61.46       | 6.54       | 16.73           |
| incident | J0             | 3.43       | 3.0          | 6.0       | 10        | 20.32        | 35.16       | 24.66      | 14.38           |
| incident | J1             | 4.89       | 4.0          | 9.0       | 18        | 28.84        | 53.3        | 18.45      | 14.08           |
| incident | J2             | 15.35      | 14.0         | 27.0      | 42        | 91.17        | 162.06      | 22.84      | 14.55           |
| incident | J3             | 4.19       | 3.0          | 9.0       | 15        | 24.79        | 51.5        | 6.45       | 14.13           |
| low      | J0             | 1.26       | 1.0          | 3.0       | 5         | 7.49         | 17.0        | 19.13      | 5.11            |
| low      | J1             | 1.44       | 1.0          | 3.0       | 8         | 8.53         | 18.7        | 13.11      | 4.86            |
| low      | J2             | 2.27       | 2.0          | 4.0       | 9         | 13.47        | 26.3        | 17.0       | 5.14            |
| low      | J3             | 1.31       | 1.0          | 3.0       | 8         | 7.8          | 19.3        | 5.66       | 4.93            |
| medium   | J0             | 2.79       | 3.0          | 5.0       | 10        | 16.53        | 30.5        | 21.6       | 10.98           |
| medium   | J1             | 3.38       | 3.0          | 6.0       | 13        | 20.21        | 38.0        | 13.03      | 10.86           |
| medium   | J2             | 6.75       | 5.0          | 13.0      | 32        | 39.99        | 77.82       | 19.17      | 10.98           |
| medium   | J3             | 3.19       | 2.0          | 7.0       | 18        | 18.99        | 42.1        | 6.46       | 11.03           |
| peak     | J0             | 4.87       | 5.0          | 8.0       | 11        | 28.51        | 48.0        | 36.5       | 23.14           |
| peak     | J1             | 7.47       | 7.0          | 13.0      | 24        | 44.27        | 78.46       | 13.27      | 23.3            |
| peak     | J2             | 50.33      | 52.0         | 74.0      | 92        | 298.84       | 442.2       | 21.82      | 20.41           |
| peak     | J3             | 7.43       | 6.0          | 14.0      | 31        | 43.93        | 83.83       | 6.8        | 23.23           |

## 3. Scenario Aggregate Summary
| scenario | mean_queue | p90_queue | mean_speed | mean_throughput | mean_delay |
| -------- | ---------- | --------- | ---------- | --------------- | ---------- |
| high     | 9.59       | 25.0      | 2.9        | 16.77           | 17.97      |
| incident | 6.96       | 17.0      | 2.95       | 14.28           | 18.06      |
| low      | 1.57       | 4.0       | 3.75       | 5.01            | 13.35      |
| medium   | 4.03       | 8.0       | 3.27       | 10.96           | 14.98      |
| peak     | 17.53      | 56.0      | 2.59       | 22.52           | 19.6       |

## 4. Throughput Capacity & Reconciliation
| junction | mean_throughput | mean_arrivals | max_interval_throughput | approach_capacity_veh_min | reconciliation_abs_diff | within_capacity |
| -------- | --------------- | ------------- | ----------------------- | ------------------------- | ----------------------- | --------------- |
| J0       | 14.11           | 14.16         | 43                      | 60.0                      | 0.05                    | True            |
| J1       | 14.01           | 14.06         | 46                      | 60.0                      | 0.05                    | True            |
| J2       | 13.5            | 13.82         | 39                      | 60.0                      | 0.32                    | True            |
| J3       | 14.01           | 14.13         | 47                      | 60.0                      | 0.12                    | True            |
