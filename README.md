# AI-Driven Quantum Optimization for Dynamic Urban Mobility Management

Research capstone prototype for traffic forecasting and signal-plan analysis on a four-junction corridor in HSR Layout, Bengaluru. It combines a SUMO/TraCI microsimulation on an OpenStreetMap-derived road network, a spatio-temporal graph neural network (ST-GNN), a signal-plan cost model, a 16-variable QUBO, classical and simulated-quantum solvers, controller evaluations, and a Streamlit dashboard.

> **Research status:** This is a simulation study, not a deployed traffic-control system. The S2 QUBO is **scientifically not validated** on held-out joint SUMO states. Solver agreement with that QUBO shows that solvers can optimize the encoded objective; it does not show that the objective reliably predicts traffic outcomes. No quantum advantage or real-world traffic improvement is claimed.

## Contents

- [Project overview](#project-overview)
- [End-to-end workflow](#end-to-end-workflow)
- [Road network, junctions, plans and scenarios](#road-network-junctions-plans-and-scenarios)
- [Dataset and data splits](#dataset-and-data-splits)
- [ST-GNN: model, training and results](#st-gnn-model-training-and-results)
- [S1: signal-plan cost model](#s1-signal-plan-cost-model)
- [S2: QUBO objective and validation](#s2-qubo-objective-and-validation)
- [S3: solver comparison](#s3-solver-comparison)
- [Phase 8: controller evaluation](#phase-8-controller-evaluation)
- [Dashboard: what it shows and how it works](#dashboard-what-it-shows-and-how-it-works)
- [Metric formulas and how to judge them](#metric-formulas-and-how-to-judge-them)
- [Setup and commands](#setup-and-commands)
- [Code and evidence map](#code-and-evidence-map)
- [Limitations and interpretation rules](#limitations-and-interpretation-rules)

## Project overview

| Component | Project details |
|---|---|
| Study area | HSR Layout, Bengaluru; OSM extract bounds are recorded in sumo_files/cluster_config.json |
| Network | OpenStreetMap-derived SUMO road network, simulated with SUMO 1.28 and TraCI |
| Signal cluster | Four connected signalized junctions, J0–J3 |
| Candidate plans | Four traffic-light programs per junction |
| ST-GNN prediction | Future queue lengths at 60, 120 and 180 simulated seconds |
| S1 cost model | Predicted target-junction delay for a candidate signal plan |
| S2 QUBO | 16 binary variables representing four plan choices at four junctions |
| S3 solvers | ExactSolver, simulated annealing (SA), QAOA using Qiskit Aer |
| Dashboard | Saved research results plus a live SUMO-GUI/TraCI simulation |
| Live control | Optional adaptive controller using ST-GNN and queue-pressure fallback; it does not use the research QUBO |

## End-to-end workflow

1. Convert a real OSM extract into a SUMO network, identify four connected traffic lights, and define scenarios, routes and signal plans.
2. Run SUMO with TraCI and collect traffic measurements and saved simulation-state snapshots.
3. Train/evaluate the ST-GNN to forecast future queues from recent network-wide measurements.
4. Replay identical saved states under candidate signal plans to create S1 target-junction delay labels. Fit a leakage-audited Ridge model using only information available before choosing a plan.
5. Estimate pairwise plan interactions from matched joint SUMO rollouts, combine them with the S1 costs, and construct the S2 QUBO.
6. Use exhaustive search, SA and QAOA to compare solvers on the same serialized QUBO.
7. Compare controllers in paired SUMO experiments and assemble results into saved CSV/JSON reports and charts.
8. The Streamlit dashboard reads saved evaluation artifacts for research pages. Its separate Live SUMO page controls and displays a real running SUMO-GUI session.

The S1/S2/S3 chain is an offline research experiment. Live SUMO's optional adaptive controller is a separate implementation: it uses ST-GNN queue forecasts when available and observed pressure as a fallback. It does not use the S2 QUBO.

## Road network, junctions, plans and scenarios

### Network and junctions

The network comes from an official OpenStreetMap extract. The SUMO network contains a larger local road graph; this study focuses on four connected signalized intersections:

| ID | Location |
|---|---|
| J0 | 14th Main & 15th Cross |
| J1 | 14th Main & 17th Cross |
| J2 | 19th Main & 17th Cross |
| J3 | 14th Main & Sarjapur Road |

The ST-GNN represents these locations as four graph nodes and uses 12 directed road connections with distance-derived weights. The Live SUMO screen is not a hand-drawn map or browser animation: it embeds refreshed screenshots from the active SUMO-GUI process.

### Signal plans

| Plan | Intended meaning |
|---|---|
| 0 | Baseline timing from the converted SUMO network |
| 1 | Main-corridor-heavy green allocation |
| 2 | Side-road-heavy green allocation |
| 3 | Baseline phase durations with a shifted coordination offset |

Programs use the actual SUMO phase-state strings at each traffic light. J2 has more phases than the other junctions. Exact phase states, durations, offsets and traffic-light IDs are in sumo_files/cluster_config.json. Signal-program changes must preserve legal states and clearance.

### Scenarios

Demand values below are configured flows from configs/scenarios.yaml, not measured Bengaluru counts.

| Scenario | Configured demand | Special condition |
|---|---:|---|
| Low | 1,200 vehicles/hour | Off-peak demand |
| Medium | 2,600 vehicles/hour | Regular demand |
| High | 4,000 vehicles/hour | High demand |
| Peak | 5,600 vehicles/hour | Near-saturation demand |
| Incident | 3,400 vehicles/hour | Lane closure on a 14th Main approach from 1,200 to 1,800 seconds |

Route seeds are 0–4. Experiment artifacts may also record the SUMO random seed separately (some scripts use route seed + 100). A simulation seed is a reproducibility setting, not an independent real-world traffic sample.

## Dataset and data splits

### What the project stores

- sumo_files/routes/: generated scenario and seed route files.
- data/raw/: SUMO traffic time-series CSVs plus a manifest. Each run has fixed/varied signal-mode output.
- data/states/: serialized SUMO states and an index. Reloading a saved state lets candidate interventions start from matched traffic conditions.
- data/rollouts_s1_v2.csv: single-junction candidate-plan outcomes for audited S1.
- data/features_s1_v2.csv: S1 pre-decision features for the saved states and plans.
- data/joint_rollouts_s2_v3.csv: S2 pair-intervention evidence and complete joint-plan validation/test rollouts.
- Files with checkpoint in their names support resumability; results/archive_pre_repair_20261112/ holds prior evidence. Versioned/checkpoint files should not automatically be treated as equivalent to final artifacts.

The ST-GNN raw features at each junction and observation time are:

1. queue_length: number of halted/queued vehicles.
2. vehicle_count: observed vehicles.
3. mean_speed_mps: mean speed in metres per second.
4. mean_delay_s: measured delay.
5. throughput: crossings during the logging interval.
6. arrivals: vehicles entering during the interval.

These are simulated observations, not field sensor data. For ST-GNN, the future queue is the target and historical values are predictors. For S1, the measured delay after applying a candidate plan is a label, not a predictor.

### Split policy

The principal split is by seed to keep a seed's simulation runs out of the other partitions:

| Split | Seeds | Purpose |
|---|---|---|
| Train | 0, 1, 2 | Fit ST-GNN scaler/weights and S1 model; estimate S2 costs/interactions |
| Validation | 3 | Choose/check model settings without using final test results |
| Held-out test | 4 | Final saved generalization check |

The ST-GNN pipeline loads 50 scenario × seed × signal-mode raw CSVs. It uses 3 past 60-second observations and predicts the next 3 intervals. Targets within the first 300 simulated seconds are skipped. The saved seed-4 split has 530 windows from 10 raw test runs. Each window contributes four junctions × three horizons, so the consolidated test metrics use 530 × 4 × 3 = 6,360 paired values.

For audited S1, the feature table contains 45 snapshot IDs: 15 train snapshots (five scenarios × train seeds 0–2, one each), 15 validation snapshots (five scenarios × three saved times for seed 3), and 15 test snapshots (five scenarios × three saved times for seed 4). Each snapshot is replayed for four candidate plans at each of four target junctions: 45 × 4 × 4 = 720 interventions and 2,880 measured-junction outcome rows. During an individual-junction experiment, only the target junction gets the candidate plan; other junctions stay on baseline. The S1 training table uses only the target junction's outcome. Seed 4 is held out.

For S2, training seeds 0–2 alone estimate pair-interaction terms. Validation and test use complete four-junction plan grids. Held-out SUMO outcomes are used for diagnosis, not model fitting.

### Data caveats

The data is generated by the project's SUMO network, route demand and signal logic. Results are conditional on these assumptions. A SUMO seed is not a new city or a new traffic regime. In S2 v3, only two validation and two test snapshots have complete 256-plan grids.

## ST-GNN: model, training and results

### Model architecture and training

Implementation: scripts/stage2_stgnn.py and src/stgnn.py.

1. Each junction/time step has six features from the table above.
2. The model takes the three most recent observations (180 seconds).
3. Two graph convolution layers (PyG GCNConv, 32 hidden units) propagate information over the four-node, 12-directed-edge graph at each time.
4. A GRU with 32 hidden units encodes temporal change for each junction.
5. A small fully connected prediction head produces queue predictions at +60, +120 and +180 seconds.

Training uses standardized inputs/targets, mean squared error, Adam (learning rate 0.005, weight decay 0.0001), batch size 32 and up to 45 epochs. The StandardScaler is fit on training data only. The model checkpoint with the lowest validation loss is saved. The fitted model and scaler are models/stgnn.pt and models/scaler.pkl.

Two baselines are saved:

- Persistence: use the latest observed queue as the prediction at every future horizon.
- Historical average: use that junction's mean queue in training data for all horizons.

### Consolidated held-out test metrics

These results are built from the saved model/scaler and seed-4 test runs; generating the consolidated evaluation does not retrain them.

| Predictor | MAE (vehicles) | RMAE | RMSE (vehicles) | R² | Pearson r | Spearman r |
|---|---:|---:|---:|---:|---:|---:|
| ST-GNN | 2.4096 | 30.78% | 3.6305 | 0.9119 | 0.9579 | 0.7799 |
| Persistence | 3.5445 | 45.28% | 5.3481 | 0.8087 | 0.9019 | 0.5702 |
| Historical average* | 6.017 | — | 10.311 | — | — | — |

*Historical-average MAE/RMSE are from models/metrics.json, the original Phase 3 benchmark; the consolidated dashboard table reports ST-GNN and persistence. ST-GNN MAE at 60, 120 and 180 seconds is 2.3782, 2.4406 and 2.4099 vehicles. It beats persistence overall in this saved test set. That is a useful result for this simulation setup, not proof of broad traffic forecasting performance. Results vary by scenario: low-demand R² is below zero, meaning the ST-GNN performs worse than a constant test-mean predictor under the R² squared-error comparison for that subset.

## S1: signal-plan cost model

### What it predicts

S1 estimates a target junction's delay after a candidate signal plan is applied. It uses a Ridge regression model with validation-selected regularization (saved alpha = 50 in audited S1 v2).

Its 46 features are grouped as follows:

- Six pre-decision observations: queue distance, delay, arrivals, throughput, speed and vehicle count.
- Four actual ST-GNN queue forecasts: H1, H2, H3 and their mean.
- One-hot plan ID and one-hot target-junction ID.
- Plan-conditioned interactions: plan-specific junction indicators and pre-decision queue, ST-GNN and arrivals inputs.

All predictors are known before the candidate rollout. Post-rollout queue, throughput and delay are excluded from inputs. The rollout delay is the response. Train seeds fit the Ridge coefficients and scaler; validation seed 3 selects alpha; seed 4 is held out.

### Held-out S1 results

The dashboard's consolidated S1 v2 result has 240 plan outcomes: 15 seed-4 snapshots × four junctions × four plans. The top-1 denominator is 60 complete snapshot × target-junction groups.

| Metric | Test result |
|---|---:|
| Delay MAE | 6.2112 seconds |
| Delay RMSE | 9.0872 seconds |
| Delay RMAE | 39.60% |
| R² | 0.2847 |
| Spearman across plan outcomes | 0.6775 |
| Correct best-plan choice | 23/60 = 38.3% |
| Four-plan random-choice reference | 25% |

Ranking is above the simple random-choice reference overall but not stable enough to call delay predictions strong. Top-1 agreement by junction is J0 4/15, J1 4/15, J2 7/15 and J3 8/15. J0/J1 are weak. The test set is limited and does not establish generalization.

Top-1 is computed by taking the minimum predicted-delay plan within each snapshot × target-junction group and checking whether it equals the plan with the minimum measured SUMO delay. Regression errors use seconds. See [Metric formulas](#metric-formulas-and-how-to-judge-them).

## S2: QUBO objective and validation

### Formulation

For junction i and plan k, binary variable x(i,k) is 1 if that plan is chosen. There are 16 variables. The energy is:

    E(x) = sum(i,k) c(i,k) x(i,k)
         + sum((i,j),k,l) J(i,j,k,l) x(i,k) x(j,l)
         + A sum(i) (sum(k) x(i,k) - 1)^2

- c(i,k): S1-predicted delay for choosing plan k at junction i, using pre-decision features and ST-GNN forecasts.
- J(i,j,k,l): pairwise interaction for choosing plans k and l together. It is estimated from matched SUMO rollouts using d_ij - d_i - d_j + d_0 (joint delay minus the two separate interventions plus baseline reference).
- A: one-hot penalty (saved v3 value approximately 86.0268), which penalizes selecting no plan or multiple plans at a junction.

Exactly one plan per junction gives 4^4 = 256 feasible plan vectors among 2^16 = 65,536 bitstrings. S2 v3 uses 810 training pair rollouts and evaluates all 256 joint plan vectors at each of two validation and two test snapshots (1,024 held-out joint rollouts total).

### Held-out result: not scientifically validated

For each held-out snapshot, QUBO's lowest-energy plan vector is compared to the vector with the lowest measured total SUMO delay among the same 256 candidates.

| Test measure | Result | Meaning |
|---|---:|---|
| Predicted joint optimum matches SUMO optimum | 0/2 | Both held-out test optima were missed |
| Mean optimization regret | 9.5212 seconds | Actual SUMO delay at selected plan minus actual delay at best tested plan |
| Pooled Spearman over 512 vectors | −0.1134 | Poor ordering across all outcomes pooled over the two snapshots |
| Mean within-snapshot Spearman over 2 states | 0.1598 | Weak average rank correlation within each state |
| Feasible unconstrained QUBO minima | 2/2 | Both minima were one-hot; this is constraint feasibility, not traffic accuracy |

The incident test snapshot has 15.7586 seconds regret and Spearman about −0.0007. The peak snapshot has 3.2839 seconds regret and Spearman about 0.3202. These results fail scientific validation. The S2 objective should not be used for closed-loop control or presented as a reliable traffic optimizer.

A solver may correctly minimize QUBO energy while the QUBO's predicted traffic ranking is wrong. Solver mechanics and objective validity are separate questions.

## S3: solver comparison

All methods use the same serialized QUBO input models/qubo_s2_v3.npz. Solver output energy and plan are independently recomputed. QAOA results use the Qiskit Aer classical simulator, not quantum hardware.

| Solver | Plan vector J0–J3 | Best energy | Runtime | Feasibility evidence |
|---|---|---:|---:|---|
| ExactSolver | [3, 3, 1, 3] | −291.7752 | 0.0528 s | Enumerated 65,536 bitstrings; 256 satisfy one-hot |
| Simulated Annealing | [3, 3, 1, 3] | −291.7752 | 0.2827 s total, 10 runs | 2,500/2,500 raw samples feasible; exact plan in all 10 runs |
| QAOA (Aer) | [3, 3, 1, 3] best result | −291.7752 | 21.4871 s total | 150/15,000 raw samples feasible (1.0%); exact plan in 1/10 seeds |

Feasibility denominators differ: ExactSolver's 256/65,536 is the fraction of the full search space that is feasible; SA/QAOA rates count each sampler's raw output samples. These are not directly comparable. QAOA's best run matched ExactSolver, but only 1/10 seeds did so and only 1% of raw samples were feasible. None of this validates S2's traffic objective.

## Phase 8: controller evaluation

The saved Phase 8 comparison contains 40 SUMO episodes: four controllers × five scenarios × two seeds. Each controller has the same scenario/seed route, initial conditions and 3,600-second episode within a paired condition. Control decisions occur every 90 simulated seconds; QAOA is sampled at 900, 1,800 and 2,700 seconds. Means/standard deviations below summarize ten episodes per controller.

| Controller | Waiting time (veh-s) | Mean travel time (s) | Average queue (veh) | Maximum queue (veh) | Throughput (crossings) | Time loss/delay (s) | Solver runtime mean (s) | Raw feasibility | Fallbacks / decisions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed time | 86,313 ± 73,474 | 72.29 ± 24.58 | 6.96 ± 5.57 | 33.40 ± 30.54 | 3,441.2 ± 1,556.3 | 103,311 ± 92,566 | n/a | n/a | 0/400 |
| Max pressure | 91,459 ± 76,703 | 73.80 ± 25.78 | 7.09 ± 5.66 | 33.50 ± 31.06 | 3,439.7 ± 1,552.2 | 108,839 ± 95,647 | n/a | n/a | 0/400 |
| SA + QUBO | 89,528 ± 78,506 | 72.18 ± 25.41 | 6.88 ± 5.83 | 31.60 ± 30.96 | 3,442.5 ± 1,555.3 | 106,689 ± 97,953 | 0.01 | 100% sampled | 390/400 |
| QAOA Aer sampled | 88,431 ± 76,142 | 72.36 ± 24.47 | 7.12 ± 5.78 | 33.30 ± 30.39 | 3,439.3 ± 1,551.2 | 105,617 ± 95,597 | 147.93 ± 464.35 | 0.6% sampled | 207/400 |

These results are noisy, descriptive simulation outcomes, not a demonstrated optimization improvement. Fixed time has the lowest mean waiting time here. SA has slightly lower mean travel time and average queue but higher waiting/time loss and falls back on 390/400 decisions (97.5%). QAOA has high/variable runtime, low sample feasibility and many fallbacks. The underlying QUBO is scientifically unvalidated.

### Separate adaptive-versus-fixed experiment

The latest commit adds five additional paired low-demand comparisons, one per route seed, each 600 seconds. Across the saved comparisons, adaptive averages 2,532 waiting vehicle-seconds versus 2,796 for fixed time; its average queue is 1.03 versus 1.455 vehicles; total recorded junction crossings are 937 versus 948. Adaptive waiting is lower in four of five runs and higher for seed 0. This is one scenario with five paired runs, not the four-controller Phase 8 experiment and not general evidence of improvement.

## Dashboard: what it shows and how it works

Start it with:

    streamlit run dashboard/app.py --server.port 8765

### Research pages

The app loads saved JSON/CSV evaluation artifacts and charts from results/project_evaluation.json, results/project_evaluation.csv and results/graphs/. Missing or malformed evidence is reported; it should not fabricate metrics.

| Page | What it displays |
|---|---|
| Project Overview | Study corridor, system flow, ST-GNN test seed, S1 plan agreement and S2 validation status |
| Traffic Prediction (ST-GNN) | Saved MAE/RMAE/RMSE/R², baselines, horizon and junction breakdowns, prediction graph |
| Cost Prediction (S1) | Delay prediction metrics and top-1 plan agreement by junction/scenario |
| QUBO Optimization (S2) | Predicted vs actual SUMO optima, regret, correlations and feasibility warning |
| Solver Comparison (S3) | Solver plan, energy, runtime and sampled/exhaustive feasibility |
| Traffic Performance | Saved Phase 8 controller summaries and graphs |
| Limitations | Scope and interpretation caveats |
| Live SUMO | Actual live SUMO-GUI screenshot and current TraCI telemetry |
### Live SUMO behavior
The Live SUMO page starts sumo-gui.exe on the HSR network and selected route file. The UI controls and observes that same SUMO/TraCI process. TraCI requests a fresh SUMO-GUI screenshot, which Streamlit embeds; vehicles and signal phases are from SUMO itself.
Controls include scenario, route seed, Start/restart, Pause/resume, one-second step and Stop. The page displays simulation time, active/stopped vehicles, queue length, waiting vehicle-seconds, throughput, junction telemetry, controller/prediction status and recent decision history.
The Automatic AI Control toggle currently defaults to on. Its controller uses live ST-GNN queue forecasts when available and a live queue-pressure fallback otherwise. It is not an S2/QUBO controller.
**Current code note:** this dashboard revision does not expose the four manual per-junction plan selectors from the earlier simple-dashboard version. When automatic control is off, SUMO's native signal program stays active. This is a documented gap from the earlier manual-selector requirement.
The dashboard only controls local SUMO simulation. It does not connect to or change municipal traffic lights.
## Metric formulas and how to judge them
For n paired finite predictions and observations, let y be actual, ŷ be predicted, and ȳ be the mean actual value.
| Metric | Formula | Meaning |
|---|---|---|
| MAE | (1/n) Σ |ŷᵢ − yᵢ| | Average absolute error; vehicles for queue prediction, seconds for delay prediction. Lower is better. |
| RMSE | sqrt((1/n) Σ (ŷᵢ − yᵢ)²) | Penalizes large misses more strongly than MAE. Lower is better. |
| RMAE (%) | 100 × MAE / mean(abs(yᵢ)) | MAE relative to mean absolute actual value. This project uses RMAE, not MAPE. Lower is better; undefined if denominator is zero. |
| R² | 1 − Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)² | 1 is perfect; 0 equals a test-mean constant prediction under squared error; negative is worse than that reference. |
| Pearson r | Linear correlation of actual values and predictions | Strength of linear association; not an error measure. |
| Spearman ρ | Correlation of the ranks of actual and predicted values | Whether values are ordered similarly; not an error measured in the target's units. |
**“Accuracy” is not an ST-GNN metric here.** The ST-GNN predicts a continuous queue, not a category, so it has no class labels for classification accuracy. R² = 0.91 must not be described as “91% accuracy.” For S1, top-1 plan agreement is the fraction of groups where the model picks the same lowest-delay plan as SUMO; it is a separate ranking score.
For optimization/control:
- QUBO energy is an encoded score, not SUMO delay in seconds.
- Feasibility means exactly one plan is selected per junction; it does not mean that the selected plan is traffic-optimal.
- Regret is actual SUMO delay at the QUBO-selected plan minus actual delay at the best of the same evaluated plan grid; zero means the selected plan was best on that specific state.
- Solver agreement is agreement with ExactSolver on the QUBO, not agreement with SUMO.
- Waiting time is accumulated stopped-vehicle seconds on selected approaches. Travel-time averages use completed trips; unfinished trips are excluded. Delay is SUMO time loss. Throughput counts junction crossings, so one vehicle can count at more than one junction.
- Runtime, raw-sample feasibility and fallback counts describe solver/controller operation. A fallback means the optimizer did not apply its intended result at that decision.
Always read a metric's target unit, data split, version, grouping and denominator. Means/standard deviations over a small number of simulations are descriptive; they do not establish statistical significance or causal effects.
## Setup and commands
### Windows prerequisites
- Windows and SUMO 1.28.0; verified install path: C:\Program Files (x86)\SUMO.
- Python 3.13 (verified with 3.13.7) and dependencies from requirements.txt.
- SUMO-GUI for the live dashboard. Set SUMO_HOME if your SUMO installation does not configure it.
- The OSM network, routes and saved artifacts are in the project checkout.
    cd C:\Users\Jyotsna\Desktop\Capstone
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
Start the dashboard:
    streamlit run dashboard/app.py --server.port 8765
Useful entry points:
    python run_all.py --help
    python run_all.py --all --smoke
    python scripts/stage2_stgnn.py
    python scripts/rebuild_s1_cost_model.py
    python scripts/rebuild_s2_qubo_v3.py
    python scripts/rebuild_s3_solvers_v3.py
    python scripts/run_phase8_evaluation.py --help
    python scripts/evaluate_adaptive_control.py --help
    pytest tests/ -v
The full SUMO experiment matrix and QAOA simulation take longer than smoke mode. These scripts can regenerate or create artifacts; inspect their options and output paths before running and preserve old evidence when comparing versions. No cloud token is needed for saved QAOA results. The recorded QAOA is a classical Qiskit Aer simulation; no verified quantum-hardware run is included.
## Code and evidence map
| Area | Main files |
|---|---|
| Project decisions/status | PROJECT_SPEC.md |
| Configurations | configs/base.yaml, configs/scenarios.yaml, configs/qubo.yaml |
| OSM/SUMO network, routes and TLS plans | sumo_files/ |
| Scenario generation/telemetry | src/scenario_runner.py, src/data_validator.py, scripts/, data/raw/ |
| ST-GNN | scripts/stage2_stgnn.py, src/stgnn.py, models/stgnn.pt, models/scaler.pkl, models/metrics.json |
| S1 audited model | scripts/stage_s1_cost_model_leakfree.py, data/rollouts_s1_v2.csv, data/features_s1_v2.csv, models/cost_model_s1_v2.pkl, models/cost_model_params.json |
| S2 audited QUBO | scripts/rebuild_s2_qubo_v3.py, data/joint_rollouts_s2_v3.csv, models/qubo_s2_v3.npz, models/qubo_s2_v3.json, results/qubo_metrics_s2_v3.json |
| S3 solver results | scripts/rebuild_s3_solvers_v3.py, results/solver_metrics_s3_v3.json |
| Phase 7/8 evaluation | src/phase7_controller.py, scripts/run_phase8_evaluation.py, results/phase8_eval_v1/ |
| Adaptive comparison | src/adaptive_signal_controller.py, scripts/evaluate_adaptive_control.py, results/adaptive_ai_verified_seed*/ |
| Consolidated report/dashboard metrics | src/project_evaluation.py, results/project_evaluation.json, results/project_evaluation.csv, results/evaluation_report.md, results/graphs/ |
| Dashboard | dashboard/app.py, dashboard/live_simulation.py, dashboard/data.py, dashboard/evaluation_views.py |
| Tests/logs | tests/, logs/ |

## Limitations and interpretation rules
1. **S2 is not validated.** Its two held-out joint snapshots show 0/2 optimum agreement and weak plan-ranking correlations. Do not use the S2 objective for closed-loop control until it passes broader held-out SUMO tests.
2. **S1 metric files contain multiple versions.** The dashboard uses consolidated S1 v2 numbers in results/project_evaluation.json and results/evaluation_report.md. results/cost_model_metrics.json is a separate earlier pipeline artifact with different counts/results. Do not mix them; inspect each file's version, split and denominator.
3. **The evaluation report summarizes saved evidence.** It does not retrain models or rerun SUMO/solvers during report generation.
4. **Small test scope.** ST-GNN is scored on seed 4; S1 has 60 test groups; S2 has two test snapshots; Phase 8 has two seeds per scenario; the separate adaptive comparison has five seeds in one scenario.
5. **Scenario variability.** ST-GNN is notably weaker on the low-demand subset. Aggregate scores hide this; use per-scenario and per-junction breakdowns.
6. **Phase 8 does not show optimization improvement.** SA falls back on 390/400 decisions; QAOA is slow, rarely feasible at the sample level and frequently falls back. S2 itself is not validated.
7. **Dashboard gap.** Live SUMO automatic control is separate from QUBO, defaults on, and current code lacks manual per-junction plan selectors.
8. **No quantum advantage.** Saved QAOA runs use a classical simulator; no hardware outcome is established.
9. **No deployment evidence.** This study uses simulated demand and no field sensor data or real traffic signal hardware.
Use the versioned artifacts as evidence. A passing unit test or QUBO-feasibility check validates code/mechanics within its test; it does not prove predictive accuracy, policy benefit or real-world safety.
