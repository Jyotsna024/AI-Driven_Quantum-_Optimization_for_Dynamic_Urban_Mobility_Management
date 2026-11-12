# AI-Driven Quantum Optimization for Dynamic Urban Mobility Management
## PROJECT_SPEC.md — Single source of truth. Re-read at start of each phase.

---

## ENVIRONMENT (verified 2026-11-12)
- OS: Windows 11 (PowerShell)
- SUMO: 1.28.0 at `C:\Program Files (x86)\SUMO\`; `SUMO_HOME` set
- Python: 3.13.7; pip 26.2.1
- PyTorch: 2.9.0+cpu (no CUDA)
- PyG (torch_geometric): 2.8.1
- dimod: 0.12.23, dwave-neal: 0.6.0, dwave-samplers: 1.8.0
- qiskit: 2.6.2, qiskit-aer: 0.18.2, qiskit-algorithms: 0.4.0
- streamlit: 1.51.0, matplotlib: 3.10.6, scikit-learn: 1.7.2
- scipy: 1.16.2, pandas: 2.3.3, numpy: 2.2.6, networkx: 3.5
- requests: 2.32.5, pytest: 8.4.2, pyyaml: 6.0.3
- Quantum hardware tokens: DWAVE_API_TOKEN / IBMQ_TOKEN not set -> Qiskit Aer simulator default

---

## DECISIONS

| Decision | Value |
|---|---|
| Area | HSR Layout, Bengaluru (OpenStreetMap API official extract, 7.89 MB) |
| Intersections | 4 signalised, connected corridor cluster (14th Main & 17th Cross) |
| Expansion gate | Only after Phase 8 passes on 4; ask user |
| Plans K | 4 per intersection (baseline, main-road-heavy, side-road-heavy, shifted-offset) |
| QUBO variables | N×K = 16 |
| Scenarios | low, medium, high, peak, incident; seeds=5 |
| Control interval | 60 simulated seconds |
| Prediction horizon | 1–3 intervals |
| Quantum priority | (1) Qiskit Aer QAOA (always); (2) D-Wave Leap if DWAVE_API_TOKEN; (3) IBM Quantum if token exists |
| Dashboard | Streamlit, built last, reads saved results only |
| Smoke mode | `--smoke` flag: shorter simulation runs for quick CI tests |

---

## DIRECTORY LAYOUT
```
Capstone/
├── PROJECT_SPEC.md          ← this file
├── README.md
├── requirements.txt
├── run_all.py               ← single entry point
├── configs/
│   ├── base.yaml
│   ├── scenarios.yaml
│   └── qubo.yaml
├── sumo_files/
│   ├── hsr_raw.osm          ← genuine OSM extract (7.89 MB)
│   ├── hsr.net.xml          ← netconvert output (4.45 MB, 600 junctions, 5 TLS)
│   ├── cluster_config.json  ← cluster metadata, coordinates, and K=4 plans
│   ├── net_sketch.png       ← layout visualization
│   ├── routes/              ← per-scenario .rou.xml
│   └── *.sumocfg
├── data/
│   ├── raw/                 ← untouched SUMO output CSVs
│   └── processed/           ← cleaned, feature-engineered
├── src/
│   ├── env_check.py
│   ├── network_builder.py   ← Phase 1
│   ├── scenario_runner.py   ← Phase 2
│   ├── data_validator.py    ← Phase 2
│   ├── stgnn.py             ← Phase 3
│   ├── cost_model.py        ← Phase 4
│   ├── qubo_builder.py      ← Phase 5
│   ├── solvers.py           ← Phase 6
│   ├── controller.py        ← Phase 7
│   └── evaluator.py         ← Phase 8
├── tests/
│   └── test_*.py
├── models/
├── results/
├── logs/
└── dashboard/
    └── app.py               ← Phase 9
```

---

## SIGNAL PLANS (documented per intersection from ACTUAL SUMO phases)
Plans are defined over each junction's ACTUAL SUMO phases from netconvert.

| Index | Junction / TLS ID | Street Location | Coords (UTM x,y) | Actual Phases | Plan 0 (baseline) | Plan 1 (main-heavy) | Plan 2 (side-heavy) | Plan 3 (shifted) |
|---|---|---|---|---|---|---|---|---|
| J0 | `joinedS_12067305543_12067305544` | 14th Main & 15th Cross | (635.5, 908.0) | 4 (cycle 90s) | 42s G, 3s y, 42s G, 3s y | 54.6s G (main), 29.4s G (side) | 29.4s G (main), 54.6s G (side) | Baseline + 15s offset |
| J1 | `GS_cluster_6633301856_7432359269_7432359283_7432359284` | 14th Main & 17th Cross | (626.6, 523.2) | 4 (cycle 90s) | 39s G, 6s y, 39s G, 6s y | 50.7s G (main), 27.3s G (side) | 27.3s G (main), 50.7s G (side) | Baseline + 15s offset |
| J2 | `cluster_10049312966_10860559949_11178931798_11178931802_#5more` | 19th Main & 17th Cross | (1323.8, 510.9) | 8 (cycle 90s) | 27s G, 6s y, 6s G, 6s y, 27s G, 6s y, 6s G, 6s y | Primary greens x1.3, minor x0.7 | Minor greens x1.3, primary x0.7 | Baseline + 15s offset |
| J3 | `GS_cluster_12067305554_6017833806_6017833807_6131076159` | 14th Main & Sarjapur Road | (818.1, 1805.0) | 4 (cycle 90s) | 39s G, 6s y, 39s G, 6s y | 50.7s G (main), 27.3s G (side) | 27.3s G (main), 50.7s G (side) | Baseline + 15s offset |

---

## QUBO FORMULATION
- Variables: x[i,k] ∈ {0,1}, i=junction, k=plan
- Energy: Σ_i Σ_k c[i,k]·x[i,k]  +  Σ_(i,j)∈E Σ_k Σ_l J[i,j,k,l]·x[i,k]·x[j,l]  +  A·Σ_i(Σ_k x[i,k]−1)²
- c[i,k]: cost model output (Webster/HCM delay + ST-GNN predicted demand)
- J[i,j,k,l]: pairwise coordination terms from rollout pairs
- A: penalty tuned so optima are feasible (Phase 5)

---

## PROGRESS CHECKLIST

### STEP 0: Spec + Setup
- [x] PROJECT_SPEC.md created
- [x] requirements.txt written
- [x] Directory structure created
- [x] configs/base.yaml written
- [x] README.md stub

### Phase 1: ENV + NETWORK
- [x] Environment verified (SUMO 1.28.0, Python 3.13.7, PyTorch 2.9.0+cpu, PyG 2.8.1, dimod, neal, qiskit, qiskit-aer)
- [x] Genuine OpenStreetMap data for HSR Layout downloaded via official OSM API (7.89 MB, 26,204 nodes, 5,343 ways)
- [x] netconvert run -> hsr.net.xml (4.45 MB, 600 non-internal junctions, 5 TLS systems)
- [x] 4-intersection cluster selected along 14th Main & 17th Cross corridors (J0: 14th Main & 15th Cross, J1: 14th Main & 17th Cross, J2: 19th Main & 17th Cross, J3: 14th Main & Sarjapur Rd)
- [x] TLS phases inspected and verified directly from SUMO netconvert output
- [x] K=4 signal plans defined over each junction's ACTUAL SUMO phases (baseline, main-heavy, side-heavy, shifted-offset)
- [x] Road connectivity verified: 12 connected directed paths between cluster junctions
- [x] Net sketch image produced: sumo_files/net_sketch.png
- [x] SUMO TraCI loading tested and verified cleanly with all 4 active TLS
- [x] test_phase1.py passes
- [x] Phase 1 report written


### Phase 2: SCENARIOS + DATA
- [x] Route generation for 12 connected cluster paths across 5 scenarios x 5 seeds
- [x] Flow calibration (low < medium < high < peak without permanent gridlock)
- [x] Incident injection configuration (medium-high demand, 600s lane block at t=1200s, recovery)
- [x] Signal safety validation (valid SUMO TLS programs, safe cycle boundary transitions, zero conflicting greens)
- [x] 60s TraCI interval logging (halting queue, count, mean speed, wait time, throughput, arrivals, plan ID)
- [x] State snapshotting (saveState every 300s for t>=600, index CSV, reload verification, determinism test)
- [x] Smoke test execution (--smoke: 1 scenario, 1 seed, 600s)
- [x] Full run matrix execution (50 runs: 5 scenarios x 5 seeds x [fixed-time, varied-plan])
- [x] Data validation script (`scripts/validate_data.py` -> `results/data_validation.md`)
- [x] Phase 2 pytest suite (`tests/test_phase2.py`: 9/9 PASS)
- [x] Phase 2 report written (see below)

### Phase 3: ST-GNN
- [x] Graph construction (nodes=junctions, edges=roads, 12 directed connections)
- [x] Dataset split by seed: Train (seeds 0, 1, 2), Val (seed 3), Held-Out Test (seed 4)
- [x] Model trained (PyG GCNConv + GRU + Multi-horizon head), baselines computed (Persistence, Historical Average)
- [x] MAE/RMSE across H=1, H=2, H=3 on held-out test reported
- [x] Model (`models/stgnn.pt`) + scaler (`models/scaler.pkl`) + metrics (`models/metrics.json`) saved
- [x] Prediction plot (`results/stgnn_predictions.png`) generated
- [x] Phase 3 pytest suite (`tests/test_phase3.py`: 4/4 PASS)
- [x] Phase 3 report written (see below)

### Phase 4: PLAN COST MODEL
- [ ] Ground-truth rollouts (singles + pairs) via saveState/loadState
- [ ] Webster/HCM cost model implemented
- [ ] Calibrated on training scenarios
- [ ] Validated on held-out: Spearman ρ, top-1 agreement
- [ ] Passes quality gate (Spearman ρ >= 0.5); if not, improved
- [ ] Phase 4 report written

### Phase 5: QUBO
- [ ] QUBO matrix built from cost model + pairwise terms
- [ ] Penalty A tuned
- [ ] ExactSolver unit test on 16 variables
- [ ] Oracle check vs 256-combo brute force
- [ ] QUBO serialized
- [ ] Phase 5 report written

### Phase 6: SOLVERS
- [ ] Simulated Annealing (dwave-neal) on serialized QUBO
- [ ] QAOA (Qiskit Aer) on serialized QUBO (≤20 qubits) or sub-instance
- [ ] D-Wave Leap (if token available)
- [ ] IBM Quantum (if token available)
- [ ] Energy, feasibility, runtime logged per solver
- [ ] Phase 6 report written

### Phase 7: CLOSED LOOP
- [ ] Closed-loop controller implemented
- [ ] Fixed-time, SA, QAOA, max-pressure controllers
- [ ] TraCI integration working
- [ ] Phase 7 report written

### Phase 8: EVALUATION
- [ ] All controllers evaluated on identical scenarios/seeds
- [ ] Avg waiting time, queue, travel time, throughput, runtime
- [ ] Mean ± std, % vs fixed-time computed
- [ ] CSV + Matplotlib figures saved
- [ ] Expansion gate: ask user about scaling to 6 intersections
- [ ] Phase 8 report written

### Phase 9: DASHBOARD
- [ ] Streamlit app reads saved results
- [ ] Traffic state, predicted vs actual, cost-model validation, controller comparison
- [ ] Phase 9 report written

### Phase 10: FINALIZE
- [ ] Clean reproducibility test (fresh run)
- [ ] RESULTS_SUMMARY.md with honest findings + limitations
- [ ] README with exact Windows setup + run steps
- [ ] Phase 10 report written

## PHASE REPORTS

### Phase 1 Report: Environment & Real Network Setup
- **Done**: Environment verified and fully provisioned; genuine HSR Layout OpenStreetMap XML (7.89 MB, 26,204 nodes, 5,343 ways, 22 traffic signals) downloaded from official OSM API; converted via SUMO `netconvert` into `hsr.net.xml` (4.45 MB, 600 junctions, 5 TLS systems); 4 adjacent signalised intersections selected along the 14th Main & 17th Cross arterial corridors; actual SUMO TLS phase structures inspected; K=4 signal plans defined per intersection; road connectivity verified (12 connected directed pairs); layout sketch generated (`net_sketch.png`); SUMO TraCI loading and runtime control tested and verified cleanly.
- **Files**:
  - [`src/network_builder.py`](file:///c:/Users/Jyotsna/Desktop/Capstone/src/network_builder.py)
  - [`sumo_files/hsr_raw.osm`](file:///c:/Users/Jyotsna/Desktop/Capstone/sumo_files/hsr_raw.osm) (7.89 MB)
  - [`sumo_files/hsr.net.xml`](file:///c:/Users/Jyotsna/Desktop/Capstone/sumo_files/hsr.net.xml) (4.45 MB)
  - [`sumo_files/cluster_config.json`](file:///c:/Users/Jyotsna/Desktop/Capstone/sumo_files/cluster_config.json)
  - [`sumo_files/net_sketch.png`](file:///c:/Users/Jyotsna/Desktop/Capstone/sumo_files/net_sketch.png)
  - [`tests/test_phase1.py`](file:///c:/Users/Jyotsna/Desktop/Capstone/tests/test_phase1.py)
- **Key numbers**:
  - Network junctions: 600 non-internal
  - Traffic light systems: 5 total in network, 4 selected in cluster
  - Cluster junctions: J0 (14th Main & 15th Cross, 4 phases), J1 (14th Main & 17th Cross, 4 phases), J2 (19th Main & 17th Cross, 8 phases), J3 (14th Main & Sarjapur Rd, 4 phases)
  - Signal plans: K=4 per intersection (16 QUBO decision variables)
  - Connected paths: 12 directed routes among cluster intersections (shortest distances 349m to 778m)
- **Problems encountered & solutions**: Overpass API returned HTTP 406 / timeouts; successfully resolved by downloading directly from the official OpenStreetMap API bounding box endpoint (`api.openstreetmap.org/api/0.6/map?bbox=...`) using standard user-agent headers. No synthetic networks or fabricated geometries were used.
- **Next**: Phase 2 (Scenarios & Data Generation) — completed.

### Phase 2 Report: Scenarios & Data Generation (Repaired & Fully Validated)
- **Status**: COMPLETE & APPROVED (All 8 Stage 1 Acceptance Gates Passed, 9/9 Pytest Tests Passed).
- **Done**:
  - Replaced terminating OD routes with 39 verified through-movement connections across all 4 cluster junctions (J0–J3).
  - Recalibrated scenario flows: low (300 veh/h/junc), medium (650 veh/h/junc), high (1000 veh/h/junc), peak (1400 veh/h/junc), incident (850 veh/h/junc).
  - Configured 4 distinct static signal programs (>=30% green share difference: baseline, main-heavy, side-heavy, shifted-offset) embedded in `hsr.net.xml`.
  - Archived previous dataset safely to `data/archive/raw_phase2_v1/` and `data/archive/states_phase2_v1/`.
  - Executed full 50-run simulation matrix (5 scenarios x 5 seeds x [fixed, varied]) logging rich metrics: `queue_length`, `queue_length_m`, `max_lane_queue`, `queue_major`, `queue_minor`, `mean_speed_mps` (NaN for empty), `mean_delay_s`, `waiting_time_s`, `throughput`, `arrivals`.
  - Saved and indexed 275 state snapshots for Phase 4 cost rollouts.
  - Successfully evaluated all 8 acceptance gates: Gate 1 (peak mean queue >= 3 & P90 >= 8 at all junctions), Gate 2 (low < med < high < peak queue ordering at all junctions), Gate 3 (best vs worst plan delay diff >= 10% at all junctions), Gate 4 (best plan varies across snapshots), Gate 5 (throughput reconciles with arrivals and stays <= capacity), Gate 6 (final 600s throughput > 0, zero gridlock), Gate 7 (bit-exact determinism on reload), Gate 8 (incident creates measurable speed drop: 1.18 m/s vs 2.20 m/s).
- **Files**:
  - [`scripts/stage1_fix_phase2_data.py`](file:///c:/Users/Jyotsna/Desktop/Capstone/scripts/stage1_fix_phase2_data.py)
  - [`configs/scenarios.yaml`](file:///c:/Users/Jyotsna/Desktop/Capstone/configs/scenarios.yaml)
  - [`data/raw/manifest.csv`](file:///c:/Users/Jyotsna/Desktop/Capstone/data/raw/manifest.csv) (50 runs, 12,000 observations)
  - [`data/states/snapshot_index.csv`](file:///c:/Users/Jyotsna/Desktop/Capstone/data/states/snapshot_index.csv) (275 snapshots)
  - [`results/data_validation.md`](file:///c:/Users/Jyotsna/Desktop/Capstone/results/data_validation.md)
  - [`results/per_junction_analysis.csv`](file:///c:/Users/Jyotsna/Desktop/Capstone/results/per_junction_analysis.csv)
  - [`results/scenario_summary.csv`](file:///c:/Users/Jyotsna/Desktop/Capstone/results/scenario_summary.csv)
  - [`results/plan_sensitivity.csv`](file:///c:/Users/Jyotsna/Desktop/Capstone/results/plan_sensitivity.csv)
  - [`results/throughput_reconciliation.csv`](file:///c:/Users/Jyotsna/Desktop/Capstone/results/throughput_reconciliation.csv)
  - [`tests/test_phase2.py`](file:///c:/Users/Jyotsna/Desktop/Capstone/tests/test_phase2.py)
- **Key numbers**:
  - Runs: 50 completed, 0 failed
  - Peak mean queue (post-warmup): J0 = 4.87, J1 = 7.47, J2 = 50.33, J3 = 7.43 (Gate 1 PASS)
  - Peak P90 queue: J0 = 8.0, J1 = 13.0, J2 = 74.0, J3 = 14.0 (Gate 1 PASS)
  - Mean queue hierarchy (low -> med -> high -> peak): J0 (1.26 -> 2.79 -> 3.92 -> 4.87), J1 (1.44 -> 3.38 -> 5.32 -> 7.47), J2 (2.27 -> 6.75 -> 23.97 -> 50.33), J3 (1.31 -> 3.19 -> 5.15 -> 7.43) (Gate 2 PASS)
  - Plan sensitivity: 4/4 junctions exhibit > 10% delay sensitivity to signal plans (Gate 3 PASS)
  - Peak final 600s throughput: 22.55 veh/min (Gate 6 PASS)
  - Incident active speed: 1.18 m/s vs 2.20 m/s baseline (Gate 8 PASS)
  - Tests: 9/9 PASS in `tests/test_phase2.py`, 6/6 PASS in `tests/test_phase1.py`
- **Next**: Phase 3 (ST-GNN Traffic Prediction) — completed.

### Phase 3 Report: Spatio-Temporal GNN Traffic Prediction
- **Status**: COMPLETE & APPROVED (4/4 Pytest Tests Passed).
- **Done**:
  - Constructed directed spatial graph of 4 cluster intersections with 12 distance-weighted edges.
  - Implemented Spatio-Temporal Graph Neural Network (`SpatioTemporalGNN`) combining PyTorch Geometric `GCNConv` spatial feature propagation, a Gated Recurrent Unit (`GRU`, hidden dim 32) across lookback $T=3$ (180s), and a multi-horizon linear projection head for $H=1, 2, 3$ (60s, 120s, 180s ahead).
  - Strictly split by seed: Train (seeds 0, 1, 2: 1,590 windows), Validation (seed 3: 530 windows), Held-Out Test (seed 4: 530 windows).
  - Fitted and persisted `StandardScaler` (`models/scaler.pkl`) on train set features.
  - Trained model using Adam optimizer (`lr=0.005`, MSE loss, 45 epochs), achieving best validation loss of 0.0721.
  - Evaluated on held-out Test set (seed 4) against Persistence and Historical Average baselines.
  - Generated prediction visualization figure (`results/stgnn_predictions.png`) and report (`results/stgnn_report.md`).
- **Files**:
  - [`scripts/stage2_stgnn.py`](file:///c:/Users/Jyotsna/Desktop/Capstone/scripts/stage2_stgnn.py)
  - [`src/stgnn.py`](file:///c:/Users/Jyotsna/Desktop/Capstone/src/stgnn.py)
  - [`models/stgnn.pt`](file:///c:/Users/Jyotsna/Desktop/Capstone/models/stgnn.pt)
  - [`models/scaler.pkl`](file:///c:/Users/Jyotsna/Desktop/Capstone/models/scaler.pkl)
  - [`models/metrics.json`](file:///c:/Users/Jyotsna/Desktop/Capstone/models/metrics.json)
  - [`results/stgnn_predictions.png`](file:///c:/Users/Jyotsna/Desktop/Capstone/results/stgnn_predictions.png)
  - [`results/stgnn_report.md`](file:///c:/Users/Jyotsna/Desktop/Capstone/results/stgnn_report.md)
  - [`tests/test_phase3.py`](file:///c:/Users/Jyotsna/Desktop/Capstone/tests/test_phase3.py)
- **Key numbers**:
  - Overall MAE: ST-GNN = **2.410 veh** vs Persistence = 3.544 veh vs Historical Average = 6.017 veh (32% error reduction vs Persistence, 60% vs HA)
  - Overall RMSE: ST-GNN = **3.630 veh** vs Persistence = 5.348 veh vs Historical Average = 10.311 veh
  - Horizon 1 (60s) MAE: ST-GNN = **2.378 veh** vs Persistence = 3.892 veh (39% error reduction)
  - Horizon 2 (120s) MAE: ST-GNN = **2.441 veh** vs Persistence = 4.024 veh
  - Horizon 3 (180s) MAE: ST-GNN = **2.410 veh** vs Persistence = 2.718 veh
  - Tests: 4/4 PASS in `tests/test_phase3.py`
- **Next**: Phase 4 (Plan Cost Model) — ground-truth rollouts (singles + pairs) via TraCI `saveState`/`loadState`, Webster/HCM analytical cost model calibration on train seeds, validation on held-out seeds (Spearman $\rho > 0.6$, top-1 agreement).

---
*Last updated: Phase 3 complete*

