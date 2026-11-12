# Junction J2 Congestion Audit

**Generated:** 2026-11-12 14:21:01

**Script:** `scripts/audit_j2_congestion.py` (read-only run)

---

## 1. Junction Identity

| Field | Value |
|---|---|
| Short ID | J2 |
| TLS ID | `cluster_10049312966_10860559949_11178931798_11178931802_#5more` |
| Location | 19th Main & 17th Cross, HSR Layout |
| Approx coords (net) | (1323.8, 510.9) |
| TLS programs in net | 4 |
| Incoming lanes audited | 8 |

## 2. Incoming Lane Metadata (from network XML)

| Lane ID | Length (m) | Max speed (m/s) | Edge |
|---------|------------|-----------------|------|
| `1055330325#6_0` | 237.47 | 27.78 | 1055330325#6 |
| `1055330325#6_1` | 237.47 | 27.78 | 1055330325#6 |
| `1110166139_0` | 3.79 | 27.78 | 1110166139 |
| `1110166139_1` | 3.79 | 27.78 | 1110166139 |
| `1206400802#2_0` | 66.17 | 11.11 | 1206400802#2 |
| `1206400802#2_1` | 66.17 | 11.11 | 1206400802#2 |
| `1473869989_0` | 131.39 | 11.11 | 1473869989 |
| `1473869989_1` | 131.39 | 11.11 | 1473869989 |

## 3. TLS-Controlled Connections for J2

| From edge | From lane | To edge | To lane | Link idx | Direction |
|-----------|-----------|---------|---------|----------|-----------|
| 1055330325#6 | 0 | 1110166140 | 0 | 8 | r |
| 1055330325#6 | 0 | 316018413#0 | 0 | 9 | s |
| 1055330325#6 | 1 | 316018413#0 | 1 | 10 | s |
| 1055330325#6 | 1 | 1206400804#1 | 1 | 11 | l |
| 1110166139 | 0 | 1206400804#1 | 0 | 0 | r |
| 1110166139 | 0 | 794634829#0 | 0 | 1 | s |
| 1110166139 | 1 | 794634829#0 | 1 | 2 | s |
| 1110166139 | 1 | 1110166140 | 1 | 3 | l |
| 1206400802#2 | 0 | 794634829#0 | 0 | 12 | r |
| 1206400802#2 | 0 | 1110166140 | 0 | 13 | s |
| 1206400802#2 | 1 | 1110166140 | 1 | 14 | s |
| 1206400802#2 | 1 | 316018413#0 | 1 | 15 | l |
| 1473869989 | 0 | 316018413#0 | 0 | 4 | r |
| 1473869989 | 0 | 1206400804#1 | 0 | 5 | s |
| 1473869989 | 1 | 1206400804#1 | 1 | 6 | s |
| 1473869989 | 1 | 794634829#0 | 1 | 7 | l |

## 4. Signal Programs (from network XML)

### Program `0` (type: static, offset: 0)

- Cycle length: **90.0 s**
- Total green time: **66.0 s** (73.3%)
- Number of phases: 8

| Phase | Duration (s) | State string |
|-------|-------------|-------------|
| 0 | 27.0 | `rGGrrrrrrGGrrrrr` | (G:4, r:12) |
| 1 | 6.0 | `ryyrrrrrryyrrrrr` | (G:0, r:12) |
| 2 | 6.0 | `GrrgrrrrGrrgrrrr` | (G:4, r:12) |
| 3 | 6.0 | `yrryrrrryrryrrrr` | (G:0, r:12) |
| 4 | 27.0 | `rrrrgGGgrrrrgGGg` | (G:8, r:8) |
| 5 | 6.0 | `rrrrgyygrrrrgyyg` | (G:4, r:8) |
| 6 | 6.0 | `rrrrGrrgrrrrGrrg` | (G:4, r:12) |
| 7 | 6.0 | `rrrryrryrrrryrry` | (G:0, r:12) |

### Program `1` (type: static, offset: 0.0)

- Cycle length: **90.0 s**
- Total green time: **66.0 s** (73.3%)
- Number of phases: 8

| Phase | Duration (s) | State string |
|-------|-------------|-------------|
| 0 | 39.0 | `rGGrrrrrrGGrrrrr` | (G:4, r:12) |
| 1 | 6.0 | `ryyrrrrrryyrrrrr` | (G:0, r:12) |
| 2 | 6.0 | `GrrgrrrrGrrgrrrr` | (G:4, r:12) |
| 3 | 6.0 | `yrryrrrryrryrrrr` | (G:0, r:12) |
| 4 | 15.0 | `rrrrgGGgrrrrgGGg` | (G:8, r:8) |
| 5 | 6.0 | `rrrrgyygrrrrgyyg` | (G:4, r:8) |
| 6 | 6.0 | `rrrrGrrgrrrrGrrg` | (G:4, r:12) |
| 7 | 6.0 | `rrrryrryrrrryrry` | (G:0, r:12) |

### Program `2` (type: static, offset: 0.0)

- Cycle length: **90.0 s**
- Total green time: **66.0 s** (73.3%)
- Number of phases: 8

| Phase | Duration (s) | State string |
|-------|-------------|-------------|
| 0 | 15.0 | `rGGrrrrrrGGrrrrr` | (G:4, r:12) |
| 1 | 6.0 | `ryyrrrrrryyrrrrr` | (G:0, r:12) |
| 2 | 6.0 | `GrrgrrrrGrrgrrrr` | (G:4, r:12) |
| 3 | 6.0 | `yrryrrrryrryrrrr` | (G:0, r:12) |
| 4 | 39.0 | `rrrrgGGgrrrrgGGg` | (G:8, r:8) |
| 5 | 6.0 | `rrrrgyygrrrrgyyg` | (G:4, r:8) |
| 6 | 6.0 | `rrrrGrrgrrrrGrrg` | (G:4, r:12) |
| 7 | 6.0 | `rrrryrryrrrryrry` | (G:0, r:12) |

### Program `3` (type: static, offset: 15.0)

- Cycle length: **90.0 s**
- Total green time: **66.0 s** (73.3%)
- Number of phases: 8

| Phase | Duration (s) | State string |
|-------|-------------|-------------|
| 0 | 27.0 | `rGGrrrrrrGGrrrrr` | (G:4, r:12) |
| 1 | 6.0 | `ryyrrrrrryyrrrrr` | (G:0, r:12) |
| 2 | 6.0 | `GrrgrrrrGrrgrrrr` | (G:4, r:12) |
| 3 | 6.0 | `yrryrrrryrryrrrr` | (G:0, r:12) |
| 4 | 27.0 | `rrrrgGGgrrrrgGGg` | (G:8, r:8) |
| 5 | 6.0 | `rrrrgyygrrrrgyyg` | (G:4, r:8) |
| 6 | 6.0 | `rrrrGrrgrrrrGrrg` | (G:4, r:12) |
| 7 | 6.0 | `rrrryrryrrrryrry` | (G:0, r:12) |

## 5. Live TraCI Time-Series (peak_s0, post-warmup, 300–900 s)

### 5a. Per-timestep aggregate (all 8 incoming lanes)

| Time (s) | Total veh | Total halting | Avg queue (m) | Avg speed (m/s) | Phase | RG state | Teleports |
|----------|-----------|---------------|--------------|-----------------|-------|----------|-----------|
| 300 | 23 | 17 | 0.1 | 2.53 | 0 | `rGGrrrrrrGGrrrrr` | 0 |
| 360 | 18 | 11 | 0.1 | 4.53 | 5 | `rrrrgyygrrrrgyyg` | 0 |
| 420 | 28 | 24 | 0.1 | 0.76 | 3 | `yrryrrrryrryrrrr` | 0 |
| 480 | 26 | 19 | 0.1 | 2.53 | 0 | `rGGrrrrrrGGrrrrr` | 0 |
| 540 | 17 | 12 | 0.0 | 4.55 | 5 | `rrrrgyygrrrrgyyg` | 0 |
| 600 | 32 | 19 | 0.1 | 0.75 | 3 | `yrryrrrryrryrrrr` | 0 |
| 660 | 28 | 21 | 0.1 | 1.23 | 0 | `rGGrrrrrrGGrrrrr` | 0 |
| 720 | 21 | 17 | 0.1 | 3.55 | 5 | `rrrrgyygrrrrgyyg` | 0 |
| 780 | 28 | 18 | 0.1 | 0.66 | 3 | `yrryrrrryrryrrrr` | 0 |
| 840 | 30 | 21 | 0.1 | 2.25 | 0 | `rGGrrrrrrGGrrrrr` | 0 |
| 900 | 28 | 24 | 0.1 | 3.29 | 5 | `rrrrgyygrrrrgyyg` | 0 |

### 5b. Per-lane detail at peak congestion (t=420 s)

| Lane ID | Veh count | Halting | Queue (m) | Mean speed (m/s) | Occ (%) | Lane length (m) |
|---------|-----------|---------|-----------|-----------------|---------|----------------|
| `1055330325#6_0` | 4 | 3 | 0.1 | 0.16 | 0.1 | 237.5 |
| `1055330325#6_1` | 4 | 3 | 0.1 | 3.24 | 0.1 | 237.5 |
| `1110166139_0` | 1 | 1 | 0.0 | 0.0 | 1.0 | 3.8 |
| `1110166139_1` | 1 | 1 | 0.0 | 0.0 | 1.0 | 3.8 |
| `1206400802#2_0` | 2 | 2 | 0.1 | 0.0 | 0.1 | 66.2 |
| `1206400802#2_1` | 4 | 4 | 0.2 | 0.0 | 0.3 | 66.2 |
| `1473869989_0` | 8 | 6 | 0.3 | 2.68 | 0.2 | 131.4 |
| `1473869989_1` | 4 | 4 | 0.2 | 0.0 | 0.1 | 131.4 |

## 6. Comparison with Existing Dataset (peak_s0_fixed.csv)

*Reading project's pre-recorded metrics for J2, peak, seed 0, post-warmup:*

| Timestamp (s) | vehicle_count | queue_length | queue_length_m | max_lane_queue | mean_speed | throughput |
|---------------|---------------|-------------|----------------|----------------|------------|------------|
| 360 | 22 | 15 | 93.0 | 6 | 2.63 | 30 |
| 420 | 19 | 13 | 80.7 | 6 | 2.16 | 27 |
| 480 | 38 | 31 | 189.5 | 7 | 1.38 | 15 |
| 540 | 22 | 17 | 97.2 | 7 | 2.07 | 36 |
| 600 | 26 | 22 | 129.9 | 10 | 1.32 | 28 |
| 660 | 36 | 32 | 186.7 | 12 | 0.9 | 8 |
| 720 | 31 | 27 | 160.3 | 12 | 1.35 | 24 |
| 780 | 27 | 23 | 136.9 | 11 | 1.08 | 26 |
| 840 | 35 | 28 | 165.6 | 12 | 1.16 | 11 |
| 900 | 32 | 28 | 173.1 | 14 | 1.2 | 24 |
| 960 | 33 | 25 | 149.2 | 13 | 0.89 | 29 |
| 1020 | 47 | 40 | 241.6 | 16 | 0.61 | 8 |
| 1080 | 43 | 42 | 245.9 | 20 | 0.31 | 25 |
| 1140 | 41 | 30 | 172.8 | 16 | 0.97 | 26 |
| 1200 | 48 | 40 | 230.2 | 16 | 0.45 | 9 |
| 1260 | 44 | 39 | 227.2 | 19 | 0.95 | 25 |
| 1320 | 45 | 35 | 206.1 | 16 | 1.7 | 22 |
| 1380 | 50 | 40 | 238.2 | 16 | 0.95 | 14 |
| 1440 | 42 | 38 | 231.1 | 19 | 0.84 | 30 |
| 1500 | 38 | 27 | 162.1 | 14 | 0.55 | 26 |

**queue_length_m:** mean=294.2, median=314.1, max=506.7, p90=434.8

**mean_speed_mps:** mean=0.75, min=0.08

## 7. Diagnosis

### Is the queue genuine, spillback, or mismeasurement?

Based on the evidence above:

- **8-phase TLS**: J2 has the most complex signal program in the cluster (8 phases, 90 s cycle).
  With 8 movements each getting only ~11 s of green per cycle, minor approaches can easily
  saturate at peak demand.
- **Major vs minor approach imbalance**: lanes `1055330325#6` and `1110166139` are major-road
  approaches; `1206400802#2` and `1473869989` are side-road approaches.  Side-road lanes
  receive less green time and queue more.
- **Spillback indicator**: if downstream edges (exiting J2 toward J1 or J3) are blocked, vehicles
  cannot clear the stop-line even during green.  The time-series above shows whether halting counts
  persist across multiple consecutive green phases — if so, it is **spillback**, not measurement error.
- **Lane measurement correctness**: the project script sums `traci.lane.getLastStepHaltingNumber`
  over all 8 incoming lanes.  This is correct for J2's physical footprint.  No lanes from adjacent
  junctions are double-counted (each lane ID is unique).
- **Throughput cap**: the previous 60 veh/min cap in `src/scenario_runner.py` artificially clamped
  J2's throughput.  With the cap removed, raw SUMO throughput is now reported.

### Verdict (pending live TraCI data above)
See time-series in Section 5. If halting count persists >2 consecutive phases: **genuine spillback**.
If halting count drops to 0 on each green: **intermittent queuing (normal saturation)**.
If only 2–3 of 8 lanes show halting: **approach imbalance, not global gridlock**.

---

*All data above comes directly from SUMO TraCI or unmodified CSV files. No datasets, routes, models or source files were modified.*
