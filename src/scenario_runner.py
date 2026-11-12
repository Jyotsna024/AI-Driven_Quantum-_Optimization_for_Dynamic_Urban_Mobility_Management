"""
scenario_runner.py – Phase 2: Scenarios & Data Generation
Generates routes for 5 scenarios x 5 seeds over the 12 cluster paths.
Executes 50 SUMO runs (fixed-time baseline & varied-plan).
Logs 60s intersection metrics with accurate throughput, queue, waiting time, arrivals.
Saves periodic simulation snapshots for Phase 4 rollouts.
Supports multiprocessing, resume safety, and --smoke mode.
"""

import os
import sys
import copy
import json
import random
import argparse
import subprocess
import multiprocessing as mp
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import yaml

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

SUMO_DIR = PROJECT_ROOT / "sumo_files"
ROUTES_DIR = SUMO_DIR / "routes"
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
STATES_DIR = PROJECT_ROOT / "data" / "states"
SUMO_BIN = Path(os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\SUMO")) / "bin" / "sumo.exe"

ROUTES_DIR.mkdir(parents=True, exist_ok=True)
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
STATES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load Cluster Configuration & 12 Connected Routes
# ---------------------------------------------------------------------------
CFG_PATH = SUMO_DIR / "cluster_config.json"
if not CFG_PATH.exists():
    raise FileNotFoundError(f"Missing {CFG_PATH}. Run Phase 1 first.")

CLUSTER_CFG = json.loads(CFG_PATH.read_text(encoding="utf-8"))
CLUSTER_TLS_IDS = CLUSTER_CFG["cluster_tls_ids"]
TLS_META = CLUSTER_CFG["tls_meta"]
PLANS_DATA = CLUSTER_CFG["plans"]

NET_PATH = SUMO_DIR / "hsr.net.xml"
NET_TREE = ET.parse(NET_PATH).getroot()

# Map edges to lanes
EDGE_LANES_MAP = {}
for e in NET_TREE.findall(".//edge"):
    EDGE_LANES_MAP[e.get("id")] = [l.get("id") for l in e.findall("lane")]

# Extract the 12 routes
from extract_cluster_routes import routes_12, incoming_edges_per_tls

# Determine controlled / incoming lanes per TLS
TLS_INCOMING_LANES = {}
for tid, edges in incoming_edges_per_tls.items():
    lanes = []
    for eid in edges:
        lanes.extend(EDGE_LANES_MAP.get(eid, []))
    TLS_INCOMING_LANES[tid] = lanes

# ---------------------------------------------------------------------------
# Load Scenario Settings from scenarios.yaml
# ---------------------------------------------------------------------------
SCENARIOS_YAML = PROJECT_ROOT / "configs" / "scenarios.yaml"
with open(SCENARIOS_YAML, "r", encoding="utf-8") as f:
    SCENARIO_CONFIGS = yaml.safe_load(f)


def get_split(seed: int) -> str:
    """Determine dataset split based on seed."""
    seeds_cfg = SCENARIO_CONFIGS["seeds"]
    if seed in seeds_cfg["train"]:
        return "train"
    elif seed in seeds_cfg["val"]:
        return "val"
    elif seed in seeds_cfg["test"]:
        return "test"
    return "train"


def generate_routes_file(scenario_name: str, seed: int, duration_s: int) -> Path:
    """Generate SUMO route file for a specific scenario and seed."""
    out_file = ROUTES_DIR / f"{scenario_name}_s{seed}.rou.xml"
    if out_file.exists() and out_file.stat().st_size > 500:
        return out_file

    sc_cfg = SCENARIO_CONFIGS["scenarios"][scenario_name]
    flow_rate = sc_cfg["flow_veh_per_hour"]
    fleet_cfg = SCENARIO_CONFIGS["fleet"]

    rng = random.Random(seed * 1000 + 7)
    root = ET.Element("routes")

    # Define vehicle types (no sublane model)
    ET.SubElement(
        root, "vType", id="car", accel="2.6", decel="4.5", sigma="0.5",
        length="4.5", maxSpeed="14.0", vClass="passenger"
    )
    ET.SubElement(
        root, "vType", id="motorcycle", accel="3.0", decel="5.0", sigma="0.5",
        length="2.2", maxSpeed="14.0", vClass="motorcycle"
    )
    ET.SubElement(
        root, "vType", id="auto_rickshaw", accel="1.8", decel="3.5", sigma="0.5",
        length="2.8", maxSpeed="10.0", vClass="taxi"
    )

    # Routes
    route_ids = []
    for r in routes_12:
        rid = f"r_{r['from_short']}_{r['to_short']}"
        route_ids.append(rid)
        ET.SubElement(root, "route", id=rid, edges=" ".join(r["edge_path"]))

    # Generate departures
    p_dep = flow_rate / 3600.0
    veh_count = 0
    p_car = fleet_cfg["car_fraction"]
    p_moto = p_car + fleet_cfg["motorcycle_fraction"]

    for t in range(duration_s):
        for rid in route_ids:
            if rng.random() < (p_dep / len(route_ids)):
                r_val = rng.random()
                if r_val < p_car:
                    vtype = "car"
                elif r_val < p_moto:
                    vtype = "motorcycle"
                else:
                    vtype = "auto_rickshaw"
                veh_id = f"veh_{veh_count}_{t}"
                ET.SubElement(
                    root, "vehicle", id=veh_id, type=vtype, route=rid,
                    depart=str(float(t)), departLane="best", departSpeed="max"
                )
                veh_count += 1

    ET.indent(root)
    ET.ElementTree(root).write(str(out_file), encoding="utf-8", xml_declaration=True)
    return out_file


# ---------------------------------------------------------------------------
# Single Simulation Run Execution
# ---------------------------------------------------------------------------
def run_simulation(args_tuple):
    """
    Execute one SUMO simulation run (scenario, seed, mode, duration).
    Returns (run_id, status, total_rows, error_msg).
    """
    import traci

    scenario_name, seed, mode, duration_s, smoke = args_tuple
    run_id = f"{scenario_name}_s{seed}_{mode}"
    out_csv = DATA_RAW_DIR / f"{run_id}.csv"

    # Resume safety check
    expected_rows = (duration_s // 60) * len(CLUSTER_TLS_IDS)
    if out_csv.exists() and not smoke:
        try:
            df = pd.read_csv(out_csv)
            if len(df) == expected_rows:
                return run_id, "SKIPPED_EXISTING", len(df), None
        except Exception:
            pass

    # Generate route file if needed
    rou_path = generate_routes_file(scenario_name, seed, duration_s)

    sc_cfg = SCENARIO_CONFIGS["scenarios"][scenario_name]
    is_incident_scenario = sc_cfg.get("incident", False)
    incident_edge = sc_cfg.get("blocked_edge")
    incident_lane = sc_cfg.get("blocked_lane")
    incident_start = sc_cfg.get("incident_start_s", 1200)
    incident_end = sc_cfg.get("incident_end_s", 1800)

    # State snapshot directory for varied-plan runs
    save_snapshots = (mode == "varied")
    snap_dir = STATES_DIR / f"{scenario_name}_s{seed}"
    if save_snapshots:
        snap_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(SUMO_BIN),
        "-n", str(NET_PATH),
        "-r", str(rou_path),
        "--seed", str(seed + 100),
        "--no-step-log", "true",
        "--no-warnings", "true",
        "--time-to-teleport", "300",
    ]

    try:
        traci.start(cmd)
    except Exception as e:
        return run_id, "FAILED_START", 0, str(e)

    # Initialize all TLS to program 0
    for tid in CLUSTER_TLS_IDS:
        traci.trafficlight.setProgram(tid, "0")

    # Tracking state
    current_plan_ids = {tid: 0 for tid in CLUSTER_TLS_IDS}
    target_plan_ids = {tid: 0 for tid in CLUSTER_TLS_IDS}
    last_phase_ids = {tid: traci.trafficlight.getPhase(tid) for tid in CLUSTER_TLS_IDS}

    # Throughput and arrival tracking
    prev_vehs_on_incoming = {tid: set() for tid in CLUSTER_TLS_IDS}
    throughput_window = {tid: 0 for tid in CLUSTER_TLS_IDS}
    arrivals_window = {tid: 0 for tid in CLUSTER_TLS_IDS}

    log_rows = []
    snapshots_taken = []
    orig_incident_speed = None

    rng_varied = random.Random(seed * 5000 + 13)

    try:
        for step in range(duration_s):
            t = step + 1

            # Handle Incident
            if is_incident_scenario and incident_lane:
                if t == incident_start:
                    orig_incident_speed = traci.lane.getMaxSpeed(incident_lane)
                    traci.lane.setMaxSpeed(incident_lane, 0.1) # Block lane
                elif t == incident_end and orig_incident_speed is not None:
                    traci.lane.setMaxSpeed(incident_lane, orig_incident_speed) # Recover lane

            # Varied Plan Decisions every 60s
            if mode == "varied" and step > 0 and step % 60 == 0:
                for tid in CLUSTER_TLS_IDS:
                    target_plan_ids[tid] = rng_varied.randint(0, 3)

            # Signal Safety: Safe plan transitions strictly at cycle boundary (phase 0)
            for tid in CLUSTER_TLS_IDS:
                cur_phase = traci.trafficlight.getPhase(tid)
                # If entering phase 0 from the final phase, transition to target plan
                if cur_phase == 0 and last_phase_ids[tid] != 0:
                    if target_plan_ids[tid] != current_plan_ids[tid]:
                        new_pid = target_plan_ids[tid]
                        traci.trafficlight.setProgram(tid, str(new_pid))
                        current_plan_ids[tid] = new_pid
                last_phase_ids[tid] = cur_phase

            # Advance simulation step
            traci.simulationStep()

            # Update per-second vehicle tracking for throughput and arrivals
            for tid in CLUSTER_TLS_IDS:
                lanes = TLS_INCOMING_LANES[tid]
                curr_vehs = set()
                for lid in lanes:
                    curr_vehs.update(traci.lane.getLastStepVehicleIDs(lid))

                # Throughput: departed from incoming lanes into junction
                departed = prev_vehs_on_incoming[tid] - curr_vehs
                throughput_window[tid] += len(departed)

                # Arrivals: newly appeared on incoming lanes
                entered = curr_vehs - prev_vehs_on_incoming[tid]
                arrivals_window[tid] += len(entered)

                prev_vehs_on_incoming[tid] = curr_vehs

            # Periodic state snapshotting (every 300s for t >= 600 in varied mode)
            if save_snapshots and t >= 600 and t % 300 == 0:
                snap_path = snap_dir / f"state_{t}.xml"
                traci.simulation.saveState(str(snap_path))
                snapshots_taken.append({
                    "path": str(snap_path.resolve()),
                    "scenario": scenario_name,
                    "seed": seed,
                    "sim_time_s": t,
                    "file_size_bytes": snap_path.stat().st_size if snap_path.exists() else 0,
                })

            # Logging at every 60-second interval
            if t % 60 == 0:
                is_warmup = (t <= 300)
                is_inc_active = is_incident_scenario and (incident_start <= t < incident_end)
                split_name = get_split(seed)

                for tid in CLUSTER_TLS_IDS:
                    meta = TLS_META[tid]
                    lanes = TLS_INCOMING_LANES[tid]

                    halting_count = sum(traci.lane.getLastStepHaltingNumber(lid) for lid in lanes)
                    veh_count = sum(traci.lane.getLastStepVehicleNumber(lid) for lid in lanes)

                    # Mean speed across incoming vehicles
                    speeds = []
                    for lid in lanes:
                        vids = traci.lane.getLastStepVehicleIDs(lid)
                        speeds.extend(traci.vehicle.getSpeed(v) for v in vids)
                    mean_speed = (sum(speeds) / len(speeds)) if speeds else 0.0

                    # Waiting time
                    waiting_time = 0.0
                    for lid in lanes:
                        vids = traci.lane.getLastStepVehicleIDs(lid)
                        waiting_time += sum(traci.vehicle.getWaitingTime(v) for v in vids)

                    cur_phase = traci.trafficlight.getPhase(tid)
                    tp = throughput_window[tid]
                    arr = arrivals_window[tid]

                    log_rows.append({
                        "run_id": run_id,
                        "scenario": scenario_name,
                        "seed": seed,
                        "mode": mode,
                        "split": split_name,
                        "timestamp_s": t,
                        "is_warmup": is_warmup,
                        "junction_id": tid,
                        "junction_short": meta["short_id"],
                        "vehicle_count": veh_count,
                        "queue_length": halting_count,
                        "mean_speed_mps": round(mean_speed, 2),
                        "waiting_time_s": round(waiting_time, 1),
                        "current_plan_id": current_plan_ids[tid],
                        "current_phase": cur_phase,
                        "throughput": tp,
                        "arrivals": arr,
                        "is_incident_active": is_inc_active,
                    })

                    # Reset interval counters
                    throughput_window[tid] = 0
                    arrivals_window[tid] = 0

        traci.close()
    except Exception as e:
        try:
            traci.close()
        except Exception:
            pass
        return run_id, "FAILED_RUN", len(log_rows), str(e)

    # Save CSV
    df = pd.DataFrame(log_rows)
    df.to_csv(out_csv, index=False)

    return run_id, "COMPLETED", len(df), snapshots_taken


# ---------------------------------------------------------------------------
# Runner Entry Point with Multiprocessing
# ---------------------------------------------------------------------------
def run_all_scenarios(smoke: bool = False, max_workers: int = None):
    """
    Run full matrix (5 scenarios x 5 seeds x 2 modes = 50 runs).
    If smoke=True: runs 1 scenario (medium), 1 seed (0), 600s, 2 modes = 2 runs.
    """
    if smoke:
        scenarios = ["medium"]
        seeds = [0]
        duration_s = 600
        print("[mode] SMOKE TEST MODE (1 scenario, seed 0, 600s)")
    else:
        scenarios = ["low", "medium", "high", "peak", "incident"]
        seeds = [0, 1, 2, 3, 4]
        duration_s = 3600
        print("[mode] FULL DATASET GENERATION (5 scenarios x 5 seeds x 2 modes = 50 runs)")

    modes = ["fixed", "varied"]
    tasks = []
    for sc in scenarios:
        for sd in seeds:
            for md in modes:
                tasks.append((sc, sd, md, duration_s, smoke))

    print(f"[plan] Total planned runs: {len(tasks)}")

    if max_workers is None:
        cpu_cnt = os.cpu_count() or 4
        max_workers = max(1, min(10, cpu_cnt - 2))

    print(f"[pool] Running with {max_workers} parallel workers ...")

    all_snapshots = []
    completed_runs = []
    failed_runs = []

    # Manifest records
    manifest_records = []

    # Use multiprocessing pool
    with mp.Pool(processes=max_workers) as pool:
        for idx, result in enumerate(pool.imap_unordered(run_simulation, tasks), 1):
            run_id, status, rows, extra = result
            print(f"[{idx}/{len(tasks)}] {run_id:<25} -> {status} ({rows} rows)")

            # Parse metadata
            sc, rest = run_id.split("_s")
            sd, md = rest.split("_")
            sd = int(sd)
            sc_cfg = SCENARIO_CONFIGS["scenarios"][sc]

            record = {
                "run_id": run_id,
                "scenario": sc,
                "seed": sd,
                "mode": md,
                "split": get_split(sd),
                "status": status,
                "total_rows": rows,
                "blocked_edge": sc_cfg.get("blocked_edge", "none"),
                "blocked_lane": sc_cfg.get("blocked_lane", "none"),
                "incident_start_s": sc_cfg.get("incident_start_s", 0) if sc_cfg.get("incident") else None,
                "incident_end_s": sc_cfg.get("incident_end_s", 0) if sc_cfg.get("incident") else None,
                "duration_s": duration_s,
            }
            manifest_records.append(record)

            if status in ("COMPLETED", "SKIPPED_EXISTING"):
                completed_runs.append(run_id)
                if isinstance(extra, list):
                    all_snapshots.extend(extra)
            else:
                failed_runs.append((run_id, extra))

    # Write Manifest
    manifest_df = pd.DataFrame(manifest_records)
    manifest_path = DATA_RAW_DIR / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"[manifest] Saved manifest to {manifest_path}")

    # Write Snapshot Index
    if all_snapshots:
        snap_df = pd.DataFrame(all_snapshots)
        snap_index_path = STATES_DIR / "snapshot_index.csv"
        # If exists, append/update
        if snap_index_path.exists():
            existing_snap = pd.read_csv(snap_index_path)
            combined = pd.concat([existing_snap, snap_df]).drop_duplicates(subset=["path"])
            combined.to_csv(snap_index_path, index=False)
        else:
            snap_df.to_csv(snap_index_path, index=False)
        print(f"[snapshots] Indexed {len(all_snapshots)} snapshots to {snap_index_path}")

    print("\n" + "=" * 65)
    print(f"RUN SUMMARY: Planned: {len(tasks)} | Completed: {len(completed_runs)} | Failed: {len(failed_runs)}")
    if failed_runs:
        print("FAILED RUNS:")
        for fid, err in failed_runs:
            print(f"  {fid}: {err}")
    print("=" * 65)

    return completed_runs, failed_runs


def main(smoke: bool = False):
    parser = argparse.ArgumentParser(description="Phase 2: Scenarios & Data Generation")
    parser.add_argument("--smoke", action="store_true", help="Quick sanity run (2 runs, 600s)")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    args, _ = parser.parse_known_args()

    is_smoke = smoke or args.smoke
    completed, failed = run_all_scenarios(smoke=is_smoke, max_workers=args.workers)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
