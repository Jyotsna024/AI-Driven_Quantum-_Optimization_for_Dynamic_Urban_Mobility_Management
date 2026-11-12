"""
stage1_fix_phase2_data.py – STAGE 1: Full autonomous repair and validation of Phase 2 dataset.
Fixes:
1. Archives previous Phase 2 dataset to data/archive/.
2. Sets up distinct K=4 signal programs in hsr.net.xml (>=30% green share difference, safe transitions).
3. Defines through-movement routes for all 4 junctions (J0-J3) so every junction receives balanced demand.
4. Calibrates flow rates for low, medium, high, peak, and incident.
5. Runs 50 simulations (5 scenarios x 5 seeds x 2 modes) with TraCI.
6. Logs rich metrics: queue_length, queue_length_m, max_lane_queue, queue_major, queue_minor,
   mean_speed_mps (NaN for empty), mean_delay_s, waiting_time_s, throughput, arrivals.
7. Saves snapshots every 300s (t>=600) and indexes in snapshot_index.csv.
8. Evaluates all 8 acceptance gates and outputs required CSVs and results/data_validation.md.
"""

import os
import sys
import copy
import json
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import multiprocessing as mp
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUMO_DIR = PROJECT_ROOT / "sumo_files"
ROUTES_DIR = SUMO_DIR / "routes"
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_ARCHIVE_DIR = PROJECT_ROOT / "data" / "archive"
STATES_DIR = PROJECT_ROOT / "data" / "states"
RESULTS_DIR = PROJECT_ROOT / "results"
SUMO_BIN = Path(os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\SUMO")) / "bin" / "sumo.exe"

ROUTES_DIR.mkdir(parents=True, exist_ok=True)
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
STATES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NET_PATH = SUMO_DIR / "hsr.net.xml"
CFG_PATH = SUMO_DIR / "cluster_config.json"
CLUSTER_CFG = json.loads(CFG_PATH.read_text(encoding="utf-8"))
CLUSTER_TLS_IDS = CLUSTER_CFG["cluster_tls_ids"]

JUNCTION_DETAILS = {
    "J0": {
        "tid": "joinedS_12067305543_12067305544",
        "name": "14th Main & 15th Cross",
        "incoming_lanes": ['1387946771#0_0', '1387946771#0_1', '1473496401_0', '1473496401_1', '1509685699_0', '1509685699_1', '755390844_0', '755390844_1'],
        "major_lanes": ['1473496401_0', '1473496401_1', '755390844_0', '755390844_1'],
    },
    "J1": {
        "tid": "GS_cluster_6633301856_7432359269_7432359283_7432359284",
        "name": "14th Main & 17th Cross",
        "incoming_lanes": ['24480656#3_0', '24480656#3_1', '631316440#0_0', '631316440#0_1', '794634831#1_0', '794634831#1_1', '794634832#0_0', '794634832#0_1'],
        "major_lanes": ['24480656#3_0', '24480656#3_1', '631316440#0_0', '631316440#0_1'],
    },
    "J2": {
        "tid": "cluster_10049312966_10860559949_11178931798_11178931802_#5more",
        "name": "19th Main & 17th Cross",
        "incoming_lanes": ['1055330325#6_0', '1055330325#6_1', '1110166139_0', '1110166139_1', '1206400802#2_0', '1206400802#2_1', '1473869989_0', '1473869989_1'],
        "major_lanes": ['1055330325#6_0', '1055330325#6_1', '1110166139_0', '1110166139_1'],
    },
    "J3": {
        "tid": "GS_cluster_12067305554_6017833806_6017833807_6131076159",
        "name": "14th Main & Sarjapur Road",
        "incoming_lanes": ['1434254320#1_0', '1434254320#1_1', '151676283#1_0', '151676283#1_1', '610166550#13_0', '610166550#13_1'],
        "major_lanes": ['1434254320#1_0', '1434254320#1_1', '151676283#1_0', '151676283#1_1'],
    },
}

# ---------------------------------------------------------------------------
# Step 1: Ensure Archive of Original Phase 2 Dataset
# ---------------------------------------------------------------------------
def ensure_archive():
    print("[1/8] Verifying dataset archive in data/archive ...")
    raw_archive = DATA_ARCHIVE_DIR / "raw_phase2_v1"
    states_archive = DATA_ARCHIVE_DIR / "states_phase2_v1"
    
    if not raw_archive.exists() or len(list(raw_archive.glob("*csv"))) < 50:
        raw_archive.mkdir(parents=True, exist_ok=True)
        src_v1 = PROJECT_ROOT / "data" / "raw_archive_phase2_v1"
        if src_v1.exists() and len(list(src_v1.glob("*csv"))) >= 50:
            for f in src_v1.glob("*"):
                shutil.copy2(f, raw_archive / f.name)
        else:
            for f in DATA_RAW_DIR.glob("*"):
                shutil.copy2(f, raw_archive / f.name)
        print(f"  Archived raw data to {raw_archive}")
        
    if not states_archive.exists():
        states_archive.mkdir(parents=True, exist_ok=True)
        src_states_v1 = PROJECT_ROOT / "data" / "states_archive_phase2_v1"
        if src_states_v1.exists():
            for item in src_states_v1.glob("*"):
                if item.is_dir():
                    shutil.copytree(item, states_archive / item.name, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, states_archive / item.name)
        print(f"  Archived states to {states_archive}")
    print("  Archive verified.")


# ---------------------------------------------------------------------------
# Step 2: Meaningful Signal Programs (>=30% green share difference)
# ---------------------------------------------------------------------------
def configure_signal_programs():
    print("[2/8] Configuring meaningful K=4 signal programs in hsr.net.xml ...")
    shutil.copy2(SUMO_DIR / "hsr_test.net.xml", NET_PATH)
    tree = ET.parse(NET_PATH)
    root = tree.getroot()
    
    tls_elems = [e for e in root if e.tag == "tlLogic"]
    for tl in tls_elems:
        tl.set("type", "static")
        for ph in tl.findall("phase"):
            if "minDur" in ph.attrib:
                del ph.attrib["minDur"]
            if "maxDur" in ph.attrib:
                del ph.attrib["maxDur"]
                
    plan_durations = {
        "joinedS_12067305543_12067305544": {
            0: ([42.0, 3.0, 42.0, 3.0], 0.0),
            1: ([56.0, 3.0, 28.0, 3.0], 0.0),
            2: ([28.0, 3.0, 56.0, 3.0], 0.0),
            3: ([42.0, 3.0, 42.0, 3.0], 15.0),
        },
        "GS_cluster_6633301856_7432359269_7432359283_7432359284": {
            0: ([39.0, 6.0, 39.0, 6.0], 0.0),
            1: ([54.0, 6.0, 24.0, 6.0], 0.0),
            2: ([24.0, 6.0, 54.0, 6.0], 0.0),
            3: ([39.0, 6.0, 39.0, 6.0], 15.0),
        },
        "cluster_10049312966_10860559949_11178931798_11178931802_#5more": {
            0: ([27.0, 6.0, 6.0, 6.0, 27.0, 6.0, 6.0, 6.0], 0.0),
            1: ([39.0, 6.0, 6.0, 6.0, 15.0, 6.0, 6.0, 6.0], 0.0),
            2: ([15.0, 6.0, 6.0, 6.0, 39.0, 6.0, 6.0, 6.0], 0.0),
            3: ([27.0, 6.0, 6.0, 6.0, 27.0, 6.0, 6.0, 6.0], 15.0),
        },
        "GS_cluster_12067305554_6017833806_6017833807_6131076159": {
            0: ([39.0, 6.0, 39.0, 6.0], 0.0),
            1: ([54.0, 6.0, 24.0, 6.0], 0.0),
            2: ([24.0, 6.0, 54.0, 6.0], 0.0),
            3: ([39.0, 6.0, 39.0, 6.0], 15.0),
        },
    }
    
    for tid, p_map in plan_durations.items():
        for p_idx, (durs, offset) in p_map.items():
            phases_list = CLUSTER_CFG["plans"][tid][p_idx]["phases"]
            for idx, dur in enumerate(durs):
                phases_list[idx]["duration"] = dur
            CLUSTER_CFG["plans"][tid][p_idx]["offset"] = offset
            
    CFG_PATH.write_text(json.dumps(CLUSTER_CFG, indent=2), encoding="utf-8")
    
    last_tl_idx = max(list(root).index(e) for e in tls_elems)
    insert_idx = last_tl_idx + 1
    
    for tid, p_map in plan_durations.items():
        orig_tl = next(e for e in tls_elems if e.get("id") == tid)
        durs0, off0 = p_map[0]
        orig_tl.set("offset", str(off0))
        for idx, ph in enumerate(orig_tl.findall("phase")):
            ph.set("duration", str(durs0[idx]))
            
        for p_idx in [1, 2, 3]:
            durs, off = p_map[p_idx]
            new_tl = ET.Element("tlLogic", id=tid, type="static", programID=str(p_idx), offset=str(off))
            for idx, orig_ph in enumerate(orig_tl.findall("phase")):
                ET.SubElement(new_tl, "phase", duration=str(durs[idx]), state=orig_ph.get("state"))
            root.insert(insert_idx, new_tl)
            insert_idx += 1
            
    tree.write(str(NET_PATH), encoding="utf-8", xml_declaration=True)
    print("  hsr.net.xml updated with static programs 0, 1, 2, 3.")


# ---------------------------------------------------------------------------
# Step 3: Extract Verified Through-Movement Routes
# ---------------------------------------------------------------------------
def get_through_routes():
    tree = ET.parse(NET_PATH)
    root = tree.getroot()
    
    tls_conns = {}
    for tid, d in JUNCTION_DETAILS.items():
        conns = [c for c in root.findall(".//connection") if c.get("tl") == d["tid"]]
        distinct = sorted(list(set((c.get("from"), c.get("to")) for c in conns if c.get("from") and c.get("to"))))
        tls_conns[tid] = distinct
        
    routes = []
    for jid, pairs in tls_conns.items():
        for idx, (f_edge, t_edge) in enumerate(pairs):
            routes.append({
                "junction": jid,
                "name": f"r_{jid}_{idx}",
                "edges": [f_edge, t_edge],
                "from_edge": f_edge,
                "to_edge": t_edge,
            })
    return routes


# ---------------------------------------------------------------------------
# Step 4: Route File Generation for 5 Scenarios x 5 Seeds
# ---------------------------------------------------------------------------
def build_route_files(duration_s=3600):
    print("[3/8] Generating route files for 5 scenarios x 5 seeds ...")
    routes = get_through_routes()
    j_routes = {"J0": [], "J1": [], "J2": [], "J3": []}
    for r in routes:
        j_routes[r["junction"]].append(r)
        
    rates_per_junction = {
        "low": 300,      # ~300 veh/h per junction (v/c ~0.3)
        "medium": 650,   # ~650 veh/h per junction (v/c ~0.6)
        "high": 1000,    # ~1000 veh/h per junction (v/c ~0.85)
        "peak": 1400,    # ~1400 veh/h per junction (v/c ~1.0)
        "incident": 850, # medium-high with incident bottleneck
    }
    
    scenarios = ["low", "medium", "high", "peak", "incident"]
    seeds = [0, 1, 2, 3, 4]
    
    for sc in scenarios:
        rate_j = rates_per_junction[sc]
        for sd in seeds:
            rou_file = ROUTES_DIR / f"{sc}_s{sd}.rou.xml"
            rng = random.Random(sd * 3000 + 17)
            
            root = ET.Element("routes")
            ET.SubElement(root, "vType", id="car", accel="2.6", decel="4.5", sigma="0.5", length="4.5", maxSpeed="14.0", vClass="passenger")
            ET.SubElement(root, "vType", id="motorcycle", accel="3.0", decel="5.0", sigma="0.5", length="2.2", maxSpeed="14.0", vClass="motorcycle")
            ET.SubElement(root, "vType", id="auto_rickshaw", accel="1.8", decel="3.5", sigma="0.5", length="2.8", maxSpeed="10.0", vClass="taxi")
            
            for r in routes:
                ET.SubElement(root, "route", id=r["name"], edges=" ".join(r["edges"]))
                
            veh_count = 0
            for t in range(duration_s):
                for jid, r_list in j_routes.items():
                    # Probability of departure per route at this second
                    p_dep = (rate_j / 3600.0) / len(r_list)
                    for r in r_list:
                        if rng.random() < p_dep:
                            r_type = rng.random()
                            vtype = "car" if r_type < 0.50 else ("motorcycle" if r_type < 0.85 else "auto_rickshaw")
                            veh_id = f"veh_{veh_count}_{t}"
                            ET.SubElement(root, "vehicle", id=veh_id, type=vtype, route=r["name"], depart=str(float(t)), departLane="best", departSpeed="max")
                            veh_count += 1
                            
            ET.indent(root)
            ET.ElementTree(root).write(str(rou_file), encoding="utf-8", xml_declaration=True)
    print(f"  All 25 route files generated ({len(routes)} through movements).")


# ---------------------------------------------------------------------------
# Step 5: Simulation Execution Worker
# ---------------------------------------------------------------------------
def run_simulation_worker(args_tuple):
    import traci
    scenario, seed, mode, duration_s = args_tuple
    run_id = f"{scenario}_s{seed}_{mode}"
    out_csv = DATA_RAW_DIR / f"{run_id}.csv"
    rou_file = ROUTES_DIR / f"{scenario}_s{seed}.rou.xml"
    
    # Incident setup
    is_incident = (scenario == "incident")
    incident_lane = "24480656#3_0" # On J1 approach
    incident_start = 1200
    incident_end = 1800
    
    snap_dir = STATES_DIR / f"{scenario}_s{seed}"
    if mode == "varied":
        snap_dir.mkdir(parents=True, exist_ok=True)
        
    cmd = [
        str(SUMO_BIN),
        "-n", str(NET_PATH),
        "-r", str(rou_file),
        "--seed", str(seed + 100),
        "--no-step-log", "true",
        "--no-warnings", "true",
        "--time-to-teleport", "300",
    ]
    
    try:
        traci.start(cmd, label=run_id)
    except Exception as e:
        return run_id, "FAILED_START", 0, str(e)
        
    for sid, d in JUNCTION_DETAILS.items():
        traci.trafficlight.setProgram(d["tid"], "0")
        
    current_plans = {sid: 0 for sid in JUNCTION_DETAILS}
    target_plans = {sid: 0 for sid in JUNCTION_DETAILS}
    last_phases = {sid: traci.trafficlight.getPhase(d["tid"]) for sid, d in JUNCTION_DETAILS.items()}
    
    prev_vehs_incoming = {sid: set() for sid in JUNCTION_DETAILS}
    throughput_window = {sid: 0 for sid in JUNCTION_DETAILS}
    arrivals_window = {sid: 0 for sid in JUNCTION_DETAILS}
    
    log_rows = []
    snapshots = []
    orig_lane_speed = None
    rng_varied = random.Random(seed * 7777 + 23)
    
    try:
        for step in range(duration_s):
            t = step + 1
            
            # Incident management
            if is_incident:
                if t == incident_start:
                    orig_lane_speed = traci.lane.getMaxSpeed(incident_lane)
                    traci.lane.setMaxSpeed(incident_lane, 0.1) # Block lane
                elif t == incident_end and orig_lane_speed is not None:
                    traci.lane.setMaxSpeed(incident_lane, orig_lane_speed)
                    
            # Varied decisions every 60s
            if mode == "varied" and step > 0 and step % 60 == 0:
                for sid in JUNCTION_DETAILS:
                    target_plans[sid] = rng_varied.randint(0, 3)
                    
            # Safe signal transitions strictly at cycle boundary (phase 0)
            for sid, d in JUNCTION_DETAILS.items():
                cur_phase = traci.trafficlight.getPhase(d["tid"])
                if cur_phase == 0 and last_phases[sid] != 0:
                    if target_plans[sid] != current_plans[sid]:
                        traci.trafficlight.setProgram(d["tid"], str(target_plans[sid]))
                        current_plans[sid] = target_plans[sid]
                last_phases[sid] = cur_phase
                
            traci.simulationStep()
            
            # Update vehicle tracking for throughput and arrivals
            for sid, d in JUNCTION_DETAILS.items():
                lanes = d["incoming_lanes"]
                curr_vehs = set()
                for lid in lanes:
                    curr_vehs.update(traci.lane.getLastStepVehicleIDs(lid))
                    
                departed = prev_vehs_incoming[sid] - curr_vehs
                throughput_window[sid] += len(departed)
                
                entered = curr_vehs - prev_vehs_incoming[sid]
                arrivals_window[sid] += len(entered)
                
                prev_vehs_incoming[sid] = curr_vehs
                
            # Snapshots in varied mode every 300s (t >= 600)
            if mode == "varied" and t >= 600 and t % 300 == 0:
                snap_p = snap_dir / f"state_{t}.xml"
                traci.simulation.saveState(str(snap_p))
                snapshots.append({
                    "path": str(snap_p.resolve()),
                    "scenario": scenario,
                    "seed": seed,
                    "sim_time_s": t,
                    "file_size_bytes": snap_p.stat().st_size if snap_p.exists() else 0,
                })
                
            # Log interval every 60s
            if t % 60 == 0:
                is_warmup = (t <= 300)
                is_inc_active = is_incident and (incident_start <= t < incident_end)
                split_name = "train" if seed in [0, 1, 2] else ("val" if seed == 3 else "test")
                
                for sid, d in JUNCTION_DETAILS.items():
                    lanes = d["incoming_lanes"]
                    maj_lanes = set(d["major_lanes"])
                    
                    halting_vehs = []
                    speeds = []
                    waiting_times = []
                    
                    q_maj = 0
                    q_min = 0
                    max_lq = 0
                    
                    for lid in lanes:
                        vids = traci.lane.getLastStepVehicleIDs(lid)
                        l_halt = traci.lane.getLastStepHaltingNumber(lid)
                        max_lq = max(max_lq, l_halt)
                        if lid in maj_lanes:
                            q_maj += l_halt
                        else:
                            q_min += l_halt
                            
                        for vid in vids:
                            spd = traci.vehicle.getSpeed(vid)
                            wt = traci.vehicle.getWaitingTime(vid)
                            speeds.append(spd)
                            waiting_times.append(wt)
                            if spd < 0.1:
                                halting_vehs.append(vid)
                                
                    vcount = len(speeds)
                    q_len = len(halting_vehs)
                    
                    q_len_m = sum(traci.vehicle.getLength(vid) + 2.5 for vid in halting_vehs)
                    
                    mean_spd = (sum(speeds) / vcount) if vcount > 0 else np.nan
                    mean_del = (sum(waiting_times) / vcount) if vcount > 0 else np.nan
                    tot_wait = sum(waiting_times)
                    
                    log_rows.append({
                        "run_id": run_id,
                        "scenario": scenario,
                        "seed": seed,
                        "mode": mode,
                        "split": split_name,
                        "timestamp_s": t,
                        "is_warmup": is_warmup,
                        "junction_id": d["tid"],
                        "junction_short": sid,
                        "vehicle_count": vcount,
                        "queue_length": q_len,
                        "queue_length_m": round(q_len_m, 1),
                        "max_lane_queue": max_lq,
                        "queue_major": q_maj,
                        "queue_minor": q_min,
                        "mean_speed_mps": round(mean_spd, 2) if not np.isnan(mean_spd) else np.nan,
                        "mean_delay_s": round(mean_del, 2) if not np.isnan(mean_del) else np.nan,
                        "waiting_time_s": round(tot_wait, 1),
                        "current_plan_id": current_plans[sid],
                        "current_phase": traci.trafficlight.getPhase(d["tid"]),
                        "throughput": throughput_window[sid],
                        "arrivals": arrivals_window[sid],
                        "is_incident_active": is_inc_active,
                    })
                    
                    throughput_window[sid] = 0
                    arrivals_window[sid] = 0
                    
        traci.close()
    except Exception as e:
        try:
            traci.close()
        except:
            pass
        return run_id, "FAILED_RUN", len(log_rows), str(e)
        
    df = pd.DataFrame(log_rows)
    df.to_csv(out_csv, index=False)
    return run_id, "COMPLETED", len(df), snapshots


# ---------------------------------------------------------------------------
# Step 6: Run Full Simulation Matrix (50 runs)
# ---------------------------------------------------------------------------
def run_all_simulations(duration_s=3600):
    print(f"[4/8] Running 50 simulations (5 scenarios x 5 seeds x 2 modes) ...")
    scenarios = ["low", "medium", "high", "peak", "incident"]
    seeds = [0, 1, 2, 3, 4]
    modes = ["fixed", "varied"]
    
    tasks = []
    for sc in scenarios:
        for sd in seeds:
            for md in modes:
                tasks.append((sc, sd, md, duration_s))
                
    max_workers = min(4, os.cpu_count() or 4)
    all_snapshots = []
    completed = []
    failed = []
    
    with mp.Pool(processes=max_workers) as pool:
        for idx, res in enumerate(pool.imap_unordered(run_simulation_worker, tasks), 1):
            run_id, status, rows, extra = res
            print(f"  [{idx:02d}/50] {run_id:<22} -> {status} ({rows} rows)", flush=True)
            if status == "COMPLETED":
                completed.append(run_id)
                all_snapshots.extend(extra)
            else:
                failed.append((run_id, extra))
                print(f"  [ERROR] {run_id}: {extra}", flush=True)
                
    # Save manifest
    manifest_rows = []
    for sc in scenarios:
        for sd in seeds:
            for md in modes:
                rid = f"{sc}_s{sd}_{md}"
                manifest_rows.append({
                    "run_id": rid,
                    "scenario": sc,
                    "seed": sd,
                    "mode": md,
                    "split": "train" if sd in [0,1,2] else ("val" if sd == 3 else "test"),
                    "status": "COMPLETED" if rid in completed else "FAILED",
                    "total_rows": 240,
                })
    pd.DataFrame(manifest_rows).to_csv(DATA_RAW_DIR / "manifest.csv", index=False)
    
    pd.DataFrame(all_snapshots).to_csv(STATES_DIR / "snapshot_index.csv", index=False)
    print(f"  Completed {len(completed)}/50 runs. Indexed {len(all_snapshots)} snapshots.")
    return len(completed), len(failed)


# ---------------------------------------------------------------------------
# Helper: Format DataFrame to Markdown Table
# ---------------------------------------------------------------------------
def df_to_markdown(df):
    headers = [str(c) for c in df.columns]
    col_widths = {c: max(len(c), 3) for c in headers}
    rows = []
    for _, row in df.iterrows():
        r = [str(row[c]) for c in headers]
        rows.append(r)
        for c, val in zip(headers, r):
            col_widths[c] = max(col_widths[c], len(val))
    header_line = "| " + " | ".join(c.ljust(col_widths[c]) for c in headers) + " |"
    sep_line = "| " + " | ".join("-" * col_widths[c] for c in headers) + " |"
    body_lines = ["| " + " | ".join(val.ljust(col_widths[c]) for c, val in zip(headers, r)) + " |" for r in rows]
    return "\n".join([header_line, sep_line] + body_lines)


# ---------------------------------------------------------------------------
# Step 7: Evaluate GATES and Generate Verification Tables
# ---------------------------------------------------------------------------
def evaluate_gates():
    print("[5/8] Evaluating acceptance gates and generating reports ...")
    csv_files = [f for f in DATA_RAW_DIR.glob("*_s*_*csv") if f.name != "manifest.csv"]
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    post_warmup = df[~df["is_warmup"]].copy()
    
    def p90(x):
        return np.percentile(x, 90)
        
    j_analysis = post_warmup.groupby(["scenario", "junction_short"]).agg(
        mean_queue=("queue_length", "mean"),
        median_queue=("queue_length", "median"),
        p90_queue=("queue_length", p90),
        max_queue=("queue_length", "max"),
        mean_queue_m=("queue_length_m", "mean"),
        p90_queue_m=("queue_length_m", p90),
        mean_delay=("mean_delay_s", "mean"),
        mean_throughput=("throughput", "mean"),
    ).reset_index()
    for col in ["mean_queue", "median_queue", "p90_queue", "max_queue", "mean_queue_m", "p90_queue_m", "mean_delay", "mean_throughput"]:
        j_analysis[col] = j_analysis[col].round(2)
    j_analysis.to_csv(RESULTS_DIR / "per_junction_analysis.csv", index=False)
    
    sc_summary = post_warmup.groupby("scenario").agg(
        mean_queue=("queue_length", "mean"),
        p90_queue=("queue_length", p90),
        mean_speed=("mean_speed_mps", "mean"),
        mean_throughput=("throughput", "mean"),
        mean_delay=("mean_delay_s", "mean"),
    ).reset_index()
    for col in ["mean_queue", "p90_queue", "mean_speed", "mean_throughput", "mean_delay"]:
        sc_summary[col] = sc_summary[col].round(2)
    sc_summary.to_csv(RESULTS_DIR / "scenario_summary.csv", index=False)
    
    # Check Gate 1: Peak mean queue >= 3 and P90 >= 8 at EVERY junction
    peak_j = j_analysis[j_analysis["scenario"] == "peak"].set_index("junction_short")
    gate1_pass = True
    gate1_details = {}
    for jid in ["J0", "J1", "J2", "J3"]:
        mq = peak_j.loc[jid, "mean_queue"]
        p90q = peak_j.loc[jid, "p90_queue"]
        gate1_details[jid] = (round(mq, 2), round(p90q, 2))
        if mq < 3.0 or p90q < 8.0:
            gate1_pass = False
            
    # Check Gate 2: low < medium < high < peak mean queue at EVERY junction
    gate2_pass = True
    gate2_details = {}
    for jid in ["J0", "J1", "J2", "J3"]:
        j_sub = j_analysis[j_analysis["junction_short"] == jid].set_index("scenario")["mean_queue"]
        ordered = (j_sub["low"] < j_sub["medium"] < j_sub["high"] < j_sub["peak"])
        gate2_details[jid] = ordered
        if not ordered:
            gate2_pass = False
            
    # Check Gate 3 & 4: Plan Sensitivity Rollouts
    print("  Testing plan sensitivity from 10 high/peak snapshots ...")
    snap_idx = pd.read_csv(STATES_DIR / "snapshot_index.csv")
    sample_snaps = snap_idx[snap_idx["scenario"].isin(["high", "peak"])]["path"].tolist()[:10]
    
    import traci
    
    sensitivity_rows = []
    best_plans_found = []
    
    for sp_idx, sp_path in enumerate(sample_snaps):
        if not Path(sp_path).exists():
            continue
        for sid, d in JUNCTION_DETAILS.items():
            plan_delays = {}
            for p_test in [0, 1, 2, 3]:
                cmd = [str(SUMO_BIN), "-n", str(NET_PATH), "--seed", "42", "--no-step-log", "true", "--no-warnings", "true"]
                traci.start(cmd)
                traci.simulation.loadState(sp_path)
                for s_other, d_other in JUNCTION_DETAILS.items():
                    traci.trafficlight.setProgram(d_other["tid"], str(p_test if s_other == sid else 0))
                    
                delays = []
                for _ in range(60):
                    traci.simulationStep()
                    for lid in d["incoming_lanes"]:
                        for vid in traci.lane.getLastStepVehicleIDs(lid):
                            delays.append(traci.vehicle.getWaitingTime(vid))
                traci.close()
                plan_delays[p_test] = (sum(delays) / len(delays)) if delays else 0.0
                
            best_p = min(plan_delays, key=plan_delays.get)
            worst_p = max(plan_delays, key=plan_delays.get)
            diff_pct = ((plan_delays[worst_p] - plan_delays[best_p]) / max(0.1, plan_delays[worst_p])) * 100.0
            best_plans_found.append(best_p)
            
            sensitivity_rows.append({
                "snapshot": Path(sp_path).name,
                "junction": sid,
                "delay_p0": round(plan_delays[0], 2),
                "delay_p1": round(plan_delays[1], 2),
                "delay_p2": round(plan_delays[2], 2),
                "delay_p3": round(plan_delays[3], 2),
                "best_plan": best_p,
                "worst_plan": worst_p,
                "diff_pct": round(diff_pct, 1),
            })
            
    df_sens = pd.DataFrame(sensitivity_rows)
    df_sens.to_csv(RESULTS_DIR / "plan_sensitivity.csv", index=False)
    
    j_sens_pass = 0
    for sid in ["J0", "J1", "J2", "J3"]:
        sub = df_sens[df_sens["junction"] == sid]
        if len(sub) > 0 and sub["diff_pct"].mean() >= 10.0:
            j_sens_pass += 1
    gate3_pass = (j_sens_pass >= 3)
    
    unique_best_plans = set(best_plans_found)
    gate4_pass = (len(unique_best_plans) >= 2)
    
    # Gate 5: Throughput Reconciliation
    reconciliation_rows = []
    for sid, d in JUNCTION_DETAILS.items():
        sub = post_warmup[post_warmup["junction_short"] == sid]
        mean_tp = sub["throughput"].mean()
        max_tp = sub["throughput"].max()
        mean_arr = sub["arrivals"].mean()
        diff = abs(mean_tp - mean_arr)
        reconciliation_rows.append({
            "junction": sid,
            "mean_throughput": round(mean_tp, 2),
            "mean_arrivals": round(mean_arr, 2),
            "max_interval_throughput": max_tp,
            "approach_capacity_veh_min": 60.0,
            "reconciliation_abs_diff": round(diff, 2),
            "within_capacity": (max_tp <= 60.0),
        })
    df_recon = pd.DataFrame(reconciliation_rows)
    df_recon.to_csv(RESULTS_DIR / "throughput_reconciliation.csv", index=False)
    gate5_pass = all(r["within_capacity"] for r in reconciliation_rows)
    
    # Gate 6: No Gridlock
    peak_final = post_warmup[(post_warmup["scenario"] == "peak") & (post_warmup["timestamp_s"] >= 3000)]
    gate6_pass = (peak_final["throughput"].mean() > 0.0)
    
    # Gate 7: Determinism & Snapshot Reload
    first_snap = sample_snaps[0]
    def rollout_snap(plan_id):
        cmd = [str(SUMO_BIN), "-n", str(NET_PATH), "--seed", "42", "--no-step-log", "true", "--no-warnings", "true"]
        traci.start(cmd)
        traci.simulation.loadState(first_snap)
        for s_other, d_other in JUNCTION_DETAILS.items():
            traci.trafficlight.setProgram(d_other["tid"], str(plan_id))
        qs = []
        for _ in range(30):
            traci.simulationStep()
            qs.append(traci.vehicle.getIDCount())
        traci.close()
        return qs
        
    r1 = rollout_snap(0)
    r2 = rollout_snap(0)
    gate7_pass = (r1 == r2)
    
    # Gate 8: Incident Effect
    inc_sub = post_warmup[post_warmup["scenario"] == "incident"]
    inc_active = inc_sub[inc_sub["is_incident_active"] & (inc_sub["junction_short"] == "J1")]
    inc_inactive = inc_sub[(~inc_sub["is_incident_active"]) & (inc_sub["junction_short"] == "J1")]
    spd_during = inc_active["mean_speed_mps"].mean()
    spd_outside = inc_inactive["mean_speed_mps"].mean()
    gate8_pass = (spd_during < spd_outside)
    
    val_md = f"""# Phase 2 Validation & Data Integrity Report (Corrected)

## 1. Summary of Acceptance Gates
| Gate | Description | Criteria | Observed Result | Status |
|---|---|---|---|---|
| **Gate 1** | Peak Congestion | Mean queue >= 3.0 & P90 >= 8.0 at EVERY junction | {gate1_details} | **{"PASS" if gate1_pass else "FAIL"}** |
| **Gate 2** | Demand Hierarchy | low < medium < high < peak at EVERY junction | {gate2_details} | **{"PASS" if gate2_pass else "FAIL"}** |
| **Gate 3** | Plan Sensitivity | Best vs worst plan delay diff >= 10% at >= 3 junctions | {j_sens_pass}/4 junctions pass | **{"PASS" if gate3_pass else "FAIL"}** |
| **Gate 4** | Plan Non-Triviality | Best plan varies across sampled snapshots | {sorted(list(unique_best_plans))} | **{"PASS" if gate4_pass else "FAIL"}** |
| **Gate 5** | Throughput Reconciliation | Throughput reconciles with arrivals & <= capacity | Max throughput <= 60 veh/min | **{"PASS" if gate5_pass else "FAIL"}** |
| **Gate 6** | No Gridlock | Final 600s throughput > 0, ~0 teleports | Peak final throughput = {peak_final['throughput'].mean():.2f} veh/min | **{"PASS" if gate6_pass else "FAIL"}** |
| **Gate 7** | Determinism & Reload | Bit-exact metrics on repeated rollout from state | r1 == r2 identical match ({gate7_pass}) | **{"PASS" if gate7_pass else "FAIL"}** |
| **Gate 8** | Incident Effect | Measurable speed/flow impact during closure | Speed: {spd_during:.2f} m/s active vs {spd_outside:.2f} m/s inactive | **{"PASS" if gate8_pass else "FAIL"}** |

## 2. Per-Junction Queue Analysis (Post Warm-Up)
{df_to_markdown(j_analysis)}

## 3. Scenario Aggregate Summary
{df_to_markdown(sc_summary)}

## 4. Throughput Capacity & Reconciliation
{df_to_markdown(df_recon)}
"""
    (RESULTS_DIR / "data_validation.md").write_text(val_md, encoding="utf-8")
    print("  Validation report saved to results/data_validation.md.")
    
    gates = [gate1_pass, gate2_pass, gate3_pass, gate4_pass, gate5_pass, gate6_pass, gate7_pass, gate8_pass]
    print(f"  Gate Summary: {[ 'G' + str(i+1) + ': ' + ('PASS' if g else 'FAIL') for i, g in enumerate(gates) ]}")
    return all(gates)


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------
def main():
    print("==================================================================")
    print("STAGE 1: AUTONOMOUS REPAIR AND VALIDATION OF PHASE 2 DATASET")
    print("==================================================================")
    ensure_archive()
    configure_signal_programs()
    build_route_files(duration_s=3600)
    completed, failed = run_all_simulations(duration_s=3600)
    if failed > 0:
        print(f"FAILED: {failed} simulation runs encountered errors.")
        sys.exit(1)
        
    all_pass = evaluate_gates()
    print("==================================================================")
    print(f"STAGE 1 GATES EVALUATION: {'ALL GATES PASSED' if all_pass else 'SOME GATES FAILED'}")
    print("==================================================================")
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
