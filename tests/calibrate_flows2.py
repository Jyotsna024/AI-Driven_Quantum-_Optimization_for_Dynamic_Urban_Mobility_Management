import os
import sys
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import random
import traci

net_p = Path("sumo_files/hsr.net.xml").resolve()
cfg_p = Path("sumo_files/cluster_config.json")
cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
sumo_bin = Path(os.environ.get("SUMO_HOME")) / "bin" / "sumo.exe"

sys.path.insert(0, str(Path("tests").resolve()))
from extract_cluster_routes import routes_12

cluster_tls = cfg["cluster_tls_ids"]

def generate_routes_for_scenario(out_path: Path, veh_rate_per_hour: float, seed: int, duration_s: int = 1200):
    random.seed(seed)
    root = ET.Element("routes")
    ET.SubElement(root, "vType", id="car", accel="2.6", decel="4.5", sigma="0.5", length="4.5", maxSpeed="14.0", vClass="passenger")
    ET.SubElement(root, "vType", id="motorcycle", accel="3.0", decel="5.0", sigma="0.5", length="2.2", maxSpeed="14.0", vClass="motorcycle")
    ET.SubElement(root, "vType", id="auto_rickshaw", accel="1.8", decel="3.5", sigma="0.5", length="2.8", maxSpeed="10.0", vClass="taxi")
    
    route_ids = []
    for r in routes_12:
        rid = f"r_{r['from_short']}_{r['to_short']}"
        route_ids.append(rid)
        ET.SubElement(root, "route", id=rid, edges=" ".join(r["edge_path"]))
        
    p_dep = veh_rate_per_hour / 3600.0
    veh_count = 0
    for t in range(duration_s):
        for rid in route_ids:
            if random.random() < (p_dep / len(route_ids)):
                r_type = random.random()
                vtype = "car" if r_type < 0.5 else ("motorcycle" if r_type < 0.85 else "auto_rickshaw")
                veh_id = f"veh_{veh_count}_{t}"
                ET.SubElement(root, "vehicle", id=veh_id, type=vtype, route=rid, depart=str(float(t)), departLane="best", departSpeed="max")
                veh_count += 1
                
    ET.indent(root)
    ET.ElementTree(root).write(str(out_path), encoding="utf-8", xml_declaration=True)
    return veh_count

def run_test_sim(rou_file: Path, duration_s: int = 1200):
    cmd = [str(sumo_bin), "-n", str(net_p), "-r", str(rou_file), "--no-step-log", "true", "--no-warnings", "true", "--time-to-teleport", "300"]
    traci.start(cmd)
    
    tls_lanes = {tid: set(traci.trafficlight.getControlledLanes(tid)) for tid in cluster_tls}
    all_controlled_lanes = set()
    for lset in tls_lanes.values():
        all_controlled_lanes.update(lset)
        
    queues = []
    teleports = 0
    throughputs = []
    
    for step in range(duration_s):
        traci.simulationStep()
        t = traci.simulation.getTime()
        teleports += traci.simulation.getStartingTeleportNumber()
        if t >= 300: # post warm-up
            step_queue = sum(traci.lane.getLastStepHaltingNumber(lid) for lid in all_controlled_lanes)
            queues.append(step_queue)
            
    traci.close()
    mean_q = sum(queues) / len(queues) if queues else 0.0
    return mean_q, teleports

print("Calibrating flow rates with exact controlled lanes:")
test_rates = {
    "low": 240,
    "medium": 600,
    "high": 1200,
    "peak": 1800
}

results = {}
for name, rate in test_rates.items():
    rou_p = Path(f"sumo_files/calib_{name}.rou.xml")
    n_vehs = generate_routes_for_scenario(rou_p, rate, 42, 1200)
    mq, tel = run_test_sim(rou_p, 1200)
    results[name] = {"rate": rate, "vehs": n_vehs, "mean_queue": round(mq, 3), "teleports": tel}
    print(f"Scenario {name:<7} (rate={rate:>4} veh/h): mean queue = {mq:>6.3f}, teleports = {tel}")

is_ordered = results["low"]["mean_queue"] < results["medium"]["mean_queue"] < results["high"]["mean_queue"] < results["peak"]["mean_queue"]
print("Strictly ordered (low < medium < high < peak):", is_ordered)
