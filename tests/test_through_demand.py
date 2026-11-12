import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import random
import traci

sumo_bin = Path(os.environ.get("SUMO_HOME")) / "bin" / "sumo.exe"
net_p = Path("sumo_files/hsr.net.xml").resolve()

# Load through routes from build_through_routes
sys.path.insert(0, str(Path("scripts").resolve()))
from build_through_routes import through_routes

# Generate a test route file with 2400 veh/h total across all 39 movements (~60 veh/h per movement)
root = ET.Element("routes")
ET.SubElement(root, "vType", id="car", accel="2.6", decel="4.5", sigma="0.5", length="4.5", maxSpeed="14.0", vClass="passenger")
ET.SubElement(root, "vType", id="motorcycle", accel="3.0", decel="5.0", sigma="0.5", length="2.2", maxSpeed="14.0", vClass="motorcycle")
ET.SubElement(root, "vType", id="auto_rickshaw", accel="1.8", decel="3.5", sigma="0.5", length="2.8", maxSpeed="10.0", vClass="taxi")

route_ids = []
for idx, r in enumerate(through_routes):
    rid = r["name"]
    route_ids.append((rid, r["junction"]))
    ET.SubElement(root, "route", id=rid, edges=" ".join(r["edges"]))

# Generate departures over 600s
random.seed(42)
veh_count = 0
p_dep = 3200.0 / 3600.0 # ~3200 veh/h network demand

for t in range(600):
    for rid, junc in route_ids:
        if random.random() < (p_dep / len(route_ids)):
            r_val = random.random()
            vtype = "car" if r_val < 0.5 else ("motorcycle" if r_val < 0.85 else "auto_rickshaw")
            veh_id = f"veh_{veh_count}_{t}"
            ET.SubElement(root, "vehicle", id=veh_id, type=vtype, route=rid, depart=str(float(t)), departLane="best", departSpeed="max")
            veh_count += 1

test_rou_p = Path("sumo_files/test_through.rou.xml")
ET.indent(root)
ET.ElementTree(root).write(str(test_rou_p), encoding="utf-8", xml_declaration=True)
print(f"Generated {veh_count} vehicles across {len(through_routes)} through-routes")

# Run in SUMO
traci.start([str(sumo_bin), "-n", str(net_p), "-r", str(test_rou_p), "--no-step-log", "true", "--no-warnings", "true"])

from diagnose_phase2 import junction_details

queues_per_junc = {sid: [] for sid in junction_details}
throughput_per_junc = {sid: 0 for sid in junction_details}

prev_incoming_vehs = {sid: set() for sid in junction_details}

for step in range(600):
    traci.simulationStep()
    t = traci.simulation.getTime()
    
    for sid, d in junction_details.items():
        lanes = d["incoming_lanes"]
        curr_vehs = set()
        for lid in lanes:
            curr_vehs.update(traci.lane.getLastStepVehicleIDs(lid))
            
        departed = prev_incoming_vehs[sid] - curr_vehs
        # Only count if vehicle actually moved to an outgoing edge
        throughput_per_junc[sid] += len(departed)
        prev_incoming_vehs[sid] = curr_vehs
        
        if t >= 300: # post warm-up
            q = sum(traci.lane.getLastStepHaltingNumber(lid) for lid in lanes)
            queues_per_junc[sid].append(q)

traci.close()

print("\n--- RESULTS AT 3200 veh/h ---")
for sid in junction_details:
    qs = queues_per_junc[sid]
    mean_q = sum(qs) / len(qs) if qs else 0.0
    import numpy as np
    p90_q = np.percentile(qs, 90) if qs else 0.0
    max_q = max(qs) if qs else 0
    tp = throughput_per_junc[sid]
    print(f"Junction {sid}: mean_queue = {mean_q:.2f}, P90 = {p90_q:.1f}, max = {max_q}, total throughput = {tp}")
