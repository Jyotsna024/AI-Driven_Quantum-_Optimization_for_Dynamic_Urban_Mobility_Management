import os
from pathlib import Path
import traci

sumo_bin = Path(os.environ.get("SUMO_HOME")) / "bin" / "sumo.exe"
net_p = Path("sumo_files/hsr.net.xml").resolve()
rou_p = Path("sumo_files/test.rou.xml").resolve()

traci.start([str(sumo_bin), "-n", str(net_p), "-r", str(rou_p), "--no-step-log", "true", "--no-warnings", "true"])

import json
cfg = json.loads(Path("sumo_files/cluster_config.json").read_text(encoding="utf-8"))

# Incoming lanes per TLS
from extract_cluster_routes import incoming_edges_per_tls
incoming_lanes_per_tls = {}
for tid, edges in incoming_edges_per_tls.items():
    lanes = []
    for e in edges:
        # Get lanes for this edge from net
        num_lanes = 2 # typically 1 or 2
        for l_idx in range(5):
            lid = f"{e}_{l_idx}"
            try:
                traci.lane.getLength(lid)
                lanes.append(lid)
            except:
                break
    incoming_lanes_per_tls[tid] = lanes
    print(f"TLS {tid[:15]} has {len(lanes)} incoming lanes: {lanes}")

prev_vehicles_on_incoming = {tid: set() for tid in incoming_lanes_per_tls}
throughput_window = {tid: 0 for tid in incoming_lanes_per_tls}
arrivals_window = {tid: 0 for tid in incoming_lanes_per_tls}

for step in range(120):
    traci.simulationStep()
    for tid, lanes in incoming_lanes_per_tls.items():
        curr_vehs = set()
        for lid in lanes:
            curr_vehs.update(traci.lane.getLastStepVehicleIDs(lid))
            
        # Throughput: was on incoming lanes, now not on incoming lanes (passed stop-line into junction)
        departed = prev_vehicles_on_incoming[tid] - curr_vehs
        # Filter for vehicles still in simulation (or reached destination)
        throughput_window[tid] += len(departed)
        
        # Arrivals: on incoming lanes, wasn't there last step
        entered = curr_vehs - prev_vehicles_on_incoming[tid]
        arrivals_window[tid] += len(entered)
        
        prev_vehicles_on_incoming[tid] = curr_vehs
        
    if (step + 1) % 60 == 0:
        sim_time = traci.simulation.getTime()
        print(f"\n--- 60s Interval at t={sim_time} ---")
        for tid in incoming_lanes_per_tls:
            lanes = incoming_lanes_per_tls[tid]
            halting = sum(traci.lane.getLastStepHaltingNumber(lid) for lid in lanes)
            vcount = sum(traci.lane.getLastStepVehicleNumber(lid) for lid in lanes)
            tp = throughput_window[tid]
            arr = arrivals_window[tid]
            print(f"  {tid[:15]}: count={vcount}, queue={halting}, throughput={tp}, arrivals={arr}")
            throughput_window[tid] = 0
            arrivals_window[tid] = 0

traci.close()
print("Throughput tracking verified!")
