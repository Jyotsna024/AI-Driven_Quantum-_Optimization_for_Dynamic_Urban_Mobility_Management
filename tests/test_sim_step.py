import os
import sys
from pathlib import Path
import traci

net_p = Path("sumo_files/hsr.net.xml").resolve()
rou_p = Path("sumo_files/test.rou.xml").resolve()
sumo_bin = Path(os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\SUMO")) / "bin" / "sumo.exe"

cmd = [
    str(sumo_bin),
    "-n", str(net_p),
    "-r", str(rou_p),
    "--no-step-log", "true",
    "--no-warnings", "true",
    "--time-to-teleport", "-1"
]

print("Starting simulation test...")
traci.start(cmd)

tls_ids = ["joinedS_12067305543_12067305544", "GS_cluster_6633301856_7432359269_7432359283_7432359284", "cluster_10049312966_10860559949_11178931798_11178931802_#5more", "GS_cluster_12067305554_6017833806_6017833807_6131076159"]

# Run for 120 steps
for step in range(120):
    traci.simulationStep()
    t = traci.simulation.getTime()
    if step % 30 == 0:
        total_veh = traci.vehicle.getIDCount()
        print(f"Step {step} (t={t}): total vehicles in sim = {total_veh}")

# Test saveState
state_dir = Path("data/states/test")
state_dir.mkdir(parents=True, exist_ok=True)
state_file = state_dir / "state_120.xml"
traci.simulation.saveState(str(state_file))
print("Saved state to:", state_file, "size:", state_file.stat().st_size)

# Step 10 more steps
for _ in range(10):
    traci.simulationStep()
print(f"After 10 steps: time = {traci.simulation.getTime()}")

# Test loadState
traci.simulation.loadState(str(state_file))
print(f"After loadState: time = {traci.simulation.getTime()}")
assert traci.simulation.getTime() == 120.0, "Time mismatch after loadState!"

traci.close()
print("TraCI closed cleanly. saveState / loadState verified!")
