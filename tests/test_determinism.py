import os
from pathlib import Path
import traci
import json

sumo_bin = Path(os.environ.get("SUMO_HOME")) / "bin" / "sumo.exe"
net_p = Path("sumo_files/hsr.net.xml").resolve()
rou_p = Path("sumo_files/calib_medium.rou.xml").resolve()
cfg_p = Path("sumo_files/cluster_config.json")
cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
cluster_tls = cfg["cluster_tls_ids"]

# Step 1: Run to t=600 and save state
state_dir = Path("data/states/determinism_test")
state_dir.mkdir(parents=True, exist_ok=True)
snap_path = state_dir / "snap_600.xml"

traci.start([str(sumo_bin), "-n", str(net_p), "-r", str(rou_p), "--seed", "42", "--no-step-log", "true", "--no-warnings", "true"])
for step in range(600):
    traci.simulationStep()
traci.simulation.saveState(str(snap_path))
print(f"Saved initial snapshot to {snap_path} ({snap_path.stat().st_size} bytes)")

def rollout_from_snapshot(plan_id: int, rollout_steps: int = 120):
    traci.simulation.loadState(str(snap_path))
    
    # Set explicit plan on all 4 junctions
    # We will build logic definitions
    # Note: getControlledLanes
    lanes = []
    for tid in cluster_tls:
        lanes.extend(traci.trafficlight.getControlledLanes(tid))
        # Set phase or program
        traci.trafficlight.setPhase(tid, 0)
        
    queues = []
    throughputs = 0
    prev_vehs = set()
    for l in lanes:
        prev_vehs.update(traci.lane.getLastStepVehicleIDs(l))
        
    for s in range(rollout_steps):
        traci.simulationStep()
        curr_vehs = set()
        for l in lanes:
            curr_vehs.update(traci.lane.getLastStepVehicleIDs(l))
            
        departed = prev_vehs - curr_vehs
        throughputs += len(departed)
        prev_vehs = curr_vehs
        
        q = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
        queues.append(q)
        
    mean_q = sum(queues) / len(queues)
    return mean_q, throughputs

# Run 1 with plan 1
mq1, tp1 = rollout_from_snapshot(1, 120)
print(f"Run 1 (plan 1): mean_q={mq1:.6f}, throughput={tp1}")

# Run 2 with SAME plan 1
mq2, tp2 = rollout_from_snapshot(1, 120)
print(f"Run 2 (plan 1): mean_q={mq2:.6f}, throughput={tp2}")

# Check exact match
assert mq1 == mq2 and tp1 == tp2, f"Determinism failed: ({mq1}, {tp1}) != ({mq2}, {tp2})"
print("Bit-exact match on identical plan verified!")

# Run 3 with DIFFERENT plan (e.g. force different phase offsets or durations)
# Let's force all phases to stay in all-red
for tid in cluster_tls:
    traci.trafficlight.setPhase(tid, 1) # yellow/red phase
mq3, tp3 = rollout_from_snapshot(2, 120)
print(f"Run 3 (different settings): mean_q={mq3:.6f}, throughput={tp3}")

traci.close()
print("Determinism test PASSED completely!")
