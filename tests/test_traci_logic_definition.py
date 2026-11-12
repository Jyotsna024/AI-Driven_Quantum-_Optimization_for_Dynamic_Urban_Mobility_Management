import os
from pathlib import Path
import traci

sumo_bin = Path(os.environ.get("SUMO_HOME")) / "bin" / "sumo.exe"
net_p = Path("sumo_files/hsr.net.xml").resolve()

traci.start([str(sumo_bin), "-n", str(net_p), "--no-step-log", "true", "--no-warnings", "true"])

tid = "GS_cluster_6633301856_7432359269_7432359283_7432359284"
comp = traci.trafficlight.getCompleteRedYellowGreenDefinition(tid)
print(f"Current logics for {tid[:20]}: {len(comp)}")
base_logic = comp[0]
print(f"Base logic: programID={base_logic.programID}, type={base_logic.type}, phases={len(base_logic.phases)}")

# Create a modified logic for Plan 1 (main heavy)
import copy
new_logic = copy.deepcopy(base_logic)
new_logic.programID = "1"
# Modify phase durations
for idx, p in enumerate(new_logic.phases):
    if "G" in p.state or "g" in p.state:
        p.duration = 50.0 if idx == 0 else 25.0
    print(f"  Plan 1 Phase {idx}: dur={p.duration}, state={p.state}")

# Apply new logic
traci.trafficlight.setCompleteRedYellowGreenDefinition(tid, new_logic)
print(f"After setting logic: current program = {traci.trafficlight.getProgram(tid)}")

# Verify controlled lanes are still there!
lanes = traci.trafficlight.getControlledLanes(tid)
print(f"Controlled lanes after setCompleteRedYellowGreenDefinition: {len(lanes)}")
assert len(lanes) == 16, f"Expected 16 lanes, got {len(lanes)}"

# Switch back to program 0
traci.trafficlight.setCompleteRedYellowGreenDefinition(tid, base_logic)
lanes = traci.trafficlight.getControlledLanes(tid)
print(f"Controlled lanes after switching back to base: {len(lanes)}")
assert len(lanes) == 16

traci.close()
print("TraCI dynamic logic definition verified successfully with full lane preservation!")
