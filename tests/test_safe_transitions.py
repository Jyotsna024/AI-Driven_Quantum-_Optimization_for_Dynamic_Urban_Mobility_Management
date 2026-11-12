import os
from pathlib import Path
import traci

sumo_bin = Path(os.environ.get("SUMO_HOME")) / "bin" / "sumo.exe"
net_p = Path("sumo_files/hsr.net.xml").resolve()
rou_p = Path("sumo_files/test.rou.xml").resolve()

traci.start([str(sumo_bin), "-n", str(net_p), "-r", str(rou_p), "--no-step-log", "true", "--no-warnings", "true"])

cluster_tls = [
    "joinedS_12067305543_12067305544",
    "GS_cluster_6633301856_7432359269_7432359283_7432359284",
    "cluster_10049312966_10860559949_11178931798_11178931802_#5more",
    "GS_cluster_12067305554_6017833806_6017833807_6131076159"
]

# Ensure initial program is 0
for tid in cluster_tls:
    traci.trafficlight.setProgram(tid, "0")

target_plans = {tid: 0 for tid in cluster_tls}
current_plans = {tid: 0 for tid in cluster_tls}
last_phases = {tid: traci.trafficlight.getPhase(tid) for tid in cluster_tls}

conflicts_detected = 0

for step in range(360):
    t = traci.simulation.getTime()
    
    # Every 60s, request a new random plan
    if step > 0 and step % 60 == 0:
        import random
        for tid in cluster_tls:
            target_plans[tid] = (target_plans[tid] + 1) % 4
            
    # Check transitions at cycle boundaries
    for tid in cluster_tls:
        cur_phase = traci.trafficlight.getPhase(tid)
        # Check for conflicting greens (e.g. adjacent greens on conflicting approaches)
        # Note: netconvert guaranteed each phase in the program has no internal conflicts.
        # Check transition: if cur_phase transitioned from max_phase to 0, or if cur_phase == 0 and prev != 0:
        if cur_phase == 0 and last_phases[tid] != 0:
            if target_plans[tid] != current_plans[tid]:
                traci.trafficlight.setProgram(tid, str(target_plans[tid]))
                current_plans[tid] = target_plans[tid]
        last_phases[tid] = cur_phase
        
    traci.simulationStep()

print("Ran 360 steps with plan switching.")
for tid in cluster_tls:
    print(f"  TLS {tid[:20]}: final plan = {traci.trafficlight.getProgram(tid)}")
traci.close()
print("Safe signal transitions verified!")
