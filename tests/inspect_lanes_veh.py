import os
from pathlib import Path
import traci

sumo_bin = Path(os.environ.get("SUMO_HOME")) / "bin" / "sumo.exe"
net_p = Path("sumo_files/hsr.net.xml").resolve()
rou_p = Path("sumo_files/calib_peak.rou.xml").resolve()

traci.start([str(sumo_bin), "-n", str(net_p), "-r", str(rou_p), "--no-step-log", "true", "--no-warnings", "true"])

cluster_tls = [
    "joinedS_12067305543_12067305544",
    "GS_cluster_6633301856_7432359269_7432359283_7432359284",
    "cluster_10049312966_10860559949_11178931798_11178931802_#5more",
    "GS_cluster_12067305554_6017833806_6017833807_6131076159"
]

for tid in cluster_tls:
    controlled_lanes = traci.trafficlight.getControlledLanes(tid)
    print(f"TLS {tid[:15]} controlled lanes ({len(controlled_lanes)}): {controlled_lanes}")

# Step 300 steps and print vehicle speeds and lanes
for step in range(300):
    traci.simulationStep()
    if step % 50 == 0:
        vehs = traci.vehicle.getIDList()
        print(f"Step {step}: {len(vehs)} active vehicles")
        for vid in vehs[:5]:
            lane = traci.vehicle.getLaneID(vid)
            speed = traci.vehicle.getSpeed(vid)
            print(f"  {vid}: lane={lane}, speed={speed:.2f}")

traci.close()
