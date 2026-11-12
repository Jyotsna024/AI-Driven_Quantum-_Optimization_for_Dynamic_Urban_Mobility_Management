import os
from pathlib import Path
import traci

sumo_bin = Path(os.environ.get("SUMO_HOME")) / "bin" / "sumo.exe"
net_p = Path("sumo_files/hsr.net.xml").resolve()
rou_p = Path("sumo_files/calib_medium.rou.xml").resolve()

# Test incident on lane '24480656#3_0'
target_lane = "24480656#3_0"

def run_incident_test(block_lane: bool):
    traci.start([str(sumo_bin), "-n", str(net_p), "-r", str(rou_p), "--no-step-log", "true", "--no-warnings", "true"])
    orig_speed = traci.lane.getMaxSpeed(target_lane)
    speeds_during_incident = []
    
    for step in range(600):
        t = traci.simulation.getTime()
        if block_lane and 200 <= t < 400:
            traci.lane.setMaxSpeed(target_lane, 0.1)
        elif block_lane and t >= 400:
            traci.lane.setMaxSpeed(target_lane, orig_speed)
            
        traci.simulationStep()
        if 200 <= t < 400:
            speeds_during_incident.append(traci.lane.getLastStepMeanSpeed(target_lane))
            
    traci.close()
    mean_spd = sum(speeds_during_incident) / len(speeds_during_incident)
    return mean_spd

spd_normal = run_incident_test(block_lane=False)
spd_incident = run_incident_test(block_lane=True)
print(f"Normal mean speed on incident lane: {spd_normal:.2f} m/s")
print(f"Incident mean speed on incident lane: {spd_incident:.2f} m/s")
print(f"Speed reduction: {((spd_normal - spd_incident) / spd_normal * 100):.1f}%")
assert spd_incident < spd_normal, "Incident failed to reduce speed on target lane!"
print("Incident injection verified!")
