import os
import sys
from pathlib import Path
import traci

net_p = Path("sumo_files/hsr_test.net.xml").resolve()
sumo_bin = Path(os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\SUMO")) / "bin" / "sumo.exe"

print(f"Testing TraCI loading with net: {net_p}")
cmd = [str(sumo_bin), "-n", str(net_p), "--no-step-log", "true", "--no-warnings", "true"]
try:
    traci.start(cmd)
    sim_time = traci.simulation.getTime()
    tls_ids = traci.trafficlight.getIDList()
    junction_ids = traci.junction.getIDList()
    print(f"TraCI loaded successfully! sim_time={sim_time}")
    print(f"TLS count in TraCI: {len(tls_ids)}")
    print(f"TLS IDs: {tls_ids}")
    print(f"Total junctions in TraCI: {len(junction_ids)}")
    for tid in tls_ids:
        ryg = traci.trafficlight.getRedYellowGreenState(tid)
        comp = traci.trafficlight.getCompleteRedYellowGreenDefinition(tid)
        print(f"  TLS {tid}: state={ryg}, programs={len(comp)}, phases={len(comp[0].phases) if comp else 0}")
    traci.close()
    print("TraCI closed cleanly. Verified!")
except Exception as e:
    print("TraCI load error:", e)
    try:
        traci.close()
    except:
        pass
    raise
