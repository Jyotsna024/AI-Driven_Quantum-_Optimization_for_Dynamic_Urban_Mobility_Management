import xml.etree.ElementTree as ET
from pathlib import Path
import json
import traci
import os

net_p = Path("sumo_files/hsr.net.xml")
cfg_p = Path("sumo_files/cluster_config.json")
cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
sumo_bin = Path(os.environ.get("SUMO_HOME")) / "bin" / "sumo.exe"

# Copy from hsr_test.net.xml
import shutil
shutil.copy("sumo_files/hsr_test.net.xml", "sumo_files/hsr.net.xml")

tree = ET.parse(net_p)
root = tree.getroot()

# Set all original tlLogic type="static"
tls_elems = [e for e in root if e.tag == "tlLogic"]
for tl in tls_elems:
    tl.set("type", "static")
    # Remove minDur / maxDur if present so it's purely static
    for ph in tl.findall("phase"):
        if "minDur" in ph.attrib:
            del ph.attrib["minDur"]
        if "maxDur" in ph.attrib:
            del ph.attrib["maxDur"]

# Add plans 1, 2, 3 as static right after original tlLogic
last_tl_idx = max(list(root).index(e) for e in tls_elems)
insert_idx = last_tl_idx + 1

plans_dict = cfg["plans"]
for tid, plans in plans_dict.items():
    for p in plans[1:]:
        pid = str(p["plan_id"])
        new_tl = ET.Element("tlLogic", id=tid, type="static", programID=pid, offset=str(p["offset"]))
        for ph in p["phases"]:
            ET.SubElement(new_tl, "phase", duration=str(ph["duration"]), state=ph["state"])
        root.insert(insert_idx, new_tl)
        insert_idx += 1

tree.write(str(net_p), encoding="utf-8", xml_declaration=True)
print("Saved net.xml with static TLS and programs 0..3")

# Test saveState and loadState
rou_p = Path("sumo_files/calib_medium.rou.xml").resolve()
traci.start([str(sumo_bin), "-n", str(net_p), "-r", str(rou_p), "--no-step-log", "true", "--no-warnings", "true"])

# Set all to plan 1
for tid in cfg["cluster_tls_ids"]:
    traci.trafficlight.setProgram(tid, "1")
    
for _ in range(30):
    traci.simulationStep()
    
traci.simulation.saveState("data/states/test_static_snap.xml")
print("Saved snapshot in program 1!")

# Step more
for _ in range(10):
    traci.simulationStep()
    
# Reload state!
traci.simulation.loadState("data/states/test_static_snap.xml")
print("SUCCESSFULLY LOADED SNAPSHOT in program 1!")
for tid in cfg["cluster_tls_ids"]:
    print(f"  TLS {tid[:20]}: program = {traci.trafficlight.getProgram(tid)}, phase = {traci.trafficlight.getPhase(tid)}")

traci.close()
print("All static tests passed!")
