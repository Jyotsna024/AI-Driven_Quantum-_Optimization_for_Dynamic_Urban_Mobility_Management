import xml.etree.ElementTree as ET
from pathlib import Path
import json
import traci
import os

net_p = Path("sumo_files/hsr.net.xml")
cfg_p = Path("sumo_files/cluster_config.json")
cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
sumo_bin = Path(os.environ.get("SUMO_HOME")) / "bin" / "sumo.exe"

# Start clean from hsr_test.net.xml
import shutil
shutil.copy("sumo_files/hsr_test.net.xml", "sumo_files/hsr.net.xml")

tree = ET.parse(net_p)
root = tree.getroot()

plans_dict = cfg["plans"]

# Find location of original tlLogic elements
tls_elems = [e for e in root if e.tag == "tlLogic"]
last_tl_idx = max(list(root).index(e) for e in tls_elems)

insert_idx = last_tl_idx + 1
for tid, plans in plans_dict.items():
    orig_tl = next(e for e in tls_elems if e.get("id") == tid)
    orig_type = orig_tl.get("type", "actuated")
    for p in plans[1:]:
        pid = str(p["plan_id"])
        new_tl = ET.Element("tlLogic", id=tid, type=orig_type, programID=pid, offset=str(p["offset"]))
        for ph in p["phases"]:
            ET.SubElement(new_tl, "phase", duration=str(ph["duration"]), state=ph["state"])
        root.insert(insert_idx, new_tl)
        insert_idx += 1

tree.write(str(net_p), encoding="utf-8", xml_declaration=True)
print(f"Inserted programs directly after tlLogic at index {last_tl_idx}!")

# Test TraCI
traci.start([str(sumo_bin), "-n", str(net_p), "--no-step-log", "true", "--no-warnings", "true"])
all_good = True
for tid in cfg["cluster_tls_ids"]:
    lanes = traci.trafficlight.getControlledLanes(tid)
    progs = traci.trafficlight.getCompleteRedYellowGreenDefinition(tid)
    print(f"TLS {tid[:20]}: controlled lanes = {len(lanes)}, programs = {[p.programID for p in progs]}")
    if len(lanes) == 0:
        all_good = False
traci.close()
print("Success:", all_good)
