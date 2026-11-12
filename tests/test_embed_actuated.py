import xml.etree.ElementTree as ET
from pathlib import Path
import json
import traci
import os

net_p = Path("sumo_files/hsr.net.xml")
cfg_p = Path("sumo_files/cluster_config.json")
cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
sumo_bin = Path(os.environ.get("SUMO_HOME")) / "bin" / "sumo.exe"

tree = ET.parse(net_p)
root = tree.getroot()

plans_dict = cfg["plans"]

# Check original tlLogic types
tl_types = {}
for tl in root.findall(".//tlLogic"):
    tl_types[tl.get("id")] = tl.get("type", "actuated")

# Remove any existing programs 1, 2, 3
for tl in list(root.findall(".//tlLogic")):
    if tl.get("programID") in ("1", "2", "3"):
        root.remove(tl)

# Add programs 1, 2, 3 with type="actuated" matching original
for tid, plans in plans_dict.items():
    orig_type = tl_types.get(tid, "actuated")
    for p in plans[1:]:
        pid = str(p["plan_id"])
        new_tl = ET.Element("tlLogic", id=tid, type=orig_type, programID=pid, offset=str(p["offset"]))
        for ph in p["phases"]:
            ET.SubElement(new_tl, "phase", duration=str(ph["duration"]), state=ph["state"])
        root.append(new_tl)

tree.write(str(net_p), encoding="utf-8", xml_declaration=True)
print("Updated hsr.net.xml with actuated programs 1, 2, 3")

# Now test TraCI!
traci.start([str(sumo_bin), "-n", str(net_p), "--no-step-log", "true", "--no-warnings", "true"])
for tid in cfg["cluster_tls_ids"]:
    lanes = traci.trafficlight.getControlledLanes(tid)
    print(f"TLS {tid[:20]}: controlled lanes = {len(lanes)}")
    assert len(lanes) > 0, f"TLS {tid} lost its controlled lanes!"
traci.close()
print("All controlled lanes preserved with programs 0, 1, 2, 3 in hsr.net.xml!")
