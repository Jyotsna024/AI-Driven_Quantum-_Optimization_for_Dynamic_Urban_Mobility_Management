import xml.etree.ElementTree as ET
from pathlib import Path
import json

net_p = Path("sumo_files/hsr.net.xml")
cfg_p = Path("sumo_files/cluster_config.json")
cfg = json.loads(cfg_p.read_text(encoding="utf-8"))

plans_dict = cfg["plans"]
tree = ET.parse(net_p)
root = tree.getroot()

# For each TLS in cluster, add programID="1", "2", "3" if not present
for tid, plans in plans_dict.items():
    # Find existing program 0
    orig_tl = root.find(f".//tlLogic[@id='{tid}']")
    if orig_tl is None:
        continue
    orig_tl.set("programID", "0")
    
    # Check if programs 1, 2, 3 already exist
    for p in plans[1:]:
        pid = str(p["plan_id"])
        existing = root.find(f".//tlLogic[@id='{tid}'][@programID='{pid}']")
        if existing is not None:
            root.remove(existing)
            
        new_tl = ET.Element("tlLogic", id=tid, type="static", programID=pid, offset=str(p["offset"]))
        for ph in p["phases"]:
            ET.SubElement(new_tl, "phase", duration=str(ph["duration"]), state=ph["state"])
        root.append(new_tl)

tree.write(str(net_p), encoding="utf-8", xml_declaration=True)
print("Updated hsr.net.xml with all K=4 signal programs for cluster junctions!")
