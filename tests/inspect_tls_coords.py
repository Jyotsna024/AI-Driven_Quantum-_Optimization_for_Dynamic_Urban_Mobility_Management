import xml.etree.ElementTree as ET
from pathlib import Path
import json

net = Path("sumo_files/hsr_test.net.xml")
root = ET.parse(net).getroot()

# Map TLS ID to junctions controlled
tls_junctions = {}
for jn in root.findall(".//junction"):
    tl = jn.get("tl")
    if tl:
        if tl not in tls_junctions:
            tls_junctions[tl] = []
        tls_junctions[tl].append(jn)

tls_list = root.findall(".//tlLogic")
for tl in tls_list:
    tid = tl.get("id")
    controlled = tls_junctions.get(tid, [])
    # Calculate average coordinates
    if controlled:
        avg_x = sum(float(j.get("x", 0)) for j in controlled) / len(controlled)
        avg_y = sum(float(j.get("y", 0)) for j in controlled) / len(controlled)
    else:
        # Check if junction has the same id
        jn = root.find(f".//junction[@id='{tid}']")
        if jn is not None:
            avg_x = float(jn.get("x", 0))
            avg_y = float(jn.get("y", 0))
            controlled = [jn]
        else:
            avg_x, avg_y = 0.0, 0.0
    phases = tl.findall("phase")
    print(f"TLS: {tid}")
    print(f"  Coords: ({avg_x:.1f}, {avg_y:.1f})")
    print(f"  Controlled junctions: {[j.get('id') for j in controlled]}")
    print(f"  Phases ({len(phases)}):")
    for idx, p in enumerate(phases):
        print(f"    p{idx}: dur={p.get('duration')} state={p.get('state')}")
