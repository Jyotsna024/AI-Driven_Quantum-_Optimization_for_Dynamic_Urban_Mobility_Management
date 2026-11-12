import xml.etree.ElementTree as ET
from pathlib import Path
import json

net = ET.parse("sumo_files/hsr.net.xml").getroot()
cfg = json.loads(Path("sumo_files/cluster_config.json").read_text())

print("Connections and internal lanes per TLS:")
for tid, meta in cfg["tls_meta"].items():
    conns = [c for c in net.findall(".//connection") if c.get("tl") == tid]
    print(f"\nTLS {meta['short_id']}: {len(conns)} connections")
    via_lanes = [c.get("via") for c in conns if c.get("via")]
    print(f"  Internal via lanes ({len(via_lanes)}): {via_lanes[:4]}...")
    from_lanes = [f"{c.get('from')}_{c.get('fromLane')}" for c in conns]
    to_lanes = [f"{c.get('to')}_{c.get('toLane')}" for c in conns]
    print(f"  From lanes ({len(set(from_lanes))}): {sorted(list(set(from_lanes)))}")
    print(f"  To lanes ({len(set(to_lanes))}): {sorted(list(set(to_lanes)))}")
