import xml.etree.ElementTree as ET
from pathlib import Path

net = Path("sumo_files/hsr_test.net.xml")
print("Net file size:", net.stat().st_size)
root = ET.parse(net).getroot()

tls_list = root.findall(".//tlLogic")
print(f"TLS programs count: {len(tls_list)}")
for tl in tls_list:
    phases = tl.findall("phase")
    print(f"  TLS id: {tl.get('id')} type={tl.get('type')} phases={len(phases)}")
    for idx, ph in enumerate(phases):
        print(f"    ph {idx}: dur={ph.get('duration')} state={ph.get('state')}")

junctions = [j for j in root.findall(".//junction") if not j.get("id", "").startswith(":")]
tls_junctions = [j for j in junctions if j.get("type") in ("traffic_light", "traffic_light_right_on_red")]
print(f"\nTotal non-internal junctions: {len(junctions)}")
print(f"Signalised junctions (type=traffic_light): {len(tls_junctions)}")
for j in tls_junctions:
    print(f"  Junction {j.get('id')} at ({float(j.get('x')):.1f}, {float(j.get('y')):.1f}) tl={j.get('tl')}")
