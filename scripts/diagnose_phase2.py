import xml.etree.ElementTree as ET
from pathlib import Path
import json
import networkx as nx

net_path = Path("sumo_files/hsr.net.xml")
cfg_path = Path("sumo_files/cluster_config.json")
net = ET.parse(net_path).getroot()
cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

print("=== 1a. Junction Incoming / Outgoing Edges & Controlled Lanes ===")
junction_details = {}
for tid, meta in cfg["tls_meta"].items():
    sid = meta["short_id"]
    conns = [c for c in net.findall(".//connection") if c.get("tl") == tid]
    from_edges = sorted(list(set(c.get("from") for c in conns)))
    to_edges = sorted(list(set(c.get("to") for c in conns)))
    
    # Lanes
    incoming_lanes = []
    for c in conns:
        lane_id = f"{c.get('from')}_{c.get('fromLane')}"
        if lane_id not in incoming_lanes:
            incoming_lanes.append(lane_id)
            
    junction_details[sid] = {
        "tid": tid,
        "name": meta["name"],
        "from_edges": from_edges,
        "to_edges": to_edges,
        "incoming_lanes": incoming_lanes,
    }
    print(f"\nJunction {sid} ({meta['name']}):")
    print(f"  Incoming edges ({len(from_edges)}): {from_edges}")
    print(f"  Outgoing edges ({len(to_edges)}): {to_edges}")
    print(f"  Incoming controlled lanes ({len(incoming_lanes)}): {incoming_lanes}")

# Graph analysis: How do routes pass THROUGH junctions?
G = nx.DiGraph()
for e in net.findall(".//edge"):
    if e.get("function") == "internal":
        continue
    eid = e.get("id")
    frm = e.get("from")
    to = e.get("to")
    length = float(e.find("lane").get("length", 100))
    G.add_edge(frm, to, id=eid, length=length)

print("\n=== 1b. Checking Existing 12 Routes Traversal ===")
# Read previous routes
rou_file = Path("sumo_files/routes/medium_s0.rou.xml")
if rou_file.exists():
    rou_tree = ET.parse(rou_file).getroot()
    routes = rou_tree.findall("route")
    for r in routes:
        rid = r.get("id")
        edges = r.get("edges").split()
        first_edge = edges[0]
        last_edge = edges[-1]
        
        # Check which junctions this route traverses
        traversed = []
        for sid, d in junction_details.items():
            # If an incoming edge to junction is in the route AND NOT the last edge, it traverses the junction!
            for idx, e in enumerate(edges[:-1]):
                if e in d["from_edges"] and edges[idx+1] in d["to_edges"]:
                    traversed.append(sid)
                    break
            # Also check if it terminates at the incoming edge
            if last_edge in d["from_edges"]:
                traversed.append(f"{sid}(terminates)")
        print(f"Route {rid:<12} (len {len(edges)} edges): traverses -> {traversed}")
