import xml.etree.ElementTree as ET
from pathlib import Path
import json
import networkx as nx

net_p = Path("sumo_files/hsr.net.xml")
cfg_p = Path("sumo_files/cluster_config.json")
net = ET.parse(net_p).getroot()
cfg = json.loads(cfg_p.read_text(encoding="utf-8"))

G = nx.DiGraph()
edge_map = {}
for e in net.findall(".//edge"):
    if e.get("function") == "internal":
        continue
    eid = e.get("id")
    frm = e.get("from")
    to = e.get("to")
    lanes = [l.get("id") for l in e.findall("lane")]
    length = float(e.find("lane").get("length", 100))
    G.add_edge(frm, to, id=eid, lanes=lanes, length=length)
    edge_map[eid] = {"from": frm, "to": to, "lanes": lanes, "length": length}

# For each TLS, check its connections
tls_conns = {}
for tid, meta in cfg["tls_meta"].items():
    sid = meta["short_id"]
    conns = [c for c in net.findall(".//connection") if c.get("tl") == tid]
    tls_conns[sid] = [(c.get("from"), c.get("to")) for c in conns]

# Build routes that traverse through each connection
# For a connection (from_edge, to_edge) of junction J:
# A valid route through J is [from_edge, to_edge] or [upstream_edge, from_edge, to_edge, downstream_edge]!
through_routes = []
route_names = []

for sid, conns_list in tls_conns.items():
    distinct_pairs = list(set(conns_list))
    print(f"\nJunction {sid} distinct (from, to) connection pairs ({len(distinct_pairs)}):")
    for idx, (f_edge, t_edge) in enumerate(distinct_pairs):
        # Find if f_edge can be preceded by an upstream edge
        up_edges = [u for u, v, d in G.in_edges(edge_map[f_edge]["from"], data=True)]
        # Find if t_edge can be succeeded by a downstream edge
        down_edges = [v for u, v, d in G.out_edges(edge_map[t_edge]["to"], data=True)]
        
        # Build path: if up_edges, take first; then f_edge, t_edge, down_edge
        path = [f_edge, t_edge]
        # Check if f_edge -> t_edge is directly connected
        through_routes.append({
            "junction": sid,
            "name": f"r_{sid}_{idx}",
            "edges": path,
            "from_edge": f_edge,
            "to_edge": t_edge
        })
        print(f"  Conn {idx}: {f_edge} -> {t_edge}")

print(f"\nTotal through routes constructed: {len(through_routes)}")
