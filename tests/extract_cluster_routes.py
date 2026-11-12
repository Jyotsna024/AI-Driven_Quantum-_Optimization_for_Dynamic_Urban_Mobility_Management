import xml.etree.ElementTree as ET
from pathlib import Path
import networkx as nx
import json

net_p = Path("sumo_files/hsr.net.xml")
cfg_p = Path("sumo_files/cluster_config.json")
root = ET.parse(net_p).getroot()
cfg = json.loads(cfg_p.read_text(encoding="utf-8"))

# Load graph
G = nx.DiGraph()
edge_map = {}
for e in root.findall(".//edge"):
    if e.get("function") == "internal":
        continue
    eid = e.get("id")
    frm = e.get("from")
    to = e.get("to")
    lanes = [l.get("id") for l in e.findall("lane")]
    length = float(e.find("lane").get("length", 100))
    G.add_edge(frm, to, id=eid, lanes=lanes, length=length)
    edge_map[eid] = {"from": frm, "to": to, "lanes": lanes, "length": length}

tls_meta = cfg["tls_meta"]
incoming_edges_per_tls = {}
for tid, meta in tls_meta.items():
    cj_set = set(meta["controlled_junctions"])
    in_edges = []
    for eid, edata in edge_map.items():
        if edata["to"] in cj_set and edata["from"] not in cj_set:
            in_edges.append(eid)
    incoming_edges_per_tls[tid] = in_edges

# Now find the exact edge sequences for the 12 OD pairs between cluster intersections
tls_to_node = {
    "joinedS_12067305543_12067305544": "12067305543",
    "GS_cluster_6633301856_7432359269_7432359283_7432359284": "cluster_6633301856_7432359269_7432359283_7432359284",
    "cluster_10049312966_10860559949_11178931798_11178931802_#5more": "cluster_10049312966_10860559949_11178931798_11178931802_#5more",
    "GS_cluster_12067305554_6017833806_6017833807_6131076159": "cluster_12067305554_6017833806_6017833807_6131076159",
}

routes_12 = []
for u_tid, u_node in tls_to_node.items():
    for v_tid, v_node in tls_to_node.items():
        if u_tid == v_tid:
            continue
        if nx.has_path(G, u_node, v_node):
            node_path = nx.shortest_path(G, u_node, v_node, weight="length")
            # Convert node path to edge path
            edge_path = []
            for i in range(len(node_path) - 1):
                edata = G.get_edge_data(node_path[i], node_path[i+1])
                edge_path.append(edata["id"])
            routes_12.append({
                "from_tls": u_tid,
                "to_tls": v_tid,
                "from_short": tls_meta[u_tid]["short_id"],
                "to_short": tls_meta[v_tid]["short_id"],
                "edge_path": edge_path,
                "length_m": round(sum(edge_map[e]["length"] for e in edge_path), 1)
            })

if __name__ == "__main__":
    print("TLS Controlled junctions and their incoming edges:")
    for tid, meta in tls_meta.items():
        print(f"  {meta['short_id']} ({meta['name']}): {len(incoming_edges_per_tls[tid])} incoming edges -> {incoming_edges_per_tls[tid]}")
    print(f"\nTotal extracted cluster routes: {len(routes_12)}")
    for r in routes_12:
        print(f"  {r['from_short']} -> {r['to_short']}: {len(r['edge_path'])} edges, {r['length_m']}m. First: {r['edge_path'][0]}, Last: {r['edge_path'][-1]}")
