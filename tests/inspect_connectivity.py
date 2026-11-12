import xml.etree.ElementTree as ET
from pathlib import Path
import networkx as nx

net = Path("sumo_files/hsr_test.net.xml")
root = ET.parse(net).getroot()

G = nx.DiGraph()

# Add edges
for edge in root.findall(".//edge"):
    if edge.get("function") == "internal":
        continue
    eid = edge.get("id")
    frm = edge.get("from")
    to = edge.get("to")
    name = edge.get("name", "")
    # length from lanes
    lanes = edge.findall("lane")
    length = float(lanes[0].get("length", 100)) if lanes else 100.0
    speed = float(lanes[0].get("speed", 13.89)) if lanes else 13.89
    G.add_edge(frm, to, id=eid, name=name, length=length, speed=speed)

print(f"NetworkX graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# Check connectivity between the TLS junctions
tls_nodes = [
    "12067305543",
    "12067305544",
    "cluster_10049312966_10860559949_11178931798_11178931802_#5more",
    "cluster_12067305554_6017833806_6017833807_6131076159",
    "cluster_2387385680_3826045400",
    "cluster_6633301856_7432359269_7432359283_7432359284"
]

for u in tls_nodes:
    for v in tls_nodes:
        if u != v:
            if nx.has_path(G, u, v):
                path = nx.shortest_path(G, u, v, weight="length")
                dist = nx.shortest_path_length(G, u, v, weight="length")
                print(f"Path {u[:15]} -> {v[:15]}: {len(path)} hops, {dist:.1f}m")
                # print edge names along path
                edge_names = []
                for i in range(len(path)-1):
                    edata = G.get_edge_data(path[i], path[i+1])
                    if edata and edata.get('name'):
                        edge_names.append(edata['name'])
                print(f"  Streets: {set(edge_names)}")
