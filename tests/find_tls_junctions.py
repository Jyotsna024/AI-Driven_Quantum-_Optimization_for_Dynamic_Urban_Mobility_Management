import xml.etree.ElementTree as ET

net = ET.parse("sumo_files/hsr_test.net.xml").getroot()

tls_ids = [
    "GS_cluster_12067305554_6017833806_6017833807_6131076159",
    "GS_cluster_6633301856_7432359269_7432359283_7432359284",
    "cluster_10049312966_10860559949_11178931798_11178931802_#5more",
    "cluster_2387385680_3826045400",
    "joinedS_12067305543_12067305544",
]

for tid in tls_ids:
    print(f"\nTLS: {tid}")
    # check connections with this tl
    conns = [c for c in net.findall(".//connection") if c.get("tl") == tid]
    print(f"  Connections controlled: {len(conns)}")
    from_edges = {c.get("from") for c in conns}
    to_edges = {c.get("to") for c in conns}
    
    # find junctions at the boundary of these edges
    incident_junctions = set()
    for e in net.findall(".//edge"):
        eid = e.get("id")
        if eid in from_edges or eid in to_edges:
            incident_junctions.add(e.get("to"))
            incident_junctions.add(e.get("from"))
            
    # get junctions with type traffic_light or coordinates
    j_coords = []
    for jid in incident_junctions:
        j = net.find(f".//junction[@id='{jid}']")
        if j is not None and not jid.startswith(":"):
            j_coords.append((jid, j.get("type"), float(j.get("x")), float(j.get("y"))))
    print(f"  Incident junctions ({len(j_coords)}):")
    for jid, jtype, x, y in j_coords:
        if jtype == "traffic_light":
            print(f"    * TLS Junction {jid} ({x:.1f}, {y:.1f})")
        else:
            print(f"      Junction {jid} type={jtype} ({x:.1f}, {y:.1f})")
