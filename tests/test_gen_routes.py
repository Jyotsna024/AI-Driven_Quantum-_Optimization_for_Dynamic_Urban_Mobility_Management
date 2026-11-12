import xml.etree.ElementTree as ET
from pathlib import Path
import random

def generate_sample_routes(out_file: Path, veh_rate_per_hour: float, seed: int, duration_s: int = 600):
    random.seed(seed)
    
    # Load 12 routes
    import json
    cfg = json.loads(Path("sumo_files/cluster_config.json").read_text(encoding="utf-8"))
    
    # We will import the routes list
    from extract_cluster_routes import routes_12
    
    root = ET.Element("routes")
    
    # Define vTypes
    # car (50%), motorcycle (35%), auto_rickshaw (15%)
    vt_car = ET.SubElement(root, "vType", id="car", accel="2.6", decel="4.5", sigma="0.5", length="4.5", maxSpeed="14.0", vClass="passenger")
    vt_moto = ET.SubElement(root, "vType", id="motorcycle", accel="3.0", decel="5.0", sigma="0.5", length="2.2", maxSpeed="14.0", vClass="motorcycle")
    vt_auto = ET.SubElement(root, "vType", id="auto_rickshaw", accel="1.8", decel="3.5", sigma="0.5", length="2.8", maxSpeed="10.0", vClass="taxi")
    
    # Add route elements
    route_ids = []
    for idx, r in enumerate(routes_12):
        rid = f"r_{r['from_short']}_{r['to_short']}"
        route_ids.append(rid)
        ET.SubElement(root, "route", id=rid, edges=" ".join(r["edge_path"]))
        
    # Generate vehicle departures
    # Expected departures per second = veh_rate_per_hour / 3600
    p_dep = veh_rate_per_hour / 3600.0
    veh_count = 0
    for t in range(duration_s):
        # sample departures this second
        # Each second, can have 0 or more departures distributed across the 12 routes
        for rid in route_ids:
            if random.random() < (p_dep / len(route_ids)):
                r_type = random.random()
                if r_type < 0.50:
                    vtype = "car"
                elif r_type < 0.85:
                    vtype = "motorcycle"
                else:
                    vtype = "auto_rickshaw"
                veh_id = f"veh_{veh_count}_{t}"
                ET.SubElement(root, "vehicle", id=veh_id, type=vtype, route=rid, depart=str(float(t)), departLane="best", departSpeed="max")
                veh_count += 1
                
    ET.indent(root)
    tree = ET.ElementTree(root)
    tree.write(str(out_file), encoding="utf-8", xml_declaration=True)
    print(f"Generated {veh_count} vehicles in {out_file} (rate: {veh_rate_per_hour} veh/h)")

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path("tests").resolve()))
    generate_sample_routes(Path("sumo_files/test.rou.xml"), 600, 42, 600)
