"""
network_builder.py – Phase 1
Downloads genuine OpenStreetMap data for HSR Layout, Bengaluru using OSM API endpoints,
runs SUMO netconvert, selects 4 connected signalised intersections,
inspects actual SUMO signal phases, generates K=4 signal plans per junction,
verifies road connectivity, generates a network sketch, and verifies TraCI loading.
"""

import os
import sys
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
import networkx as nx
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUMO_DIR = PROJECT_ROOT / "sumo_files"
SUMO_BIN = Path(os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\SUMO")) / "bin"
NETCONVERT = str(SUMO_BIN / "netconvert.exe")
SUMO_EXE = str(SUMO_BIN / "sumo.exe")

SUMO_DIR.mkdir(parents=True, exist_ok=True)

# Bounding box for HSR Layout Sector 1-2 / 14th Main & 17th Cross corridor
# Format: min_lon, min_lat, max_lon, max_lat
HSR_BBOX = {
    "min_lon": 77.6360,
    "min_lat": 12.9100,
    "max_lon": 77.6490,
    "max_lat": 12.9270,
}

# The 4 target adjacent signalised intersections identified along 14th Main & 17th Cross
TARGET_TLS_META = {
    "joinedS_12067305543_12067305544": {
        "short_id": "J0",
        "name": "14th Main & 15th Cross",
        "description": "Commercial junction on 14th Main corridor near 15th Cross",
        "approx_coords": (635.5, 908.0),
        "controlled_junctions": ["12067305543", "12067305544"],
    },
    "GS_cluster_6633301856_7432359269_7432359283_7432359284": {
        "short_id": "J1",
        "name": "14th Main & 17th Cross",
        "description": "Major 4-way intersection connecting 14th Main and 17th Cross",
        "approx_coords": (626.6, 523.2),
        "controlled_junctions": ["cluster_6633301856_7432359269_7432359283_7432359284"],
    },
    "cluster_10049312966_10860559949_11178931798_11178931802_#5more": {
        "short_id": "J2",
        "name": "19th Main & 17th Cross",
        "description": "Multi-phase intersection on 17th Cross connecting to 19th Main",
        "approx_coords": (1323.8, 510.9),
        "controlled_junctions": ["cluster_10049312966_10860559949_11178931798_11178931802_#5more"],
    },
    "GS_cluster_12067305554_6017833806_6017833807_6131076159": {
        "short_id": "J3",
        "name": "14th Main & Sarjapur Road",
        "description": "Arterial junction at the intersection of 14th Main and Sarjapur Road",
        "approx_coords": (818.1, 1805.0),
        "controlled_junctions": ["cluster_12067305554_6017833806_6017833807_6131076159"],
    },
}


def download_osm(out_path: Path) -> None:
    """Download genuine OSM data for HSR Layout. If it fails, STOP and report."""
    if out_path.exists() and out_path.stat().st_size > 100_000:
        print(f"[skip] Genuine OSM already present at {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
        return

    bbox = HSR_BBOX
    # Primary: Official OpenStreetMap API bounding box endpoint
    osm_url = f"https://api.openstreetmap.org/api/0.6/map?bbox={bbox['min_lon']},{bbox['min_lat']},{bbox['max_lon']},{bbox['max_lat']}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) CapstoneTrafficProject/1.0"}

    print(f"[download] Fetching real OSM from {osm_url} ...")
    try:
        resp = requests.get(osm_url, headers=headers, timeout=120)
        resp.raise_for_status()
        if b"<osm" not in resp.content[:300]:
            raise ValueError("Response does not contain valid OSM XML")
        out_path.write_bytes(resp.content)
        size_kb = out_path.stat().st_size / 1024
        print(f"[ok] Genuine OSM saved: {out_path} ({size_kb:.1f} KB)")
        return
    except Exception as e:
        print(f"[error] OSM API failed: {e}")

    # Fallback to Overpass API interpreter
    overpass_urls = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]
    query = f"""[out:xml][timeout:90];
(
  way["highway"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
  node["highway"="traffic_signals"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
  node(w);
);
out body;
>;
out skel qt;
"""
    for url in overpass_urls:
        try:
            print(f"[download] Trying Overpass {url} ...")
            resp = requests.post(url, data={"data": query}, headers=headers, timeout=120)
            resp.raise_for_status()
            if b"<osm" in resp.content[:300]:
                out_path.write_bytes(resp.content)
                size_kb = out_path.stat().st_size / 1024
                print(f"[ok] OSM saved from Overpass: {out_path} ({size_kb:.1f} KB)")
                return
        except Exception as e:
            print(f"[warn] {url} failed: {e}")

    raise RuntimeError(
        "Failed to download genuine OpenStreetMap data for HSR Layout from all endpoints. "
        "Stopping per project instructions (no synthetic fallback allowed without explicit user permission)."
    )


def run_netconvert(osm_path: Path, net_path: Path) -> None:
    """Run netconvert on real OSM data."""
    if net_path.exists() and net_path.stat().st_size > 100_000:
        print(f"[skip] Net already exists at {net_path} ({net_path.stat().st_size / 1024:.1f} KB)")
        return

    cmd = [
        NETCONVERT,
        "--osm-files", str(osm_path),
        "--output-file", str(net_path),
        "--geometry.remove",
        "--roundabouts.guess",
        "--ramps.guess",
        "--junctions.join",
        "--junctions.join-dist", "25",
        "--tls.guess-signals",
        "--tls.discard-simple",
        "--tls.join",
        "--tls.default-type", "actuated",
        "--keep-edges.by-vclass", "passenger",
        "--remove-edges.isolated",
        "--no-turnarounds.tls",
        "--proj.utm",
        "--output.street-names",
    ]
    print("[netconvert] Converting genuine OSM to SUMO net.xml ...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("STDERR:", res.stderr[-2000:])
        raise RuntimeError("netconvert execution failed")
    print(f"[ok] Net successfully generated: {net_path} ({net_path.stat().st_size / 1024:.1f} KB)")


def parse_network(net_path: Path):
    """Parse net.xml, extract TLS programs, junctions, and connectivity graph."""
    tree = ET.parse(net_path)
    root = tree.getroot()

    tls_dict = {}
    for tl in root.findall(".//tlLogic"):
        tid = tl.get("id")
        phases = []
        for p in tl.findall("phase"):
            phases.append({
                "duration": float(p.get("duration", 30)),
                "state": p.get("state", ""),
                "minDur": float(p.get("minDur", 5)) if p.get("minDur") else None,
                "maxDur": float(p.get("maxDur", 60)) if p.get("maxDur") else None,
            })
        tls_dict[tid] = {
            "id": tid,
            "type": tl.get("type"),
            "programID": tl.get("programID", "0"),
            "offset": float(tl.get("offset", "0")),
            "phases": phases,
        }

    junctions = {}
    for jn in root.findall(".//junction"):
        jid = jn.get("id")
        if jid.startswith(":"):
            continue
        junctions[jid] = {
            "id": jid,
            "x": float(jn.get("x", 0)),
            "y": float(jn.get("y", 0)),
            "type": jn.get("type"),
            "tl": jn.get("tl"),
        }

    # Graph of edges
    G = nx.DiGraph()
    for edge in root.findall(".//edge"):
        if edge.get("function") == "internal":
            continue
        eid = edge.get("id")
        frm = edge.get("from")
        to = edge.get("to")
        name = edge.get("name", "")
        lanes = edge.findall("lane")
        length = float(lanes[0].get("length", 100)) if lanes else 100.0
        speed = float(lanes[0].get("speed", 13.89)) if lanes else 13.89
        G.add_edge(frm, to, id=eid, name=name, length=length, speed=speed)

    return tls_dict, junctions, G


def build_signal_plans(tls_dict: dict, cluster_tls_ids: list) -> dict:
    """
    Define K=4 signal plans per junction over each junction's ACTUAL SUMO phases.
    Plans:
      0: baseline (original durations from netconvert)
      1: main-road-heavy split (green time scaled 1.3x for primary phases, minor 0.7x)
      2: side-road-heavy split (green time scaled 1.3x for secondary phases, major 0.7x)
      3: shifted offset (+15s offset for green wave coordination)
    """
    plans = {}

    def scale_phases(phases, major_factor, minor_factor):
        scaled = []
        green_indices = [i for i, p in enumerate(phases) if "G" in p["state"] or "g" in p["state"]]
        # Primary green phases: alternating even indices of green phases
        primary_set = set(green_indices[::2])
        for i, p in enumerate(phases):
            new_p = dict(p)
            if "G" in p["state"] or "g" in p["state"]:
                factor = major_factor if i in primary_set else minor_factor
                new_p["duration"] = max(10.0, round(p["duration"] * factor, 1))
            scaled.append(new_p)
        return scaled

    for tid in cluster_tls_ids:
        tl_data = tls_dict.get(tid)
        if not tl_data:
            raise KeyError(f"TLS {tid} not found in SUMO network")

        orig_phases = tl_data["phases"]
        orig_offset = tl_data["offset"]

        plans[tid] = [
            {
                "plan_id": 0,
                "name": "baseline",
                "description": "Original fixed-time durations from netconvert",
                "phases": [dict(p) for p in orig_phases],
                "offset": orig_offset,
            },
            {
                "plan_id": 1,
                "name": "main_heavy",
                "description": "Main-corridor phases scaled x1.3 green; side approaches x0.7",
                "phases": scale_phases(orig_phases, 1.3, 0.7),
                "offset": orig_offset,
            },
            {
                "plan_id": 2,
                "name": "side_heavy",
                "description": "Side-road approaches scaled x1.3 green; main corridor x0.7",
                "phases": scale_phases(orig_phases, 0.7, 1.3),
                "offset": orig_offset,
            },
            {
                "plan_id": 3,
                "name": "shifted_offset",
                "description": "Baseline phase durations with +15s coordinated progression offset",
                "phases": [dict(p) for p in orig_phases],
                "offset": (orig_offset + 15.0) % 90.0,
            },
        ]
    return plans


def verify_cluster_connectivity(G: nx.DiGraph, cluster_tls_ids: list):
    """Verify that paths exist between cluster junctions in the road network."""
    connectivity_report = []
    # Map TLS to a representative junction node
    tls_to_node = {
        "joinedS_12067305543_12067305544": "12067305543",
        "GS_cluster_6633301856_7432359269_7432359283_7432359284": "cluster_6633301856_7432359269_7432359283_7432359284",
        "cluster_10049312966_10860559949_11178931798_11178931802_#5more": "cluster_10049312966_10860559949_11178931798_11178931802_#5more",
        "GS_cluster_12067305554_6017833806_6017833807_6131076159": "cluster_12067305554_6017833806_6017833807_6131076159",
    }

    connected_pairs = 0
    for u_tid in cluster_tls_ids:
        u_node = tls_to_node[u_tid]
        for v_tid in cluster_tls_ids:
            if u_tid == v_tid:
                continue
            v_node = tls_to_node[v_tid]
            if nx.has_path(G, u_node, v_node):
                dist = nx.shortest_path_length(G, u_node, v_node, weight="length")
                hops = len(nx.shortest_path(G, u_node, v_node))
                connectivity_report.append({
                    "from_tls": u_tid,
                    "to_tls": v_tid,
                    "distance_m": round(dist, 1),
                    "hops": hops,
                })
                connected_pairs += 1

    print(f"[connectivity] Found {connected_pairs} connected directed pairs among 4 cluster intersections")
    return connectivity_report


def draw_net_sketch(junctions: dict, cluster_tls_ids: list, out_path: Path):
    """Draw a schematic of the genuine HSR Layout road network and cluster intersections."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot all network junctions as faint background points
        xs = [j["x"] for j in junctions.values()]
        ys = [j["y"] for j in junctions.values()]
        ax.scatter(xs, ys, s=4, color="#cccccc", alpha=0.5, label="HSR Network Junctions")

        colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
        handles = []
        for idx, tid in enumerate(cluster_tls_ids):
            meta = TARGET_TLS_META[tid]
            x, y = meta["approx_coords"]
            color = colors[idx]
            ax.scatter(x, y, s=500, color=color, edgecolors="black", linewidths=2, zorder=10)
            ax.annotate(
                f"{meta['short_id']}: {meta['name']}",
                (x, y),
                textcoords="offset points",
                xytext=(15, 10),
                fontsize=9,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1.5, alpha=0.9),
                zorder=11,
            )
            handles.append(mpatches.Patch(color=color, label=f"{meta['short_id']} ({meta['name']})"))

        ax.set_title(
            "HSR Layout, Bengaluru – 4-Intersection Signal Cluster (Phase 1)\n"
            "Genuine OpenStreetMap Network (14th Main & 17th Cross Corridors)",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("UTM Easting (m)")
        ax.set_ylabel("UTM Northing (m)")
        ax.legend(handles=handles, loc="upper left", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
        print(f"[ok] Net sketch generated: {out_path}")
    except Exception as e:
        print(f"[warn] Could not draw net sketch: {e}")


def verify_traci_load(net_path: Path, cluster_tls_ids: list):
    """Verify that SUMO can start with this network and TraCI can control all 4 TLS."""
    import traci
    cmd = [SUMO_EXE, "-n", str(net_path), "--no-step-log", "true", "--no-warnings", "true"]
    try:
        traci.start(cmd)
        tls_in_sim = traci.trafficlight.getIDList()
        for tid in cluster_tls_ids:
            if tid not in tls_in_sim:
                raise RuntimeError(f"TLS {tid} missing in active TraCI simulation")
            state = traci.trafficlight.getRedYellowGreenState(tid)
            print(f"  [TraCI verify] {tid[:25]}... active state: {state}")
        traci.close()
        print("[ok] TraCI simulation load and TLS inspection verified successfully")
        return True
    except Exception as e:
        try:
            traci.close()
        except:
            pass
        raise RuntimeError(f"TraCI simulation verification failed: {e}")


def main():
    osm_path = SUMO_DIR / "hsr_raw.osm"
    net_path = SUMO_DIR / "hsr.net.xml"
    sketch_path = SUMO_DIR / "net_sketch.png"
    cfg_path = SUMO_DIR / "cluster_config.json"

    print("=== PHASE 1: Real HSR Layout Network Builder ===")
    # 1. Download real OSM
    download_osm(osm_path)

    # 2. netconvert
    run_netconvert(osm_path, net_path)

    # 3. Parse network
    tls_dict, junctions, G = parse_network(net_path)
    print(f"[parse] Found {len(tls_dict)} TLS programs, {len(junctions)} non-internal junctions")

    cluster_tls_ids = list(TARGET_TLS_META.keys())
    for tid in cluster_tls_ids:
        if tid not in tls_dict:
            raise RuntimeError(f"Required target TLS {tid} not found in net.xml!")

    # 4. Signal plans
    plans = build_signal_plans(tls_dict, cluster_tls_ids)

    # 5. Connectivity verification
    connectivity_report = verify_cluster_connectivity(G, cluster_tls_ids)

    # 6. Save cluster config
    config_data = {
        "network_source": "OpenStreetMap_official_API",
        "bbox": HSR_BBOX,
        "cluster_tls_ids": cluster_tls_ids,
        "tls_meta": TARGET_TLS_META,
        "plans": plans,
        "connectivity": connectivity_report,
    }
    cfg_path.write_text(json.dumps(config_data, indent=2), encoding="utf-8")
    print(f"[ok] Cluster configuration saved to {cfg_path}")

    # 7. Draw sketch
    draw_net_sketch(junctions, cluster_tls_ids, sketch_path)

    # 8. TraCI verification
    verify_traci_load(net_path, cluster_tls_ids)

    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE: Real HSR Layout Cluster Verified")
    print("=" * 70)
    for tid in cluster_tls_ids:
        meta = TARGET_TLS_META[tid]
        tl = tls_dict[tid]
        cycle = sum(p["duration"] for p in tl["phases"])
        print(f"  {meta['short_id']}: {meta['name']} | Coords: {meta['approx_coords']} | Phases: {len(tl['phases'])} | Cycle: {cycle:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
