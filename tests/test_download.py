import requests
from pathlib import Path

# Bbox: min_lon, min_lat, max_lon, max_lat
# 77.6360, 12.9100, 77.6490, 12.9270
url_osm = "https://api.openstreetmap.org/api/0.6/map?bbox=77.6360,12.9100,77.6490,12.9270"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) CapstoneProject/1.0"}

print("Testing official OSM API...")
try:
    r = requests.get(url_osm, headers=headers, timeout=60)
    print("OSM API status:", r.status_code, "Content length:", len(r.content))
    if r.status_code == 200 and b"<osm" in r.content:
        print("SUCCESS from official OSM API!")
        out_p = Path("sumo_files/hsr_raw.osm")
        out_p.write_bytes(r.content)
        print("Wrote to", out_p, "size:", len(r.content))
except Exception as e:
    print("OSM API error:", e)

# Also test Overpass
url_op = "https://overpass-api.de/api/interpreter"
q = """[out:xml][timeout:60];
(
  way["highway"](12.9100,77.6360,12.9270,77.6490);
  node["highway"="traffic_signals"](12.9100,77.6360,12.9270,77.6490);
  node(w);
);
out body;
>;
out skel qt;
"""
print("Testing Overpass POST...")
try:
    r2 = requests.post(url_op, data={"data": q}, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) CapstoneTraffic/1.0"}, timeout=60)
    print("Overpass status:", r2.status_code, "Content length:", len(r2.content))
except Exception as e:
    print("Overpass error:", e)
