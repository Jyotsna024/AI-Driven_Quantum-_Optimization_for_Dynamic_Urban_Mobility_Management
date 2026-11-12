import json
from pathlib import Path

cfg = json.loads(Path("sumo_files/cluster_config.json").read_text(encoding="utf-8"))
for tid, plans in cfg["plans"].items():
    print(tid[:25])
    for p in plans:
        print(f"  plan {p['plan_id']}: {len(p['phases'])} phases")
        for idx, ph in enumerate(p["phases"]):
            print(f"    ph {idx}: dur={ph.get('duration')} state={ph.get('state')}")
