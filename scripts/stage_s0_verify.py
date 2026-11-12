"""
scripts/stage_s0_verify.py  –  S0 verification checks.
Outputs: results/s0_verify.json  (all checks with actual values)
         logs/pytest_s0.txt      (created by calling pytest at the end)
Run: python scripts/stage_s0_verify.py
"""
import os, sys, re, json, time, math, pickle
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "src"))

OUT = {}
START = time.time()

# ── S0-1  throughput cap grep ─────────────────────────────────────────────
cap_hits = []
for f in sorted((PROJECT / "src").glob("*.py")) + sorted((PROJECT / "scripts").glob("*.py")):
    txt = f.read_text(errors="replace")
    for i, line in enumerate(txt.splitlines(), 1):
        if re.search(r"min\s*\(.*60", line) and "throughput" in txt[max(0, txt.find(line)-200):txt.find(line)+200]:
            cap_hits.append({"file": str(f.relative_to(PROJECT)), "line": i, "content": line.strip()})
OUT["s0_1_throughput_cap"] = {
    "hits": cap_hits,
    "verdict": "CAP_FOUND" if cap_hits else "NO_CAP_FOUND"
}
print(f"S0-1 throughput cap: {OUT['s0_1_throughput_cap']['verdict']}  ({len(cap_hits)} hits)")

# ── S0-2  J2 peak queue analysis from existing raw CSVs (read-only) ───────
raw_dir = PROJECT / "data" / "raw"
peak_j2 = []
for f in sorted(raw_dir.glob("peak_*.csv")):
    df = pd.read_csv(f)
    j2 = df[(df["junction_short"]=="J2") & (df["is_warmup"]==False)]
    if not j2.empty:
        peak_j2.append(j2[["timestamp_s","queue_length","queue_length_m","mean_speed_mps","throughput"]].copy().assign(run=f.stem))

if peak_j2:
    all_j2 = pd.concat(peak_j2, ignore_index=True)
    q_m = all_j2["queue_length_m"].dropna()
    q_v = all_j2["queue_length"].dropna()
    # plateau check: does max_queue grow monotonically?
    ts_mean = all_j2.groupby("timestamp_s")["queue_length"].mean()
    is_monotone_grow = all(ts_mean.diff().dropna() >= 0)
    OUT["s0_2_j2_peak"] = {
        "n_rows": len(all_j2),
        "queue_veh_mean": round(float(q_v.mean()),2),
        "queue_veh_p90": round(float(q_v.quantile(0.9)),2),
        "queue_veh_max": round(float(q_v.max()),2),
        "queue_m_mean": round(float(q_m.mean()),2),
        "queue_m_p90": round(float(q_m.quantile(0.9)),2),
        "queue_m_max": round(float(q_m.max()),2),
        "throughput_mean": round(float(all_j2["throughput"].mean()),2),
        "mean_speed_mean": round(float(all_j2["mean_speed_mps"].dropna().mean()),3),
        "monotone_growth": bool(is_monotone_grow),
        "verdict": "PLATEAU_CONGESTION" if not is_monotone_grow else "MONOTONE_GROWTH_RISK"
    }
else:
    OUT["s0_2_j2_peak"] = {"verdict": "NO_DATA"}
print(f"S0-2 J2 peak: queue_m_mean={OUT['s0_2_j2_peak'].get('queue_m_mean')}  verdict={OUT['s0_2_j2_peak']['verdict']}")

# also compare J2 vs other junctions at peak
other_j = {}
for jid in ["J0","J1","J3"]:
    rows = []
    for f in sorted(raw_dir.glob("peak_*.csv")):
        df = pd.read_csv(f)
        sub = df[(df["junction_short"]==jid) & (df["is_warmup"]==False)]
        if not sub.empty:
            rows.append(sub)
    if rows:
        c = pd.concat(rows)["queue_length"]
        other_j[jid] = {"mean": round(float(c.mean()),2), "p90": round(float(c.quantile(0.9)),2)}
OUT["s0_2_j2_vs_others"] = other_j
print(f"S0-2 J others: {other_j}")

# ── S0-3  Persistence baseline at multiple lags ───────────────────────────
metrics_path = PROJECT / "models" / "metrics.json"
if metrics_path.exists():
    m = json.loads(metrics_path.read_text())
    OUT["s0_3_existing_metrics"] = m
    print(f"S0-3 existing metrics: {list(m.keys())}")
else:
    OUT["s0_3_existing_metrics"] = {"verdict": "METRICS_MISSING"}

# compute persistence at lags from raw data  
lag_maes = {}
all_runs = []
manifest = raw_dir / "manifest.csv"
if manifest.exists():
    mf = pd.read_csv(manifest)
    test_runs = mf[mf["split"]=="test"]["run_id"].tolist()
    for rid in test_runs:
        fp = raw_dir / f"{rid}.csv"
        if fp.exists():
            df = pd.read_csv(fp)
            df = df[df["is_warmup"]==False].copy()
            all_runs.append(df)
    if all_runs:
        combined = pd.concat(all_runs, ignore_index=True)
        combined = combined.sort_values(["run_id","junction_short","timestamp_s"])
        for lag_s in [60, 90, 120, 180]:
            lag_steps = lag_s // 60
            errs = []
            for (rid, jid), grp in combined.groupby(["run_id","junction_short"]):
                q = grp["queue_length"].values
                if len(q) > lag_steps:
                    pred = q[:-lag_steps]
                    actual = q[lag_steps:]
                    errs.extend(np.abs(actual - pred).tolist())
            lag_maes[f"lag_{lag_s}s"] = round(float(np.mean(errs)),3) if errs else None
OUT["s0_3_persistence_lags"] = lag_maes
print(f"S0-3 persistence lags: {lag_maes}")

# ── S0-4  ST-GNN metrics: val+test separately, per junction ──────────────
# Load metrics.json — it was saved by stage2_stgnn.py
if metrics_path.exists():
    m = json.loads(metrics_path.read_text())
    # check if val metrics are stored separately
    has_val_separate = "val" in m or any("val" in str(k) for k in m)
    has_per_junction = any("J0" in str(v) or "per_junction" in str(k) for k,v in m.items())
    OUT["s0_4_metrics_audit"] = {
        "keys_found": list(m.keys()),
        "has_val_separate": has_val_separate,
        "has_per_junction": has_per_junction,
        "needs_extended_eval": not (has_val_separate and has_per_junction)
    }
else:
    OUT["s0_4_metrics_audit"] = {"needs_extended_eval": True, "reason": "metrics.json missing"}
print(f"S0-4 metrics audit: needs_extended_eval={OUT['s0_4_metrics_audit'].get('needs_extended_eval')}")

# ── Write output ──────────────────────────────────────────────────────────
OUT["elapsed_s"] = round(time.time()-START, 2)
(PROJECT / "results").mkdir(exist_ok=True)
(PROJECT / "results" / "s0_verify.json").write_text(json.dumps(OUT, indent=2))
print(f"\nS0 verify done in {OUT['elapsed_s']}s  ->  results/s0_verify.json")
print("Run pytest next: pytest -q tests/test_phase1.py tests/test_phase2.py tests/test_phase3.py")
