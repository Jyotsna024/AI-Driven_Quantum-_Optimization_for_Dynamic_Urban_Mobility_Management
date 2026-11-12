"""
validate_data.py – Comprehensive validation of Phase 2 dataset & simulation outputs.
Generates results/data_validation.md and enforces all required validation gates.
"""

import os
import sys
import glob
import json
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
STATES_DIR = PROJECT_ROOT / "data" / "states"
RESULTS_DIR = PROJECT_ROOT / "results"
SUMO_DIR = PROJECT_ROOT / "sumo_files"
SUMO_BIN = Path(os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\SUMO")) / "bin" / "sumo.exe"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLUMNS = [
    "run_id", "scenario", "seed", "mode", "split", "timestamp_s",
    "is_warmup", "junction_id", "junction_short", "vehicle_count",
    "queue_length", "mean_speed_mps", "waiting_time_s", "current_plan_id",
    "current_phase", "throughput", "arrivals", "is_incident_active"
]


def load_all_data():
    """Load and concatenate all raw simulation CSVs."""
    csv_files = list(DATA_RAW_DIR.glob("*_s*_*csv"))
    if not csv_files:
        raise FileNotFoundError(f"No simulation CSVs found in {DATA_RAW_DIR}")

    dfs = []
    for f in csv_files:
        if f.name == "manifest.csv":
            continue
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"[warn] Could not read {f}: {e}")

    if not dfs:
        raise ValueError("No valid run CSVs loaded!")

    full_df = pd.concat(dfs, ignore_index=True)
    return full_df, csv_files


def run_validation():
    print("=== STARTING PHASE 2 DATA VALIDATION ===")
    full_df, csv_files = load_all_data()
    total_runs = full_df["run_id"].nunique()
    total_rows = len(full_df)
    print(f"Loaded {total_rows} total rows across {total_runs} distinct runs ({len(csv_files)} files)")

    report_lines = []
    report_lines.append("# Phase 2 Data Validation Report\n")
    report_lines.append(f"- **Total runs evaluated**: {total_runs}")
    report_lines.append(f"- **Total observation rows**: {total_rows}")
    report_lines.append(f"- **Data directory**: `{DATA_RAW_DIR}`\n")

    # Gate 1: Missing values check
    missing_counts = full_df[REQUIRED_COLUMNS].isnull().sum()
    total_missing = missing_counts.sum()
    print(f"Missing values check: {total_missing} missing across required columns")
    report_lines.append("## 1. Data Integrity & Completeness")
    report_lines.append(f"- **Missing values**: {total_missing} (Gate: must be 0)")
    assert total_missing == 0, f"Gate failed: found {total_missing} missing values!"

    # Gate 2: Duplicate check
    dup_mask = full_df.duplicated(subset=["run_id", "timestamp_s", "junction_id"])
    total_dups = dup_mask.sum()
    print(f"Duplicate rows check: {total_dups} duplicates")
    report_lines.append(f"- **Duplicate rows (run_id, timestamp_s, junction_id)**: {total_dups} (Gate: must be 0)")
    assert total_dups == 0, f"Gate failed: found {total_dups} duplicate rows!"

    # Gate 3: Timestamp grid continuity
    expected_timestamps = set(range(60, 3601, 60))
    time_continuity_errors = 0
    for rid, grp in full_df.groupby("run_id"):
        observed_ts = set(grp["timestamp_s"].unique())
        if observed_ts != expected_timestamps:
            # Check if smoke run (up to 600s)
            if observed_ts == set(range(60, 601, 60)):
                continue
            time_continuity_errors += 1
    print(f"Timestamp continuity errors: {time_continuity_errors}")
    report_lines.append(f"- **Timestamp continuity errors (60s grid)**: {time_continuity_errors} (Gate: must be 0)")
    assert time_continuity_errors == 0, f"Gate failed: {time_continuity_errors} runs have broken timestamp continuity!"

    # Gate 4: Scenario Demand Separation (post warm-up t > 300)
    post_warmup = full_df[~full_df["is_warmup"]]
    scenario_stats = post_warmup.groupby("scenario").agg(
        mean_queue=("queue_length", "mean"),
        max_queue=("queue_length", "max"),
        mean_speed=("mean_speed_mps", "mean"),
        mean_throughput=("throughput", "mean"),
        mean_waiting=("waiting_time_s", "mean"),
        mean_arrivals=("arrivals", "mean"),
    ).reset_index()

    print("\nScenario Summary (Post Warm-Up t > 300s):")
    print(scenario_stats.to_string(index=False))

    headers = list(scenario_stats.columns)
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    table_lines = [header_line, sep_line]
    for _, row in scenario_stats.iterrows():
        r_str = "| " + " | ".join(f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v) for v in row) + " |"
        table_lines.append(r_str)
    report_lines.append("\n".join(table_lines))

    # Verify low < medium < high < peak ordering for queue
    sc_map = scenario_stats.set_index("scenario")["mean_queue"].to_dict()
    if all(s in sc_map for s in ["low", "medium", "high", "peak"]):
        q_low = sc_map["low"]
        q_med = sc_map["medium"]
        q_high = sc_map["high"]
        q_peak = sc_map["peak"]
        ordered = q_low < q_med < q_high < q_peak
        print(f"\nDemand Check: low ({q_low:.2f}) < med ({q_med:.2f}) < high ({q_high:.2f}) < peak ({q_peak:.2f}) -> {ordered}")
        report_lines.append(f"\n- **Demand check (low < med < high < peak)**: {ordered} (low={q_low:.2f}, med={q_med:.2f}, high={q_high:.2f}, peak={q_peak:.2f})")
        assert ordered, f"Demand ordering check failed: low={q_low}, med={q_med}, high={q_high}, peak={q_peak}"

    # Gate 5: Peak Gridlock Check
    peak_df = post_warmup[post_warmup["scenario"] == "peak"]
    if len(peak_df) > 0:
        final_600s = peak_df[peak_df["timestamp_s"] >= 3000]
        final_tp = final_600s["throughput"].mean()
        print(f"Peak final 600s throughput: {final_tp:.2f} veh/min/junction")
        report_lines.append(f"- **Peak final 600s mean throughput**: {final_tp:.2f} veh/min/junction (Gate: must be > 0)")
        assert final_tp > 0, "Gate failed: peak scenario gridlocked with 0 throughput!"

    # Gate 6: Incident Impact Verification
    inc_runs = full_df[full_df["scenario"] == "incident"]
    if len(inc_runs) > 0:
        inc_active = inc_runs[inc_runs["is_incident_active"]]
        inc_inactive = inc_runs[(inc_runs["timestamp_s"] > 300) & (~inc_runs["is_incident_active"])]
        speed_during_inc = inc_active["mean_speed_mps"].mean()
        speed_outside_inc = inc_inactive["mean_speed_mps"].mean()
        print(f"Incident check: mean speed during incident = {speed_during_inc:.2f} m/s vs outside = {speed_outside_inc:.2f} m/s")
        report_lines.append(f"- **Incident verification**: Speed during incident={speed_during_inc:.2f} m/s vs baseline={speed_outside_inc:.2f} m/s")

    # Gate 7: Snapshot Verification & Reloadability Check
    snap_index_file = STATES_DIR / "snapshot_index.csv"
    if snap_index_file.exists():
        snap_df = pd.read_csv(snap_index_file)
        snap_count = len(snap_df)
        print(f"\nSnapshots indexed: {snap_count}")
        # Test reloading first 3 snapshots
        import traci
        sample_snaps = snap_df["path"].dropna().tolist()[:3]
        reload_successes = 0
        for sp in sample_snaps:
            if Path(sp).exists():
                try:
                    traci.start([str(SUMO_BIN), "-n", str(SUMO_DIR / "hsr.net.xml"), "--no-step-log", "true", "--no-warnings", "true"])
                    traci.simulation.loadState(sp)
                    reload_successes += 1
                    traci.close()
                except Exception as e:
                    try:
                        traci.close()
                    except:
                        pass
                    print(f"[error] Reload failed for {sp}: {e}")
        print(f"Sample snapshot reload test: {reload_successes}/{len(sample_snaps)} succeeded")
        report_lines.append(f"- **Snapshots indexed**: {snap_count}")
        report_lines.append(f"- **Snapshot reload test**: {reload_successes}/{len(sample_snaps)} passed")
        assert reload_successes == len(sample_snaps), "Gate failed: some snapshots failed to reload!"

    # Gate 8: Fixed vs Varied Mode check
    mode_comp = full_df.groupby("mode")["current_plan_id"].nunique().to_dict()
    print(f"Plans observed by mode: {mode_comp}")
    report_lines.append(f"- **Plans used by mode**: {mode_comp} (fixed should be 1, varied should be 4)")
    assert mode_comp.get("fixed", 1) == 1, "Fixed mode unexpectedly used multiple plans!"
    assert mode_comp.get("varied", 0) > 1, "Varied mode failed to switch plans!"

    report_lines.append("\n## 3. Overall Gate Verdict: ALL PASS\n")
    report_md = "\n".join(report_lines)
    out_file = RESULTS_DIR / "data_validation.md"
    out_file.write_text(report_md, encoding="utf-8")
    print(f"\n[ok] Validation report written to {out_file}")
    return True


if __name__ == "__main__":
    run_validation()
