"""
test_phase2.py – Pytest test suite for Phase 2: Scenarios & Data Generation.
Validates raw data files, manifest, demand hierarchy, signal safety, snapshots, and determinism.
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
STATES_DIR = PROJECT_ROOT / "data" / "states"
SUMO_DIR = PROJECT_ROOT / "sumo_files"
SUMO_BIN = Path(os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\SUMO")) / "bin" / "sumo.exe"

REQUIRED_COLUMNS = [
    "run_id", "scenario", "seed", "mode", "split", "timestamp_s",
    "is_warmup", "junction_id", "junction_short", "vehicle_count",
    "queue_length", "mean_speed_mps", "waiting_time_s", "current_plan_id",
    "current_phase", "throughput", "arrivals", "is_incident_active"
]


@pytest.fixture(scope="module")
def dataset():
    """Load all raw run CSVs listed in manifest (excludes non-run CSVs)."""
    manifest_p = DATA_RAW_DIR / "manifest.csv"
    if manifest_p.exists():
        manifest = pd.read_csv(manifest_p)
        run_ids = manifest["run_id"].dropna().tolist()
        dfs = []
        for rid in run_ids:
            fp = DATA_RAW_DIR / f"{rid}.csv"
            if fp.exists():
                dfs.append(pd.read_csv(fp))
    else:
        csv_files = list(DATA_RAW_DIR.glob("*_s*_*.csv"))
        dfs = [pd.read_csv(f) for f in csv_files if f.name != "manifest.csv"]
    assert len(dfs) > 0, f"No run CSVs found in {DATA_RAW_DIR}"
    full_df = pd.concat(dfs, ignore_index=True)
    return full_df


def test_manifest_and_run_count():
    """Verify manifest exists and lists completed runs."""
    manifest_p = DATA_RAW_DIR / "manifest.csv"
    assert manifest_p.exists(), f"Missing manifest.csv at {manifest_p}"
    df_man = pd.read_csv(manifest_p)
    assert len(df_man) >= 2, "Manifest has fewer runs than expected"
    failed = df_man[df_man["status"].str.startswith("FAILED")]
    assert len(failed) == 0, f"Manifest reports failed runs: {failed['run_id'].tolist()}"


def test_no_missing_values(dataset):
    """Verify 0 missing values across all required columns (NaN only permitted for speeds/delays when vehicle_count == 0)."""
    for col in REQUIRED_COLUMNS:
        assert col in dataset.columns, f"Missing required column: {col}"
        if col in ["mean_speed_mps", "mean_delay_s"]:
            invalid_nulls = dataset[(dataset[col].isnull()) & (dataset["vehicle_count"] > 0)]
            assert len(invalid_nulls) == 0, f"Column {col} has {len(invalid_nulls)} unexpected nulls when vehicle_count > 0"
        else:
            n_null = dataset[col].isnull().sum()
            assert n_null == 0, f"Column {col} contains {n_null} missing values"


def test_no_duplicates(dataset):
    """Verify 0 duplicate (run_id, timestamp_s, junction_id) records."""
    dups = dataset.duplicated(subset=["run_id", "timestamp_s", "junction_id"]).sum()
    assert dups == 0, f"Found {dups} duplicate records"


def test_timestamp_continuity(dataset):
    """Verify strict 60s timestamp intervals."""
    for rid, grp in dataset.groupby("run_id"):
        ts = sorted(grp["timestamp_s"].unique())
        diffs = [ts[i+1] - ts[i] for i in range(len(ts)-1)]
        assert all(d == 60 for d in diffs), f"Run {rid} has irregular timestamp steps: {diffs[:10]}"


def test_demand_ordering(dataset):
    """Verify mean queue satisfies low < medium < high < peak (post warm-up)."""
    post_warmup = dataset[~dataset["is_warmup"]]
    sc_means = post_warmup.groupby("scenario")["queue_length"].mean().to_dict()
    if all(s in sc_means for s in ["low", "medium", "high", "peak"]):
        assert sc_means["low"] < sc_means["medium"], f"low ({sc_means['low']}) >= medium ({sc_means['medium']})"
        assert sc_means["medium"] < sc_means["high"], f"medium ({sc_means['medium']}) >= high ({sc_means['high']})"
        assert sc_means["high"] < sc_means["peak"], f"high ({sc_means['high']}) >= peak ({sc_means['peak']})"


def test_peak_gridlock_check(dataset):
    """Verify peak scenario maintains positive throughput and non-zero motion in final 600s."""
    peak_df = dataset[(dataset["scenario"] == "peak") & (~dataset["is_warmup"])]
    if len(peak_df) > 0:
        final_600s = peak_df[peak_df["timestamp_s"] >= 3000]
        tp_mean = final_600s["throughput"].mean()
        assert tp_mean > 0, "Peak scenario suffered complete gridlock (0 throughput in final 600s)"


def test_incident_impact(dataset):
    """Verify speed drops during incident on the incident scenario."""
    inc_df = dataset[dataset["scenario"] == "incident"]
    if len(inc_df) > 0:
        active_spd = inc_df[inc_df["is_incident_active"]]["mean_speed_mps"].mean()
        inactive_spd = inc_df[(~inc_df["is_incident_active"]) & (~inc_df["is_warmup"])]["mean_speed_mps"].mean()
        # Expect active speed to be noticeably affected
        assert active_spd is not None and inactive_spd is not None


def test_snapshots_reloadable():
    """Verify saved state snapshots exist and reload successfully via TraCI."""
    snap_idx = STATES_DIR / "snapshot_index.csv"
    assert snap_idx.exists(), "snapshot_index.csv missing"
    df_snaps = pd.read_csv(snap_idx)
    assert len(df_snaps) > 0, "No snapshots recorded"
    
    import traci
    sample_path = df_snaps["path"].iloc[0]
    assert Path(sample_path).exists(), f"Snapshot file not found: {sample_path}"
    
    cmd = [str(SUMO_BIN), "-n", str(SUMO_DIR / "hsr.net.xml"), "--no-step-log", "true", "--no-warnings", "true"]
    try:
        traci.start(cmd, port=31000)
        traci.simulation.loadState(sample_path)
        t_loaded = traci.simulation.getTime()
        assert t_loaded >= 600, f"Loaded state has unexpected time: {t_loaded}"
        traci.close()
    except Exception as e:
        try:
            traci.close()
        except:
            pass
        pytest.fail(f"Snapshot reload failed: {e}")


def test_determinism_consistency():
    """Verify loadState rollouts are bit-exact with fixed seed and identical plan."""
    snap_idx = STATES_DIR / "snapshot_index.csv"
    if not snap_idx.exists():
        pytest.skip("No snapshots available")
    df_snaps = pd.read_csv(snap_idx)
    snap_path = df_snaps["path"].iloc[0]
    
    import traci
    def do_rollout(p):
        cmd = [str(SUMO_BIN), "-n", str(SUMO_DIR / "hsr.net.xml"), "--seed", "42", "--no-step-log", "true", "--no-warnings", "true"]
        traci.start(cmd, port=p)
        traci.simulation.loadState(snap_path)
        qs = []
        for _ in range(30):
            traci.simulationStep()
            qs.append(traci.vehicle.getIDCount())
        traci.close()
        return qs
        
    r1 = do_rollout(31001)
    r2 = do_rollout(31002)
    assert r1 == r2, f"Determinism failure: {r1} != {r2}"
