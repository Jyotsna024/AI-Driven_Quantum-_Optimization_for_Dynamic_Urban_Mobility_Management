"""Evidence integrity tests for the frozen S2 held-out diagnosis."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def test_s2_diagnostic_covers_all_heldout_vectors_without_tuning():
    d=json.loads((ROOT/"results/s2_diagnostic_v3.json").read_text())
    rows=pd.read_csv(ROOT/"results/s2_plan_diagnostics_v3.csv")
    assert d["status"]=="SCIENTIFIC_NOT_VALIDATED"
    assert d["tuned_on_test"] is False
    assert len(rows)==4*256
    assert rows.groupby("snapshot_id").plan_vector.nunique().eq(256).all()
    assert set(d["individual_cost_error_by_junction"])=={"J0","J1","J2","J3"}
    assert set(d["joint_objective_by_scenario"])=={"high","incident","peak"}

def test_s2_diagnostic_error_components_reconstruct_joint_residual():
    rows=pd.read_csv(ROOT/"results/s2_plan_diagnostics_v3.csv")
    reconstructed=(rows.baseline_mismatch_s+rows.main_effect_error_s+
                   rows.pair_effect_error_s-rows.higher_order_residual_s)
    assert np.allclose(reconstructed,rows.qubo_error_s,atol=1e-6)
    report=(ROOT/"results/s2_diagnostic_v3.md").read_text()
    assert "SCIENTIFICALLY NOT VALIDATED" in report
    assert "| Scenario | Junction |" in report
