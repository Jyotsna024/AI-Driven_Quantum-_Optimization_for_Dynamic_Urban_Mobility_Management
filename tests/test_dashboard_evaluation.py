"""Dashboard data loading and Streamlit section rendering checks."""
from pathlib import Path
import sys

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))
from data import load_evaluation_bundle


def test_loader_reads_consolidated_json_csv_and_graphs():
    bundle = load_evaluation_bundle(ROOT / "results")
    assert bundle["evaluation"]["s2_qubo"]["mean_snapshot_spearman"]["value"] is not None
    assert {"system", "scope", "metric", "value"}.issubset(bundle["metrics"].columns)
    assert "solver_comparison.png" in bundle["graphs"]
    assert "stgnn_prediction_comparison.png" in bundle["graphs"]
    assert bundle["errors"] == []


def test_loader_handles_missing_evaluation_artifacts(tmp_path):
    bundle = load_evaluation_bundle(tmp_path)
    assert bundle["evaluation"] is None
    assert bundle["metrics"].empty
    assert bundle["graphs"] == {}
    assert any("project_evaluation.json is missing" in error for error in bundle["errors"])
    assert any("project_evaluation.csv is missing" in error for error in bundle["errors"])


def test_all_dashboard_sections_render_without_starting_sumo():
    sections = [
        "Project Overview",
        "Traffic Prediction (ST-GNN)",
        "Cost Prediction (S1)",
        "QUBO Optimization (S2)",
        "Solver Comparison (S3)",
        "Traffic Performance",
        "Limitations",
        "Live SUMO",
    ]
    for section in sections:
        app = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=20)
        app.session_state["dashboard_page"] = section
        app.run()
        assert not app.exception, f"{section} render failed: {app.exception}"
        assert len(app.title) == 1, f"{section} did not render its page title"
        assert app.sidebar.radio[0].value == section
