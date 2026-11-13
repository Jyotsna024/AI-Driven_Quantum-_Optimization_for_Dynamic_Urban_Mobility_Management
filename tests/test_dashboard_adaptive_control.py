from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "dashboard" / "app.py"


def test_dashboard_sections_render_and_live_page_exposes_automatic_control():
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    radio = app.radio(key="dashboard_page")
    for page in (
        "Project Overview",
        "Traffic Prediction (ST-GNN)",
        "Cost Prediction (S1)",
        "QUBO Optimization (S2)",
        "Solver Comparison (S3)",
        "Traffic Performance",
        "Limitations",
        "Live SUMO",
    ):
        radio.set_value(page).run()
        assert not app.exception, f"{page} raised: {[x.value for x in app.exception]}"

    assert any(toggle.label == "Automatic AI Control" for toggle in app.toggle)
    assert {widget.label for widget in app.selectbox} == {"Scenario", "Route seed"}
    text = " ".join(item.value for item in app.markdown)
    assert "AI control ON" in text

