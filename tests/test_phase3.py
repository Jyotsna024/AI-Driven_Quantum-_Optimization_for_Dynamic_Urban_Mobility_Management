"""
test_phase3.py – Test suite for Phase 3: Spatio-Temporal GNN Traffic Prediction.
Validates model artifacts, feature scalers, predictor inference, and benchmark evaluations.
"""

import os
import sys
import json
import pickle
from pathlib import Path
import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stgnn import SpatioTemporalGNN, STGNNPredictor, JUNCTION_ORDER

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


def test_artifacts_exist():
    """Verify all required Phase 3 model and evaluation artifacts exist."""
    assert (MODELS_DIR / "stgnn.pt").exists(), "models/stgnn.pt missing"
    assert (MODELS_DIR / "scaler.pkl").exists(), "models/scaler.pkl missing"
    assert (MODELS_DIR / "metrics.json").exists(), "models/metrics.json missing"
    assert (RESULTS_DIR / "stgnn_predictions.png").exists(), "results/stgnn_predictions.png missing"
    assert (RESULTS_DIR / "stgnn_report.md").exists(), "results/stgnn_report.md missing"


def test_scaler_validity():
    """Verify scaler loads correctly and contains valid feature metadata."""
    with open(MODELS_DIR / "scaler.pkl", "rb") as f:
        meta = pickle.load(f)
    assert "feature_scaler" in meta
    assert "target_mean" in meta and "target_std" in meta
    assert meta["target_std"] > 0, "Target standard deviation must be positive"
    assert len(meta["feature_cols"]) == 6
    assert meta["junction_order"] == ["J0", "J1", "J2", "J3"]


def test_predictor_inference():
    """Verify STGNNPredictor loads and produces valid multi-horizon predictions."""
    predictor = STGNNPredictor()
    dummy_input = np.ones((4, 3, 6), dtype=np.float32)
    # Dummy traffic: queue=5, count=10, speed=10, delay=5, throughput=15, arrivals=15
    dummy_input[:, :, 0] = 5.0
    dummy_input[:, :, 1] = 10.0
    dummy_input[:, :, 2] = 10.0
    dummy_input[:, :, 3] = 5.0
    dummy_input[:, :, 4] = 15.0
    dummy_input[:, :, 5] = 15.0

    preds = predictor.predict(dummy_input)
    assert preds.shape == (4, 3), f"Expected prediction shape (4, 3), got {preds.shape}"
    assert not np.isnan(preds).any(), "Predictions contain NaN values"
    assert (preds >= 0).all(), "Predicted queue lengths must be non-negative"


def test_benchmark_metrics():
    """Verify ST-GNN benchmark metrics beat or remain competitive against baselines."""
    with open(MODELS_DIR / "metrics.json", "r") as f:
        metrics = json.load(f)

    assert "ST-GNN" in metrics
    assert "Persistence" in metrics
    assert "Historical_Average" in metrics

    stgnn_mae = metrics["ST-GNN"]["Overall_MAE"]
    ha_mae = metrics["Historical_Average"]["Overall_MAE"]
    assert stgnn_mae < ha_mae, f"ST-GNN MAE ({stgnn_mae}) must outperform Historical Average ({ha_mae})"
    assert metrics["ST-GNN"]["H1_MAE"] > 0
    assert metrics["ST-GNN"]["H2_MAE"] > 0
    assert metrics["ST-GNN"]["H3_MAE"] > 0
