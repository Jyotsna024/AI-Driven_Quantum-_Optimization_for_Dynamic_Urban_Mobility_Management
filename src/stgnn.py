"""
stgnn.py – Phase 3: Spatio-Temporal Graph Neural Network for Traffic Prediction
Architecture:
- Spatial: Graph Convolution (PyG GCNConv) over the 4-junction road adjacency graph.
- Temporal: Gated Recurrent Unit (GRU) capturing temporal traffic dynamics across T past intervals.
- Multi-horizon prediction head: predicts target queue lengths over H future intervals (H=1..3).
- Inference helper: loads trained weights and scaler to generate real-time near-future demand predictions.
"""

import os
import sys
import json
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

JUNCTION_ORDER = ["J0", "J1", "J2", "J3"]
J_TO_IDX = {j: i for i, j in enumerate(JUNCTION_ORDER)}

EDGE_INDEX = torch.tensor([
    [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
    [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2],
], dtype=torch.long)

EDGE_WEIGHTS = torch.tensor([
    358.3, 771.2, 778.0,
    349.0, 649.2, 1127.0,
    776.7, 648.4, 1554.7,
    855.1, 1196.6, 1609.5
], dtype=torch.float32)

NORM_EDGE_WEIGHTS = 100.0 / EDGE_WEIGHTS


class SpatioTemporalGNN(nn.Module):
    def __init__(self, in_features: int = 6, hidden_dim: int = 32, num_nodes: int = 4, horizon: int = 3):
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.hidden_dim = hidden_dim

        self.gcn1 = GCNConv(in_features, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, horizon),
        )

    def forward(self, x_seq, edge_index, edge_weight=None):
        """
        x_seq: [batch_size, num_nodes, seq_len, in_features]
        Output: [batch_size, num_nodes, horizon]
        """
        batch_size, num_nodes, seq_len, in_feat = x_seq.shape
        
        b_edge_indices = []
        b_edge_weights = []
        for b in range(batch_size):
            offset = b * num_nodes
            b_edge_indices.append(edge_index + offset)
            if edge_weight is not None:
                b_edge_weights.append(edge_weight)
                
        batched_edges = torch.cat(b_edge_indices, dim=1)
        batched_weights = torch.cat(b_edge_weights, dim=0) if edge_weight is not None else None

        gru_inputs = []
        for t in range(seq_len):
            xt = x_seq[:, :, t, :].contiguous().view(batch_size * num_nodes, in_feat)
            h = torch.relu(self.gcn1(xt, batched_edges, batched_weights))
            h = torch.relu(self.gcn2(h, batched_edges, batched_weights))
            gru_inputs.append(h.view(batch_size * num_nodes, 1, self.hidden_dim))

        gru_seq = torch.cat(gru_inputs, dim=1)
        _, h_last = self.gru(gru_seq)
        h_last = h_last.squeeze(0)

        preds = self.fc_out(h_last)
        preds = preds.view(batch_size, num_nodes, self.horizon)
        return preds


class STGNNPredictor:
    """Convenience inference class for Phase 4 & 7 TraCI closed loop."""
    def __init__(self, model_path: Path = None, scaler_path: Path = None):
        if model_path is None:
            model_path = MODELS_DIR / "stgnn.pt"
        if scaler_path is None:
            scaler_path = MODELS_DIR / "scaler.pkl"

        assert model_path.exists(), f"Model missing at {model_path}"
        assert scaler_path.exists(), f"Scaler missing at {scaler_path}"

        with open(scaler_path, "rb") as f:
            meta = pickle.load(f)
            self.scaler = meta["feature_scaler"]
            self.target_mean = meta["target_mean"]
            self.target_std = meta["target_std"]
            self.feature_cols = meta["feature_cols"]

        ckpt = torch.load(model_path, map_location="cpu")
        self.model = SpatioTemporalGNN(
            in_features=ckpt["in_features"],
            hidden_dim=ckpt["hidden_dim"],
            num_nodes=ckpt["num_nodes"],
            horizon=ckpt["horizon"]
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def predict(self, raw_features_history):
        """
        raw_features_history: numpy array [num_nodes=4, seq_len=3, num_features=6]
        Returns: predicted queues [num_nodes=4, horizon=3]
        """
        N, T, F = raw_features_history.shape
        flat = raw_features_history.reshape(-1, F)
        normed = self.scaler.transform(flat).reshape(1, N, T, F).astype(np.float32)

        with torch.no_grad():
            preds_norm = self.model(torch.from_numpy(normed), EDGE_INDEX, NORM_EDGE_WEIGHTS).numpy()[0]
            
        preds = preds_norm * self.target_std + self.target_mean
        return np.clip(preds, 0.0, None)
