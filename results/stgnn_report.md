# Phase 3: Spatio-Temporal GNN (ST-GNN) Traffic Prediction Report

## 1. Executive Summary
- **Model**: Spatio-Temporal Graph Neural Network (`SpatioTemporalGNN`)
- **Spatial Topology**: 4 nodes (J0, J1, J2, J3) connected by 12 physical directed road links with distance-weighted edges.
- **Temporal Dynamics**: GCN feature propagation across lookback window $T=3$ (180s) coupled with Gated Recurrent Units (GRU, hidden dimension 32) and multi-horizon projection head.
- **Prediction Target**: Multi-horizon intersection queue lengths ($H=1$: 60s, $H=2$: 120s, $H=3$: 180s ahead).
- **Split Strategy**: Strictly partitioned by simulation random seed:
  - **Train**: Seeds 0, 1, 2 (30 runs, 1,650 sequence windows)
  - **Validation**: Seed 3 (10 runs, 550 sequence windows)
  - **Held-Out Test**: Seed 4 (10 runs, 550 sequence windows)

---

## 2. Test Set Benchmark Comparison (Held-Out Seed 4)

| Model Architecture | Overall MAE (veh) | Overall RMSE (veh) | H1 MAE (60s) | H2 MAE (120s) | H3 MAE (180s) |
|---|---|---|---|---|---|
| **ST-GNN (Ours)** | **2.41** | **3.63** | **2.378** | **2.441** | **2.41** |
| **Persistence Baseline** | 3.544 | 5.348 | 3.892 | 4.024 | 2.718 |
| **Historical Average** | 6.017 | 10.311 | 5.959 | 6.021 | 6.07 |

---

## 3. Analysis & Key Takeaways
1. **Dynamic Tracking**: The ST-GNN captures spatial wave propagation between adjacent intersections along 14th Main and 17th Cross, achieving lower error compared to static historical averages and reactive persistence.
2. **Multi-Horizon Horizon Scaling**: Error gracefully scales across the horizons ($H_1=2.378 \rightarrow H_2=2.441 \rightarrow H_3=2.41$), maintaining tight predictive bounds for the Phase 4/5 QUBO control formulation.
3. **Artifact Integrity**:
   - Model weights saved at `models/stgnn.pt`
   - Fitted StandardScaler saved at `models/scaler.pkl`
   - Evaluation metrics serialized at `models/metrics.json`
   - Prediction visualization saved at `results/stgnn_predictions.png`
