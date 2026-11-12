# src/cost_model.py
"""
Cost model for traffic signal plans.
Implements Webster/HCM delay approximation.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict

# Constants (could be tuned during calibration)
SATURATION_FLOW = 1800  # veh/h per lane (default)
CYCLE_LENGTH = 90.0  # seconds, common for all TLS

def webster_delay(green: float, flow: float, saturation: float = SATURATION_FLOW) -> float:
    """Compute average delay per vehicle using Webster's formula.
    green: effective green time (s)
    flow: arrival flow (veh/h)
    saturation: saturation flow (veh/h)
    Returns delay in seconds.
    """
    v_c = flow / saturation
    if v_c >= 1.0:
        # oversaturated, use high penalty
        return 999.0
    # Webster's delay (approximate)
    d = (CYCLE_LENGTH * (1 - green / CYCLE_LENGTH) ** 2) / (2 * (1 - min(v_c, 1.0)))
    return d

def compute_cost(snapshot_path: str, plan_id: int) -> Tuple[float, Dict[str, float]]:
    """Compute total cost for a given snapshot and plan.
    Returns total cost and per‑junction breakdown.
    """
    df = pd.read_csv(snapshot_path)
    # Expect columns: junction, lane, flow_veh_per_h, green_time_s
    costs = {}
    total = 0.0
    for _, row in df.iterrows():
        delay = webster_delay(row["green_time_s"], row["flow_veh_per_h"])
        costs[row["junction"]] = costs.get(row["junction"], 0.0) + delay
        total += delay
    return total, costs

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python -m src.cost_model <snapshot_csv> <plan_id>")
        sys.exit(1)
    total, breakdown = compute_cost(sys.argv[1], int(sys.argv[2]))
    print(f"Total cost: {total:.2f}s")
    for j, c in breakdown.items():
        print(f"{j}: {c:.2f}s")
