"""Read-only, artifact-backed loaders used by the Streamlit dashboard."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

def load_json(path:Path):
    path=Path(path)
    if not path.exists(): return None
    return json.loads(path.read_text(encoding="utf-8"))

def load_network(path:Path):
    return load_json(path) or {}

def load_phase8(directory:Path,suffix=None):
    directory=Path(directory)
    if suffix is None:
        payload=load_json(directory/"phase8_evaluation.json") or {}
        return pd.DataFrame(payload.get("episodes",[]))
    parts=[]
    for path in sorted(directory.glob(f"*{suffix}")):
        try:
            frame=pd.read_csv(path)
            if not frame.empty: parts.append(frame)
        except (OSError,pd.errors.ParserError,pd.errors.EmptyDataError,UnicodeDecodeError):
            continue
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()

def load_phase7_table(path:Path):
    path=Path(path)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

def load_evaluation_bundle(results_dir:Path):
    """Load the consolidated evaluation, its long-form CSV, and saved graphs.

    Each artifact is optional so a partial checkout still renders useful
    dashboard pages with clear missing-data messages.
    """
    results_dir=Path(results_dir)
    errors=[]
    evaluation=None
    evaluation_path=results_dir/"project_evaluation.json"
    if evaluation_path.is_file():
        try:
            evaluation=load_json(evaluation_path)
        except (OSError,json.JSONDecodeError) as exc:
            errors.append(f"Could not read project_evaluation.json: {exc}")
    else:
        errors.append("results/project_evaluation.json is missing")

    metrics_path=results_dir/"project_evaluation.csv"
    try:
        metrics=pd.read_csv(metrics_path) if metrics_path.is_file() else pd.DataFrame()
    except (OSError,pd.errors.ParserError,pd.errors.EmptyDataError,UnicodeDecodeError) as exc:
        metrics=pd.DataFrame()
        errors.append(f"Could not read project_evaluation.csv: {exc}")
    if not metrics_path.is_file():
        errors.append("results/project_evaluation.csv is missing")

    graph_names=(
        "stgnn_prediction_comparison.png",
        "waiting_time_comparison.png",
        "queue_length_comparison.png",
        "throughput_comparison.png",
        "solver_comparison.png",
    )
    graph_dir=results_dir/"graphs"
    graphs={name:graph_dir/name for name in graph_names if (graph_dir/name).is_file()}
    missing=[name for name in graph_names if name not in graphs]
    if missing:
        errors.append("Missing graph(s): " + ", ".join(missing))
    return {"evaluation":evaluation,"metrics":metrics,"graphs":graphs,"errors":errors}
