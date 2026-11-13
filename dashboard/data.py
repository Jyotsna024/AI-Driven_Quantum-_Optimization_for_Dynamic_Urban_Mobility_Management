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
