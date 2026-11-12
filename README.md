# AI-Driven Quantum Optimization for Dynamic Urban Mobility Management
## Capstone Research Prototype – HSR Layout, Bengaluru

> **Scope**: 4 signalised intersections, SUMO simulation, ST-GNN traffic prediction,
> QUBO-based signal plan optimisation solved by Simulated Annealing and QAOA.
> No live traffic signals, no city-scale claims, no guaranteed quantum advantage.

---

## Windows Setup (exact steps)

### Prerequisites
- **SUMO 1.28.0** installed at `C:\Program Files (x86)\SUMO\`; env var `SUMO_HOME` set
- **Python 3.13** (tested with 3.13.7)
- Internet access for Overpass OSM download (Phase 1 only)

### Install dependencies
```powershell
cd C:\Users\Jyotsna\Desktop\Capstone
pip install -r requirements.txt
```

> **Note**: `pyosmium` does not support Python 3.13; OSM data is fetched via
> the Overpass HTTP API using `requests` instead.

### Environment variables (optional, for real quantum hardware)
```powershell
# Never commit these – set in your shell only
$env:DWAVE_API_TOKEN = "your-token-here"      # enables D-Wave Leap hybrid
$env:IBMQ_TOKEN = "your-token-here"           # enables IBM Quantum
```

---

## Run the Pipeline

```powershell
# Full pipeline
python run_all.py --all

# Single phase
python run_all.py --phase 1

# Multiple phases
python run_all.py --phase 1 2 3

# Quick smoke test (short simulations)
python run_all.py --all --smoke
```

## Run Tests
```powershell
pytest tests/ -v
```

## Launch Dashboard (Phase 9+)
```powershell
streamlit run dashboard/app.py
```

---

## Directory Structure
```
Capstone/
├── PROJECT_SPEC.md     ← decisions, checklist, phase reports
├── README.md           ← this file
├── requirements.txt
├── run_all.py          ← pipeline entry point
├── configs/            ← YAML configs (base, scenarios, qubo)
├── sumo_files/         ← OSM, net.xml, routes, .sumocfg
├── data/
│   ├── raw/            ← untouched SUMO output (never modified)
│   └── processed/      ← cleaned features for ML
├── src/                ← pipeline modules (one per phase)
├── tests/              ← pytest suites (one per phase)
├── models/             ← saved ST-GNN model + scaler
├── results/            ← evaluation CSVs + figures
├── logs/               ← run logs
└── dashboard/          ← Streamlit app (Phase 9)
```

---

## Reproducibility
All seeds are fixed (see `configs/base.yaml`).
All results come from executed SUMO simulations; no data is fabricated.
Simulator results and hardware results are reported and labelled separately.

---

## Honest Limitations
- Quantum advantage is NOT claimed. SA and QAOA results are compared objectively.
- QAOA runs on the Qiskit Aer **simulator**; real-hardware results only if tokens provided.
- Network is a small synthetic cluster; not a city-scale deployment.
- ST-GNN prediction horizon is 1–3 control intervals (60–180 simulated seconds).
