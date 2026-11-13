"""Pure checks for live feature layout, one-hot penalty, and program safety."""
import itertools
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import phase7_controller as controller

def test_runtime_feature_builder_uses_s1_model_schema_and_stgnn_history():
    cfg,tls,incoming,model,predictor,pairwise,meta=controller.load_static()
    rows=[]
    for t in (60,120,180):
        rows.append({j:{"queue_length":t//60,"queue_length_m":3.5,"vehicle_count":4,
          "mean_speed_mps":5.0,"mean_delay_s":2.0,"throughput":1,"arrivals":2} for j in controller.JUNCTIONS})
    forecast,costs=controller.online_features(rows,"low",predictor,model)
    assert forecast.shape==(4,3) and costs.shape==(4,4)
    assert np.isfinite(forecast).all() and np.isfinite(costs).all()
    assert set(tls)==set(controller.JUNCTIONS)
    assert all(incoming[j] for j in controller.JUNCTIONS)

def test_runtime_qubo_penalty_enforces_all_one_hot_groups():
    rng=np.random.default_rng(4821)
    costs=rng.uniform(0,35,(4,4)); pairs=rng.normal(0,0.5,(6,4,4))
    q,a=controller.make_qubo(costs,pairs,10)
    bits=((np.arange(1<<16,dtype=np.uint32)[:,None]>>np.arange(16,dtype=np.uint32))&1).astype(float)
    e=np.einsum("bi,ij,bj->b",bits,q,bits)
    grouped=bits.reshape(-1,4,4).sum(axis=2); feas=(grouped==1).all(axis=1)
    assert feas.sum()==256 and (~feas).sum()==65280
    assert e[~feas].min()>e[feas].max()

def test_controller_smoke_evidence_marks_qaoa_no_sample_fallback():
    d=ROOT/"results/phase7_smoke_qaoa_v1/qaoa_aer_sampled_low_s0_300_metrics.json"
    assert d.exists()
    m=json.loads(d.read_text())
    assert m["solver_calls"]==1
    assert m["raw_solver_samples"]==512
    assert m["feasible_raw_samples"]==0
    assert m["fallback_count"]==1
    assert m["s2_objective_status"]=="SCIENTIFICALLY_NOT_VALIDATED"

def test_max_pressure_smoke_changed_programs_only_at_clearance_boundaries():
    p=ROOT/"results/phase7_smoke_max_v1/max_pressure_low_s0_300_signal_updates.csv"
    assert p.exists()
    updates=pd.read_csv(p)
    assert len(updates)>0
    assert set(updates.transition)=={"cycle boundary after yellow clearance"}
    cfg=json.loads((ROOT/"sumo_files/cluster_config.json").read_text())
    for _,r in updates.iterrows():
        tid=next(t for t,m in cfg["tls_meta"].items() if m["short_id"]==r.junction)
        phases=cfg["plans"][tid][int(r.applied_plan_id)]["phases"]
        assert r.signal_state==phases[0]["state"]
