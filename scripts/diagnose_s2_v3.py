"""Held-out diagnostic decomposition for the frozen S2 v3 objective; no tuning."""
from __future__ import annotations
import itertools, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
J = ["J0", "J1", "J2", "J3"]
PLANS = list(itertools.product(range(4), repeat=4))
PAIRS = list(itertools.combinations(range(4), 2))
S1F = ROOT / "data/features_s1_v2.csv"
S1R = ROOT / "data/rollouts_s1_v2.csv"
JOINT = ROOT / "data/joint_rollouts_s2_v3.csv"
MODEL = ROOT / "models/cost_model_s1_v2.pkl"
QFILE = ROOT / "models/qubo_s2_v3.npz"
META = ROOT / "models/qubo_s2_v3.json"
METRICS = ROOT / "results/qubo_metrics_s2_v3.json"
CSV_OUT = ROOT / "results/s2_plan_diagnostics_v3.csv"
JSON_OUT = ROOT / "results/s2_diagnostic_v3.json"
MD_OUT = ROOT / "results/s2_diagnostic_v3.md"

def corr(x, y):
    if np.ptp(np.asarray(x, float)) == 0 or np.ptp(np.asarray(y, float)) == 0:
        return None
    v = float(spearmanr(x, y).statistic)
    return v if np.isfinite(v) else None

def regression(yhat, y):
    a, b = np.asarray(yhat, float), np.asarray(y, float)
    return {"n": len(a), "mae_s": float(np.mean(np.abs(a-b))),
            "rmse_s": float(np.sqrt(np.mean((a-b)**2))),
            "spearman": corr(a,b)}

def main():
    s1f, s1r, joint = pd.read_csv(S1F), pd.read_csv(S1R), pd.read_csv(JOINT)
    meta, oldm = json.loads(META.read_text()), json.loads(METRICS.read_text())
    model = pickle.loads(MODEL.read_bytes())
    with np.load(QFILE) as f:
        q_by_state, q_ids = f["validation_Q"], f["validation_snapshot_ids"].tolist()
        a_by_state = f["validation_A"].astype(float).tolist()
    if len(joint[(joint.experiment == "full_joint_validation")]) != 4*256*4:
        raise RuntimeError("Held-out rollout grid is incomplete")
    joint["plan_vector"] = joint.plan_vector.map(lambda s: json.dumps(json.loads(s), separators=(",", ":")))
    rows, states, local_rows, comp_rows = [], {}, [], []
    for sid, block in joint[joint.experiment == "full_joint_validation"].groupby("snapshot_id", sort=True):
        scen, split = str(block.iloc[0].scenario), str(block.iloc[0].split)
        outcomes = block.groupby("plan_vector").mean_delay_s.sum().to_dict()
        if len(outcomes) != 256: raise RuntimeError(f"{sid}: expected all 256 outcomes")
        actual = np.array([outcomes[json.dumps(list(p), separators=(",", ":"))] for p in PLANS])
        # Genuine S1 model costs at this snapshot for each target-junction intervention.
        xf = s1f[(s1f.snapshot_id == sid)].copy()
        if len(xf) != 16: raise RuntimeError(f"{sid}: expected 16 S1 feature rows")
        cstate = np.zeros((4,4))
        pred_local = model["model"].predict(model["scaler"].transform(xf[model["feature_cols"]]))
        for pos, (_, r) in enumerate(xf.iterrows()): cstate[J.index(r.target_junction), int(r.applied_plan_id)] = max(0., float(pred_local[pos]))
        singles = s1r[(s1r.snapshot_id == sid)]
        singleton_total = {}
        singleton_local = {}
        for i,jid in enumerate(J):
            for p in range(4):
                g = singles[(singles.target_junction == jid) & (singles.applied_plan_id == p)]
                if len(g) != 4: raise RuntimeError(f"{sid} {jid}/{p}: missing target rollout rows")
                singleton_total[(i,p)] = float(g.mean_delay_s.sum())
                loc = g[g.measured_junction == jid]
                actual_loc = float(loc.mean_delay_s.iloc[0])
                pred_loc = cstate[i,p]
                local_rows.append({"snapshot_id":sid,"scenario":scen,"split":split,"junction":jid,
                                   "plan":p,"predicted_local_delay_s":pred_loc,"actual_local_delay_s":actual_loc,
                                   "local_residual_s":pred_loc-actual_loc})
                singleton_local[(i,p)] = actual_loc
        d0 = float(np.mean([singleton_total[(i,0)] for i in range(4)]))
        main_actual = {(i,p):singleton_total[(i,p)]-d0 for i in range(4) for p in range(4)}
        # Exact pair corrections on held-out joint SUMO outcomes, isolated with other plans at baseline.
        pair_actual = {}
        for pi,(i,j) in enumerate(PAIRS):
            for a in range(4):
                for b in range(4):
                    v=[0,0,0,0]; v[i],v[j]=a,b
                    yij=outcomes[json.dumps(v,separators=(",",":"))]
                    pair_actual[(i,j,a,b)] = yij-d0-main_actual[(i,a)]-main_actual[(j,b)]
        pair_train=np.asarray(meta["pairwise_array_s"],float)
        offset=float(meta["offset_s"])
        qidx=q_ids.index(sid); q=q_by_state[qidx]; astate=a_by_state[qidx]
        add_pred=[]; pair_pred=[]; full_pred=[]; high=[]; actual_pair_sum=[]; train_pair_sum=[]; main_mismatch=[]
        for vec,y in zip(PLANS,actual):
            inc_pred=sum(cstate[i,p]-cstate[i,0] for i,p in enumerate(vec))
            inc_actual=sum(main_actual[(i,p)] for i,p in enumerate(vec))
            jp=sum(pair_train[k,vec[i],vec[j]] for k,(i,j) in enumerate(PAIRS))
            ja=sum(pair_actual[(i,j,vec[i],vec[j])] for i,j in PAIRS)
            h=float(y-d0-inc_actual-ja)
            baseline_error=offset+float(cstate[:,0].sum())-d0
            linerr=inc_pred-inc_actual
            pairerr=jp-ja
            # Additive and full QUBO costs have same training-derived baseline offset.
            ap=offset+float(cstate[:,0].sum())+inc_pred
            fp=ap+jp
            add_pred.append(ap); pair_pred.append(jp); full_pred.append(fp)
            high.append(h); actual_pair_sum.append(ja); train_pair_sum.append(jp)
            main_mismatch.append(linerr)
            rows.append({"snapshot_id":sid,"scenario":scen,"split":split,"plan_vector":json.dumps(vec),
                         "actual_joint_delay_s":float(y),"additive_prediction_s":ap,
                         "trained_pair_sum_s":float(jp),"qubo_prediction_s":fp,
                         "actual_pair_sum_s":float(ja),"higher_order_residual_s":h,
                         "baseline_mismatch_s":baseline_error,"main_effect_error_s":float(linerr),
                         "pair_effect_error_s":float(pairerr),"qubo_error_s":float(fp-y)})
        # Validate Q arithmetic against the independently reconstructed expression.
        x=np.zeros((256,16))
        for n,v in enumerate(PLANS): x[n,np.arange(4)*4+np.asarray(v)]=1
        qhat=np.einsum("bi,ij,bj->b",x,q,x)+4*astate+offset
        if not np.allclose(qhat,full_pred,atol=1e-6): raise RuntimeError(f"{sid}: QUBO prediction decomposition mismatch")
        best_actual=int(np.argmin(actual)); best_add=int(np.argmin(add_pred)); best_q=int(np.argmin(full_pred))
        states[sid]={"scenario":scen,"split":split,"seed":int(block.iloc[0].seed),"n_plan_vectors":256,
          "actual_joint_vs_additive":regression(add_pred,actual),"actual_joint_vs_full_qubo":regression(full_pred,actual),
          "additive_spearman":corr(add_pred,actual),"qubo_spearman":corr(full_pred,actual),
          "additive_best_vector":list(PLANS[best_add]),"qubo_best_vector":list(PLANS[best_q]),
          "sumo_best_vector":list(PLANS[best_actual]),"additive_regret_s":float(actual[best_add]-actual[best_actual]),
          "qubo_regret_s":float(actual[best_q]-actual[best_actual]),
          "actual_system_baseline_s":d0,"qubo_baseline_prediction_s":offset+float(cstate[:,0].sum()),
          "baseline_mismatch_s":float(offset+cstate[:,0].sum()-d0),
          "main_effect_error":regression(main_mismatch,np.zeros(256)),
          "trained_pair_vs_measured_pair":regression(train_pair_sum,actual_pair_sum),
          "pair_error_rmse_s":float(np.sqrt(np.mean((np.asarray(train_pair_sum)-actual_pair_sum)**2))),
          "higher_order_residual":{"mean_s":float(np.mean(high)),"mae_s":float(np.mean(np.abs(high))),
                                   "rmse_s":float(np.sqrt(np.mean(np.asarray(high)**2))),
                                   "max_abs_s":float(np.max(np.abs(high)))}}
        for i,jid in enumerate(J):
            for p in range(4):
                comp_rows.append({"snapshot_id":sid,"scenario":scen,"split":split,"junction":jid,"plan":p,
                  "predicted_plan_increment_s":float(cstate[i,p]-cstate[i,0]),
                  "actual_system_increment_s":float(main_actual[(i,p)]),
                  "increment_error_s":float(cstate[i,p]-cstate[i,0]-main_actual[(i,p)])})
    plan_df=pd.DataFrame(rows); local_df=pd.DataFrame(local_rows); comp_df=pd.DataFrame(comp_rows)
    plan_df.to_csv(CSV_OUT,index=False)
    def grouped_local(group_cols):
        return {"|".join(map(str,k if isinstance(k,tuple) else (k,))): regression(g.predicted_local_delay_s,g.actual_local_delay_s)
                for k,g in local_df.groupby(group_cols,sort=True)}
    by_scenario={}
    for scen,g in plan_df.groupby("scenario",sort=True):
        by_scenario[scen]={"n_state_vectors":len(g),"actual_joint_vs_additive":regression(g.additive_prediction_s,g.actual_joint_delay_s),
          "actual_joint_vs_qubo":regression(g.qubo_prediction_s,g.actual_joint_delay_s),
          "additive_spearman":corr(g.additive_prediction_s,g.actual_joint_delay_s),"qubo_spearman":corr(g.qubo_prediction_s,g.actual_joint_delay_s),
          "mean_qubo_regret_s":float(np.mean([states[s]["qubo_regret_s"] for s in states if states[s]["scenario"]==scen]))}
    by_junction={jid:regression(g.predicted_local_delay_s,g.actual_local_delay_s) for jid,g in local_df.groupby("junction",sort=True)}
    by_scenario_junction={}
    for (sc,jid),g in local_df.groupby(["scenario","junction"],sort=True):
        by_scenario_junction[f"{sc}|{jid}"]=regression(g.predicted_local_delay_s,g.actual_local_delay_s)
    component_by_scenario={}
    for sc,g in comp_df.groupby("scenario",sort=True):
        component_by_scenario[sc]={jid:{"n":len(h),"increment_mae_s":float(np.mean(np.abs(h.increment_error_s))),
             "increment_rmse_s":float(np.sqrt(np.mean(h.increment_error_s**2))),
             "predicted_mean_increment_s":float(h.predicted_plan_increment_s.mean()),
             "actual_mean_system_increment_s":float(h.actual_system_increment_s.mean())}
             for jid,h in g.groupby("junction",sort=True)}
    validation=[v for v in states.values() if v["split"]=="val"]
    testing=[v for v in states.values() if v["split"]=="test"]
    doc={"status":"SCIENTIFIC_NOT_VALIDATED","tuned_on_test":False,
      "data":"Frozen S2 v3 QUBO, held-out joint SUMO grid, S1 v2 cost model and held-out single-plan rollouts.",
      "decomposition":"baseline + S1-predicted local increments + train-mean pair terms; residual decomposed against observed single and pair system effects into baseline mismatch, main-effect error, pair-effect error, and higher-order residual.",
      "states":states,"validation_mean_qubo_spearman":float(np.mean([v["qubo_spearman"] for v in validation])),
      "test_mean_qubo_spearman":float(np.mean([v["qubo_spearman"] for v in testing])),
      "validation_mean_qubo_regret_s":float(np.mean([v["qubo_regret_s"] for v in validation])),
      "test_mean_qubo_regret_s":float(np.mean([v["qubo_regret_s"] for v in testing])),
      "individual_cost_error_by_junction":by_junction,"individual_cost_error_by_scenario_and_junction":by_scenario_junction,
      "joint_objective_by_scenario":by_scenario,"system_increment_component_by_scenario_and_junction":component_by_scenario,
      "conclusion":"The frozen QUBO does not preserve held-out SUMO ranking consistently. Investigate main-effect scale/spillover and omitted higher-order effects; pair-error and higher-order residuals are quantified per state. No test-data tuning was performed. S2 remains scientifically NOT VALIDATED; stop tuning.",
      "outputs":{"plan_table":str(CSV_OUT.relative_to(ROOT)),"report":str(MD_OUT.relative_to(ROOT))}}
    JSON_OUT.write_text(json.dumps(doc,indent=2,allow_nan=False),encoding="utf-8")
    lines=["# S2 v3 Held-out Diagnosis","","**Status: SCIENTIFICALLY NOT VALIDATED.** Frozen model only; no tuning on validation or test outcomes.","",
      "The table decomposes QUBO error into baseline mismatch, per-junction main-effect error, train-learned pair error, and omitted higher-order residual. Pair effects and higher-order residuals are estimated from the complete held-out 256-plan SUMO grid.","",
      "| State | Additive Spearman | QUBO Spearman | QUBO regret (s) | Pair RMSE (s) | Higher-order RMSE (s) |","|---|---:|---:|---:|---:|---:|"]
    for sid,v in states.items(): lines.append(f"| {sid} | {v['additive_spearman']:.3f} | {v['qubo_spearman']:.3f} | {v['qubo_regret_s']:.3f} | {v['pair_error_rmse_s']:.3f} | {v['higher_order_residual']['rmse_s']:.3f} |")
    lines += ["",f"Mean validation QUBO Spearman: {doc['validation_mean_qubo_spearman']:.3f}; mean validation regret: {doc['validation_mean_qubo_regret_s']:.3f} s.",
      f"Mean test QUBO Spearman: {doc['test_mean_qubo_spearman']:.3f}; mean test regret: {doc['test_mean_qubo_regret_s']:.3f} s.","",
      "## Error sources","",f"Individual S1 costs by junction: `{json.dumps(by_junction,sort_keys=True)}`.","",
      "Per-state component diagnostics and scenario/junction tables are in `s2_diagnostic_v3.json`; the 1,024 plan-level rows are in `s2_plan_diagnostics_v3.csv`.","",
      "| Scenario | Junction | Local cost MAE (s) | Predicted increment MAE (s) | Mean predicted / actual system increment (s) |",
      "|---|---|---:|---:|---:|"]
    for sc, js in component_by_scenario.items():
        for jid, m in js.items():
            loc=by_scenario_junction[f"{sc}|{jid}"]
            lines.append(f"| {sc} | {jid} | {loc['mae_s']:.3f} | {m['increment_mae_s']:.3f} | {m['predicted_mean_increment_s']:.3f} / {m['actual_mean_system_increment_s']:.3f} |")
    lines += ["", "The additive model is equal or better on the high and peak snapshots; pair terms only slightly change the incident rankings. Large J0 single-cost errors and J0/J2 main-effect increment errors are evident, especially for incident and peak demand. The residual decomposition shows pair-term mismatch is large at peak, while higher-order residual is also material there. On incident snapshots, poor ranking is mainly present even before pair terms; the tested higher-order residual is smaller than the main-effect error. These held-out findings are diagnostic only; no tuning followed."]
    MD_OUT.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"Wrote {JSON_OUT.name}, {MD_OUT.name}, {CSV_OUT.name}; S2 scientifically NOT VALIDATED")

if __name__ == "__main__": main()
