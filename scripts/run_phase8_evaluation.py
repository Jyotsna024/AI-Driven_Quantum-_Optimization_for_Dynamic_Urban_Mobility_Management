"""Resumable identical-condition SUMO evaluation for four Phase 7 controllers."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from phase7_controller import run_episode,JUNCTIONS,S2_STATUS
SCENARIOS=["low","medium","high","peak","incident"]
SEEDS=[0,1]
CONTROLLERS=["fixed_time","max_pressure","sa_qubo","qaoa_aer_sampled"]
OUT=ROOT/"results/phase8_eval_v1"

def atomic_json(path,payload):
    tmp=path.with_suffix(path.suffix+".partial")
    tmp.write_text(json.dumps(payload,indent=2,allow_nan=False),encoding="utf-8")
    tmp.replace(path)

def _save_episode(out,runid,metrics,telemetry,decisions,updates,forecasts):
    atomic_json(out/f"{runid}_metrics.json",metrics)
    for name,rows in (("telemetry",telemetry),("decisions",decisions),("signal_updates",updates),("forecasts",forecasts)):
        pd.DataFrame(rows).to_csv(out/f"{runid}_{name}.csv",index=False)

def _read_completed(out,runid):
    p=out/f"{runid}_metrics.json"
    if not p.exists(): return None
    try:
        d=json.loads(p.read_text())
        return d if d.get("status")=="COMPLETE" else None
    except Exception: return None

def _aggregate(out,episodes):
    df=pd.DataFrame(episodes)
    if df.empty: raise RuntimeError("No completed evaluation episodes")
    df.to_csv(out/"episode_metrics.csv",index=False)
    numeric=["total_waiting_time_vehicle_s","mean_completed_trip_travel_time_s","total_time_loss_s",
      "average_queue_vehicles","maximum_queue_vehicles","average_queue_m","maximum_queue_m",
      "throughput_total_junction_crossings","mean_solver_runtime_s","total_solver_runtime_s",
      "raw_sample_feasibility_rate","fallback_rate","forecast_mae_vehicles"]
    summaries={}
    for controller,g in df.groupby("controller",sort=True):
        summaries[controller]={}
        for col in numeric:
            v=pd.to_numeric(g[col],errors="coerce").dropna()
            summaries[controller][col]={"n":int(len(v)),"mean":float(v.mean()) if len(v) else None,
              "std":float(v.std(ddof=1)) if len(v)>1 else (0. if len(v)==1 else None)}
        summaries[controller]["fallback_count_total"]=int(g.fallback_count.sum())
        summaries[controller]["decision_count_total"]=int(g.decision_count.sum())
        summaries[controller]["raw_solver_samples_total"]=int(g.raw_solver_samples.sum())
        summaries[controller]["feasible_raw_samples_total"]=int(g.feasible_raw_samples.sum())
    paired=[]
    for (sc,seed),group in df.groupby(["scenario","seed"],sort=True):
        base=group[group.controller=="fixed_time"]
        if len(base)!=1: continue
        b=base.iloc[0]
        for _,r in group[group.controller!="fixed_time"].iterrows():
            paired.append({"scenario":sc,"seed":int(seed),"controller":r.controller,
              "waiting_time_delta_vehicle_s":float(r.total_waiting_time_vehicle_s-b.total_waiting_time_vehicle_s),
              "travel_time_delta_s":None if pd.isna(r.mean_completed_trip_travel_time_s) or pd.isna(b.mean_completed_trip_travel_time_s) else float(r.mean_completed_trip_travel_time_s-b.mean_completed_trip_travel_time_s),
              "time_loss_delta_s":float(r.total_time_loss_s-b.total_time_loss_s),
              "average_queue_delta_vehicles":float(r.average_queue_vehicles-b.average_queue_vehicles),
              "throughput_delta_crossings":int(r.throughput_total_junction_crossings-b.throughput_total_junction_crossings)})
    pairdf=pd.DataFrame(paired); pairdf.to_csv(out/"paired_controller_deltas.csv",index=False)
    pair_summary={}
    if not pairdf.empty:
        for controller,g in pairdf.groupby("controller",sort=True):
            pair_summary[controller]={}
            for col in [c for c in pairdf.columns if c.endswith("_delta_vehicle_s") or c.endswith("_delta_s") or c.endswith("_delta_vehicles") or c.endswith("_delta_crossings")]:
                values=pd.to_numeric(g[col],errors="coerce").dropna()
                pair_summary[controller][col]={"n_pairs":int(len(values)),"mean":float(values.mean()) if len(values) else None,
                    "std":float(values.std(ddof=1)) if len(values)>1 else None}
    def clean(v):
        if isinstance(v,dict): return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)): return [clean(x) for x in v]
        if isinstance(v,np.generic): v=v.item()
        if isinstance(v,float) and not np.isfinite(v): return None
        if pd.isna(v) if not isinstance(v,(dict,list,tuple)) else False: return None
        return v
    payload={"status":"COMPLETE" if len(df)==len(SCENARIOS)*len(SEEDS)*len(CONTROLLERS) else "INCOMPLETE",
      "evaluation_version":"phase8_eval_v1","n_expected_episodes":len(SCENARIOS)*len(SEEDS)*len(CONTROLLERS),
      "n_completed_episodes":len(df),"scenarios":SCENARIOS,"seeds":SEEDS,"controllers":CONTROLLERS,
      "identical_scenario_seed_initial_conditions":True,"duration_s":3600,"decision_interval_s":90,
      "qaoa_sample_times_s":[900,1800,2700],"s2_objective_status":S2_STATUS,
      "results_are_simulation_only":True,"no_optimization_improvement_claim":True,
      "controller_summary":summaries,"paired_deltas_vs_fixed_time":pair_summary,
      "episodes":[{k:clean(v) for k,v in r.items()} for r in df.to_dict("records")]}
    atomic_json(out/"phase8_evaluation.json",payload)
    return payload,df,pairdf

def _plots(out,df):
    labels={"fixed_time":"Fixed time","max_pressure":"Max pressure","sa_qubo":"SA + QUBO","qaoa_aer_sampled":"QAOA Aer (sampled)"}
    fields=[("mean_completed_trip_travel_time_s","Mean completed-trip travel time (s)"),
            ("average_queue_vehicles","Mean queue (vehicles)"),
            ("total_time_loss_s","Episode time loss (s)"),
            ("throughput_total_junction_crossings","Junction crossings")]
    for col,title in fields:
        stats=df.groupby("controller")[col].agg(["mean","std"]).reindex(CONTROLLERS)
        fig,ax=plt.subplots(figsize=(9,5)); ax.bar([labels[x] for x in stats.index],stats["mean"],yerr=stats["std"].fillna(0),capsize=4,color="#4c78a8")
        ax.set_ylabel(title); ax.set_title(title+" across identical scenario-seed episodes"); ax.tick_params(axis="x",rotation=15); ax.grid(axis="y",alpha=.2); fig.tight_layout()
        fig.savefig(out/f"{col}_by_controller.png",dpi=160); plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(11,8))
    for ax,(col,title) in zip(axs.ravel(),[("average_queue_vehicles","Average queue"),("maximum_queue_vehicles","Maximum queue"),("total_waiting_time_vehicle_s","Waiting time (veh-s)"),("mean_completed_trip_travel_time_s","Travel time (s)")]):
        data=[pd.to_numeric(df[df.controller==c][col],errors="coerce").dropna().values for c in CONTROLLERS]
        ax.boxplot(data,tick_labels=[labels[c] for c in CONTROLLERS]);ax.set_title(title);ax.tick_params(axis="x",rotation=20);ax.grid(axis="y",alpha=.2)
    fig.tight_layout();fig.savefig(out/"controller_metrics_boxplots.png",dpi=160);plt.close(fig)

def _report(out,payload,df,pairdf):
    lines=["# Phase 8 Controller Evaluation","",f"**Status:** {payload['status']} ({payload['n_completed_episodes']}/{payload['n_expected_episodes']} episodes).",
      "","Evaluation is SUMO-only. Every controller uses the same scenario, seed, route file, duration, and SUMO seed within each paired condition. Optimization results are experimental because S2 v3 is scientifically not validated.",
      "No claim of optimization improvement is made from these results.","",
      "## Controller metrics","","| Controller | Travel time mean ± SD (s) | Waiting time mean ± SD (veh-s) | Avg queue | Max queue | Throughput crossings | Delay (time loss, s) | Solver runtime mean (s) | Raw feasibility | Fallbacks / decisions |",
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    names={"fixed_time":"Fixed time","max_pressure":"Max pressure","sa_qubo":"SA + QUBO","qaoa_aer_sampled":"QAOA Aer sampled"}
    for c,g in df.groupby("controller",sort=False):
        def m(col):
            x=pd.to_numeric(g[col],errors="coerce").dropna()
            return "n/a" if x.empty else f"{x.mean():.2f} ± {x.std(ddof=1) if len(x)>1 else 0.:.2f}"
        feas=pd.to_numeric(g.raw_sample_feasibility_rate,errors="coerce").dropna()
        ftxt="n/a" if feas.empty else f"{feas.mean():.3f}"
        lines.append(f"| {names[c]} | {m('mean_completed_trip_travel_time_s')} | {m('total_waiting_time_vehicle_s')} | {m('average_queue_vehicles')} | {m('maximum_queue_vehicles')} | {m('throughput_total_junction_crossings')} | {m('total_time_loss_s')} | {m('mean_solver_runtime_s')} | {ftxt} | {int(g.fallback_count.sum())}/{int(g.decision_count.sum())} |")
    lines += ["","## Paired differences vs fixed time","","Negative values mean lower measured outcome for the listed delta; they do not imply statistical or causal significance.","",
      "| Controller | Waiting-time delta (veh-s) | Travel-time delta (s) | Time-loss delta (s) | Queue delta (veh) | Throughput delta |",
      "|---|---:|---:|---:|---:|---:|"]
    for c,g in pairdf.groupby("controller",sort=True):
        lines.append(f"| {names[c]} | {g.waiting_time_delta_vehicle_s.mean():.2f} | {g.travel_time_delta_s.mean():.2f} | {g.time_loss_delta_s.mean():.2f} | {g.average_queue_delta_vehicles.mean():.2f} | {g.throughput_delta_crossings.mean():.1f} |")
    lines += ["","## Artifacts","", "- `phase8_evaluation.json`: full run configuration, episode metrics, and summary.","- `episode_metrics.csv` and `paired_controller_deltas.csv`.","- Per-episode telemetry, decision logs, applied-signal updates, and forecast CSVs.","- Controller comparison graphs are saved as PNG files in this directory.","","## Limitations","",
      "- Two seeds per scenario provide a small sample; means and standard deviations are descriptive.",
      "- Waiting time is integrated stopped-vehicle seconds on the four selected junction approaches. Travel time is mean for completed route vehicles; incomplete trips are excluded.",
      "- Delay is SUMO vehicle time loss for observed vehicles on the selected approaches. Junction throughput counts approach-lane exits and can count the same trip at multiple junctions.",
      "- S2 v3 failed held-out ranking/regret diagnosis. SA/QAOA controller actions are exploratory; fallbacks and feasibility are reported.",
      "- Plan 3 may be rejected by the safe-switch check when its offset would change the current signal state; baseline is queued and the rejection is logged.",
      "- QAOA Aer is classical simulation; no quantum hardware or quantum advantage is claimed."]
    (out/"PHASE8_EVALUATION_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--smoke",action="store_true");parser.add_argument("--out",default="results/phase8_eval_v1");parser.add_argument("--max-episodes",type=int,default=None);a=parser.parse_args()
    out=ROOT/a.out;out.mkdir(parents=True,exist_ok=True)
    matrix=[(s,seed,c) for s in SCENARIOS for seed in SEEDS for c in CONTROLLERS]
    if a.smoke: matrix=[("low",0,c) for c in CONTROLLERS]
    if a.max_episodes: matrix=matrix[:a.max_episodes]
    manifest_path=out/"run_manifest.json"
    try: manifest=json.loads(manifest_path.read_text())
    except Exception: manifest={"version":1,"completed":[],"failures":[]}
    for idx,(sc,seed,c) in enumerate(matrix,1):
        runid=f"{c}_{sc}_s{seed}_3600"
        if a.smoke: runid=f"{c}_{sc}_s{seed}_600"
        if runid in manifest["completed"] and _read_completed(out,runid): print(f"SKIP {runid}");continue
        existing=_read_completed(out,runid)
        if existing: manifest["completed"].append(runid); continue
        print(f"RUN {idx}/{len(matrix)} {runid}",flush=True)
        try:
            metrics,tele,dec,updates,forecasts=run_episode(sc,seed,c,out,600 if a.smoke else 3600,(180,) if a.smoke else (900,1800,2700),runid)
            metrics["status"]="COMPLETE";_save_episode(out,runid,metrics,tele,dec,updates,forecasts)
            if runid not in manifest["completed"]: manifest["completed"].append(runid)
            manifest["failures"]=[x for x in manifest["failures"] if x.get("run_id")!=runid]
        except Exception as ex:
            manifest["failures"]=[x for x in manifest["failures"] if x.get("run_id")!=runid]
            manifest["failures"].append({"run_id":runid,"error":repr(ex)})
            print(f"FAILED {runid}: {ex!r}",flush=True)
        atomic_json(manifest_path,manifest)
    episode=[]
    for runid in manifest["completed"]:
        m=_read_completed(out,runid)
        if m: episode.append(m)
    if episode:
        payload,df,pairdf=_aggregate(out,episode)
        _plots(out,df);_report(out,payload,df,pairdf)
    print(f"Phase 8 resumable evaluator finished: {len(episode)} completed; {len(manifest['failures'])} failure(s).")

if __name__=="__main__":main()
