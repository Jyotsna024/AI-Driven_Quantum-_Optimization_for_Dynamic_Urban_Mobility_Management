"""Simulation-only Phase 7 closed-loop SUMO controller and run recorder."""
from __future__ import annotations
import argparse, json, math, os, pickle, sys, time, xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/"scripts"))
from stgnn import STGNNPredictor
from rebuild_s3_solvers_v3 import feasible as onehot_feasible, energy as qubo_energy
from rebuild_s3_solvers_v3 import qubo_to_ising
import dimod, neal
from qiskit import transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

JUNCTIONS=["J0","J1","J2","J3"]
PLAN_IDS=[0,1,2,3]
DECISION_S=90
S2_STATUS="SCIENTIFICALLY_NOT_VALIDATED"

def load_static():
    config=json.loads((ROOT/"sumo_files/cluster_config.json").read_text(encoding="utf-8"))
    with (ROOT/"models/cost_model_s1_v2.pkl").open("rb") as f: cost_model=pickle.load(f)
    stgnn=STGNNPredictor()
    qmeta=json.loads((ROOT/"models/qubo_s2_v3.json").read_text(encoding="utf-8"))
    pairwise=np.asarray(qmeta["pairwise_array_s"],float)
    tls_by_j={m["short_id"]:tid for tid,m in config["tls_meta"].items()}
    net=ET.parse(ROOT/"sumo_files/hsr.net.xml").getroot()
    edge_lanes={e.get("id"):[l.get("id") for l in e.findall("lane")] for e in net.findall(".//edge")}
    incoming={j:[] for j in JUNCTIONS}
    for c in net.findall(".//connection"):
        tid=c.get("tl")
        if tid in config["cluster_tls_ids"]:
            j=config["tls_meta"][tid]["short_id"]
            for lane in edge_lanes.get(c.get("from"),[]):
                if lane not in incoming[j]: incoming[j].append(lane)
    if any(not incoming[j] for j in JUNCTIONS): raise RuntimeError("Missing controlled incoming lanes")
    # Every candidate program must preserve state sequence and yellow clearance states.
    for tid in config["cluster_tls_ids"]:
        plans=config["plans"][tid]
        state_seq=[[p["state"] for p in plan["phases"]] for plan in plans]
        if any(seq!=state_seq[0] for seq in state_seq[1:]):
            raise RuntimeError(f"Signal plan {tid}: candidate programs alter legal phase sequence")
        for states in state_seq[0]:
            if "y" in states.lower(): continue
        if any("y" not in s.lower() for s in state_seq[0][1::2]):
            raise RuntimeError(f"Signal plan {tid}: expected yellow clearance phases")
    return config,tls_by_j,incoming,cost_model,stgnn,pairwise,qmeta

def online_features(history,scenario,predictor,cost_model):
    """Create S1-schema pre-decision covariates from three observed 60s samples."""
    if len(history)<3: raise ValueError("ST-GNN requires three 60-second observations")
    latest=history[-1]
    raw=np.asarray([[h[j][k] for h in history[-3:]] for j in JUNCTIONS for k in predictor.feature_cols],dtype=np.float32)
    raw=raw.reshape(4,len(predictor.feature_cols),3).transpose(0,2,1)
    forecasts=predictor.predict(raw)
    cols=cost_model["feature_cols"]
    matrix=[]
    for jidx,j in enumerate(JUNCTIONS):
        for p in PLAN_IDS:
            pre=latest[j]
            r={"pre_queue_veh":pre["queue_length"],"pre_queue_m":pre["queue_length_m"],
               "pre_delay_s":pre["mean_delay_s"],"pre_arrivals":pre["arrivals"],
               "pre_throughput":pre["throughput"],"pre_speed_mps":pre["mean_speed_mps"],
               "pre_vehicle_count":pre["vehicle_count"],
               "stgnn_queue_h1":float(forecasts[jidx,0]),"stgnn_queue_h2":float(forecasts[jidx,1]),
               "stgnn_queue_h3":float(forecasts[jidx,2])}
            for s in ("low","medium","high","peak","incident"): r[f"scenario_{s}"]=float(s==scenario)
            for pp in PLAN_IDS: r[f"plan_{pp}"]=float(pp==p)
            for jj in JUNCTIONS: r[f"junction_{jj}"]=float(jj==j)
            for pp in PLAN_IDS:
                for jj in JUNCTIONS: r[f"plan{pp}_junction_{jj}"]=float(pp==p and jj==j)
                for h in (1,2,3): r[f"plan{pp}_queue_h{h}"]=float(pp==p)*r[f"stgnn_queue_h{h}"]
                r[f"plan{pp}_pre_queue_m"]=float(pp==p)*r["pre_queue_m"]
                r[f"plan{pp}_pre_arrivals"]=float(pp==p)*r["pre_arrivals"]
            matrix.append([r[k] for k in cols])
    scaled=cost_model["scaler"].transform(pd.DataFrame(matrix,columns=cols))
    costs=np.clip(cost_model["model"].predict(scaled).reshape(4,4),0,None)
    return forecasts,costs

def make_qubo(costs,pairwise,reference_a):
    # A conservative bound across all binary base terms makes every one-hot
    # state lower than any infeasible assignment, including runtime costs.
    abs_sum=float(np.abs(costs).sum()+np.abs(pairwise).sum())
    a=max(float(reference_a),2*abs_sum+1.0)
    q=np.zeros((16,16),float)
    for i in range(4):
        for p in range(4): q[i*4+p,i*4+p]=costs[i,p]-a
        for p in range(4):
            for op in range(p+1,4): q[i*4+p,i*4+op]=2*a
    pairs=list(__import__("itertools").combinations(range(4),2))
    for pair_idx,(i,j) in enumerate(pairs):
        for p in range(4):
            for r in range(4): q[i*4+p,j*4+r]=pairwise[pair_idx,p,r]
    return q,a

def solve_sa(q,seed):
    lin={i:float(q[i,i]) for i in range(16) if q[i,i]!=0}
    quad={(i,j):float(q[i,j]) for i in range(16) for j in range(i+1,16) if q[i,j]!=0}
    bqm=dimod.BinaryQuadraticModel(lin,quad,0.,dimod.BINARY)
    sample_set=neal.SimulatedAnnealingSampler().sample(bqm,num_reads=100,seed=int(seed))
    valid=[]
    for row in sample_set.data(fields=["sample","energy","num_occurrences"]):
        x=np.asarray([row.sample[i] for i in range(16)],int)
        if onehot_feasible(x): valid.append((float(row.energy),x))
    return (min(valid,key=lambda z:z[0])[1],len(valid),100) if valid else (None,0,100)

def solve_qaoa(q,seed):
    op,_offset,_,_=qubo_to_ising(q)
    circuit=QAOAAnsatz(cost_operator=op,reps=1); circuit.measure_all()
    sim=AerSimulator(); calls=0
    def objective(params):
        nonlocal calls
        calls+=1
        tc=transpile(circuit.assign_parameters(params),sim,seed_transpiler=int(seed))
        counts=sim.run(tc,shots=64,seed_simulator=int(seed)*1000+calls).result().get_counts()
        return sum(qubo_energy(q,[int(s[15-i]) for i in range(16)])*n for s,n in counts.items())/sum(counts.values())
    opt=minimize(objective,[0.4,0.3],method="COBYLA",options={"maxiter":6})
    tc=transpile(circuit.assign_parameters(opt.x),sim,seed_transpiler=int(seed))
    counts=sim.run(tc,shots=512,seed_simulator=int(seed)+50000).result().get_counts()
    feasible_n=sum(n for s,n in counts.items() if onehot_feasible([int(s[15-i]) for i in range(16)]))
    valid=[]
    for s,n in counts.items():
        x=np.asarray([int(s[15-i]) for i in range(16)],int)
        if onehot_feasible(x): valid.append((qubo_energy(q,x),x))
    return (min(valid,key=lambda z:z[0])[1] if valid else None,feasible_n,512)

def max_pressure_plan(conn,tls_id,junction,plan_defs,incoming_lanes):
    links=conn.trafficlight.getControlledLinks(tls_id)
    pressures=[]
    for idx,group in enumerate(links):
        p=0.
        for inc,out,_via in group:
            qin=conn.lane.getLastStepHaltingNumber(inc) if inc else 0
            qout=conn.lane.getLastStepVehicleNumber(out) if out else 0
            p+=float(qin-qout)
        pressures.append(p)
    scores=[]
    for plan in plan_defs:
        cycle=sum(float(ph["duration"]) for ph in plan["phases"])
        score=0.
        for phase in plan["phases"]:
            for idx,state in enumerate(phase["state"]):
                if idx<len(pressures) and state in "Gg": score+=float(phase["duration"])*pressures[idx]
        scores.append(score/max(cycle,1))
    return int(np.argmax(scores)),scores

def run_episode(scenario,seed,controller,out_dir,duration_s=3600,qaoa_sample_times=(900,1800,2700),run_tag="phase8_v1"):
    import traci
    cfg,tls_by_j,incoming,cost_model,predictor,pairwise,qmeta=load_static()
    sumo=Path(os.environ.get("SUMO_HOME",r"C:\Program Files (x86)\SUMO"))/"bin"/"sumo.exe"
    net=ROOT/"sumo_files/hsr.net.xml"; route=ROOT/f"sumo_files/routes/{scenario}_s{seed}.rou.xml"
    if not route.exists(): raise FileNotFoundError(route)
    label=f"{run_tag}_{scenario}_s{seed}_{controller}"
    cmd=[str(sumo),"-n",str(net),"-r",str(route),"--seed",str(seed+100),"--no-step-log","true","--no-warnings","true","--time-to-teleport","300"]
    before={}; after={}; decision_rows=[]; updates=[]; telemetry=[]; forecast_rows=[]; solver_times=[]
    fallback_count=solver_calls=solver_feasible=solver_raw=0
    traci.start(cmd,label=label); conn=traci.getConnection(label)
    tids=[tls_by_j[j] for j in JUNCTIONS]
    plan_defs={j:cfg["plans"][tls_by_j[j]] for j in JUNCTIONS}
    curr_plan={j:0 for j in JUNCTIONS}; target_plan={j:0 for j in JUNCTIONS}
    last_phase={j:conn.trafficlight.getPhase(tls_by_j[j]) for j in JUNCTIONS}
    for j in JUNCTIONS: conn.trafficlight.setProgram(tls_by_j[j],"0")
    lane_ids={j:incoming[j] for j in JUNCTIONS}
    prev_by_lane={j:{lane:set() for lane in lane_ids[j]} for j in JUNCTIONS}
    lane_tp=defaultdict(int); lane_arr=defaultdict(int); prev_j={j:set() for j in JUNCTIONS}
    history=[]; trips_depart={}; trip_times=[]; seen_loss={}; total_wait_vehicle_s=0
    dep_map={v.get("id"):float(v.get("depart",0)) for v in ET.parse(route).getroot().findall("vehicle")}
    scen_cfg=__import__("yaml").safe_load((ROOT/"configs/scenarios.yaml").read_text())["scenarios"][scenario]
    inc_lane=scen_cfg.get("blocked_lane"); inc_start=scen_cfg.get("incident_start_s",1200); inc_end=scen_cfg.get("incident_end_s",1800); old_speed=None
    qaoa_set=set(qaoa_sample_times)
    last_targets={j:0 for j in JUNCTIONS}; pending_decision={}
    try:
        for t in range(1,duration_s+1):
            if scen_cfg.get("incident") and inc_lane:
                if t==inc_start: old_speed=conn.lane.getMaxSpeed(inc_lane); conn.lane.setMaxSpeed(inc_lane,0.1)
                elif t==inc_end and old_speed is not None: conn.lane.setMaxSpeed(inc_lane,old_speed)
            pre_phase={j:conn.trafficlight.getPhase(tls_by_j[j]) for j in JUNCTIONS}
            conn.simulationStep()
            # Integrate stopped vehicle-seconds and time-loss using unique vehicles.
            all_incoming=set()
            for j in JUNCTIONS:
                curr=set()
                for lane in lane_ids[j]:
                    ids=set(conn.lane.getLastStepVehicleIDs(lane)); curr.update(ids)
                    lane_tp[(j,lane)]+=len(prev_by_lane[j][lane]-ids)
                    lane_arr[(j,lane)]+=len(ids-prev_by_lane[j][lane])
                    prev_by_lane[j][lane]=ids
                all_incoming.update(curr)
                departed=prev_j[j]-curr
                # vehicle leaves this junction's incoming approaches
                prev_j[j]=curr
            stopped=set()
            for veh in all_incoming:
                try:
                    if conn.vehicle.getSpeed(veh)<0.1: stopped.add(veh)
                    seen_loss[veh]=max(seen_loss.get(veh,0.),float(conn.vehicle.getTimeLoss(veh)))
                except Exception: pass
            total_wait_vehicle_s+=len(stopped)
            for veh in conn.simulation.getArrivedIDList():
                if veh in dep_map: trip_times.append(float(t)-dep_map[veh])
            if t%60==0:
                sample={}
                for j in JUNCTIONS:
                    qv=0; qm=0.; speeds=[]; waits=[]; vc=0; lane_q={}
                    for lane in lane_ids[j]:
                        lane_q[lane]=int(conn.lane.getLastStepHaltingNumber(lane)); qv+=lane_q[lane]
                        vids=conn.lane.getLastStepVehicleIDs(lane); vc+=len(vids)
                        for vid in vids:
                            speed=float(conn.vehicle.getSpeed(vid)); speeds.append(speed); waits.append(float(conn.vehicle.getWaitingTime(vid)))
                            if speed<0.1: qm+=float(conn.vehicle.getLength(vid))+2.5
                    row={"time_s":t,"scenario":scenario,"seed":seed,"controller":controller,"junction":j,
                      "queue_length":qv,"queue_length_m":qm,"vehicle_count":vc,
                      "mean_speed_mps":float(np.mean(speeds)) if speeds else 0.,
                      "mean_delay_s":float(np.mean(waits)) if waits else 0.,
                      "throughput":sum(lane_tp[(j,l)] for l in lane_ids[j]),
                      "arrivals":sum(lane_arr[(j,l)] for l in lane_ids[j]),
                      "applied_plan_id":curr_plan[j],"lane_queue_json":json.dumps(lane_q,sort_keys=True)}
                    sample[j]={k:row[k] for k in ("queue_length","queue_length_m","vehicle_count","mean_speed_mps","mean_delay_s","throughput","arrivals")}
                    telemetry.append(row)
                    for lane in lane_ids[j]:
                        lane_tp[(j,lane)]=0; lane_arr[(j,lane)]=0
                history.append(sample)
            if t%DECISION_S==0:
                costs=None; forecasts=None; pred_error=None
                try: forecasts,costs=online_features(history,scenario,predictor,cost_model)
                except Exception as ex: pred_error=repr(ex)
                for jidx,j in enumerate(JUNCTIONS):
                    if costs is not None and forecasts is not None:
                        for h in (1,2,3): forecast_rows.append({"scenario":scenario,"seed":seed,"controller":controller,
                            "decision_time_s":t,"target_time_s":t+60*h,"junction":j,"horizon":h,
                            "predicted_queue":float(forecasts[jidx,h-1]),"actual_queue":None})
                desired=[curr_plan[j] for j in JUNCTIONS]; method=controller; fallback=False; reason=None; raw=feas_ct=0; solve_s=0.
                pressure_scores={}
                try:
                    if controller=="fixed_time": desired=[0]*4
                    elif controller=="max_pressure":
                        for j in JUNCTIONS:
                            p,sc=max_pressure_plan(conn,tls_by_j[j],j,plan_defs[j],lane_ids[j]); desired[JUNCTIONS.index(j)]=p; pressure_scores[j]=sc
                    elif controller=="sa_qubo":
                        if costs is None: raise ValueError(pred_error or "missing cost features")
                        q,a=make_qubo(costs,pairwise,qmeta["penalty_A"]); solver_calls+=1
                        tsolve=time.perf_counter(); x,feas_ct,raw=solve_sa(q,seed*100000+t); solve_s=time.perf_counter()-tsolve; solver_times.append(solve_s)
                        solver_raw+=raw; solver_feasible+=feas_ct
                        if x is None or not onehot_feasible(x): raise ValueError("SA returned no feasible plan")
                        desired=np.asarray(x).reshape(4,4).argmax(axis=1).astype(int).tolist()
                    elif controller=="qaoa_aer_sampled":
                        if t not in qaoa_set:
                            desired=[last_targets[j] for j in JUNCTIONS]; method="qaoa_hold_previous"
                        else:
                            if costs is None: raise ValueError(pred_error or "missing cost features")
                            q,a=make_qubo(costs,pairwise,qmeta["penalty_A"]); solver_calls+=1
                            tsolve=time.perf_counter(); x,feas_ct,raw=solve_qaoa(q,seed*100000+t); solve_s=time.perf_counter()-tsolve; solver_times.append(solve_s)
                            solver_raw+=raw; solver_feasible+=feas_ct
                            if x is None or not onehot_feasible(x): raise ValueError("QAOA returned no feasible raw sample")
                            desired=np.asarray(x).reshape(4,4).argmax(axis=1).astype(int).tolist()
                    else: raise ValueError(f"unknown controller {controller}")
                except Exception as ex:
                    desired=[0]*4; fallback=True; reason=repr(ex); fallback_count+=1
                if fallback and controller not in ("fixed_time","max_pressure"): desired=[0]*4
                for idx,j in enumerate(JUNCTIONS): target_plan[j]=int(desired[idx]); last_targets[j]=int(desired[idx])
                decision={"time_s":t,"scenario":scenario,"seed":seed,"controller":controller,
                  "decision_interval_s":DECISION_S,"solver_method":method,"target_plan_vector":json.dumps(desired),
                  "actual_applied_plan_vector_at_decision":json.dumps([curr_plan[j] for j in JUNCTIONS]),
                  "predicted_cost_matrix":json.dumps(costs.tolist()) if costs is not None else None,
                  "stgnn_forecasts":json.dumps(forecasts.tolist()) if forecasts is not None else None,
                  "objective_status":S2_STATUS,"objective_warning":"S2 v3 held-out validation failed; optimization plans are experimental only.",
                  "solver_runtime_s":solve_s,"raw_solver_samples":raw,"feasible_raw_samples":feas_ct,
                  "fallback":fallback,"fallback_reason":reason or pred_error,"applied_updates_pending":False,
                  "max_pressure_scores":json.dumps(pressure_scores) if pressure_scores else None}
                for j in JUNCTIONS:
                    decision[f"applied_time_{j}"]=t if target_plan[j]==curr_plan[j] else None
                    decision[f"applied_plan_{j}"]=curr_plan[j] if target_plan[j]==curr_plan[j] else None
                    if target_plan[j]!=curr_plan[j]: pending_decision[j]=len(decision_rows)
                decision["applied_updates_pending"]=any(target_plan[j]!=curr_plan[j] for j in JUNCTIONS)
                decision_rows.append(decision)
            # Apply requested program only after the old program's clearance sequence reaches phase 0.
            for j in JUNCTIONS:
                tid=tls_by_j[j]; ph=conn.trafficlight.getPhase(tid)
                if ph==0 and pre_phase[j]!=0 and target_plan[j]!=curr_plan[j]:
                    old=curr_plan[j]; oldstate=conn.trafficlight.getRedYellowGreenState(tid)
                    try:
                        conn.trafficlight.setProgram(tid,str(target_plan[j]))
                        newstate=conn.trafficlight.getRedYellowGreenState(tid)
                        if newstate!=oldstate:
                            # No simulation step occurs between attempted switch and rollback.
                            conn.trafficlight.setProgram(tid,str(old))
                            if conn.trafficlight.getRedYellowGreenState(tid)!=oldstate:
                                conn.trafficlight.setProgram(tid,"0")
                            raise RuntimeError("candidate program would change signal state at transition")
                        curr_plan[j]=target_plan[j]
                        updates.append({"time_s":t,"scenario":scenario,"seed":seed,"controller":controller,
                          "junction":j,"old_plan_id":old,"applied_plan_id":curr_plan[j],"signal_state":newstate,
                          "transition":"cycle boundary after yellow clearance"})
                        if j in pending_decision:
                            di=pending_decision.pop(j); dr=decision_rows[di]
                            dr[f"applied_time_{j}"]=t; dr[f"applied_plan_{j}"]=curr_plan[j]
                            dr["applied_updates_pending"]=any(k in pending_decision for k in JUNCTIONS)
                    except Exception as ex:
                        target_plan[j]=0; curr_plan[j]=int(conn.trafficlight.getProgram(tid)) if str(conn.trafficlight.getProgram(tid)).isdigit() else 0
                        if j in pending_decision:
                            dr=decision_rows[pending_decision[j]]; dr["fallback"]=True
                            prior=dr.get("fallback_reason")
                            dr["fallback_reason"]=(str(prior)+"; " if prior else "")+f"{j} safe transition rejected: {ex!r}"
                            dr.setdefault("safe_transition_rejections",[]).append(j)
                            if curr_plan[j]==0:
                                dr[f"applied_time_{j}"]=t; dr[f"applied_plan_{j}"]=0
                                pending_decision.pop(j,None)
                        updates.append({"time_s":t,"scenario":scenario,"seed":seed,"controller":controller,
                          "junction":j,"old_plan_id":old,"applied_plan_id":curr_plan[j],"signal_state":conn.trafficlight.getRedYellowGreenState(tid),
                          "transition":"safe-switch rejected; controller fallback queued","error":repr(ex)})
            for j in JUNCTIONS: last_phase[j]=conn.trafficlight.getPhase(tls_by_j[j])
    finally:
        conn.close()
    # Attach realized future queues to each forecast, when the target timestamp was observed.
    actual_lookup={(r["time_s"],r["junction"]):r["queue_length"] for r in telemetry}
    for r in forecast_rows: r["actual_queue"]=actual_lookup.get((r["target_time_s"],r["junction"]))
    fc=[r for r in forecast_rows if r["actual_queue"] is not None]
    losses=list(seen_loss.values())
    lane_summary={}
    for j in JUNCTIONS:
        lane_summary[j]={}
        for lane in lane_ids[j]:
            qs=[json.loads(r["lane_queue_json"])[lane] for r in telemetry if r["junction"]==j]
            lane_summary[j][lane]={"mean_queue_vehicles":float(np.mean(qs)) if qs else 0.,"max_queue_vehicles":int(max(qs,default=0))}
    metrics={"scenario":scenario,"seed":seed,"controller":controller,"duration_s":duration_s,
      "total_waiting_time_vehicle_s":int(total_wait_vehicle_s),"completed_trip_count":len(trip_times),
      "mean_completed_trip_travel_time_s":float(np.mean(trip_times)) if trip_times else None,
      "total_time_loss_s":float(sum(losses)),"mean_time_loss_per_seen_vehicle_s":float(np.mean(losses)) if losses else None,
      "average_queue_vehicles":float(np.mean([r["queue_length"] for r in telemetry])) if telemetry else None,
      "maximum_queue_vehicles":int(max([r["queue_length"] for r in telemetry],default=0)),
      "average_queue_m":float(np.mean([r["queue_length_m"] for r in telemetry])) if telemetry else None,
      "maximum_queue_m":float(max([r["queue_length_m"] for r in telemetry],default=0)),
      "per_junction_queue":{j:{"average_queue_vehicles":float(np.mean([r["queue_length"] for r in telemetry if r["junction"]==j])) if any(r["junction"]==j for r in telemetry) else None,
         "maximum_queue_vehicles":int(max([r["queue_length"] for r in telemetry if r["junction"]==j],default=0)),
         "average_queue_m":float(np.mean([r["queue_length_m"] for r in telemetry if r["junction"]==j])) if any(r["junction"]==j for r in telemetry) else None} for j in JUNCTIONS},
      "junction_throughput_count":{j:int(sum(r["throughput"] for r in telemetry if r["junction"]==j)) for j in JUNCTIONS},
      "throughput_total_junction_crossings":int(sum(r["throughput"] for r in telemetry)),"per_lane_queue":lane_summary,
      "mean_solver_runtime_s":float(np.mean(solver_times)) if solver_times else None,"total_solver_runtime_s":float(sum(solver_times)),
      "solver_calls":solver_calls,"raw_solver_samples":solver_raw,"feasible_raw_samples":solver_feasible,
      "raw_sample_feasibility_rate":solver_feasible/solver_raw if solver_raw else None,
      "fallback_count":sum(bool(r.get("fallback")) for r in decision_rows),"decision_count":len(decision_rows),
      "fallback_rate":sum(bool(r.get("fallback")) for r in decision_rows)/len(decision_rows) if decision_rows else None,
      "forecast_mae_vehicles":float(np.mean([abs(r["predicted_queue"]-r["actual_queue"]) for r in fc])) if fc else None,
      "forecast_sample_count":len(fc),"s2_objective_status":S2_STATUS}
    return metrics,telemetry,decision_rows,updates,forecast_rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--scenario",default="low"); ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--controller",default="sa_qubo",choices=["fixed_time","max_pressure","sa_qubo","qaoa_aer_sampled"])
    ap.add_argument("--duration",type=int,default=3600); ap.add_argument("--out",default="results/phase8_v1")
    ap.add_argument("--qaoa-sample-times",type=int,nargs="*",default=[900,1800,2700]); args=ap.parse_args()
    out=ROOT/args.out; out.mkdir(parents=True,exist_ok=True)
    runid=f"{args.controller}_{args.scenario}_s{args.seed}_{args.duration}"
    metrics,telemetry,decisions,updates,forecasts=run_episode(args.scenario,args.seed,args.controller,out,args.duration,tuple(args.qaoa_sample_times),runid)
    import pandas as pd
    for name,rows in (("traffic_timeseries",telemetry),("decision_log",decisions),("signal_updates",updates),("traffic_forecasts",forecasts)):
        p=out/f"{runid}_{name}.csv"; pd.DataFrame(rows).to_csv(p,index=False)
    (out/f"{runid}_metrics.json").write_text(json.dumps(metrics,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(metrics,indent=2))

if __name__=="__main__": main()
