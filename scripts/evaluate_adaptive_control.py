"""Paired fixed-time vs adaptive SUMO test on identical scenario/seed inputs."""
from __future__ import annotations
import argparse, json, os, sys, time, uuid, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
JUNCTIONS=("J0","J1","J2","J3")

def _paths():
    home=Path(os.environ.get("SUMO_HOME",r"C:\Program Files (x86)\SUMO"))
    tools=home/"tools"
    if tools.is_dir() and str(tools) not in sys.path: sys.path.insert(0,str(tools))
    if str(ROOT/"src") not in sys.path: sys.path.insert(0,str(ROOT/"src"))
    binary=home/"bin"/"sumo.exe"
    if not binary.is_file(): raise FileNotFoundError(f"SUMO executable not found: {binary}")
    return binary

def _context():
    cfg=json.loads((ROOT/"sumo_files/cluster_config.json").read_text(encoding="utf-8"))
    tls={m["short_id"]:tid for tid,m in cfg["tls_meta"].items()}
    net=ET.parse(ROOT/"sumo_files/hsr.net.xml").getroot()
    edge_lanes={e.get("id"):[l.get("id") for l in e.findall("lane")] for e in net.findall("edge")}
    incoming={j:[] for j in JUNCTIONS}
    for c in net.findall(".//connection"):
        tid=c.get("tl")
        if tid in cfg["cluster_tls_ids"]:
            j=cfg["tls_meta"][tid]["short_id"]
            for lane in edge_lanes.get(c.get("from"),[]):
                if lane not in incoming[j]: incoming[j].append(lane)
    if any(not incoming[j] for j in JUNCTIONS): raise RuntimeError("A junction has no controlled incoming lanes")
    return cfg,tls,incoming

def run_episode(scenario:str,seed:int,mode:str,duration_s:int):
    binary=_paths()
    import traci
    from adaptive_signal_controller import AdaptiveSignalController
    from stgnn import STGNNPredictor
    route=ROOT/f"sumo_files/routes/{scenario}_s{seed}.rou.xml"
    net=ROOT/"sumo_files/hsr.net.xml"
    if not route.is_file(): raise FileNotFoundError(route)
    cfg,tls,incoming=_context()
    predictor=None; predictor_error=None
    if mode=="adaptive":
        try: predictor=STGNNPredictor()
        except Exception as exc: predictor_error=f"{type(exc).__name__}: {exc}"
    label=f"adaptive_eval_{mode}_{scenario}_{seed}_{uuid.uuid4().hex[:8]}"
    cmd=[str(binary),"-n",str(net),"-r",str(route),"--seed",str(seed+100),"--no-step-log","true","--no-warnings","true","--time-to-teleport","300"]
    traci.start(cmd,label=label)
    conn=traci.getConnection(label)
    telemetry=[]; previous={}; controller=None
    try:
        for j in JUNCTIONS: conn.trafficlight.setProgram(tls[j],"0")
        controller=AdaptiveSignalController(conn,tls,incoming,predictor,enabled=(mode=="adaptive"),predictor_error=predictor_error)
        for _ in range(int(duration_s)):
            previous={j:int(conn.trafficlight.getPhase(tls[j])) for j in JUNCTIONS}
            conn.simulationStep()
            controller.after_step(previous)
            now=int(round(float(conn.simulation.getTime())))
            if now and now%60==0:
                for j in JUNCTIONS:
                    row=dict(controller.telemetry[j]); row.update({"scenario":scenario,"seed":seed,"mode":mode,"junction":j})
                    telemetry.append(row)
        decisions=list(controller.decision_history)
        queue_values=[float(r["queue_length"]) for r in telemetry]
        throughput_by_j={j:int(sum(int(r["throughput"]) for r in telemetry if r["junction"]==j)) for j in JUNCTIONS}
        metrics={"mode":mode,"scenario":scenario,"seed":seed,"sumo_seed":seed+100,"duration_s":int(duration_s),
          "total_waiting_time_vehicle_s":round(float(controller.cumulative_stopped_vehicle_seconds),3),
          "average_queue_vehicles":float(np.mean(queue_values)) if queue_values else None,
          "maximum_queue_vehicles":int(max(queue_values,default=0)),
          "throughput_total_junction_crossings":int(sum(throughput_by_j.values())),
          "throughput_by_junction":throughput_by_j,
          "decision_count":len(decisions),
          "decisions_by_junction":{j:sum(d["junction"]==j for d in decisions) for j in JUNCTIONS},
          "green_duration_unique_s_by_junction":{j:sorted({round(float(x),2) for x in controller.green_duration_history[j]}) for j in JUNCTIONS},
          "green_duration_count_by_junction":{j:len(controller.green_duration_history[j]) for j in JUNCTIONS},
          "phase_selection_counts_by_junction":{j:{str(p):sum(d["junction"]==j and d["green_phase"]==p for d in decisions) for p in controller._phase_info[j]["service_phases"]} for j in JUNCTIONS},
          "stgnn_status":controller.model_status,"controller_status":controller.status,
          "controller_error":controller.error,"telemetry_samples":len(telemetry)//4}
        if mode=="adaptive" and not any(metrics["green_duration_count_by_junction"].values()):
            raise RuntimeError("Adaptive controller produced no green-duration decisions")
        return metrics,telemetry,decisions
    finally:
        try: conn.close(wait=True)
        except Exception:
            try: traci.close(wait=True)
            except Exception: pass

def compare(scenario="low",seed=0,duration_s=600):
    fixed,ft,fd=run_episode(scenario,seed,"fixed_time",duration_s)
    adaptive,at,ad=run_episode(scenario,seed,"adaptive",duration_s)
    deltas={}
    for key in ("total_waiting_time_vehicle_s","average_queue_vehicles","throughput_total_junction_crossings"):
        a=adaptive.get(key); f=fixed.get(key)
        deltas[key]={"adaptive_minus_fixed":a-f if a is not None and f is not None else None,
                     "relative_change_pct":(100*(a-f)/f if a is not None and f not in (None,0) else None)}
    return ({"status":"COMPLETE","scenario":scenario,"seed":seed,"sumo_seed":seed+100,"duration_s":duration_s,
            "identical_route_and_seed":True,"fixed_time":fixed,"adaptive":adaptive,"adaptive_minus_fixed":deltas},
            {"fixed_time":ft,"adaptive":at},{"fixed_time":fd,"adaptive":ad})

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--scenario",default="low")
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--duration",type=int,default=600)
    parser.add_argument("--out",default="results/adaptive_ai_compare_20261113")
    args=parser.parse_args()
    payload,telemetry,decisions=compare(args.scenario,args.seed,args.duration)
    out=ROOT/args.out
    if out.exists():
        suffix=time.strftime("%H%M%S")
        out=out.with_name(out.name+"_"+suffix)
    out.mkdir(parents=True,exist_ok=False)
    (out/"paired_comparison.json").write_text(json.dumps(payload,indent=2,allow_nan=False),encoding="utf-8")
    for mode,rows in telemetry.items(): pd.DataFrame(rows).to_csv(out/f"{mode}_telemetry.csv",index=False)
    for mode,rows in decisions.items(): pd.DataFrame(rows).to_csv(out/f"{mode}_decisions.csv",index=False)
    print(json.dumps({"result_path":str(out),**payload},indent=2,allow_nan=False))

if __name__=="__main__": main()
