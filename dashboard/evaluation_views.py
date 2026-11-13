"""Streamlit views for the saved capstone evaluation evidence."""
from __future__ import annotations

from importlib.util import find_spec
from typing import Any

import pandas as pd
import streamlit as st


def _number(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "Not available"
    if isinstance(value, (int, float)):
        return f"{value:,.{digits}f}"
    return str(value)


def _rate(block: dict | None) -> str:
    if not block:
        return "Not available"
    if block.get("total") is not None and block.get("rate") is not None:
        return f"{block.get('correct', block.get('feasible', ''))}/{block['total']} ({100*block['rate']:.1f}%)"
    return _number(block.get("value"))


def _show_graph(bundle: dict, filename: str, caption: str) -> None:
    path = bundle.get("graphs", {}).get(filename)
    if path and path.is_file():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Chart unavailable: {filename}")


def _show_missing(bundle: dict) -> None:
    for error in bundle.get("errors", []):
        st.warning(error)


def _kpi_row(metrics: list[tuple[str, Any, str | None]]) -> None:
    cols = st.columns(len(metrics))
    for col, (label, value, help_text) in zip(cols, metrics):
        col.metric(label, _number(value), help=help_text)


def render_section(section: str, bundle: dict) -> None:
    evaluation = bundle.get("evaluation") or {}
    _show_missing(bundle)

    renderers = {
        "Project Overview": _overview,
        "Traffic Prediction (ST-GNN)": _stgnn,
        "Cost Prediction (S1)": _s1,
        "QUBO Optimization (S2)": _s2,
        "Solver Comparison (S3)": _s3,
        "Traffic Performance": _traffic,
        "Limitations": _limitations,
    }
    if section in renderers:
        renderers[section](evaluation, bundle)


def _overview(e: dict, bundle: dict) -> None:
    st.title("AI-Driven Quantum Optimization for Dynamic Urban Mobility Management")
    st.markdown("Research dashboard for a four-junction HSR Layout corridor using saved SUMO experiments, ST-GNN forecasts, cost estimation, QUBO diagnostics, and solver comparisons.")
    s2=e.get("s2_qubo", {})
    s1=e.get("s1_cost_model", {})
    st.subheader("Project status")
    a,b,c,d=st.columns(4)
    a.metric("Evaluation package", "Available" if e else "Missing")
    a.caption("Saved evaluation outputs are loaded from JSON and CSV.")
    b.metric("ST-GNN test split", f"Seed {e.get('stgnn',{}).get('test_seed','—')}" if e else "—")
    c.metric("S1 top-1 agreement", _rate(s1.get("top1_plan_agreement")))
    d.metric("S2 validation", "Not validated" if s2 and not s2.get("scientific_validation_pass") else "See evaluation")
    st.subheader("System architecture and workflow")
    st.markdown("`SUMO scenarios and telemetry` → `ST-GNN queue forecasts` → `S1 cost estimates` → `S2 QUBO plan ranking` → `S3 ExactSolver / SA / QAOA analysis` → `held-out evaluation`")
    st.caption("S2 is not scientifically validated. Solver-driven live control remains disabled; the separate SUMO page preserves manual controls.")
    st.subheader("Study corridor")
    cols=st.columns(4)
    details=(("J0","14th Main & 15th Cross"),("J1","14th Main & 17th Cross"),("J2","19th Main & 17th Cross"),("J3","14th Main & Sarjapur Road"))
    for col,(junction,description) in zip(cols,details):
        with col:
            st.markdown(f"**{junction}**")
            st.caption(description)
    st.info("This is a research prototype on a small simulated corridor. It does not control real traffic signals and makes no quantum-advantage claim.")


def _stgnn(e: dict, bundle: dict) -> None:
    st.title("Traffic Prediction · ST-GNN")
    section=e.get("stgnn", {})
    model=section.get("stgnn", {}).get("overall", {})
    baseline=section.get("persistence", {}).get("overall", {})
    if not model:
        st.info("ST-GNN evaluation metrics are unavailable.")
        return
    st.caption(f"Held-out test set · seed {section.get('test_seed','—')} · {section.get('n_forecast_windows','—')} forecast windows · 60–180 second horizons")
    _kpi_row([("MAE",model.get("mae"),None),("RMAE (%)",model.get("rmae_pct"),"Percent of mean absolute actual queue"),("RMSE",model.get("rmse"),None),("R²",model.get("r2"),None)])
    st.subheader("ST-GNN versus persistence")
    compare=[]
    for label,values in (("ST-GNN",model),("Persistence",baseline)):
        compare.append({"Model":label,"MAE":values.get("mae"),"RMAE (%)":values.get("rmae_pct"),"RMSE":values.get("rmse"),"R²":values.get("r2"),"Pearson r":values.get("pearson_r"),"Spearman r":values.get("spearman_r")})
    st.dataframe(pd.DataFrame(compare),hide_index=True,use_container_width=True)
    _show_graph(bundle,"stgnn_prediction_comparison.png","Held-out actual queue versus ST-GNN and persistence means by forecast horizon")
    h=model.get("by_horizon",{})
    if h:
        st.subheader("Prediction horizon comparison")
        st.dataframe(pd.DataFrame([{"Horizon":key,"Seconds":section.get("prediction_horizon_seconds",[])[int(key[1:])-1],**value} for key,value in h.items()]),hide_index=True,use_container_width=True)
    junction=model.get("by_junction",{})
    if junction:
        st.subheader("Junction-level metrics")
        st.dataframe(pd.DataFrame([{"Junction":key,**value} for key,value in junction.items()]),hide_index=True,use_container_width=True)
    st.caption("Regression metrics use paired finite queue predictions and ground truth. No classification accuracy is used.")


def _s1(e: dict, bundle: dict) -> None:
    st.title("Cost Prediction · S1")
    s=e.get("s1_cost_model",{})
    m=s.get("regression",{})
    if not m:
        st.info("S1 held-out cost-model metrics are unavailable.")
        return
    st.caption(f"S1 v2 · held-out test seed {s.get('test_seed','—')} · {s.get('n_plan_outcomes','—')} plan outcomes")
    agreement=s.get("top1_plan_agreement",{})
    cols=st.columns(5)
    for col,label,value in zip(cols,("MAE","RMAE (%)","RMSE","Spearman","Top-1 plan agreement"),(m.get("mae"),m.get("rmae_pct"),m.get("rmse"),s.get("spearman_plan_outcomes"),_rate(agreement))):
        col.metric(label,_number(value))
    for title,key in (("By junction","by_junction"),("By scenario","by_scenario")):
        data=s.get(key,{})
        if data:
            st.subheader(title)
            st.dataframe(pd.DataFrame([{"Group":group,**metrics} for group,metrics in data.items()]),hide_index=True,use_container_width=True)
    st.caption("Top-1 agreement counts test snapshot × target-junction groups where the lowest predicted-delay plan matches the measured SUMO-delay minimum.")


def _s2(e: dict, bundle: dict) -> None:
    st.title("QUBO Optimization · S2")
    s=e.get("s2_qubo",{})
    if not s:
        st.info("S2 QUBO evaluation metrics are unavailable.")
        return
    if not s.get("scientific_validation_pass",False):
        st.error("Scientific validation warning: S2 v3 did not pass held-out predictive validation. Results are diagnostic and must not be interpreted as evidence of reliable optimization or used for closed-loop control.")
    st.caption("Predicted QUBO optima and measured SUMO optima are evaluated on identical held-out test plan grids.")
    pooled=s.get("spearman_predicted_vs_actual",{})
    snapshot=s.get("mean_snapshot_spearman",{})
    cols=st.columns(5)
    cols[0].metric("Top-1 plan agreement",_rate(s.get("top1_plan_agreement")))
    cols[1].metric("Mean optimization regret",f"{_number(s.get('mean_optimization_regret_s'))} s")
    cols[2].metric("Pooled Spearman",_number(pooled.get("value")),help=f"Across {pooled.get('n_plan_vectors','—')} plan outcomes pooled across snapshots")
    cols[3].metric("Mean snapshot Spearman",_number(snapshot.get("value")),help=f"Mean within-snapshot correlation across {snapshot.get('n_snapshots','—')} snapshots")
    cols[4].metric("Feasible global minima",_rate(s.get("feasible_solution_rate")))
    st.info("Pooled Spearman ranks all plan outcomes together. Mean snapshot-level Spearman computes each snapshot’s rank correlation separately, then averages those correlations. The aggregation methods differ; both are shown.")
    details=[]
    for sid,item in s.get("by_snapshot",{}).items():
        details.append({"Snapshot":sid,"Predicted QUBO optimum":str(item.get("predicted_optimum_plan_vector")),"Actual SUMO optimum":str(item.get("actual_sumo_optimum_plan_vector")),"Top-1 match":item.get("top1_plan_agreement"),"Pooled within-snapshot Spearman":item.get("spearman_predicted_qubo_vs_actual_sumo"),"Regret (s)":item.get("optimization_regret_s"),"Global QUBO minimum feasible":item.get("unconstrained_qubo_global_minimum_feasible")})
    if details:
        st.subheader("Held-out snapshot details")
        st.dataframe(pd.DataFrame(details),hide_index=True,use_container_width=True)
    st.caption(s.get("prediction_actual_optimum_distinction","Predicted and actual optima use different objective sources."))


def _s3(e: dict, bundle: dict) -> None:
    st.title("Solver Comparison · S3")
    s=e.get("s3_solvers",{})
    solvers=s.get("solvers",{})
    if not solvers:
        st.info("Saved S3 solver metrics are unavailable.")
        return
    rows=[]
    for name in ("ExactSolver","SimulatedAnnealing","QAOA"):
        x=solvers.get(name,{})
        feasibility=x.get("feasibility_metric",{})
        agreement=x.get("exact_plan_agreement",{})
        repeated=x.get("per_seed_exact_plan_agreement",{})
        rows.append({"Solver":name,"Best plan":str(x.get("solution_plan_vector","—")),"Best energy":x.get("best_energy"),"Runtime (s)":x.get("runtime_s_total"),"Feasibility metric":feasibility.get("label","Not available"),"Feasible / total":f"{feasibility.get('feasible','—')} / {feasibility.get('total','—')}","Feasibility rate":feasibility.get("value"),"Exact-plan agreement":("Reference optimum" if agreement.get("status")=="REFERENCE_OPTIMUM" else agreement.get("value")),"Repeated best-plan agreement":_rate(repeated)})
    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    st.info("ExactSolver’s exhaustive search-space feasible fraction uses all 65,536 bitstrings. SA/QAOA sampled-output feasibility rates use returned samples. These denominators are not directly comparable. ExactSolver defines the reference optimum; its self-match is not an independent performance result.")
    _show_graph(bundle,"solver_comparison.png","S3 total runtime, best energy, and feasibility with denominator types labeled")
    st.warning("QAOA was run using Qiskit Aer simulation. No quantum hardware was used and no quantum advantage is claimed.")
    if solvers.get("QAOA",{}).get("runtime_per_seed_status")=="NOT AVAILABLE":
        st.caption(solvers["QAOA"].get("runtime_per_seed_reason","Per-seed QAOA runtime is not available."))


def _traffic(e: dict, bundle: dict) -> None:
    st.title("Traffic Performance")
    phase=e.get("phase8_controller_comparison",{})
    if not phase or phase.get("status")!="AVAILABLE":
        st.info("Saved Phase 8 traffic outcomes are unavailable.")
        return
    st.caption(f"Descriptive SUMO outcomes · {phase.get('episode_count','—')} saved episodes")
    st.warning("These charts show simulation outcomes (waiting time, queue, and junction crossings). They are not ST-GNN prediction-accuracy metrics.")
    for graph,caption in (("waiting_time_comparison.png","Waiting time by controller"),("queue_length_comparison.png","Average queue length by controller"),("throughput_comparison.png","Junction throughput by controller")):
        _show_graph(bundle,graph,caption)
    st.subheader("Controller summary")
    rows=[]
    for controller,metrics in phase.get("metrics_by_controller",{}).items():
        row={"Controller":controller}
        for key,label in (("waiting_time","Waiting time mean (vehicle-s)"),("queue_length","Average queue mean (vehicles)"),("throughput","Junction crossings mean")):
            block=metrics.get(key,{})
            row[label]=block.get("mean")
            row[label.replace("mean","SD")]=block.get("std")
        rows.append(row)
    if rows:
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)


def _limitations(e: dict, bundle: dict) -> None:
    st.title("Limitations and Research Scope")
    s2=e.get("s2_qubo",{})
    s1=e.get("s1_cost_model",{})
    st.subheader("S2 scientific validation")
    st.error("S2 v3 held-out predictive validation failed. Its plan rankings and regret are diagnostic only; they do not establish reliable optimization and are not enabled for closed-loop control.")
    st.subheader("S1 cost prediction")
    agreement=s1.get("top1_plan_agreement",{})
    st.write(f"Held-out plan agreement is {_rate(agreement)} over snapshot × junction groups. Regression metrics summarize the saved seed-4 test outcomes; the result should not be generalized beyond the small evaluated set.")
    st.subheader("QAOA and quantum claims")
    st.write("QAOA used Qiskit Aer classical simulation. No quantum hardware result or quantum advantage is claimed.")
    st.subheader("Testing environment")
    optional=("dimod","neal","qiskit","qiskit_aer")
    missing=[name for name in optional if find_spec(name) is None]
    if missing:
        st.warning("Optional solver-test dependencies missing from this Python environment: " + ", ".join(missing) + ". This can prevent collecting the solver mechanics test module.")
    else:
        st.success("Solver test dependencies are available in this Python environment.")
    st.subheader("Additional scope limits")
    for item in e.get("limitations",[]):
        st.markdown(f"- {item}")
    st.markdown("The dashboard describes a four-junction simulated corridor. It is not a production traffic-control system.")
