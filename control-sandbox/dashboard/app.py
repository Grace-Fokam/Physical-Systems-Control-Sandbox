"""Launch from control-sandbox: python -m streamlit run dashboard/app.py."""
from pathlib import Path
import os
import pickle
import subprocess
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ctrlsandbox.conditions import MODES
from ctrlsandbox.dashboard_support import LIVE_CONTROLLERS
from ctrlsandbox.plotting import COLORS, LABELS, MODE_TITLES, aggregate, load_rows
from ctrlsandbox.systems import SYSTEMS, make_system

RESULTS = Path(__file__).resolve().parents[1] / "results"
METRICS = ("total_cost", "survived", "tracking_error", "saturation", "ms_per_step", "recovery_s")

st.set_page_config(page_title="Physical Systems Control Sandbox", layout="wide")
st.title("Physical Systems Control Sandbox")
st.caption("Mechanical control experiments · cart-pole and planar quadrotor · simulated systems")
st.write("Explore recorded robustness experiments or execute a fresh rollout using the repository's simulator.")
with st.sidebar:
    system_name = st.selectbox("Physical system", list(SYSTEMS), key="system")
    mode = st.selectbox("Disturbance scenario", MODES, format_func=MODE_TITLES.get, key="scenario")
    st.caption({"static": "Parameters shifted from the start.", "drift": "Parameters gradually shift over the episode.", "fault": "Parameters jump after one third of the episode.", "disturbance": "Nominal parameters with random gusts and extra sensor noise."}[mode])

recorded, live = st.tabs(["Recorded experiments", "Run a simulation"])
with recorded:
    st.subheader("Previously recorded results")
    path = RESULTS / f"{system_name}_episodes.csv"
    if not path.exists():
        st.warning(f"Recorded dataset unavailable: {path.name}")
    else:
        rows = load_rows(path)
        data = pd.DataFrame(rows)
        for col in ("severity", "seed", "steps", *METRICS):
            data[col] = pd.to_numeric(data[col], errors="coerce")
        st.caption(f"Source: results/{path.name} · {len(data)} episode records. CSV metrics, not stored trajectories.")
        controllers = st.multiselect("Compare controllers", list(data.controller.unique()), default=list(data.controller.unique()), format_func=lambda c: LABELS.get(c, c))
        levels = sorted(data.severity.unique())
        severity = st.select_slider("Recorded severity", options=levels, value=levels[-1])
        metric = st.selectbox("Robustness metric", METRICS, key="metric")
        selected = [r for r in rows if r["mode"] == mode and r["controller"] in controllers]
        if not controllers:
            st.info("Select at least one controller to compare.")
        else:
            curves = aggregate(selected, metric).get(mode, {})
            fig, ax = plt.subplots(figsize=(9, 4))
            for c, (x, mean, lo, hi) in curves.items():
                ax.plot(x, mean, marker="o", color=COLORS.get(c), label=LABELS.get(c, c))
                ax.fill_between(x, lo, hi, alpha=.12, color=COLORS.get(c))
            ax.axvline(severity, color="gray", linestyle="--")
            ax.set(xlabel="Severity", ylabel=metric, title=MODE_TITLES[mode])
            if curves:
                ax.legend()
            else:
                st.info("No finite values for this metric and scenario.")
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Lines show means across seeds. Shaded bands show min–max, not confidence intervals. Missing values are excluded.")
            subset = data[(data["mode"] == mode) & data.controller.isin(controllers) & np.isclose(data.severity, severity)]
            summary = subset.groupby("controller")[list(METRICS)].mean()
            summary["episodes"] = subset.groupby("controller").size()
            summary["valid_tracking_error"] = subset.groupby("controller").tracking_error.count()
            st.dataframe(summary, use_container_width=True)
            st.caption("Survived and saturation are fractions. Tracking error excludes failed episodes. Recovery is only defined for faults and may be missing. Costs include failure penalties. Compute times reflect the original machine.")
            st.download_button("Download selected episode records", subset.to_csv(index=False), f"{system_name}_{mode}_recorded.csv", "text/csv")

with live:
    st.subheader("New simulator execution")
    st.write("LQR, MPPI, and adaptive MPPI use the existing controller factories with their default planning settings. No training is started.")
    st.caption("ES, learned-model MPPI, PPO, and SAC remain in the recorded view because their factories can train models or require optional dependencies.")
    with st.form("simulation"):
        controller = st.selectbox("Live controller", LIVE_CONTROLLERS, format_func=LABELS.get)
        live_severity = st.slider("Live severity", 0.0, 1.0, 0.5, 0.05)
        horizon = st.slider("Episode length (steps)", 10, 300, 100, 10)
        seed = st.number_input("Random seed", 0, 1_000_000, 0, 1)
        submitted = st.form_submit_button("Run simulation")
    st.caption("Runs only on button press, in an isolated subprocess with a 45-second timeout. Each session retains one result. Runtime varies by host. Short episodes are not directly comparable to the 300-step benchmark.")
    if submitted:
        config = (system_name, controller, mode, live_severity, horizon, seed)
        with st.spinner("Executing actual simulator…"):
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    output = Path(tmp) / "episode.pkl"
                    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                    subprocess.run([sys.executable, "-m", "ctrlsandbox.dashboard_support", *map(str, config), str(output)], check=True, capture_output=True, text=True, timeout=45, env=env)
                    result = pickle.loads(output.read_bytes())
                st.session_state["last_run"] = (config, result)
            except subprocess.TimeoutExpired:
                st.error("Simulation exceeded 45 seconds and was stopped. Try LQR or a shorter episode.")
            except (subprocess.CalledProcessError, OSError) as exc:
                st.error(f"Simulation could not complete: {exc}")
    if "last_run" in st.session_state:
        config, (episode, metrics) = st.session_state["last_run"]
        name, ctrl, scenario, sev, steps, run_seed = config
        system = make_system(name)
        st.markdown(f"**Last completed run:** {name} · {LABELS[ctrl]} · {scenario} · severity {sev:g} · {steps} requested steps · seed {run_seed}")
        st.caption("This result retains its original configuration when controls change. Physical trajectories are reproducible by seed, wall-clock timing is not.")
        st.dataframe(pd.DataFrame([metrics]), use_container_width=True)
        st.write("Episode ended in failure." if episode.failed else "Episode completed without failure.")
        state_frame = pd.DataFrame(episode.states, columns=system.state_labels, index=np.arange(len(episode.states))*system.dt)
        action_frame = pd.DataFrame(episode.actions, columns=system.action_labels, index=np.arange(len(episode.actions))*system.dt)
        st.write("True states versus simulation time (seconds)")
        st.line_chart(state_frame)
        st.write("Applied control actions versus simulation time (seconds)")
        st.line_chart(action_frame)
        st.write("Effective parameter severity and per-step cost")
        diagnostics = pd.DataFrame({"parameter_severity": episode.severities, "cost": episode.costs}, index=action_frame.index)
        st.line_chart(diagnostics)
        st.caption("In gust/noise mode, parameter severity stays zero. The selected severity scales gusts and observation noise instead. Angles use radians, positions meters, and actions force/thrust in newtons.")
        trajectory = state_frame.add_prefix("state_").join(action_frame.add_prefix("action_")).join(diagnostics)
        trajectory.index.name = "time_s"
        st.download_button("Download new rollout", trajectory.to_csv(), f"{name}_{ctrl}_live.csv", "text/csv")
