# Interactive control dashboard

From the repository root (Python 3.10+):

```bash
cd control-sandbox
python -m pip install -e .
python -m pip install -r dashboard/requirements.txt
python -m streamlit run dashboard/app.py
```

Open the local URL printed by Streamlit. Keep the repository checkout: recorded
CSV data lives in `results/` and is not packaged as wheel data. Paths are resolved
relative to the app, not the current working directory. No RL extra is needed.
GitHub Actions validates installation, the core and dashboard tests, and the Streamlit health endpoint on pull requests and pushes to main. Optional RL tests skip when their extra dependencies are absent.

The recorded tab compares all six benchmark controllers across four scenarios,
with mean and min–max robustness curves, per-severity metric tables, episode
counts, and CSV export. It reads existing data and does not regenerate benchmarks.
Tracking error averages omit failed episodes; valid counts are shown. Recovery
can be undefined. These curves are simulation evidence, not physical validation.

The live tab calls `controller_factory`, `run_episode`, and `summarize` through
`ctrlsandbox.dashboard_support`. LQR, nominal MPPI, and adaptive MPPI retain core
defaults. Training-dependent controllers are deliberately excluded from live execution.
A fresh process/controller is created per click. Episodes are capped at 300 steps,
severity at 1, and elapsed subprocess runtime at 45 seconds. BLAS thread counts are
limited to one. Results persist only within the session and are labeled with their
original configuration. No live results are written into benchmark CSVs.

Live state/action plots use actual simulated outputs and include early failure.
Recorded data contains summary metrics only, so historical state trajectories
cannot be reconstructed. Shorter live episodes change the cost and error window,
and cannot be compared directly with the 300-step benchmark. Timing depends on
hardware. The timeout is a per-run safeguard, not a deployment-wide concurrency
limit: public hosting should additionally cap concurrent sessions at the host.
This is a mechanical control research demo, separate from Fovra's energy prototype.

Lightweight checks (with dashboard requirements and pytest installed):

```bash
python -m pytest -q tests/test_dashboard.py
```
