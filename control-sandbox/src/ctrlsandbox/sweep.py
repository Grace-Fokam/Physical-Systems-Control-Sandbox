"""Robustness sweeps: every controller x condition mode x severity x seed.

Example::

    ctrl-sweep --system quadrotor2d --seeds 5 --out results/
    python -m ctrlsandbox.sweep --system cartpole --modes static,fault --quick

Writes ``<out>/<system>_episodes.csv`` (one row per episode) and robustness
figures. Re-running reuses trained policies and models cached in ``<out>/cache``.
"""

from __future__ import annotations

import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .conditions import MODES, ConditionSchedule
from .registry import DEFAULT_CONTROLLERS, controller_factory
from .runner import run_episode, summarize
from .systems import make_system

FIELDS = [
    "system", "mode", "severity", "controller", "seed",
    "total_cost", "survived", "steps", "tracking_error", "saturation", "ms_per_step", "recovery_s",
]


def _run_block(args):
    """One controller on one (mode, severity) across seeds. Runs in a worker process."""
    system_name, controller, mode, severity, seeds, horizon, cache = args
    system = make_system(system_name)
    make = controller_factory(controller, system, cache_dir=cache)
    rows = []
    for seed in seeds:
        schedule = ConditionSchedule(system, mode=mode, severity=severity, horizon=horizon)
        ep = run_episode(system, make(), schedule, seed=seed)
        m = summarize(system, ep, schedule)
        rows.append({
            "system": system_name, "mode": mode, "severity": severity, "controller": controller,
            "seed": seed, **m, "recovery_s": m.get("recovery_s", ""),
        })
    return rows


def run_sweep(system_name, controllers, modes, severities, seeds, horizon=300, out="results", workers=1, verbose=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "cache"
    system = make_system(system_name)
    # train everything once up front so workers only load from cache
    for c in controllers:
        t0 = time.time()
        controller_factory(c, system, cache_dir=cache)
        if verbose:
            print(f"[setup] {c:14s} ready ({time.time() - t0:.1f}s)", flush=True)
    jobs = [
        (system_name, c, mode, float(sev), list(seeds), horizon, str(cache))
        for mode in modes for sev in severities for c in controllers
    ]
    rows = []
    t0 = time.time()
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            for i, block in enumerate(pool.map(_run_block, jobs), 1):
                rows += block
                if verbose:
                    _progress(i, len(jobs), block, t0)
    else:
        for i, job in enumerate(jobs, 1):
            block = _run_block(job)
            rows += block
            if verbose:
                _progress(i, len(jobs), block, t0)
    path = out / f"{system_name}_episodes.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path, rows


def _progress(i, n, block, t0):
    r = block[0]
    cost = np.mean([b["total_cost"] for b in block])
    surv = np.mean([b["survived"] for b in block])
    print(
        f"[{i:3d}/{n}] {r['mode']:11s} sev={r['severity']:.2f} {r['controller']:14s} "
        f"cost={cost:9.1f} survived={surv:.2f}  ({time.time() - t0:.0f}s)",
        flush=True,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", default="cartpole", choices=["cartpole", "quadrotor2d"])
    ap.add_argument("--controllers", default=",".join(DEFAULT_CONTROLLERS))
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--severities", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true", help="3 severities, 2 seeds, 200 steps")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args(argv)
    severities = [float(s) for s in a.severities.split(",")]
    seeds, horizon = a.seeds, a.horizon
    if a.quick:
        severities, seeds, horizon = [0.0, 0.5, 1.0], 2, 200
    path, _ = run_sweep(
        a.system, a.controllers.split(","), a.modes.split(","), severities, range(seeds),
        horizon=horizon, out=a.out, workers=a.workers,
    )
    print(f"wrote {path}")
    if not a.no_plot:
        from .plotting import plot_robustness

        for fig in plot_robustness(path, Path(a.out)):
            print(f"wrote {fig}")
    from .plotting import summary_table

    md = Path(a.out) / f"{a.system}_summary.md"
    sev = max(severities)
    md.write_text(f"Mean episode cost at severity {sev:g} (share of episodes without failure)\n\n" + summary_table(path, sev))
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
