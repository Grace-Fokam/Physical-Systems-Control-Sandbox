"""Robustness figures: one small multiple per condition mode, one line per controller."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Fixed categorical order: a controller keeps its colour in every figure.
COLORS = {
    "lqr": "#2a78d6",
    "mppi": "#eb6834",
    "adaptive_mppi": "#1baf7a",
    "es": "#eda100",
    "es_dr": "#e87ba4",
    "learned_mppi": "#008300",
    "ppo": "#6250d6",
    "sac": "#e34948",
}
LABELS = {
    "lqr": "LQR",
    "mppi": "MPPI (nominal)",
    "adaptive_mppi": "Adaptive MPPI",
    "es": "ES policy",
    "es_dr": "ES policy + DR",
    "learned_mppi": "Learned-model MPPI",
    "ppo": "PPO",
    "sac": "SAC",
}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MODE_TITLES = {
    "static": "Static shift",
    "drift": "Gradual drift",
    "fault": "Abrupt fault",
    "disturbance": "Gusts + sensor noise",
}


def load_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def aggregate(rows, metric):
    """{mode: {controller: (severities, mean, min, max)}} across seeds."""
    groups = defaultdict(list)
    for r in rows:
        v = r[metric]
        if v not in ("", "nan"):
            groups[(r["mode"], r["controller"], float(r["severity"]))].append(float(v))
    out = defaultdict(dict)
    for mode, ctrl in {(m, c) for m, c, _ in groups}:
        sev = sorted(s for m, c, s in groups if m == mode and c == ctrl)
        vals = [np.array(groups[(mode, ctrl, s)]) for s in sev]
        out[mode][ctrl] = (
            np.array(sev),
            np.array([v.mean() for v in vals]),
            np.array([v.min() for v in vals]),
            np.array([v.max() for v in vals]),
        )
    return out


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _figure(rows, metric, ylabel, title, out_path, log=False, ylim=None, band=True, subtitle=None):
    data = aggregate(rows, metric)
    modes = [m for m in MODE_TITLES if m in data]
    order = [c for c in COLORS if any(c in data[m] for m in modes)]
    fig, axes = plt.subplots(1, len(modes), figsize=(3.4 * len(modes), 3.4), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, mode in zip(axes[0], modes):
        _style(ax)
        for ctrl in order:
            if ctrl not in data[mode]:
                continue
            s, mean, lo, hi = data[mode][ctrl]
            c = COLORS[ctrl]
            if band:  # range across seeds: honest for small n and bimodal (fail / no fail) outcomes
                ax.fill_between(s, lo, hi, color=c, alpha=0.10, linewidth=0)
            ax.plot(s, mean, color=c, linewidth=2, marker="o", markersize=4, label=LABELS.get(ctrl, ctrl))
        ax.set_title(MODE_TITLES[mode], color=INK, fontsize=11, loc="left")
        ax.set_xlabel("severity", color=MUTED, fontsize=9)
        if log:
            ax.set_yscale("log")
        if ylim:
            ax.set_ylim(*ylim)
    axes[0][0].set_ylabel(ylabel, color=MUTED, fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    ncol = min(len(labels), 3 if len(modes) <= 2 else 6)
    rows_of_legend = int(np.ceil(len(labels) / ncol))
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle(title, x=0.01, y=0.985, ha="left", color=INK, fontsize=12, fontweight="bold")
    if subtitle:
        fig.text(0.01, 0.885, subtitle, ha="left", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.06 + 0.05 * rows_of_legend, 1, 0.86))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def plot_robustness(csv_path, out_dir=None):
    csv_path = Path(csv_path)
    out_dir = Path(out_dir) if out_dir else csv_path.parent
    rows = load_rows(csv_path)
    system = rows[0]["system"] if rows else "system"
    figs = [
        _figure(rows, "total_cost", "episode cost (log, lower is better)",
                f"{system}: episode cost as conditions move away from nominal",
                out_dir / f"{system}_robustness_cost.png", log=True,
                subtitle="line = mean over seeds, band = range over seeds"),
        _figure(rows, "survived", "episodes without failure",
                f"{system}: share of episodes without failure", out_dir / f"{system}_robustness_survival.png",
                ylim=(-0.05, 1.05), band=False),
    ]
    return figs


def summary_table(csv_path, severity: float = 1.0) -> str:
    """Markdown table: mean episode cost (and survival rate) per controller and mode at one severity."""
    rows = [r for r in load_rows(csv_path) if abs(float(r["severity"]) - severity) < 1e-9]
    modes = [m for m in MODE_TITLES if any(r["mode"] == m for r in rows)]
    ctrls = [c for c in COLORS if any(r["controller"] == c for r in rows)]
    head = "| Controller | " + " | ".join(MODE_TITLES[m] for m in modes) + " |"
    lines = [head, "| --- | " + " | ".join("---" for _ in modes) + " |"]
    for c in ctrls:
        cells = []
        for m in modes:
            sel = [r for r in rows if r["controller"] == c and r["mode"] == m]
            cost = np.mean([float(r["total_cost"]) for r in sel])
            surv = np.mean([float(r["survived"]) for r in sel])
            cells.append(f"{cost:,.0f} ({surv:.0%})")
        lines.append(f"| {LABELS.get(c, c)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
