import csv
import math

import numpy as np
import pytest

from ctrlsandbox import CartPole, ConditionSchedule, Quadrotor2D, run_episode, summarize
from ctrlsandbox.controllers import Controller, LQRController
from ctrlsandbox.registry import controller_factory
from ctrlsandbox.sweep import run_sweep


class DoNothing(Controller):
    def __init__(self, system):
        self.system = system

    def act(self, obs):
        return self.system.equilibrium_action()


def test_failure_is_penalized_and_recorded():
    s = CartPole()
    sched = ConditionSchedule(s, horizon=300)
    ep = run_episode(s, DoNothing(s), sched, seed=0, x0=np.array([0.0, 0.3, 0.0, 0.0]))
    m = summarize(s, ep, sched)
    assert ep.failed and m["survived"] == 0.0
    assert m["total_cost"] >= s.fail_penalty * (300 - m["steps"])
    assert math.isnan(m["tracking_error"])


def test_fault_mode_reports_recovery():
    q = Quadrotor2D()
    sched = ConditionSchedule(q, "fault", 1.0, horizon=300)
    m = summarize(q, run_episode(q, LQRController(q), sched, seed=0), sched)
    assert "recovery_s" in m


def test_registry_rejects_unknown():
    with pytest.raises(ValueError):
        controller_factory("magic", CartPole())


def test_small_sweep_writes_csv_and_figures(tmp_path):
    path, rows = run_sweep(
        "cartpole", ["lqr", "es"], ["static", "fault"], [0.0, 1.0], range(2),
        horizon=60, out=tmp_path, verbose=False,
    )
    assert len(rows) == 2 * 2 * 2 * 2
    with open(path) as f:
        assert len(list(csv.DictReader(f))) == len(rows)
    from ctrlsandbox.plotting import plot_robustness

    figs = plot_robustness(path, tmp_path)
    assert all(f.exists() for f in figs)
    # trained ES policy is cached for re-use
    assert any(p.name.startswith("cartpole_es") for p in (tmp_path / "cache").iterdir())
