from pathlib import Path
import numpy as np
import pytest
from ctrlsandbox.dashboard_support import simulate


@pytest.mark.parametrize("system", ["cartpole", "quadrotor2d"])
@pytest.mark.parametrize("controller", ["lqr", "mppi", "adaptive_mppi"])
def test_live_real_simulator(system, controller):
    ep, metrics = simulate(system, controller, "fault", .5, 10, 0)
    assert len(ep.states) == len(ep.actions) + 1
    assert metrics["steps"] == len(ep.costs)
    assert np.isfinite(metrics["total_cost"])
    assert ep.severities[-1] == .5


@pytest.mark.parametrize("args", [
    ("cartpole", "es", "static", 0., 10, 0),
    ("cartpole", "lqr", "static", float("nan"), 10, 0),
    ("cartpole", "lqr", "static", 0., 301, 0),
])
def test_reject_unbounded_or_training_runs(args):
    with pytest.raises(ValueError):
        simulate(*args)


def test_dashboard_initial_and_live_execution():
    pytest.importorskip("streamlit")
    pytest.importorskip("pandas")
    from streamlit.testing.v1 import AppTest
    app = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
    at = AppTest.from_file(str(app)).run(timeout=20)
    assert not at.exception
    assert len(at.dataframe[0].value) == 6
    at.button[0].click().run(timeout=60)
    assert not at.exception
    assert "last_run" in at.session_state
    config, (ep, metrics) = at.session_state["last_run"]
    assert config[1] == "lqr"
    assert len(ep.actions) == metrics["steps"]


@pytest.mark.parametrize("mode", ["static", "drift", "fault", "disturbance"])
def test_seeded_scenarios(mode):
    a, _ = simulate("cartpole", "lqr", mode, .75, 10, 7)
    b, _ = simulate("cartpole", "lqr", mode, .75, 10, 7)
    np.testing.assert_array_equal(a.states, b.states)
    if mode == "disturbance":
        assert not a.severities.any()


def test_recorded_filters_and_timeout(monkeypatch):
    pytest.importorskip("streamlit")
    pytest.importorskip("pandas")
    import subprocess
    from streamlit.testing.v1 import AppTest
    app = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
    at = AppTest.from_file(str(app)).run(timeout=20)
    at.selectbox(key="system").select("quadrotor2d")
    at.selectbox(key="scenario").select("disturbance")
    at.selectbox(key="metric").select("recovery_s")
    at.run(timeout=20)
    assert not at.exception
    assert at.info
    at.multiselect[0].set_value([]).run(timeout=20)
    assert not at.exception

    def timed_out(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timed_out)
    at.button[0].click().run(timeout=20)
    assert not at.exception
    assert "45 seconds" in at.error[0].value
    assert "last_run" not in at.session_state
