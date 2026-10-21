import numpy as np
import pytest

from ctrlsandbox import CartPole, ConditionSchedule, Quadrotor2D


def test_static_is_constant():
    s = CartPole()
    sched = ConditionSchedule(s, "static", 0.7, horizon=100)
    sched.reset(np.random.default_rng(0))
    assert sched.params_at(0) == sched.params_at(99) == s.shifted_params(0.7)


def test_drift_is_monotone_from_nominal_to_shift():
    s = CartPole()
    sched = ConditionSchedule(s, "drift", 1.0, horizon=100)
    sched.reset(np.random.default_rng(0))
    masses = [sched.params_at(t)["m_pole"] for t in range(100)]
    assert masses[0] == pytest.approx(s.nominal_params["m_pole"])
    assert masses[-1] == pytest.approx(s.shifted_params(1.0)["m_pole"])
    assert np.all(np.diff(masses) >= 0)


def test_fault_jumps_at_fault_step():
    s = Quadrotor2D()
    sched = ConditionSchedule(s, "fault", 1.0, horizon=90, fault_step=30)
    sched.reset(np.random.default_rng(0))
    assert sched.params_at(29)["eta_right"] == pytest.approx(1.0)
    assert sched.params_at(30)["eta_right"] == pytest.approx(0.8)


def test_gust_statistics_scale_with_severity():
    s = Quadrotor2D()
    stds = []
    for sev in (0.5, 1.0):
        sched = ConditionSchedule(s, "disturbance", sev, horizon=20000)
        sched.reset(np.random.default_rng(0))
        gusts = np.array([sched.params_at(t)["f_ext"] for t in range(20000)])
        stds.append(gusts.std())
        assert sched.params_at(0)["mass"] == s.nominal_params["mass"]
    assert stds[0] == pytest.approx(0.5 * s.disturbance_scale, rel=0.25)
    assert stds[1] == pytest.approx(1.0 * s.disturbance_scale, rel=0.25)


def test_disturbance_mode_raises_sensor_noise():
    s = CartPole()
    quiet = ConditionSchedule(s, "static", 1.0).obs_noise_std()
    loud = ConditionSchedule(s, "disturbance", 1.0).obs_noise_std()
    assert np.all(loud > quiet)


def test_bad_mode():
    with pytest.raises(ValueError):
        ConditionSchedule(CartPole(), "earthquake")
