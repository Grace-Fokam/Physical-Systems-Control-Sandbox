import numpy as np
import pytest

from ctrlsandbox import CartPole, ConditionSchedule, Quadrotor2D, run_episode, summarize
from ctrlsandbox.controllers import (
    AdaptiveMPPI,
    ESPolicyController,
    LearnedMPPI,
    LQRController,
    MPPIController,
    ParticleParamEstimator,
    RFFDynamicsModel,
    collect_transitions,
    linearize,
    train_es,
)

SYSTEMS = [CartPole(), Quadrotor2D()]


def run(system, controller, mode="static", severity=0.0, horizon=150, seed=0):
    sched = ConditionSchedule(system, mode, severity, horizon=horizon)
    return summarize(system, run_episode(system, controller, sched, seed=seed), sched)


@pytest.mark.parametrize("system", SYSTEMS, ids=lambda s: s.name)
def test_linearization_matches_small_steps(system):
    u0 = system.equilibrium_action()
    A, B = linearize(system, system.target, u0)
    dx = 1e-4 * np.ones(system.state_dim)
    du = 1e-4 * np.ones(system.action_dim)
    pred = A @ dx + B @ du
    true = system.diff(system.step(system.target + dx, u0 + du), system.target)
    assert np.allclose(pred, true, atol=1e-7)


@pytest.mark.parametrize("system", SYSTEMS, ids=lambda s: s.name)
def test_lqr_stabilizes_nominal(system):
    m = run(system, LQRController(system))
    assert m["survived"] == 1.0
    assert m["tracking_error"] < 0.1


@pytest.mark.parametrize("system", SYSTEMS, ids=lambda s: s.name)
def test_mppi_stabilizes_nominal(system):
    m = run(system, MPPIController(system, samples=128), horizon=100)
    assert m["survived"] == 1.0
    assert m["tracking_error"] < 0.3


def test_estimator_recovers_shifted_quadrotor_parameters():
    q = Quadrotor2D()
    true = q.shifted_params(1.0)
    est = ParticleParamEstimator(q, names=["mass", "eta_right"], n=256)
    est.reset(np.random.default_rng(0))
    rng = np.random.default_rng(1)
    lqr = LQRController(q)
    x = q.sample_initial_state(rng)
    for _ in range(150):
        u = lqr.act(x) + rng.normal(0, 0.3, 2)
        xn = q.step(x, u, true)
        noisy = lambda s: s + q.obs_noise_scale * rng.standard_normal(6)  # noqa: E731
        est.update(noisy(x), np.clip(u, q.action_low, q.action_high), noisy(xn))
        x = xn
    e = est.estimate()
    assert e["mass"] == pytest.approx(true["mass"], rel=0.08)
    assert e["eta_right"] == pytest.approx(true["eta_right"], rel=0.08)


def test_adaptive_mppi_beats_nominal_mppi_under_shift():
    q = Quadrotor2D()
    nominal = np.mean([run(q, MPPIController(q), severity=1.0, horizon=200, seed=s)["total_cost"] for s in range(2)])
    adaptive = np.mean([run(q, AdaptiveMPPI(q), severity=1.0, horizon=200, seed=s)["total_cost"] for s in range(2)])
    assert adaptive < nominal


def test_es_training_reduces_cost():
    s = CartPole()
    policy, history = train_es(s, iters=40, steps=150, seed=0)
    assert history[-1] < 0.2 * history[0]
    assert run(s, ESPolicyController(policy))["survived"] == 1.0


def test_policy_save_load(tmp_path):
    from ctrlsandbox.controllers import LinearPolicy

    s = CartPole()
    p = LinearPolicy(s, np.arange(5.0))
    p.save(tmp_path / "p.npz")
    q = LinearPolicy.load(s, tmp_path / "p.npz")
    assert np.allclose(p.theta, q.theta)


@pytest.mark.parametrize("system", SYSTEMS, ids=lambda s: s.name)
def test_learned_model_is_accurate_and_controls(system):
    X, U, Xn = collect_transitions(system, episodes=20, steps=120, seed=0)
    Xt, Ut, Xnt = collect_transitions(system, episodes=5, steps=120, seed=1)
    model = RFFDynamicsModel(system).fit(X, U, Xn)
    err = system.diff(model.predict(Xt, Ut), Xnt)
    rel = np.sqrt((err**2).mean(0)) / system.diff(Xnt, Xt).std(0)
    assert np.all(rel < 0.3)
    m = run(system, LearnedMPPI(system, model, samples=128), horizon=100)
    assert m["survived"] == 1.0
