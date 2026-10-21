import numpy as np
import pytest

from ctrlsandbox import CartPole, ConditionSchedule, Quadrotor2D, SandboxEnv, make_system

SYSTEMS = [CartPole(), Quadrotor2D()]


@pytest.mark.parametrize("system", SYSTEMS, ids=lambda s: s.name)
def test_equilibrium_is_stationary(system):
    p = system.nominal_params
    x = system.target
    u = system.equilibrium_action(p)
    assert np.allclose(system.step(x, u, p), x, atol=1e-10)


@pytest.mark.parametrize("system", SYSTEMS, ids=lambda s: s.name)
def test_batched_step_matches_single(system):
    rng = np.random.default_rng(0)
    xs = np.stack([system.sample_initial_state(rng) for _ in range(8)])
    us = rng.uniform(system.action_low, system.action_high, (8, system.action_dim))
    batched = system.step(xs, us)
    single = np.stack([system.step(x, u) for x, u in zip(xs, us)])
    assert np.allclose(batched, single)


@pytest.mark.parametrize("system", SYSTEMS, ids=lambda s: s.name)
def test_batched_params_broadcast(system):
    """One state, many parameter hypotheses -> one prediction per hypothesis."""
    rng = np.random.default_rng(1)
    x = system.sample_initial_state(rng)
    u = system.equilibrium_action()
    p, _ = system.sample_params(rng, 0.0, 1.0, size=16)
    out = system.step(x, u, p)
    assert out.shape == (16, system.state_dim)
    assert np.allclose(out[3], system.step(x, u, {k: (v[3] if np.ndim(v) else v) for k, v in p.items()}))


def test_shifted_params_endpoints():
    s = CartPole()
    assert s.shifted_params(0.0) == s.nominal_params
    p1 = s.shifted_params(1.0)
    assert p1["m_pole"] == pytest.approx(0.4)
    assert p1["gain"] == pytest.approx(0.6)
    assert p1["friction"] == pytest.approx(2.1)
    # extrapolation beyond the reference shift
    assert s.shifted_params(1.5)["gain"] == pytest.approx(0.4)


def test_angle_wrapping():
    s = CartPole()
    x = np.array([0.0, 3.5, 0.0, 0.0])
    assert -np.pi <= s.wrap(x)[1] < np.pi
    assert s.diff(np.array([0, np.pi - 0.1, 0, 0]), np.array([0, -np.pi + 0.1, 0, 0]))[1] == pytest.approx(-0.2)


def test_quadrotor_hover_needs_more_thrust_when_heavier():
    q = Quadrotor2D()
    u0 = q.equilibrium_action(q.nominal_params)
    u1 = q.equilibrium_action(q.shifted_params(1.0))
    assert np.all(u1 > u0)
    assert u1[1] > u1[0]  # weaker right motor must work harder


def test_make_system():
    assert make_system("cartpole").name == "cartpole"
    with pytest.raises(ValueError):
        make_system("nope")


def test_env_api_and_noise():
    s = Quadrotor2D()
    env = SandboxEnv(s, ConditionSchedule(s, horizon=10))
    obs = env.reset(seed=0)
    assert obs.shape == (6,)
    for _ in range(10):
        obs, cost, term, trunc, info = env.step(s.equilibrium_action())
    assert trunc and not term
    assert cost >= 0
    assert not np.allclose(obs, info["state"])  # sensor noise is applied


def test_env_observation_delay():
    s = CartPole()
    env = SandboxEnv(s, ConditionSchedule(s, horizon=20, delay_steps=3, noise_level=0.0))
    env.reset(seed=0)
    states = [env.state.copy()]
    obs_seen = []
    for _ in range(6):
        obs, *_rest, info = env.step(np.array([2.0]))
        states.append(info["state"])
        obs_seen.append(obs)
    # observation after step k reflects the state from step k-3
    assert np.allclose(obs_seen[5], states[3])
