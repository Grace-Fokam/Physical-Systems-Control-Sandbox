"""Optional-extra tests; skipped unless gymnasium / stable-baselines3 are installed."""

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")


def test_gym_env_checker():
    from gymnasium.utils.env_checker import check_env

    from ctrlsandbox.adapters.gym_env import GymSandboxEnv

    env = GymSandboxEnv("cartpole", horizon=50)
    check_env(env, skip_render_check=True)


def test_gym_env_randomized_episode():
    from ctrlsandbox.adapters.gym_env import GymSandboxEnv

    env = GymSandboxEnv("quadrotor2d", horizon=20, randomize=True)
    obs, _ = env.reset(seed=0)
    total = 0.0
    for _ in range(20):
        obs, r, term, trunc, _ = env.step(np.zeros(2, dtype=np.float32))
        total += r
        if term or trunc:
            break
    assert obs.shape == (6,) and np.isfinite(total)


def test_sb3_short_training(tmp_path):
    pytest.importorskip("stable_baselines3")
    from ctrlsandbox import CartPole
    from ctrlsandbox.adapters.sb3 import load_or_train

    agent = load_or_train("ppo", CartPole(), cache_dir=tmp_path, timesteps=512)
    u = agent.act(np.zeros(4))
    assert u.shape == (1,)
