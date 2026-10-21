"""PPO and SAC baselines through Stable-Baselines3 (requires the ``rl`` extra).

    ctrl-sweep --system cartpole --controllers lqr,adaptive_mppi,ppo,sac

Agents are trained once with domain randomization and cached next to the
other trained controllers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..controllers.base import Controller
from ..systems import System

TIMESTEPS = {"ppo": 300_000, "sac": 100_000}


class SB3Controller(Controller):
    def __init__(self, model, env, name: str):
        self.model = model
        self.env = env
        self.name = name

    def act(self, obs):
        action, _ = self.model.predict(np.asarray(obs, dtype=np.float32), deterministic=True)
        return self.env.to_physical(action)


def load_or_train(algo: str, system: System, cache_dir: Path | None = None, seed: int = 0, timesteps: int | None = None):
    try:
        from stable_baselines3 import PPO, SAC
    except ImportError as exc:  # pragma: no cover
        raise ImportError('stable-baselines3 is required: pip install -e ".[rl]"') from exc
    from .gym_env import GymSandboxEnv

    cls = {"ppo": PPO, "sac": SAC}[algo]
    env = GymSandboxEnv(system, randomize=True)
    path = Path(cache_dir) / f"{system.name}_{algo}_seed{seed}.zip" if cache_dir else None
    if path is not None and path.exists():
        model = cls.load(path, env=env)
    else:
        model = cls("MlpPolicy", env, seed=seed, verbose=0)
        model.learn(total_timesteps=timesteps or TIMESTEPS[algo])
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            model.save(path)
    return SB3Controller(model, env, algo)
