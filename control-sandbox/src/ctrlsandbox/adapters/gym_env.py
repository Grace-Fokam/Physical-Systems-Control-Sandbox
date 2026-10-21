"""A ``gymnasium.Env`` view of the sandbox (requires the ``rl`` extra).

Actions are normalised to ``[-1, 1]``; reward is ``-cost / reward_scale``; a failure
ends the episode with the same penalty the benchmark uses. With
``randomize=True`` each reset draws parameters with independent severities in
``[0, max_severity]`` (domain randomization for training).
"""

from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError('gymnasium is required: pip install -e ".[rl]"') from exc

from ..conditions import ConditionSchedule
from ..env import SandboxEnv
from ..systems import System, make_system


class GymSandboxEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        system: System | str = "cartpole",
        mode: str = "static",
        severity: float = 0.0,
        horizon: int = 300,
        randomize: bool = False,
        max_severity: float = 1.0,
        reward_scale: float = 10.0,
    ):
        super().__init__()
        self.system = make_system(system) if isinstance(system, str) else system
        self.schedule = ConditionSchedule(self.system, mode=mode, severity=severity, horizon=horizon)
        self.env = SandboxEnv(self.system, self.schedule)
        self.randomize = randomize
        self.max_severity = max_severity
        self.reward_scale = reward_scale
        n, m = self.system.state_dim, self.system.action_dim
        self.observation_space = spaces.Box(-np.inf, np.inf, (n,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (m,), dtype=np.float32)
        self._mid = (self.system.action_high + self.system.action_low) / 2.0
        self._span = (self.system.action_high - self.system.action_low) / 2.0

    def to_physical(self, a):
        return self._mid + self._span * np.clip(np.asarray(a, dtype=float), -1.0, 1.0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        s = int(self.np_random.integers(1 << 31))
        obs = self.env.reset(seed=s)
        if self.randomize:
            # override the schedule's parameters with a random draw for this episode
            p, _ = self.system.sample_params(self.np_random, 0.0, self.max_severity)
            self.env.schedule.params_at = lambda t, p=p: dict(p)
        return obs.astype(np.float32), {}

    def step(self, action):
        obs, cost, terminated, truncated, info = self.env.step(self.to_physical(action))
        if terminated:
            cost += self.system.fail_penalty * (self.env.horizon - self.env.t)
        return obs.astype(np.float32), -cost / self.reward_scale, terminated, truncated, info
