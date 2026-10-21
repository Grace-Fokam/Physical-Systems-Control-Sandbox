"""A small, dependency-free environment with a Gymnasium-like API.

``step`` returns ``(obs, cost, terminated, truncated, info)``; note it returns a
*cost* (lower is better). ``ctrlsandbox.adapters.gym_env`` wraps this as a real
``gymnasium.Env`` with reward = -cost for RL libraries.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .conditions import ConditionSchedule
from .systems import System


class SandboxEnv:
    def __init__(self, system: System, schedule: ConditionSchedule | None = None, horizon: int | None = None):
        self.system = system
        if schedule is None:
            schedule = ConditionSchedule(system, horizon=horizon or 300)
        self.schedule = schedule
        self.horizon = horizon or schedule.horizon
        self.rng = np.random.default_rng()
        self.t = 0
        self.state = np.zeros(system.state_dim)
        self.params = dict(system.nominal_params)
        self._obs = deque(maxlen=1)

    def reset(self, seed: int | None = None, x0: np.ndarray | None = None) -> np.ndarray:
        self.rng = np.random.default_rng(seed)
        self.schedule.reset(self.rng)
        self.t = 0
        self.state = (
            np.array(x0, dtype=float) if x0 is not None else self.system.sample_initial_state(self.rng)
        )
        self.params = self.system.shifted_params(self.schedule.effective_severity(0))
        self._obs = deque(maxlen=self.schedule.delay_steps + 1)
        first = self._noisy()
        for _ in range(self.schedule.delay_steps + 1):
            self._obs.append(first)
        return self._obs[0].copy()

    def _noisy(self) -> np.ndarray:
        std = self.schedule.obs_noise_std()
        return self.system.wrap(self.state + std * self.rng.standard_normal(self.system.state_dim))

    def step(self, u):
        u = np.clip(np.asarray(u, dtype=float), self.system.action_low, self.system.action_high)
        self.params = self.schedule.params_at(self.t)
        self.state = self.system.step(self.state, u, self.params)
        self.t += 1
        cost = float(self.system.stage_cost(self.state, u))
        terminated = bool(self.system.failed(self.state))
        truncated = self.t >= self.horizon
        self._obs.append(self._noisy())
        info = {"state": self.state.copy(), "params": dict(self.params), "t": self.t, "action": u}
        return self._obs[0].copy(), cost, terminated, truncated, info
