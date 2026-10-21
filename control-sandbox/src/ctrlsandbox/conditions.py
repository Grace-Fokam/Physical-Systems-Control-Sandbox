"""Condition schedules: how the world changes during an episode.

Four modes, each scaled by a single ``severity`` so results can be read as a
robustness curve (performance versus severity):

``static``       parameters sit at the shifted value for the whole episode
``drift``        parameters move linearly from nominal to the shifted value
``fault``        nominal until ``fault_step``, then an abrupt jump to the shifted value
``disturbance``  nominal parameters, plus random gusts and extra sensor noise
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .systems import System

MODES = ("static", "drift", "fault", "disturbance")


@dataclass
class ConditionSchedule:
    system: System
    mode: str = "static"
    severity: float = 0.0
    horizon: int = 300
    fault_step: int | None = None
    noise_level: float = 1.0  # baseline sensor noise, in units of system.obs_noise_scale
    delay_steps: int = 0  # observation latency
    gust_tau: float = 0.5  # gust correlation time (s)
    _gust: float = field(default=0.0, init=False, repr=False)
    _rng: np.random.Generator | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.fault_step is None:
            self.fault_step = self.horizon // 3

    def reset(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._gust = 0.0

    def effective_severity(self, t: int) -> float:
        if self.mode == "static":
            return self.severity
        if self.mode == "drift":
            return self.severity * min(t / max(self.horizon - 1, 1), 1.0)
        if self.mode == "fault":
            return self.severity if t >= self.fault_step else 0.0
        return 0.0  # disturbance: parameters stay nominal

    def params_at(self, t: int) -> dict:
        """Parameters for step ``t``. Stateful in ``disturbance`` mode: call once per step."""
        p = self.system.shifted_params(self.effective_severity(t))
        name = self.system.disturbance_param
        if self.mode == "disturbance" and name is not None:
            if self._rng is None:
                self._rng = np.random.default_rng()
            scale = self.severity * self.system.disturbance_scale
            a = self.system.dt / self.gust_tau
            # Ornstein-Uhlenbeck process with stationary std == scale
            self._gust += -a * self._gust + scale * np.sqrt(2 * a) * self._rng.standard_normal()
            p[name] = p[name] + self._gust
        return p

    def obs_noise_std(self) -> np.ndarray:
        level = self.noise_level
        if self.mode == "disturbance":
            level = level + 2.0 * self.severity
        return level * self.system.obs_noise_scale
