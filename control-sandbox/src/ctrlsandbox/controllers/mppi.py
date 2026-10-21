from __future__ import annotations

from typing import Callable

import numpy as np

from ..systems import System
from .base import Controller
from .lqr import linearize, lqr_gain

# per-system defaults that work out of the box
MPPI_DEFAULTS = {
    "cartpole": dict(horizon=20, samples=512, noise_sigma=[2.0], temperature=0.05),
    "quadrotor2d": dict(horizon=20, samples=512, noise_sigma=[0.6, 0.6], temperature=0.05),
}


class MPPIController(Controller):
    """Model Predictive Path Integral control (sampling-based MPC).

    The planning model is ``dynamics(x_batch, u_batch) -> x_next_batch``. By default it
    is the system's own physics evaluated at ``self.params`` (nominal unless changed),
    so swapping in a learned model or an updated parameter estimate is one assignment.

    The terminal cost is the LQR value function ``e' P e`` of the *planning model*
    linearized at the target (``terminal="lqr"``), which lets short horizons plan
    stably; ``terminal="quadratic"`` uses the system's plain terminal cost instead.
    """

    name = "mppi"

    def __init__(
        self,
        system: System,
        params=None,
        horizon: int | None = None,
        samples: int | None = None,
        noise_sigma=None,
        temperature: float | None = None,
        dynamics: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        terminal: str = "lqr",
        failure_cost: float = 1e4,
    ):
        cfg = dict(MPPI_DEFAULTS.get(system.name, MPPI_DEFAULTS["cartpole"]))
        self.system = system
        self.params = dict(system.nominal_params if params is None else params)
        self.H = horizon or cfg["horizon"]
        self.K = samples or cfg["samples"]
        sigma = noise_sigma if noise_sigma is not None else cfg["noise_sigma"]
        self.sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (system.action_dim,)).copy()
        self.temperature = temperature or cfg["temperature"]
        self.failure_cost = failure_cost
        self.terminal = terminal
        self._dynamics = dynamics
        self.rng = np.random.default_rng(0)
        self.U = None
        self.P = None

    # ---------------------------------------------------------------- model
    def dynamics(self, x, u):
        if self._dynamics is not None:
            return self._dynamics(x, u)
        return self.system.step(x, u, self.params)

    def nominal_action(self) -> np.ndarray:
        return self.system.equilibrium_action(self.params)

    def refresh_terminal(self) -> None:
        """Recompute the LQR terminal cost from the current planning model."""
        self.P = None
        if self.terminal != "lqr":
            return
        try:
            A, B = linearize(self.system, self.system.target, self.nominal_action(), step=self.dynamics)
            _, P = lqr_gain(A, B, self.system.Q, self.system.R)
            if np.all(np.isfinite(P)):
                self.P = 0.5 * (P + P.T)
        except (np.linalg.LinAlgError, ValueError):
            self.P = None  # fall back to the quadratic terminal cost

    def terminal_cost(self, x):
        if self.P is None:
            return self.system.terminal_cost(x)
        e = self.system.state_error(x)
        return np.einsum("...i,ij,...j->...", e, self.P, e)

    # --------------------------------------------------------------- control
    def reset(self, rng=None):
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.U = np.tile(self.nominal_action(), (self.H, 1))
        self.refresh_terminal()

    def rollout_costs(self, x0, U):
        sys = self.system
        x = np.repeat(np.asarray(x0, dtype=float)[None], U.shape[0], axis=0)
        cost = np.zeros(U.shape[0])
        dead = np.zeros(U.shape[0], dtype=bool)
        for t in range(self.H):
            x = self.dynamics(x, U[:, t])
            cost += sys.stage_cost(x, U[:, t])
            dead |= sys.failed(x) | ~np.all(np.isfinite(x), axis=-1)
            x = np.nan_to_num(x)
        cost += self.terminal_cost(x) + self.failure_cost * dead
        return np.nan_to_num(cost, nan=1e12, posinf=1e12)

    def act(self, obs):
        if self.U is None:
            self.reset()
        sys = self.system
        eps = self.rng.standard_normal((self.K, self.H, sys.action_dim)) * self.sigma
        eps[0] = 0.0  # always evaluate the current plan itself
        U = np.clip(self.U[None] + eps, sys.action_low, sys.action_high)
        cost = self.rollout_costs(obs, U)
        best = cost.min()
        spread = np.percentile(cost, 50) - best + 1e-9
        w = np.exp(-(cost - best) / (self.temperature * spread))
        w /= w.sum()
        self.U = np.einsum("k,khm->hm", w, U)
        u = self.U[0].copy()
        self.U = np.vstack([self.U[1:], self.nominal_action()[None]])
        return u
