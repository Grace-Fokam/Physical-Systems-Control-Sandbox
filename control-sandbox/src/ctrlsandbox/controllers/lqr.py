from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are

from ..systems import System
from .base import Controller


def linearize(system: System, x0, u0, params=None, eps: float = 1e-5, step=None):
    """Finite-difference linearization of a discrete-time step: x' ~ A dx + B du.

    ``step(x, u)`` defaults to the system physics at ``params``; pass a learned
    model's ``predict`` to linearize that instead.
    """
    params = system.nominal_params if params is None else params
    if step is None:
        step = lambda x, u: system.step(x, u, params)  # noqa: E731
    x0 = np.asarray(x0, dtype=float)
    u0 = np.asarray(u0, dtype=float)
    n, m = system.state_dim, system.action_dim
    A = np.zeros((n, n))
    B = np.zeros((n, m))
    for i in range(n):
        d = np.zeros(n)
        d[i] = eps
        A[:, i] = system.diff(step(x0 + d, u0), step(x0 - d, u0)) / (2 * eps)
    for j in range(m):
        d = np.zeros(m)
        d[j] = eps
        B[:, j] = system.diff(step(x0, u0 + d), step(x0, u0 - d)) / (2 * eps)
    return A, B


def lqr_gain(A, B, Q, R):
    """Discrete LQR gain ``K`` and Riccati solution ``P`` (diagonal Q, R given as vectors or matrices)."""
    Q = np.diag(Q) if np.ndim(Q) == 1 else Q
    R = np.diag(R) if np.ndim(R) == 1 else R
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


class LQRController(Controller):
    """Infinite-horizon discrete LQR around the target, built from a (nominal) model."""

    name = "lqr"

    def __init__(self, system: System, params=None, Q=None, R=None):
        self.system = system
        self.params = dict(system.nominal_params if params is None else params)
        self.u_eq = system.equilibrium_action(self.params)
        A, B = linearize(system, system.target, self.u_eq, self.params)
        self.K, self.P = lqr_gain(A, B, system.Q if Q is None else Q, system.R if R is None else R)

    def act(self, obs):
        e = self.system.state_error(obs)
        u = self.u_eq - self.K @ e
        return np.clip(u, self.system.action_low, self.system.action_high)
