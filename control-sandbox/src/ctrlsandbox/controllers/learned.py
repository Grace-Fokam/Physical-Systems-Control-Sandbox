"""Learned dynamics inside MPC.

``LearnedMPPI`` plans with any object exposing ``predict(x, u) -> x_next`` on
batches. ``RFFDynamicsModel`` is a small reference model (random Fourier features
plus ridge regression) so the sandbox runs end to end on its own; the companion
repo ``dynamics-learning`` provides richer models through the same interface.
"""

from __future__ import annotations

import numpy as np

from ..systems import System
from .base import Controller
from .lqr import LQRController
from .mppi import MPPIController


def featurize(system: System, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """State-action features: angles as (sin, cos), everything else scaled."""
    x = np.asarray(x, dtype=float)
    cols = []
    for i in range(system.state_dim):
        if i in system.angle_indices:
            cols += [np.sin(x[..., i]), np.cos(x[..., i])]
        else:
            cols.append(x[..., i] / system.state_scale[i])
    span = (system.action_high - system.action_low) / 2.0
    mid = (system.action_high + system.action_low) / 2.0
    u = (np.asarray(u, dtype=float) - mid) / span
    return np.concatenate([np.stack(np.broadcast_arrays(*cols), axis=-1), u], axis=-1)


class RFFDynamicsModel:
    """Predicts the state change with ridge regression on random Fourier features."""

    def __init__(self, system: System, n_features: int = 300, lengthscale: float = 1.5, ridge: float = 1e-4, seed: int = 0):
        self.system = system
        self.n_features = n_features
        self.lengthscale = lengthscale
        self.ridge = ridge
        self.rng = np.random.default_rng(seed)
        self.W = None

    def _phi(self, x, u):
        z = featurize(self.system, x, u)
        proj = z @ self.Omega + self.phase
        return np.concatenate([np.sqrt(2.0 / self.n_features) * np.cos(proj), z, np.ones(z.shape[:-1] + (1,))], axis=-1)

    def fit(self, X, U, Xn):
        d_in = featurize(self.system, X[:1], U[:1]).shape[-1]
        self.Omega = self.rng.standard_normal((d_in, self.n_features)) / self.lengthscale
        self.phase = self.rng.uniform(0, 2 * np.pi, self.n_features)
        Phi = self._phi(X, U)
        D = self.system.diff(Xn, X)
        self.d_scale = D.std(axis=0) + 1e-8
        A = Phi.T @ Phi + self.ridge * len(Phi) * np.eye(Phi.shape[1])
        self.W = np.linalg.solve(A, Phi.T @ (D / self.d_scale))
        return self

    def predict(self, x, u):
        delta = (self._phi(x, u) @ self.W) * self.d_scale
        return self.system.wrap(np.asarray(x, dtype=float) + delta)


def collect_transitions(
    system: System,
    episodes: int = 40,
    steps: int = 150,
    action_noise: float = 0.5,
    params_sampler=None,
    obs_noise: float = 0.0,
    seed: int = 0,
):
    """Exploration data: an LQR controller plus correlated action noise.

    ``action_noise`` is a fraction of the action half-range. ``params_sampler(rng)``
    optionally returns per-episode parameters (defaults to nominal).
    Returns ``(X, U, Xn)`` stacked across episodes.
    """
    rng = np.random.default_rng(seed)
    lqr = LQRController(system)
    span = (system.action_high - system.action_low) / 2.0
    Xs, Us, Xns = [], [], []
    for _ in range(episodes):
        p = params_sampler(rng) if params_sampler else dict(system.nominal_params)
        x = system.sample_initial_state(rng)
        noise = np.zeros(system.action_dim)
        for _ in range(steps):
            noise = 0.8 * noise + 0.6 * action_noise * span * rng.standard_normal(system.action_dim)
            u = np.clip(lqr.act(x) + noise, system.action_low, system.action_high)
            xn = system.step(x, u, p)
            Xs.append(x + obs_noise * system.obs_noise_scale * rng.standard_normal(system.state_dim))
            Us.append(u)
            Xns.append(xn)
            if system.failed(xn):
                break
            x = xn
    return np.array(Xs), np.array(Us), np.array(Xns)


class LearnedMPPI(MPPIController):
    """MPPI that plans with a learned model instead of the physics."""

    name = "learned_mppi"

    def __init__(self, system: System, model, samples: int = 256, **kwargs):
        super().__init__(system, dynamics=model.predict, samples=samples, **kwargs)
        self.model = model
