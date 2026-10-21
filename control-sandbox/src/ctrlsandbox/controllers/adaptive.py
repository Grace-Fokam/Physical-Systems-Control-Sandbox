from __future__ import annotations

import numpy as np

from ..systems import System
from .base import Controller
from .mppi import MPPIController


class ParticleParamEstimator:
    """Online Bayesian estimate of physical parameters from observed transitions.

    Holds ``n`` parameter hypotheses, scores each by how well it predicted the last
    observed transition, and resamples with jitter so the estimate can track drift
    and abrupt faults. A forgetting factor keeps old evidence from dominating.
    """

    def __init__(
        self,
        system: System,
        names=None,
        n: int = 256,
        noise_std=None,
        jitter: float = 0.01,
        forget: float = 0.97,
    ):
        self.system = system
        self.names = list(names or system.adapt_ranges)
        self.low = np.array([system.adapt_ranges[k][0] for k in self.names])
        self.high = np.array([system.adapt_ranges[k][1] for k in self.names])
        self.n = n
        floor = 0.5 * system.obs_noise_scale
        base = np.sqrt(2.0) * system.obs_noise_scale
        self.noise_std = np.sqrt(base**2 + floor**2) if noise_std is None else np.asarray(noise_std)
        self.jitter = jitter
        self.forget = forget
        self.rng = np.random.default_rng(0)
        self.reset()

    def reset(self, rng=None):
        if rng is not None:
            self.rng = rng
        self.particles = self.rng.uniform(self.low, self.high, (self.n, len(self.names)))
        nominal = np.array([self.system.nominal_params[k] for k in self.names])
        self.particles[0] = nominal  # nominal is always one of the hypotheses
        self.logw = np.zeros(self.n)

    def _params(self, values):
        p = dict(self.system.nominal_params)
        for i, k in enumerate(self.names):
            p[k] = values[..., i]
        return p

    def update(self, x_prev, u, x_obs):
        pred = self.system.step(x_prev, u, self._params(self.particles))
        err = self.system.diff(pred, x_obs) / self.noise_std
        loglik = -0.5 * np.sum(err**2, axis=-1)
        self.logw = self.forget * self.logw + loglik
        self.logw -= self.logw.max()
        w = np.exp(self.logw)
        w /= w.sum()
        if 1.0 / np.sum(w**2) < self.n / 2:
            idx = self._systematic_resample(w)
            self.particles = self.particles[idx]
            self.particles += self.rng.standard_normal(self.particles.shape) * self.jitter * (self.high - self.low)
            self.particles = np.clip(self.particles, self.low, self.high)
            self.logw = np.zeros(self.n)

    def _systematic_resample(self, w):
        positions = (self.rng.random() + np.arange(self.n)) / self.n
        return np.minimum(np.searchsorted(np.cumsum(w), positions), self.n - 1)

    def estimate(self) -> dict:
        w = np.exp(self.logw - self.logw.max())
        w /= w.sum()
        mean = w @ self.particles
        return {k: float(v) for k, v in zip(self.names, mean)}


class AdaptiveMPPI(Controller):
    """MPPI whose planning model is updated every step by a particle-filter estimate."""

    name = "adaptive_mppi"

    def __init__(self, system: System, estimator_kwargs=None, **mppi_kwargs):
        self.system = system
        self.mppi = MPPIController(system, **mppi_kwargs)
        self.estimator = ParticleParamEstimator(system, **(estimator_kwargs or {}))

    def reset(self, rng=None):
        rng = rng if rng is not None else np.random.default_rng(0)
        self.estimator.reset(np.random.default_rng(rng.integers(1 << 31)))
        self.mppi.params = dict(self.system.nominal_params)
        self.mppi.reset(rng)

    def observe(self, obs, action, next_obs):
        self.estimator.update(obs, action, next_obs)
        self.mppi.params.update(self.estimator.estimate())

    def act(self, obs):
        return self.mppi.act(obs)
