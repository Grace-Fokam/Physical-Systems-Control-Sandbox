"""Model-free policy search with evolution strategies (Salimans et al., 2017).

A dependency-free model-free RL baseline: a linear feedback policy trained only
from episode returns, optionally with domain randomization over the physical
parameters. The whole population is simulated in one batched rollout.
PPO and SAC via Stable-Baselines3 live in ``ctrlsandbox.adapters.sb3``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..systems import System
from .base import Controller


@dataclass
class LinearPolicy:
    """u = u_eq + scale * (b + W @ (error / state_scale)), clipped to the action bounds."""

    system: System
    theta: np.ndarray = field(default=None)

    def __post_init__(self):
        n, m = self.system.state_dim, self.system.action_dim
        if self.theta is None:
            self.theta = np.zeros(m * n + m)
        self.theta = np.asarray(self.theta, dtype=float)

    @property
    def n_params(self) -> int:
        return self.system.action_dim * (self.system.state_dim + 1)

    def act_batch(self, thetas: np.ndarray, x: np.ndarray) -> np.ndarray:
        sys = self.system
        n, m = sys.state_dim, sys.action_dim
        W = thetas[..., : m * n].reshape(*thetas.shape[:-1], m, n)
        b = thetas[..., m * n :]
        e = sys.state_error(x) / sys.state_scale
        scale = (sys.action_high - sys.action_low) / 2.0
        u = sys.equilibrium_action(sys.nominal_params) + scale * (b + np.einsum("...mn,...n->...m", W, e))
        return np.clip(u, sys.action_low, sys.action_high)

    def act(self, x: np.ndarray) -> np.ndarray:
        return self.act_batch(self.theta, x)

    def save(self, path) -> None:
        np.savez(path, theta=self.theta, system=self.system.name)

    @classmethod
    def load(cls, system: System, path) -> "LinearPolicy":
        return cls(system, np.load(path)["theta"])


def batched_rollout_costs(system, policy, thetas, params, x0, steps):
    """Total episode cost for each row of ``thetas`` (with failure penalty)."""
    x = np.array(x0, dtype=float)
    B = x.shape[0]
    cost = np.zeros(B)
    alive = np.ones(B, dtype=bool)
    for t in range(steps):
        u = policy.act_batch(thetas, x)
        x_next = system.step(x, u, params)
        c = system.stage_cost(x_next, u)
        cost += np.where(alive, c, 0.0)
        newly = alive & system.failed(x_next)
        cost += newly * system.fail_penalty * (steps - t - 1)
        alive &= ~newly
        x = np.where(alive[:, None], x_next, x)
    return cost


def train_es(
    system: System,
    iters: int = 200,
    population: int = 32,
    sigma: float = 0.1,
    lr: float = 0.05,
    rollouts: int = 4,
    steps: int = 200,
    dr_severity: float = 0.0,
    seed: int = 0,
    verbose: bool = False,
):
    """Train a ``LinearPolicy``. ``dr_severity > 0`` samples each episode's
    parameters with independent severities in ``[0, dr_severity]``.

    Returns ``(policy, history)`` where history is mean population cost per iteration.
    """
    rng = np.random.default_rng(seed)
    policy = LinearPolicy(system)
    d = policy.n_params
    theta = np.zeros(d)
    m_adam, v_adam = np.zeros(d), np.zeros(d)
    history = []
    for it in range(iters):
        eps = rng.standard_normal((population, d))
        cand = np.concatenate([theta + sigma * eps, theta - sigma * eps])  # antithetic
        P = cand.shape[0]
        # common random numbers: every candidate sees the same R scenarios
        x0 = np.stack([system.sample_initial_state(rng) for _ in range(rollouts)])
        if dr_severity > 0:
            p_r, _ = system.sample_params(rng, 0.0, dr_severity, size=rollouts)
        else:
            p_r = dict(system.nominal_params)
        params = {k: np.tile(np.broadcast_to(v, (rollouts,)), P) for k, v in p_r.items()}
        thetas = np.repeat(cand, rollouts, axis=0)
        costs = batched_rollout_costs(system, policy, thetas, params, np.tile(x0, (P, 1)), steps)
        fitness = -costs.reshape(P, rollouts).mean(axis=1)
        history.append(float(-fitness.mean()))
        ranks = np.empty(P)
        ranks[np.argsort(fitness)] = np.arange(P)
        shaped = ranks / (P - 1) - 0.5
        grad = (shaped[:population] - shaped[population:]) @ eps / (2 * population * sigma)
        # Adam ascent
        m_adam = 0.9 * m_adam + 0.1 * grad
        v_adam = 0.999 * v_adam + 0.001 * grad**2
        mh = m_adam / (1 - 0.9 ** (it + 1))
        vh = v_adam / (1 - 0.999 ** (it + 1))
        theta = theta + lr * mh / (np.sqrt(vh) + 1e-8)
        if verbose and it % 20 == 0:
            print(f"[es] iter {it:4d}  mean cost {history[-1]:.1f}")
    policy.theta = theta
    return policy, history


class ESPolicyController(Controller):
    name = "es"

    def __init__(self, policy: LinearPolicy, name: str | None = None):
        self.policy = policy
        if name:
            self.name = name

    def act(self, obs):
        return self.policy.act(obs)
