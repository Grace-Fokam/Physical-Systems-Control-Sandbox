"""Episodes and metrics."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .conditions import ConditionSchedule
from .controllers.base import Controller
from .env import SandboxEnv
from .systems import System


@dataclass
class EpisodeResult:
    states: np.ndarray  # (T+1, n) true states
    actions: np.ndarray  # (T, m)
    costs: np.ndarray  # (T,)
    observations: np.ndarray  # (T+1, n)
    severities: np.ndarray  # (T,) effective parameter severity per step
    compute_s: np.ndarray  # (T,) controller wall time per step
    failed: bool
    horizon: int


def run_episode(
    system: System,
    controller: Controller,
    schedule: ConditionSchedule,
    seed: int = 0,
    x0: np.ndarray | None = None,
) -> EpisodeResult:
    env = SandboxEnv(system, schedule)
    obs = env.reset(seed=seed, x0=x0)
    controller.reset(np.random.default_rng(seed + 10_000))
    states, actions, costs, observations, sev, compute = [env.state.copy()], [], [], [obs], [], []
    failed = False
    for t in range(schedule.horizon):
        t0 = time.perf_counter()
        u = np.asarray(controller.act(obs), dtype=float)
        compute.append(time.perf_counter() - t0)
        sev.append(schedule.effective_severity(t))
        next_obs, cost, terminated, truncated, info = env.step(u)
        controller.observe(obs, info["action"], next_obs)
        states.append(env.state.copy())
        actions.append(info["action"])
        costs.append(cost)
        observations.append(next_obs)
        obs = next_obs
        if terminated:
            failed = True
            break
        if truncated:
            break
    return EpisodeResult(
        states=np.array(states),
        actions=np.array(actions),
        costs=np.array(costs),
        observations=np.array(observations),
        severities=np.array(sev),
        compute_s=np.array(compute),
        failed=failed,
        horizon=schedule.horizon,
    )


def recovery_time(system: System, ep: EpisodeResult, start_step: int, tol: float = 0.25) -> float:
    """Seconds after ``start_step`` until the normalized error stays below ``tol``."""
    if ep.failed:
        return float("nan")
    err = system.normalized_error(ep.states[start_step + 1 :])
    above = np.nonzero(err > tol)[0]
    if len(above) == 0:
        return 0.0
    last = above[-1] + 1
    if last >= len(err):
        return float("nan")
    return float(last * system.dt)


def summarize(system: System, ep: EpisodeResult, schedule: ConditionSchedule | None = None) -> dict:
    steps = len(ep.costs)
    penalty = system.fail_penalty * (ep.horizon - steps) if ep.failed else 0.0
    sat = np.any(
        (ep.actions <= system.action_low + 1e-6) | (ep.actions >= system.action_high - 1e-6), axis=-1
    )
    tail = max(1, ep.horizon // 5)
    out = {
        "total_cost": float(ep.costs.sum() + penalty),
        "survived": float(not ep.failed),
        "steps": steps,
        "tracking_error": float("nan") if ep.failed else float(system.normalized_error(ep.states[-tail:]).mean()),
        "saturation": float(sat.mean()) if steps else 0.0,
        "ms_per_step": float(1e3 * ep.compute_s.mean()) if steps else 0.0,
    }
    if schedule is not None and schedule.mode == "fault":
        out["recovery_s"] = recovery_time(system, ep, schedule.fault_step)
    return out
