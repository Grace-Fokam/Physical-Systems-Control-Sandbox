"""Named controller factories, with on-disk caching for anything that is trained."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .controllers import (
    AdaptiveMPPI,
    ESPolicyController,
    LearnedMPPI,
    LinearPolicy,
    LQRController,
    MPPIController,
    RFFDynamicsModel,
    collect_transitions,
    train_es,
)
from .controllers.base import Controller
from .systems import System

CONTROLLERS = ("lqr", "mppi", "adaptive_mppi", "es", "es_dr", "learned_mppi", "ppo", "sac")
DEFAULT_CONTROLLERS = ("lqr", "mppi", "adaptive_mppi", "es", "es_dr", "learned_mppi")

DESCRIPTIONS = {
    "lqr": "LQR designed on the nominal model",
    "mppi": "Sampling MPC with the nominal physics model",
    "adaptive_mppi": "Sampling MPC + online particle-filter parameter estimates",
    "es": "Model-free linear policy (evolution strategies), nominal training",
    "es_dr": "Model-free linear policy (evolution strategies), domain randomization",
    "learned_mppi": "Sampling MPC with a learned dynamics model (RFF regression)",
    "ppo": "PPO via Stable-Baselines3 (optional extra)",
    "sac": "SAC via Stable-Baselines3 (optional extra)",
}


def _es_policy(system: System, dr: float, cache: Path | None, seed: int) -> LinearPolicy:
    path = cache / f"{system.name}_es_dr{dr:g}_seed{seed}.npz" if cache else None
    if path is not None and path.exists():
        return LinearPolicy.load(system, path)
    policy, _ = train_es(system, dr_severity=dr, seed=seed)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        policy.save(path)
    return policy


def _learned_model(system: System, cache: Path | None, seed: int) -> RFFDynamicsModel:
    path = cache / f"{system.name}_rff_data_seed{seed}.npz" if cache else None
    if path is not None and path.exists():
        d = np.load(path)
        X, U, Xn = d["X"], d["U"], d["Xn"]
    else:
        X, U, Xn = collect_transitions(system, seed=seed)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, X=X, U=U, Xn=Xn)
    return RFFDynamicsModel(system, seed=seed).fit(X, U, Xn)


def controller_factory(name: str, system: System, cache_dir=None, seed: int = 0) -> Callable[[], Controller]:
    """Return a zero-argument factory producing fresh controllers of type ``name``.

    Training (ES policies, the learned model, RL agents) happens once, here.
    """
    cache = Path(cache_dir) if cache_dir else None
    if name == "lqr":
        return lambda: LQRController(system)
    if name == "mppi":
        return lambda: MPPIController(system)
    if name == "adaptive_mppi":
        return lambda: AdaptiveMPPI(system)
    if name in ("es", "es_dr"):
        policy = _es_policy(system, 1.0 if name == "es_dr" else 0.0, cache, seed)
        return lambda: ESPolicyController(policy, name=name)
    if name == "learned_mppi":
        model = _learned_model(system, cache, seed)
        return lambda: LearnedMPPI(system, model)
    if name in ("ppo", "sac"):
        from .adapters.sb3 import load_or_train

        agent = load_or_train(name, system, cache_dir=cache, seed=seed)
        return lambda: agent
    raise ValueError(f"unknown controller {name!r}; choose from {CONTROLLERS}")
