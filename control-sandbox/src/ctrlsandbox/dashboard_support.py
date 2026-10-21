"""Bounded dashboard adapter using existing controllers and simulator."""
import argparse
import pickle
from pathlib import Path
import numpy as np
from .conditions import MODES, ConditionSchedule
from .registry import controller_factory
from .runner import run_episode, summarize
from .systems import SYSTEMS, make_system

LIVE_CONTROLLERS = ("lqr", "mppi", "adaptive_mppi")


def simulate(system_name, controller_name, mode, severity, horizon, seed):
    if system_name not in SYSTEMS or controller_name not in LIVE_CONTROLLERS or mode not in MODES:
        raise ValueError("Unsupported live configuration")
    if not np.isfinite(severity) or not 0 <= severity <= 1:
        raise ValueError("Severity must be between 0 and 1")
    if not isinstance(horizon, int) or not 10 <= horizon <= 300:
        raise ValueError("Horizon must be an integer between 10 and 300")
    if not isinstance(seed, int) or not 0 <= seed <= 1_000_000:
        raise ValueError("Invalid seed")
    system = make_system(system_name)
    schedule = ConditionSchedule(system, mode=mode, severity=severity, horizon=horizon)
    controller = controller_factory(controller_name, system, seed=seed)()
    episode = run_episode(system, controller, schedule, seed=seed)
    return episode, summarize(system, episode, schedule)


def main():
    parser = argparse.ArgumentParser()
    for name in ("system", "controller", "mode"):
        parser.add_argument(name)
    parser.add_argument("severity", type=float)
    parser.add_argument("horizon", type=int)
    parser.add_argument("seed", type=int)
    parser.add_argument("output", type=Path)
    a = parser.parse_args()
    result = simulate(a.system, a.controller, a.mode, a.severity, a.horizon, a.seed)
    # Only exchanged via a private TemporaryDirectory; never load user pickles.
    a.output.write_bytes(pickle.dumps(result))


if __name__ == "__main__":
    main()
