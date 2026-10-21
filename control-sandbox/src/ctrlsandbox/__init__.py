"""control-sandbox: control physical systems under changing conditions.

The package is organised around four pieces:

* ``systems``     - batched, differentiable-by-finite-difference physics models
* ``conditions``  - a scheduler that shifts parameters, injects faults and gusts
* ``controllers`` - LQR, MPPI (sampling MPC), adaptive MPPI, ES policies, learned-model MPPI
* ``runner`` / ``sweep`` - episodes, metrics and robustness sweeps
"""

from .systems import CartPole, Quadrotor2D, System, make_system, SYSTEMS
from .conditions import ConditionSchedule, MODES
from .env import SandboxEnv
from .runner import run_episode, summarize

__all__ = [
    "CartPole",
    "Quadrotor2D",
    "System",
    "make_system",
    "SYSTEMS",
    "ConditionSchedule",
    "MODES",
    "SandboxEnv",
    "run_episode",
    "summarize",
]

__version__ = "0.1.0"
