"""Add your own controller in a few lines and benchmark it against LQR.

    python examples/custom_controller.py
"""

import numpy as np

from ctrlsandbox import CartPole, ConditionSchedule, run_episode, summarize
from ctrlsandbox.controllers import Controller, LQRController


class GainScheduledLQR(Controller):
    """A toy idea: switch to a more aggressive LQR when the pole is far from upright."""

    name = "gain_scheduled_lqr"

    def __init__(self, system):
        self.system = system
        self.gentle = LQRController(system)
        self.aggressive = LQRController(system, R=system.R * 0.1)

    def act(self, obs):
        far = abs(self.system.state_error(obs)[1]) > 0.1
        return (self.aggressive if far else self.gentle).act(obs)


system = CartPole()
for severity in (0.0, 0.5, 1.0):
    sched = ConditionSchedule(system, "static", severity, horizon=300)
    for ctrl in (LQRController(system), GainScheduledLQR(system)):
        r = [summarize(system, run_episode(system, ctrl, sched, seed=s)) for s in range(5)]
        print(f"severity {severity:.1f}  {getattr(ctrl, 'name', '?'):20s}"
              f" cost {np.mean([x['total_cost'] for x in r]):8.1f}  survived {np.mean([x['survived'] for x in r]):.0%}")
