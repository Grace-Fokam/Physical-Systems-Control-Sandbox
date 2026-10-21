"""Compare four controllers on a quadrotor whose payload got heavier and whose
right motor lost 20% of its thrust. Runs in well under a minute on a laptop.

    python examples/quickstart.py
"""

import numpy as np

from ctrlsandbox import ConditionSchedule, Quadrotor2D, run_episode, summarize
from ctrlsandbox.controllers import AdaptiveMPPI, LQRController, MPPIController

system = Quadrotor2D()
controllers = {
    "LQR (nominal model)": lambda: LQRController(system),
    "MPPI (nominal model)": lambda: MPPIController(system),
    "Adaptive MPPI": lambda: AdaptiveMPPI(system),
}

for mode in ("static", "fault"):
    schedule = ConditionSchedule(system, mode=mode, severity=1.0, horizon=300)
    print(f"\n{mode} shift, severity 1.0")
    for name, make in controllers.items():
        results = [summarize(system, run_episode(system, make(), schedule, seed=s), schedule) for s in range(3)]
        cost = np.mean([r["total_cost"] for r in results])
        survived = np.mean([r["survived"] for r in results])
        print(f"  {name:22s} cost {cost:8.1f}   survived {survived:.0%}")

# What did the adaptive controller learn about the world?
ctrl = AdaptiveMPPI(system)
run_episode(system, ctrl, ConditionSchedule(system, "static", 1.0, horizon=200), seed=0)
true = system.shifted_params(1.0)
print("\nestimated vs true parameters after 4 seconds:")
for k, v in ctrl.estimator.estimate().items():
    print(f"  {k:10s} {v:6.3f}  (true {true[k]:.3f})")
