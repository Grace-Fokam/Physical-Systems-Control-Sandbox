Mean episode cost at severity 1 (share of episodes without failure)

| Controller | Static shift | Gradual drift | Abrupt fault | Gusts + sensor noise |
| --- | --- | --- | --- | --- |
| LQR | 936 (80%) | 24 (100%) | 51 (100%) | 106 (100%) |
| MPPI (nominal) | 2,661 (40%) | 27 (100%) | 85 (100%) | 922 (80%) |
| Adaptive MPPI | 98 (100%) | 22 (100%) | 37 (100%) | 78 (100%) |
| ES policy | 908 (80%) | 22 (100%) | 37 (100%) | 74 (100%) |
| ES policy + DR | 64 (100%) | 22 (100%) | 34 (100%) | 81 (100%) |
| Learned-model MPPI | 2,644 (40%) | 28 (100%) | 83 (100%) | 925 (80%) |
