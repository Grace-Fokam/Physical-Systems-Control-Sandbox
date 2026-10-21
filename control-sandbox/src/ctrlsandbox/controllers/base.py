from __future__ import annotations

import numpy as np


class Controller:
    """Interface every controller implements.

    ``act(obs)`` returns an action. ``observe(obs, action, next_obs)`` is called
    after each environment step so adaptive or learning controllers can update.
    """

    name = "controller"

    def reset(self, rng: np.random.Generator | None = None) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def observe(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray) -> None:
        pass
