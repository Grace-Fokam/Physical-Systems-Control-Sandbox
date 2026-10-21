"""Physical systems.

Every system exposes *batched* dynamics: states have shape ``(..., state_dim)``,
actions ``(..., action_dim)``, and any parameter may be a scalar or an array that
broadcasts against the leading batch dimensions. That one convention lets the same
code serve a single simulated episode, thousands of MPC rollouts, a particle filter
over parameter hypotheses, and population-based policy search.
"""

from __future__ import annotations

import numpy as np

Params = dict  # name -> float | np.ndarray


class System:
    """Base class. Subclasses define ``deriv`` and the class attributes below."""

    name: str = "system"
    state_dim: int = 0
    action_dim: int = 0
    dt: float = 0.02
    substeps: int = 1
    state_labels: tuple = ()
    action_labels: tuple = ()
    angle_indices: tuple = ()

    nominal_params: dict = {}
    # name -> ("mul", factor at severity 1) | ("add", offset at severity 1)
    shift_spec: dict = {}
    # name -> (low, high): prior range used by the adaptive estimator
    adapt_ranges: dict = {}
    # external force parameter driven by gusts in "disturbance" mode
    disturbance_param: str | None = None
    disturbance_scale: float = 0.0

    action_low = np.zeros(0)
    action_high = np.zeros(0)
    target = np.zeros(0)
    Q = np.zeros(0)  # diagonal state-cost weights (on state error)
    R = np.zeros(0)  # diagonal action-cost weights (on deviation from equilibrium)
    terminal_weight: float = 10.0
    state_scale = np.zeros(0)  # typical magnitude of each state, for normalisation
    obs_noise_scale = np.zeros(0)  # sensor-noise std per state at noise level 1
    fail_penalty: float = 20.0  # cost per remaining step after a failure

    # ------------------------------------------------------------------ physics
    def deriv(self, x: np.ndarray, u: np.ndarray, p: Params) -> np.ndarray:
        raise NotImplementedError

    def step(self, x: np.ndarray, u: np.ndarray, p: Params | None = None) -> np.ndarray:
        """One RK4 step of length ``dt`` (split into ``substeps``)."""
        p = self.nominal_params if p is None else p
        x = np.asarray(x, dtype=float)
        u = np.clip(np.asarray(u, dtype=float), self.action_low, self.action_high)
        h = self.dt / self.substeps
        for _ in range(self.substeps):
            k1 = self.deriv(x, u, p)
            k2 = self.deriv(x + 0.5 * h * k1, u, p)
            k3 = self.deriv(x + 0.5 * h * k2, u, p)
            k4 = self.deriv(x + h * k3, u, p)
            x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return self.wrap(x)

    # ------------------------------------------------------------- geometry
    def wrap(self, x: np.ndarray) -> np.ndarray:
        if not self.angle_indices:
            return x
        x = np.array(x, dtype=float, copy=True)
        for i in self.angle_indices:
            x[..., i] = (x[..., i] + np.pi) % (2 * np.pi) - np.pi
        return x

    def diff(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """``a - b`` with angles wrapped to [-pi, pi)."""
        d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
        return self.wrap(d)

    def state_error(self, x: np.ndarray) -> np.ndarray:
        return self.diff(x, self.target)

    # ---------------------------------------------------------------- costs
    def equilibrium_action(self, p: Params | None = None) -> np.ndarray:
        return np.zeros(self.action_dim)

    def stage_cost(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        e = self.state_error(x)
        du = np.asarray(u) - self.equilibrium_action(self.nominal_params)
        return np.sum(self.Q * e**2, axis=-1) + np.sum(self.R * du**2, axis=-1)

    def terminal_cost(self, x: np.ndarray) -> np.ndarray:
        e = self.state_error(x)
        return self.terminal_weight * np.sum(self.Q * e**2, axis=-1)

    def failed(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def normalized_error(self, x: np.ndarray) -> np.ndarray:
        return np.linalg.norm(self.state_error(x) / self.state_scale, axis=-1)

    # ------------------------------------------------------- initial states
    def sample_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    # ----------------------------------------------------- parameter shifts
    def shifted_params(self, severity=0.0, base: Params | None = None) -> Params:
        """Parameters moved along ``shift_spec`` by ``severity``.

        ``severity`` is a scalar (all parameters move together), a dict
        (per-parameter severity) and either may hold arrays for a batch.
        Severity 0 is nominal, 1 is the reference shift, values above 1 extrapolate.
        """
        p = dict(self.nominal_params if base is None else base)
        for name, (kind, amount) in self.shift_spec.items():
            s = severity[name] if isinstance(severity, dict) else severity
            nominal = self.nominal_params[name]
            if kind == "mul":
                p[name] = nominal * (1.0 + np.asarray(s) * (amount - 1.0))
            elif kind == "add":
                p[name] = nominal + np.asarray(s) * amount
            else:
                raise ValueError(f"unknown shift kind {kind!r}")
            if np.ndim(p[name]) == 0:
                p[name] = float(p[name])
        return p

    def sample_params(self, rng: np.random.Generator, low=0.0, high=1.0, size=None):
        """Independent per-parameter severities in ``[low, high]`` (domain randomization)."""
        sev = {n: rng.uniform(low, high, size) for n in self.shift_spec}
        return self.shifted_params(sev), sev


# ---------------------------------------------------------------------------
class CartPole(System):
    """Cart-pole balancing. ``theta = 0`` is upright; ``half_length`` is half the pole.

    Shift at severity 1: pole 4x heavier, 1.5x longer, extra cart friction, an
    actuator that delivers only 60% of the commanded force, and a 1.5 N actuator bias
    (the bias is what makes an abrupt fault visible while the pole is balanced).
    """

    name = "cartpole"
    state_dim = 4
    action_dim = 1
    dt = 0.02
    state_labels = ("x", "theta", "x_dot", "theta_dot")
    action_labels = ("force",)
    angle_indices = (1,)

    nominal_params = dict(
        m_cart=1.0, m_pole=0.1, half_length=0.5, friction=0.1, gain=1.0, bias=0.0, g=9.81, f_ext=0.0
    )
    shift_spec = {
        "m_pole": ("mul", 4.0),
        "half_length": ("mul", 1.5),
        "friction": ("add", 2.0),
        "gain": ("mul", 0.6),
        "bias": ("add", 1.5),
    }
    adapt_ranges = {
        "m_pole": (0.05, 0.7),
        "half_length": (0.3, 1.0),
        "friction": (0.0, 3.5),
        "gain": (0.3, 1.2),
        "bias": (-3.0, 3.0),
    }
    disturbance_param = "f_ext"
    disturbance_scale = 3.0

    action_low = np.array([-10.0])
    action_high = np.array([10.0])
    target = np.zeros(4)
    Q = np.array([1.0, 10.0, 0.1, 0.5])
    R = np.array([0.01])
    state_scale = np.array([1.0, 0.3, 1.0, 1.0])
    obs_noise_scale = np.array([0.005, 0.002, 0.01, 0.01])
    fail_penalty = 20.0

    def deriv(self, x, u, p):
        v, th, w = x[..., 2], x[..., 1], x[..., 3]
        mc, mp, l, g = p["m_cart"], p["m_pole"], p["half_length"], p["g"]
        force = p["gain"] * u[..., 0] + p["bias"] - p["friction"] * v + p["f_ext"]
        total = mc + mp
        s, c = np.sin(th), np.cos(th)
        temp = (force + mp * l * w**2 * s) / total
        th_acc = (g * s - c * temp) / (l * (4.0 / 3.0 - mp * c**2 / total))
        x_acc = temp - mp * l * th_acc * c / total
        return np.stack(np.broadcast_arrays(v, w, x_acc, th_acc), axis=-1)

    def failed(self, x):
        x = np.asarray(x)
        return (np.abs(x[..., 0]) > 2.4) | (np.abs(x[..., 1]) > np.pi / 3)

    def sample_initial_state(self, rng):
        return rng.uniform([-0.5, -0.2, -0.2, -0.2], [0.5, 0.2, 0.2, 0.2])


# ---------------------------------------------------------------------------
class Quadrotor2D(System):
    """Planar quadrotor hovering at the origin. State ``(x, z, theta, vx, vz, omega)``.

    Two rotors (left, right) produce thrust along the body axis. Shift at
    severity 1: a 25% heavier payload and a right motor at 80% efficiency.
    Gusts push horizontally through ``f_ext``.
    """

    name = "quadrotor2d"
    state_dim = 6
    action_dim = 2
    dt = 0.02
    state_labels = ("x", "z", "theta", "vx", "vz", "omega")
    action_labels = ("thrust_left", "thrust_right")
    angle_indices = (2,)

    nominal_params = dict(
        mass=1.0, inertia=0.02, arm=0.2, eta_left=1.0, eta_right=1.0, drag=0.1, g=9.81, f_ext=0.0
    )
    shift_spec = {"mass": ("mul", 1.25), "eta_right": ("mul", 0.8)}
    adapt_ranges = {"mass": (0.7, 1.7), "eta_right": (0.5, 1.15), "f_ext": (-5.0, 5.0)}
    disturbance_param = "f_ext"
    disturbance_scale = 3.0

    action_low = np.array([0.0, 0.0])
    action_high = np.array([8.0, 8.0])
    target = np.zeros(6)
    Q = np.array([2.0, 2.0, 0.5, 0.2, 0.2, 0.05])
    R = np.array([0.02, 0.02])
    state_scale = np.array([1.0, 1.0, 0.3, 1.0, 1.0, 1.0])
    obs_noise_scale = np.array([0.005, 0.005, 0.002, 0.01, 0.01, 0.01])
    fail_penalty = 20.0

    def deriv(self, x, u, p):
        th, vx, vz, w = x[..., 2], x[..., 3], x[..., 4], x[..., 5]
        t_left = p["eta_left"] * u[..., 0]
        t_right = p["eta_right"] * u[..., 1]
        thrust = t_left + t_right
        m = p["mass"]
        ax = (-thrust * np.sin(th) - p["drag"] * vx + p["f_ext"]) / m
        az = (thrust * np.cos(th) - p["drag"] * vz) / m - p["g"]
        alpha = p["arm"] * (t_right - t_left) / p["inertia"]
        return np.stack(np.broadcast_arrays(vx, vz, w, ax, az, alpha), axis=-1)

    def equilibrium_action(self, p=None):
        p = self.nominal_params if p is None else p
        half = p["mass"] * p["g"] / 2.0
        return np.array([half / p["eta_left"], half / p["eta_right"]], dtype=float)

    def failed(self, x):
        x = np.asarray(x)
        return (np.abs(x[..., 0]) > 3.0) | (np.abs(x[..., 1]) > 3.0) | (np.abs(x[..., 2]) > 1.2)

    def sample_initial_state(self, rng):
        lo = np.array([-0.5, -0.5, -0.1, -0.2, -0.2, -0.2])
        return rng.uniform(lo, -lo)


SYSTEMS = {"cartpole": CartPole, "quadrotor2d": Quadrotor2D}


def make_system(name: str) -> System:
    try:
        return SYSTEMS[name]()
    except KeyError as exc:  # pragma: no cover - message only
        raise ValueError(f"unknown system {name!r}; choose from {sorted(SYSTEMS)}") from exc
