"""Small linear appendage, flexible-mode, and propellant-slosh extension.

The appendage and modes are linearized about a nominal spacecraft: hinge angle
and modal displacement are generalized coordinates, while coupling loads are
reported in the spacecraft body frame.  This is not a finite-element or CFD
model; use a higher-fidelity structural model when that approximation matters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from ..constants import EARTH_MU
from .sixdof import (
    SixDOFTrajectory,
    _initial_state,
    _rhs_in_elapsed_time,
    _times,
    _validate_time_direction,
    rotate_vector,
    sixdof_rhs,
)

__all__ = [
    "Extended6DOFTrajectory",
    "FlexibleMode",
    "HingedAppendage",
    "SloshMode",
    "propagate_6dof_extended",
    "propagate_6dof_extended_variational",
]


def _vec(value, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite 3-vector.")
    return value


def _modes(value, mode_type, name):
    if value is None:
        return ()
    modes = value if isinstance(value, (tuple, list)) else (value,)
    if not all(isinstance(mode, mode_type) for mode in modes):
        raise TypeError(f"{name} must contain only {mode_type.__name__} instances.")
    return tuple(modes)


@dataclass(frozen=True)
class HingedAppendage:
    """Reduced-order hinged appendage with linear and optional cubic stiffness."""

    axis_body: np.ndarray
    inertia: float
    stiffness: float = 0.0
    damping: float = 0.0
    angle0: float = 0.0
    rate0: float = 0.0
    cubic_stiffness: float = 0.0

    def __post_init__(self):
        axis = _vec(self.axis_body, "axis_body")
        inertia = float(self.inertia)
        stiffness = float(self.stiffness)
        damping = float(self.damping)
        cubic_stiffness = float(self.cubic_stiffness)
        if (np.linalg.norm(axis) == 0 or not np.isfinite(inertia) or inertia <= 0
                or not np.isfinite(stiffness) or stiffness < 0
                or not np.isfinite(damping) or damping < 0
                or not np.isfinite(cubic_stiffness) or cubic_stiffness < 0):
            raise ValueError("hinge axis must be nonzero; inertia must be positive; stiffness/damping/cubic_stiffness nonnegative.")
        object.__setattr__(self, "axis_body", axis / np.linalg.norm(axis))
        object.__setattr__(self, "inertia", inertia)
        object.__setattr__(self, "stiffness", stiffness)
        object.__setattr__(self, "damping", damping)
        object.__setattr__(self, "cubic_stiffness", cubic_stiffness)


@dataclass(frozen=True)
class FlexibleMode:
    axis_body: np.ndarray
    effective_mass: float
    natural_frequency: float
    damping_ratio: float = 0.0
    displacement0: float = 0.0
    velocity0: float = 0.0

    def __post_init__(self):
        axis = _vec(self.axis_body, "axis_body")
        effective_mass = float(self.effective_mass)
        natural_frequency = float(self.natural_frequency)
        damping_ratio = float(self.damping_ratio)
        if np.linalg.norm(axis) == 0 or not np.isfinite(effective_mass) or effective_mass <= 0 or not np.isfinite(natural_frequency) or natural_frequency <= 0 or not np.isfinite(damping_ratio) or damping_ratio < 0:
            raise ValueError("flexible mode requires nonzero axis, positive mass/frequency, and nonnegative damping.")
        object.__setattr__(self, "axis_body", axis / np.linalg.norm(axis))
        object.__setattr__(self, "effective_mass", effective_mass)
        object.__setattr__(self, "natural_frequency", natural_frequency)
        object.__setattr__(self, "damping_ratio", damping_ratio)


@dataclass(frozen=True)
class SloshMode:
    axis_body: np.ndarray
    mass: float
    natural_frequency: float
    damping_ratio: float = 0.0
    lever_arm_body: np.ndarray = (0.0, 0.0, 0.0)
    displacement0: float = 0.0
    velocity0: float = 0.0

    def __post_init__(self):
        axis = _vec(self.axis_body, "axis_body")
        mass = float(self.mass)
        natural_frequency = float(self.natural_frequency)
        damping_ratio = float(self.damping_ratio)
        if np.linalg.norm(axis) == 0 or not np.isfinite(mass) or mass <= 0 or not np.isfinite(natural_frequency) or natural_frequency <= 0 or not np.isfinite(damping_ratio) or damping_ratio < 0:
            raise ValueError("slosh mode requires nonzero axis, positive mass/frequency, and nonnegative damping.")
        object.__setattr__(self, "axis_body", axis / np.linalg.norm(axis))
        object.__setattr__(self, "lever_arm_body", _vec(self.lever_arm_body, "lever_arm_body"))
        object.__setattr__(self, "mass", mass)
        object.__setattr__(self, "natural_frequency", natural_frequency)
        object.__setattr__(self, "damping_ratio", damping_ratio)


@dataclass(frozen=True)
class Extended6DOFTrajectory:
    """Integrated rigid and reduced-order body states.

    ``stm`` is populated by :func:`propagate_6dof_extended_variational` and
    uses the full state ordering ``[r, v, q, omega, [mass], modes]``.
    """

    trajectory: SixDOFTrajectory
    hinge: np.ndarray | None = None
    flexible: np.ndarray | None = None
    slosh: np.ndarray | None = None
    stm: np.ndarray | None = None


def propagate_6dof_extended(*, times, inertia, hinge=None, flexible=None, slosh=None,
                            bus_mass=None, r0=None, v0=None, t0=None, q0=None, omega0=None,
                            mass0=None, mu=EARTH_MU, acceleration=None, torque=None,
                            gravity_gradient=False, rtol=1e-9, atol=1e-12, method="DOP853",
                            max_step=np.inf, first_step=None, with_stm=False, stm0=None,
                            jacobian_step=1e-7):
    """Propagate rigid-body state plus hinge, flexible, and slosh modes.

    Each mode argument may be one mode or a sequence of modes. Extended arrays
    have columns ``[coordinate, rate]`` for one mode, or shape ``(samples, 2,
    modes)`` for multiple modes. Slosh coupling is the linear restoring force
    ``m*w²*x`` at ``lever_arm_body``.
    ``HingedAppendage.cubic_stiffness`` adds the restoring torque
    ``cubic_stiffness * angle**3`` in N m, and defaults to zero.
    """
    times = _times(times)
    hinges = _modes(hinge, HingedAppendage, "hinge")
    flexibles = _modes(flexible, FlexibleMode, "flexible")
    sloshes = _modes(slosh, SloshMode, "slosh")
    if not (hinges or flexibles or sloshes):
        raise ValueError("at least one extended mode is required.")
    if sloshes:
        bus_mass = float(bus_mass) if bus_mass is not None else 0.0
        if not np.isfinite(bus_mass) or bus_mass <= 0:
            raise ValueError("positive bus_mass is required for slosh force coupling.")
    state = _initial_state(orbit0=None, r0=r0, v0=v0, t0=t0, q0=q0, omega0=omega0, mass0=mass0)
    _validate_time_direction(times, state.t)
    ext0 = []
    for mode in hinges: ext0 += [mode.angle0, mode.rate0]
    for mode in flexibles: ext0 += [mode.displacement0, mode.velocity0]
    for mode in sloshes: ext0 += [mode.displacement0, mode.velocity0]
    y0 = np.concatenate(([ *state.r, *state.v, *state.q, *state.omega] + ([] if mass0 is None else [mass0]), ext0))
    rigid_n = 14 if mass0 is not None else 13
    n = y0.size
    if with_stm:
        if jacobian_step <= 0.0 or not np.isfinite(jacobian_step):
            raise ValueError("jacobian_step must be finite and positive.")
        if stm0 is None:
            phi0 = np.eye(n)
        else:
            phi0 = np.asarray(stm0, dtype=float)
            if phi0.shape != (n, n) or not np.all(np.isfinite(phi0)):
                raise ValueError(f"stm0 must be finite with shape {(n, n)}.")
    elif stm0 is not None:
        raise ValueError("stm0 requires with_stm=True.")

    def rhs(t, y):
        rigid, z = y[:rigid_n], y[rigid_n:]
        i = 0; loads_t = np.zeros(3); loads_a = np.zeros(3)
        for mode in hinges:
            angle, rate = z[i:i+2]; i += 2
            loads_t += mode.axis_body * (mode.stiffness * angle + mode.cubic_stiffness * angle**3 + mode.damping * rate)
        for mode in flexibles:
            disp, rate = z[i:i+2]; i += 2
            loads_t += mode.axis_body * (mode.effective_mass * mode.natural_frequency**2 * disp + 2 * mode.damping_ratio * mode.natural_frequency * mode.effective_mass * rate)
        for mode in sloshes:
            disp, rate = z[i:i+2]; i += 2
            force = mode.axis_body * (
                mode.mass * mode.natural_frequency**2 * disp
                + 2 * mode.damping_ratio * mode.natural_frequency * mode.mass * rate
            )
            loads_a += rotate_vector(rigid[6:10], force) / bus_mass
            loads_t += np.cross(mode.lever_arm_body, force)
        a = lambda tt, r, v, q, om: np.asarray(acceleration(tt, r, v, q, om) if acceleration else 0.0) + loads_a
        trq = lambda tt, r, v, q, om: np.asarray(torque(tt, r, v, q, om) if torque else 0.0) + loads_t
        base = sixdof_rhs(t, rigid, inertia=inertia, mu=mu, acceleration=a, torque=trq, gravity_gradient=gravity_gradient, mass_state=mass0 is not None)
        dz = []
        i = 0
        for mode in hinges:
            angle, rate = z[i:i+2]; i += 2
            dz += [rate, -(mode.stiffness * angle + mode.cubic_stiffness * angle**3 + mode.damping * rate) / mode.inertia]
        for mode in flexibles:
            disp, rate = z[i:i+2]; i += 2
            dz += [rate, -mode.natural_frequency**2 * disp - 2 * mode.damping_ratio * mode.natural_frequency * rate]
        for mode in sloshes:
            disp, rate = z[i:i+2]; i += 2
            dz += [rate, -mode.natural_frequency**2 * disp - 2 * mode.damping_ratio * mode.natural_frequency * rate]
        return np.concatenate((base, dz))

    if with_stm:
        def combined_rhs(t, combined):
            y = combined[:n]
            phi = combined[n:].reshape(n, n)
            jacobian = np.empty((n, n))
            for column in range(n):
                step = jacobian_step * max(1.0, abs(y[column]))
                delta = np.zeros(n)
                delta[column] = step
                jacobian[:, column] = (rhs(t, y + delta) - rhs(t, y - delta)) / (2.0 * step)
            return np.concatenate((rhs(t, y), (jacobian @ phi).ravel()))

        y_initial = np.concatenate((y0, phi0.ravel()))
        derivative = combined_rhs
    else:
        y_initial = y0
        derivative = rhs
    # Integrate in seconds since state.t. See propagate_6dof: SciPy's minimum
    # step is 10 * ulp(t), which is 2.4e-6 s at GPS 1.4e9 against ~5e-322 s at
    # zero, and a discontinuity needing a finer step aborts the run.
    t_ref = float(state.t)
    sol = solve_ivp(_rhs_in_elapsed_time(derivative, t_ref), (0.0, float(times[-1]) - t_ref), y_initial, t_eval=times - t_ref, rtol=rtol, atol=atol, method=method, max_step=max_step, first_step=first_step)
    if not sol.success: raise RuntimeError(sol.message)
    y = sol.y[:n].T
    stm = None if not with_stm else sol.y[n:].T.reshape((-1, n, n))
    q = np.array([item / np.linalg.norm(item) for item in y[:, 6:10]])
    trajectory = SixDOFTrajectory(sol.t + t_ref, y[:, :3], y[:, 3:6], q, y[:, 10:13], None if mass0 is None else y[:, 13], nfev=sol.nfev, message=sol.message, status=sol.status)
    j = rigid_n
    arrays = []
    for modes in (hinges, flexibles, sloshes):
        if not modes:
            arrays.append(None)
            continue
        values = y[:, j:j + 2 * len(modes)].reshape(len(times), len(modes), 2)
        arrays.append(values[:, 0, :] if len(modes) == 1 else values.transpose(0, 2, 1))
        j += 2 * len(modes)
    return Extended6DOFTrajectory(trajectory, *arrays, stm)


def propagate_6dof_extended_variational(*, times, inertia, hinge=None, flexible=None, slosh=None,
                                         bus_mass=None, r0=None, v0=None, t0=None, q0=None,
                                         omega0=None, mass0=None, mu=EARTH_MU, acceleration=None,
                                         torque=None, gravity_gradient=False, stm0=None,
                                         jacobian_step=1e-7, rtol=1e-9, atol=1e-12,
                                         method="DOP853", max_step=np.inf, first_step=None):
    """Propagate extended 6-DoF states and their full finite-difference STM.

    The STM state ordering is ``[r, v, q, omega, [mass], modes]``.  It can be
    passed directly to :func:`propagate_6dof_covariance`; use
    :func:`attitude_error_stm` for the non-redundant local attitude form.
    ``jacobian_step`` is a relative central-difference perturbation because
    the reduced-order coupling includes nonlinear hinge stiffness.
    """
    return propagate_6dof_extended(
        times=times, inertia=inertia, hinge=hinge, flexible=flexible, slosh=slosh,
        bus_mass=bus_mass, r0=r0, v0=v0, t0=t0, q0=q0, omega0=omega0, mass0=mass0,
        mu=mu, acceleration=acceleration, torque=torque, gravity_gradient=gravity_gradient,
        rtol=rtol, atol=atol, method=method, max_step=max_step, first_step=first_step,
        with_stm=True, stm0=stm0, jacobian_step=jacobian_step,
    )
