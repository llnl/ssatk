"""Bounded body-axis coefficient tables and launch aerodynamic loads."""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ..accelerations_6dof.spacecraft import (
    SpacecraftAccel, _call_optional, _parse_state_args, _validate_finite,
    _validate_positive, _value_or_spacecraft,
)
from ..coordinates.attitude import quaternion_conjugate, quaternion_from_matrix, rotate_vector

__all__ = ["AerodynamicCoefficientTable", "LaunchAerodynamics"]


class AerodynamicCoefficientTable:
    """Multilinear aerodynamic-axis coefficients, with no extrapolation.

    ``axes`` maps any evidenced subset of ``mach``, ``alpha``, ``beta`` and
    ``deflection`` to increasing knots. Angles are radians; a singleton axis
    restricts that variable to its sole value. Omitted axes explicitly mean
    independence of that variable in this model. Force and moment arrays have
    shape ``(*axis_sizes, 3)``, ordered like ``axes``; moments are right-hand
    aerodynamic-axis moments about the load adapter's declared reference point.
    The table frame has +X forward. ``source`` identifies evidence and its
    conventions, not a catalog name.
    """

    def __init__(self, axes, force_coefficients, moment_coefficients, *, source):
        if not axes or set(axes) - {"mach", "alpha", "beta", "deflection"}:
            raise ValueError("axes must be a nonempty subset of mach/alpha/beta/deflection")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must identify coefficient provenance and conventions")
        self.source = source
        self.axes = {name: np.array(values, dtype=float, copy=True) for name, values in axes.items()}
        for name, values in self.axes.items():
            if (values.ndim != 1 or not values.size or not np.isfinite(values).all()
                    or np.any(np.diff(values) <= 0) or (name == "mach" and np.any(values < 0))):
                raise ValueError(f"{name} knots must be finite and increasing; Mach must be nonnegative")
        shape = (*[len(values) for values in self.axes.values()], 3)
        coefficients = []
        for values in (force_coefficients, moment_coefficients):
            values = np.array(values, dtype=float, copy=True)
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError(f"coefficient arrays must have finite shape {shape}")
            coefficients.append(values)
        self._table = RegularGridInterpolator(tuple(self.axes.values()),
                                             np.concatenate(coefficients, axis=-1), bounds_error=True)

    def __call__(self, *, mach, alpha, beta, deflection=0.0):
        """Return (body force coefficients, body moment coefficients)."""
        variables = dict(mach=mach, alpha=alpha, beta=beta, deflection=deflection)
        if not all(np.isfinite(value) for value in variables.values()) or mach < 0:
            raise ValueError("coefficient queries must be finite; Mach must be nonnegative")
        values = self._table([[variables[name] for name in self.axes]])[0]
        return values[:3], values[3:]


class LaunchAerodynamics(SpacecraftAccel):
    """Quasi-steady aerodynamic force in GCRF and torque about current C in body.

    The table's +X is forward. ``coefficient_to_body`` is a fixed proper 3x3
    rotation from its axes to the existing vehicle body axes (identity by
    default). At the fixed ``reference_position_body`` (m), alpha
    is atan2(V_air,z, V_air,x), beta is atan2(V_air,y, hypot(V_air,x,V_air,z)).
    These velocity components are in the table frame, not necessarily body.
    ``reference_area`` is m squared; the three ``reference_lengths`` (m)
    dimensionalize roll, pitch, yaw moments; ``reynolds_length`` is metres.
    ``mass`` may supply a fixed vehicle mass for state-only
    ``SpacecraftAccel`` calls; otherwise mass is read from the Spacecraft.
    Coefficients multiply dynamic pressure, area and those moment lengths.
    This is a whole-vehicle table; no facet projection is applied again.

    ``center_of_mass_rate`` is dc/dt in fixed body coordinates (m/s), a vector
    or state callable. Its default explicitly assumes a stationary centroid.
    For a draining tank supply its derived rate; mass alone cannot identify
    individual tank rates. The reference-point velocity includes omega cross
    (reference-C) minus dc/dt. No rate damping or plume/rarefaction model is
    inferred. ``deflection`` is a prescribed radian command, scalar/callable.
    """

    spacecraft_torque_model = True
    _combined_force_torque_safe = True

    def __init__(self, atmosphere, coefficients, *, reference_area, reference_lengths,
                 reynolds_length, reference_position_body, mass=None, deflection=0.0,
                 center_of_mass_rate=(0.0, 0.0, 0.0), coefficient_to_body=None):
        self.atmosphere, self.coefficients = atmosphere, coefficients
        self.reference_area = _validate_positive(_validate_finite(reference_area, "reference_area"), "reference_area")
        self.reynolds_length = _validate_positive(_validate_finite(reynolds_length, "reynolds_length"), "reynolds_length")
        self.mass = None if mass is None else _validate_positive(_validate_finite(mass, "mass"), "mass")
        lengths = np.array(reference_lengths, dtype=float, copy=True)
        position = np.array(reference_position_body, dtype=float, copy=True)
        if lengths.shape != (3,) or not np.isfinite(lengths).all() or np.any(lengths <= 0):
            raise ValueError("reference_lengths must be three positive finite lengths")
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("reference_position_body must be a finite 3-vector")
        self.reference_lengths, self.reference_position_body = lengths, position
        self.deflection, self.center_of_mass_rate = deflection, center_of_mass_rate
        self.coefficient_rotation = quaternion_from_matrix(
            np.eye(3) if coefficient_to_body is None else coefficient_to_body)

    def loads(self, *, t, r, v, q, omega, spacecraft):
        """Return forces, translated moments, flow diagnostics and environment.

        At exactly zero airspeed loads are zero and alpha/beta are reported
        as zero by convention; the coefficient table is not queried.
        """
        mass = _validate_finite(_value_or_spacecraft(self.mass, spacecraft, "mass"), "mass")
        body = getattr(spacecraft, "body", None)
        center = np.zeros(3) if body is None else np.asarray(body.current_center_of_mass, dtype=float)
        center_rate = np.asarray(_call_optional(self.center_of_mass_rate, t, r, v, q, omega, spacecraft), dtype=float)
        for name, values in (("r", r), ("v", v), ("omega", omega), ("center", center), ("center_rate", center_rate)):
            if np.shape(values) != (3,) or not np.isfinite(values).all():
                raise ValueError(f"{name} must be a finite 3-vector")
        if np.shape(q) != (4,) or not np.isfinite(q).all():
            raise ValueError("q must be a finite quaternion")
        arm = self.reference_position_body - center
        position = np.asarray(r) + rotate_vector(q, arm)
        velocity = np.asarray(v) + rotate_vector(q, np.cross(omega, arm) - center_rate)
        environment = self.atmosphere.sample(t, position)
        for name in ("density_kg_m3", "sound_speed_m_s", "dynamic_viscosity_pa_s"):
            value = _validate_finite(environment[name], name)
            if value < 0 or (name != "density_kg_m3" and value == 0):
                raise ValueError(f"{name} must be nonnegative; sound speed and viscosity must be positive")
        air_velocity = np.asarray(environment["atmosphere_velocity_gcrf_m_s"], dtype=float)
        if air_velocity.shape != (3,) or not np.isfinite(air_velocity).all():
            raise ValueError("atmosphere velocity must be a finite GCRF 3-vector")
        air_body = rotate_vector(quaternion_conjugate(q),
                                 velocity - air_velocity)
        air_coefficient = rotate_vector(quaternion_conjugate(self.coefficient_rotation), air_body)
        speed = float(np.linalg.norm(air_coefficient))
        alpha = float(np.arctan2(air_coefficient[2], air_coefficient[0])) if speed else 0.0
        beta = float(np.arctan2(air_coefficient[1], np.hypot(air_coefficient[0], air_coefficient[2]))) if speed else 0.0
        mach = speed/environment["sound_speed_m_s"]
        pressure = 0.5*environment["density_kg_m3"]*speed**2
        deflection = _validate_finite(_call_optional(self.deflection, t, r, v, q, omega, spacecraft), "deflection")
        if speed:
            force_coefficient, moment_coefficient = self.coefficients(
                mach=mach, alpha=alpha, beta=beta, deflection=deflection)
            force = rotate_vector(self.coefficient_rotation, pressure*self.reference_area*force_coefficient)
            moment = rotate_vector(self.coefficient_rotation,
                                   pressure*self.reference_area*self.reference_lengths*moment_coefficient)
        else:
            force, moment = np.zeros(3), np.zeros(3)
        torque = moment + np.cross(arm, force)
        force_gcrf = rotate_vector(q, force)
        return {"force_body_n": force, "force_gcrf_n": force_gcrf,
                "moment_reference_body_n_m": moment, "torque_center_body_n_m": torque,
                "acceleration_gcrf_m_s2": force_gcrf/mass,
                "reference_position_body_m": self.reference_position_body.copy(),
                "center_of_mass_body_m": center, "air_velocity_body_m_s": air_body,
                "air_velocity_coefficient_m_s": air_coefficient,
                "mach": float(mach), "alpha_rad": alpha, "beta_rad": beta,
                "deflection_rad": deflection, "dynamic_pressure_pa": float(pressure),
                "reynolds_number": float(environment["density_kg_m3"]*speed*self.reynolds_length
                                         /environment["dynamic_viscosity_pa_s"]),
                "environment": environment, "coefficient_source": self.coefficients.source}

    def _acceleration_torque(self, t, r, v, q, omega, spacecraft):
        result = self.loads(t=t, r=r, v=v, q=q, omega=omega, spacecraft=spacecraft)
        return result["acceleration_gcrf_m_s2"], result["torque_center_body_n_m"]

    def acceleration(self, *, t, r, v, q, omega, spacecraft=None):
        return self._acceleration_torque(t, r, v, q, omega, spacecraft)[0]

    def torque(self, *args, **kwargs):
        spacecraft, t, r, v, q, omega = _parse_state_args(args, kwargs)
        return self._acceleration_torque(t, r, v, q, omega, spacecraft)[1]
