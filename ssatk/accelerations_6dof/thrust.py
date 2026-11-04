"""Finite-burn thrust profiles and maneuver accelerations."""

from __future__ import annotations

import csv as _csv
import json as _json
from io import StringIO as _StringIO
from pathlib import Path as _Path

import numpy as _np
from scipy.integrate import trapezoid as _trapezoid

from ..constants import STANDARD_GRAVITY
from ..coordinates.satellite_frames import frame_to_gcrf_matrix
from .spacecraft import (
    ArrayLike,
    SpacecraftAccel,
    _as_vector3,
    _call_optional,
    _parse_state_args,
    _rotate_vector,
    _unit_vector,
    _validate_positive,
)

NASA_NTRS_DIGITIZED_THRUST_CURVE_DIR = "propulsion/thrust_curves/digitized/nasa_ntrs"
THRUSTCURVE_ORG_PD_THRUST_CURVE_DIR = "propulsion/thrust_curves/solid_motor_pd/thrustcurve_org"
PACKAGED_THRUST_CURVE_COLLECTIONS = {
    "nasa_ntrs": NASA_NTRS_DIGITIZED_THRUST_CURVE_DIR,
    "digitized_nasa_ntrs": NASA_NTRS_DIGITIZED_THRUST_CURVE_DIR,
    "digitized": NASA_NTRS_DIGITIZED_THRUST_CURVE_DIR,
    "thrustcurve_org_pd": THRUSTCURVE_ORG_PD_THRUST_CURVE_DIR,
    "public_domain": THRUSTCURVE_ORG_PD_THRUST_CURVE_DIR,
    "solid_motor_pd": THRUSTCURVE_ORG_PD_THRUST_CURVE_DIR,
}

__all__ = [
    "SpacecraftManeuverAccel",
    "ThrustCurve",
    "integrated_thrust_impulse",
    "load_digitized_thrust_curve",
    "load_packaged_thrust_curve",
    "load_packaged_thrust_curve_metadata",
    "load_thrust_curve_csv",
    "load_thrust_curve_data",
    "make_finite_burn_acceleration",
    "make_maneuver_acceleration",
    "packaged_thrust_curve_index",
    "thrust_profile_constant",
    "thrust_profile_exponential",
    "thrust_profile_pulsed",
    "thrust_profile_smoothstep",
    "thrust_profile_trapezoid",
]


class ThrustCurve:
    """Interpolated scalar thrust curve in newtons.

    Curves are evaluated with linear interpolation and return ``fill_value``
    outside the tabulated time span. Store real engine data in ``ssa-data-propulsion`` or a
    user path, then load it with :func:`load_thrust_curve_csv`.
    ``bounds_error=True`` rejects extrapolation instead of using ``fill_value``.
    """

    def __init__(self, times: ArrayLike, thrusts: ArrayLike, *, fill_value: float = 0.0,
                 bounds_error: bool = False):
        self.times = _as_1d(times, "times")
        self.thrusts = _as_1d(thrusts, "thrusts")
        if self.times.size != self.thrusts.size or self.times.size < 2:
            raise ValueError("times and thrusts must have the same length >= 2.")
        if not _np.all(_np.diff(self.times) > 0.0):
            raise ValueError("times must be strictly increasing.")
        if _np.any(self.thrusts < 0.0):
            raise ValueError("thrusts must be non-negative.")
        self.fill_value = _validate_nonnegative(fill_value, "fill_value")
        self.time_breakpoints = self.times.copy()
        self.bounds_error = bool(bounds_error)

    def __call__(self, t, *args) -> float:
        if not _np.isfinite(float(t)):
            raise ValueError("curve time must be finite.")
        if self.bounds_error and not self.times[0] <= float(t) <= self.times[-1]:
            raise ValueError("curve time is outside the documented interpolation span.")
        return float(_np.interp(float(t), self.times, self.thrusts, left=self.fill_value, right=self.fill_value))

    @property
    def total_impulse(self) -> float:
        """Trapezoidal total impulse over the tabulated span in N s."""

        return float(_trapezoid(self.thrusts, self.times))


class SpacecraftManeuverAccel(SpacecraftAccel):
    """Finite thrust maneuver in a satellite-operations frame.

    ``thrust`` may be a scalar in newtons, a callable/profile, a
    :class:`ThrustCurve`, or a 3-vector thrust in the selected frame. ``frame``
    accepts the same frame names as ``ssatk.coordinates``: ``rtn``,
    ``lvlh``, ``ric``, ``vnb``, ``ntw``, ``body``, ``gcrf``/``inertial``, etc.
    The default ``frame="rtn", direction=[0, 1, 0]`` is transverse/prograde in
    the common RTN/RIC/LVLH operations convention. Use ``frame="ntw"`` for
    exact SSAPy-style ``[N, T, W]`` inputs.

    ``throttle`` (scalar or state callable) declares linear scaling of reference
    thrust and flow; zero is shutdown, otherwise ``throttle_range`` bounds it.
    Only use this law when justified by the engine data. For measured commanded
    histories, leave throttle at one so it is not applied twice.

    Optional fixed-nozzle correction is ``F = F_ref + (p_ref-p_ambient)*A_exit``.
    All pressure inputs and the validity ``pressure_range`` in Pa are required.
    Flow and Isp refer to the reference-pressure thrust, not corrected thrust.
    Do not correct an already ambient-adjusted history. No chamber dynamics,
    nozzle separation, mixture-ratio law or thrust moment is inferred.
    """

    def __init__(
        self,
        thrust,
        *,
        direction: ArrayLike | None = None,
        frame: str = "rtn",
        mass: float | None = None,
        isp: float | None = None,
        propellant_flow_rate=None,
        throttle=1.0,
        throttle_range=(0.0, 1.0),
        reference_pressure: float | None = None,
        ambient_pressure=None,
        nozzle_exit_area: float | None = None,
        pressure_range=None,
        tank_name: str | None = None,
        start: float = -_np.inf,
        stop: float = _np.inf,
    ):
        self.thrust = thrust
        self.frame = frame
        self.direction = _default_maneuver_direction(frame) if direction is None else _unit_vector(direction, "direction")
        self.mass = None if mass is None else _validate_positive(mass, "mass")
        self.isp = None if isp is None else _validate_positive(isp, "isp")
        if self.isp is not None and not _np.isfinite(self.isp):
            raise ValueError("isp must be finite.")
        if isp is not None and propellant_flow_rate is not None:
            raise ValueError("Supply either isp or independently documented propellant_flow_rate, not both.")
        self.propellant_flow_rate = propellant_flow_rate
        self.throttle = throttle
        self.throttle_range = _operating_range(throttle_range, "throttle_range")
        pressure_inputs = (reference_pressure, ambient_pressure, nozzle_exit_area, pressure_range)
        if any(value is not None for value in pressure_inputs) and not all(value is not None for value in pressure_inputs):
            raise ValueError("Pressure correction requires reference_pressure, ambient_pressure, nozzle_exit_area and pressure_range.")
        self.reference_pressure = reference_pressure
        self.ambient_pressure = ambient_pressure
        self.nozzle_exit_area = nozzle_exit_area
        self.pressure_range = None
        if reference_pressure is not None:
            self.reference_pressure = _validate_nonnegative(reference_pressure, "reference_pressure")
            self.nozzle_exit_area = _validate_positive(nozzle_exit_area, "nozzle_exit_area")
            if not _np.isfinite(self.nozzle_exit_area):
                raise ValueError("nozzle_exit_area must be finite.")
            self.pressure_range = _operating_range(pressure_range, "pressure_range")
            if not self.pressure_range[0] <= self.reference_pressure <= self.pressure_range[1]:
                raise ValueError("reference_pressure is outside pressure_range.")
        self.tank_name = None if tank_name is None else str(tank_name)
        self.start = float(start)
        self.stop = float(stop)
        self.time_breakpoints = _np.array([self.start, self.stop])
        if _np.isnan(self.start) or _np.isnan(self.stop) or self.stop < self.start:
            raise ValueError("stop must be greater than or equal to start.")

    def acceleration(self, *, t, r, v, q, omega, spacecraft=None) -> _np.ndarray:
        thrust_frame = self.thrust_vector(t, r, v, q, omega, spacecraft)
        if not _np.any(thrust_frame):
            return _np.zeros(3)
        return _frame_vector_to_gcrf(self.frame, thrust_frame, r, v, q) / _mass_from_spacecraft(self.mass, spacecraft)

    def thrust_vector(self, t, r, v, q, omega, spacecraft=None) -> _np.ndarray:
        """Return thrust vector in the selected maneuver frame in newtons."""

        value = self._reference_thrust_vector(t, r, v, q, omega, spacecraft)
        magnitude = float(_np.linalg.norm(value))
        if magnitude == 0.0 or self.reference_pressure is None:
            return value
        pressure = float(_call_optional(self.ambient_pressure, t, r, v, q, omega, spacecraft))
        if not _np.isfinite(pressure) or not self.pressure_range[0] <= pressure <= self.pressure_range[1]:
            raise ValueError("ambient_pressure is outside the documented pressure_range in Pa.")
        corrected = magnitude + (self.reference_pressure - pressure) * self.nozzle_exit_area
        if corrected < 0.0:
            raise ValueError("Pressure correction produces negative thrust; the nozzle model is invalid here.")
        return value * (corrected / magnitude)

    def _reference_thrust_vector(self, t, r, v, q, omega, spacecraft=None):

        if not (self.start <= float(t) <= self.stop):
            return _np.zeros(3)
        if self.tank_name is not None and spacecraft is not None:
            tanks = tuple(getattr(getattr(spacecraft, "body", None), "tanks", ()))
            selected = [tank for tank in tanks if tank.name == self.tank_name]
            if len(selected) != 1:
                raise ValueError(f"expected exactly one tank named {self.tank_name!r}.")
            if selected[0].usable_propellant_mass <= 0.0:
                return _np.zeros(3)
        throttle = self._throttle_value(t, r, v, q, omega, spacecraft)
        if throttle == 0.0:
            return _np.zeros(3)
        value = _call_optional(self.thrust, t, r, v, q, omega, spacecraft) if callable(self.thrust) else self.thrust
        value = _np.asarray(value, dtype=float)
        if value.shape == ():
            thrust = _validate_nonnegative(float(value), "thrust")
            return self.direction * thrust * throttle
        if value.shape == (3,):
            if _np.any(~_np.isfinite(value)):
                raise ValueError("thrust vector must be finite.")
            return value * throttle
        raise ValueError("thrust must be a scalar profile or 3-vector.")

    def _throttle_value(self, t, r, v, q, omega, spacecraft):
        throttle = float(_call_optional(self.throttle, t, r, v, q, omega, spacecraft))
        if not _np.isfinite(throttle) or (throttle != 0.0 and not self.throttle_range[0] <= throttle <= self.throttle_range[1]):
            raise ValueError("throttle is outside the documented throttle_range (zero is shutdown).")
        return throttle

    def mass_flow_rate(self, *args, **kwargs) -> float:
        """Return positive kg/s from documented flow, or infer it from Isp.

        ``propellant_flow_rate`` is a scalar or state callable, independent of
        thrust (including startup flow). It shares the burn window and tank.
        """

        if self.isp is None and self.propellant_flow_rate is None:
            return 0.0
        spacecraft, t, r, v, q, omega = _parse_state_args(args, kwargs)
        if not (self.start <= t <= self.stop):
            return 0.0
        if self.tank_name is not None and spacecraft is not None:
            tanks = tuple(getattr(getattr(spacecraft, "body", None), "tanks", ()))
            selected = [tank for tank in tanks if tank.name == self.tank_name]
            if len(selected) != 1:
                raise ValueError(f"expected exactly one tank named {self.tank_name!r}.")
            if selected[0].usable_propellant_mass <= 0.0:
                return 0.0
        if self.propellant_flow_rate is not None:
            throttle = self._throttle_value(t, r, v, q, omega, spacecraft)
            if throttle == 0.0:
                return 0.0
            flow = float(_call_optional(self.propellant_flow_rate, t, r, v, q, omega, spacecraft))
            if not _np.isfinite(flow) or flow < 0.0:
                raise ValueError("propellant_flow_rate must be finite and non-negative in kg/s.")
            return flow * throttle
        return float(_np.linalg.norm(self._reference_thrust_vector(t, r, v, q, omega, spacecraft)) / (self.isp * STANDARD_GRAVITY))


def thrust_profile_constant(thrust: float, *, start: float = -_np.inf, stop: float = _np.inf):
    """Return a constant finite-burn thrust profile in newtons."""

    thrust = _validate_nonnegative(thrust, "thrust")
    start = float(start)
    stop = float(stop)
    if stop < start:
        raise ValueError("stop must be greater than or equal to start.")

    def profile(t, *args):
        return thrust if start <= float(t) <= stop else 0.0

    profile.time_breakpoints = _np.array([start, stop])
    return profile


def thrust_profile_trapezoid(
    thrust: float,
    *,
    start: float = 0.0,
    burn_time: float,
    rise_time: float = 0.0,
    fall_time: float | None = None,
):
    """Return a textbook-style linear ramp/steady/ramp-down thrust profile."""

    return _ramped_thrust_profile(
        thrust,
        start=start,
        burn_time=burn_time,
        rise_time=rise_time,
        fall_time=fall_time,
        smooth=False,
    )


def thrust_profile_smoothstep(
    thrust: float,
    *,
    start: float = 0.0,
    burn_time: float,
    rise_time: float = 0.0,
    fall_time: float | None = None,
):
    """Return a smoothstep ramp/steady/ramp-down thrust profile."""

    return _ramped_thrust_profile(
        thrust,
        start=start,
        burn_time=burn_time,
        rise_time=rise_time,
        fall_time=fall_time,
        smooth=True,
    )


def thrust_profile_exponential(
    thrust: float,
    *,
    start: float = 0.0,
    stop: float = _np.inf,
    rise_tau: float = 0.0,
    decay_tau: float = 0.0,
):
    """Return a first-order thrust rise and optional decay profile."""

    thrust = _validate_nonnegative(thrust, "thrust")
    start = float(start)
    stop = float(stop)
    rise_tau = _validate_nonnegative(rise_tau, "rise_tau")
    decay_tau = _validate_nonnegative(decay_tau, "decay_tau")
    if stop < start:
        raise ValueError("stop must be greater than or equal to start.")

    def profile(t, *args):
        t = float(t)
        if t < start:
            return 0.0
        if t <= stop:
            return thrust if rise_tau == 0.0 else thrust * (1.0 - _np.exp(-(t - start) / rise_tau))
        if decay_tau == 0.0:
            return 0.0
        at_stop = thrust if rise_tau == 0.0 else thrust * (1.0 - _np.exp(-(stop - start) / rise_tau))
        return float(at_stop * _np.exp(-(t - stop) / decay_tau))

    profile.time_breakpoints = _np.array([start, stop])
    return profile


def thrust_profile_pulsed(
    thrust: float,
    *,
    period: float,
    duty_cycle: float,
    start: float = 0.0,
    stop: float = _np.inf,
):
    """Return a square-wave on/off thrust profile for duty-cycled burns."""

    thrust = _validate_nonnegative(thrust, "thrust")
    period = _validate_positive(period, "period")
    duty_cycle = float(duty_cycle)
    if not 0.0 <= duty_cycle <= 1.0:
        raise ValueError("duty_cycle must be in [0, 1].")
    start = float(start)
    stop = float(stop)
    if stop < start:
        raise ValueError("stop must be greater than or equal to start.")

    def profile(t, *args):
        t = float(t)
        if t < start or t > stop:
            return 0.0
        return thrust if (t - start) % period < duty_cycle * period else 0.0

    def boundaries(left, right):
        if 0.0 < duty_cycle < 1.0:
            first = max(0, int(_np.floor((left - start) / period)))
            last = int(_np.ceil((min(right, stop) - start) / period))
            on = start + _np.arange(first, max(first, last) + 1) * period
            points = _np.concatenate(([start, stop], on, on + duty_cycle * period))
        else:
            points = _np.array([start, stop])
        return _np.unique(points[(points >= start) & (points <= stop)])
    profile.time_breakpoints = boundaries
    return profile


def load_thrust_curve_csv(
    path,
    *,
    time_column: str = "time_s",
    thrust_column: str = "thrust_n",
    time_scale: float = 1.0,
    thrust_scale: float = 1.0,
    fill_value: float = 0.0,
    bounds_error: bool = False,
) -> ThrustCurve:
    """Load a scalar thrust curve from a CSV file with time and thrust columns."""

    path = _Path(path)
    with path.open(newline="") as handle:
        reader = _csv.DictReader(handle)
        rows = list(reader)
    if not rows or time_column not in rows[0] or thrust_column not in rows[0]:
        raise ValueError(f"CSV must contain {time_column!r} and {thrust_column!r} columns.")
    times = [float(row[time_column]) * time_scale for row in rows]
    thrusts = [float(row[thrust_column]) * thrust_scale for row in rows]
    return ThrustCurve(times, thrusts, fill_value=fill_value, bounds_error=bounds_error)


def load_thrust_curve_data(relative_path, **kwargs) -> ThrustCurve:
    """Load a thrust curve CSV from the installed propulsion data package."""

    from ..data import data_path

    with data_path(relative_path) as path:
        return load_thrust_curve_csv(path, **kwargs)


def packaged_thrust_curve_index(collection: str = "nasa_ntrs") -> tuple[dict[str, str], ...]:
    """Return rows from a packaged propulsion-data thrust-curve index.

    Parameters
    ----------
    collection
        ``"nasa_ntrs"`` for digitized NASA NTRS plot curves, or
        ``"thrustcurve_org_pd"`` for public-domain ThrustCurve.org RASP/RockSim
        motor files.
    """

    return tuple(_read_packaged_csv(f"{_curve_collection_dir(collection)}/index.csv"))


def load_packaged_thrust_curve_metadata(identifier: str, *, collection: str = "nasa_ntrs") -> dict:
    """Load sidecar metadata for a packaged thrust curve selected by ID/path."""

    record = _find_packaged_thrust_curve(identifier, collection=collection)
    metadata_path = record.get("metadata_path")
    if not metadata_path:
        raise ValueError(f"Thrust curve record {identifier!r} has no metadata_path column.")

    from ..data import read_data_text

    return _json.loads(read_data_text(f"{_curve_collection_dir(collection)}/{metadata_path}"))


def load_packaged_thrust_curve(
    identifier: str,
    *,
    collection: str = "nasa_ntrs",
    steady_state_thrust_n: float | None = None,
    normalized_column: str = "filtered_fraction_steady_state",
    clamp_normalized: bool = True,
    fill_value: float = 0.0,
    bounds_error: bool = False,
) -> ThrustCurve:
    """Load a named thrust curve packaged in ``ssa-data-propulsion``.

    Absolute curves with a ``thrust_n`` column load directly. Normalized curves,
    such as the RS-18 startup shape, require ``steady_state_thrust_n`` so the
    selected normalized column can be converted to newtons.
    """

    record = _find_packaged_thrust_curve(identifier, collection=collection)
    csv_path = record.get("csv_path")
    if not csv_path:
        raise ValueError(f"Thrust curve record {identifier!r} has no csv_path column.")

    resource_path = f"{_curve_collection_dir(collection)}/{csv_path}"
    rows = _read_packaged_csv(resource_path)
    columns = set(rows[0])
    if "thrust_n" in columns:
        return load_thrust_curve_data(resource_path, fill_value=fill_value, bounds_error=bounds_error)
    if normalized_column not in columns:
        raise ValueError(
            f"Packaged curve {identifier!r} lacks {normalized_column!r}; "
            f"available columns are {sorted(columns)}."
        )
    if steady_state_thrust_n is None:
        raise ValueError(
            f"Packaged curve {identifier!r} is normalized. Provide steady_state_thrust_n "
            "to convert it to an absolute ThrustCurve."
        )
    scale = _validate_nonnegative(steady_state_thrust_n, "steady_state_thrust_n")
    times = [float(row["time_s"]) for row in rows]
    thrusts = [float(row[normalized_column]) * scale for row in rows]
    if clamp_normalized:
        thrusts = [max(0.0, thrust) for thrust in thrusts]
    return ThrustCurve(times, thrusts, fill_value=fill_value, bounds_error=bounds_error)


def load_digitized_thrust_curve(identifier: str, **kwargs) -> ThrustCurve:
    """Load a digitized NASA NTRS thrust curve by NTRS ID or file stem."""

    return load_packaged_thrust_curve(identifier, collection="nasa_ntrs", **kwargs)


def integrated_thrust_impulse(profile, start: float, stop: float, *, samples: int = 1001) -> float:
    """Numerically integrate a thrust profile over ``[start, stop]`` in N s."""

    if samples < 2:
        raise ValueError("samples must be at least 2.")
    times = _np.linspace(float(start), float(stop), int(samples))
    thrusts = _np.array([float(profile(t)) for t in times], dtype=float)
    return float(_trapezoid(thrusts, times))


def make_maneuver_acceleration(thrust, **kwargs):
    return SpacecraftManeuverAccel(thrust, **kwargs)


def make_finite_burn_acceleration(thrust, **kwargs):
    return SpacecraftManeuverAccel(thrust, **kwargs)


def _as_1d(value: ArrayLike, name: str) -> _np.ndarray:
    array = _np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array.")
    if _np.any(~_np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    return array


def _validate_nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not _np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return value


def _operating_range(value, name):
    limits = _as_1d(value, name)
    if limits.shape != (2,) or limits[0] < 0.0 or limits[1] < limits[0]:
        raise ValueError(f"{name} must contain ordered non-negative lower and upper limits.")
    return tuple(limits)


def _read_packaged_csv(relative_path: str) -> list[dict[str, str]]:
    from ..data import read_data_text

    rows = list(_csv.DictReader(_StringIO(read_data_text(relative_path))))
    if not rows:
        raise ValueError(f"Packaged CSV {relative_path!r} has no data rows.")
    return rows


def _curve_collection_dir(collection: str) -> str:
    key = str(collection).strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return PACKAGED_THRUST_CURVE_COLLECTIONS[key]
    except KeyError as exc:
        choices = ", ".join(sorted(PACKAGED_THRUST_CURVE_COLLECTIONS))
        raise ValueError(f"Unknown thrust-curve collection {collection!r}. Available: {choices}.") from exc


def _find_packaged_thrust_curve(identifier: str, *, collection: str) -> dict[str, str]:
    key = _resource_key(identifier)
    matches = []
    for record in packaged_thrust_curve_index(collection):
        candidates = {
            record.get("ntrs_id", ""),
            record.get("simfile_id", ""),
            record.get("csv_path", ""),
            record.get("metadata_path", ""),
            _Path(record.get("csv_path", "")).stem,
            _Path(record.get("metadata_path", "")).stem,
            record.get("designation", ""),
            f"{record.get('manufacturer', '')}_{record.get('designation', '')}",
        }
        if key in {_resource_key(candidate) for candidate in candidates if candidate}:
            matches.append(record)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise KeyError(f"No packaged thrust curve {identifier!r} in collection {collection!r}.")
    raise KeyError(
        f"Identifier {identifier!r} matches {len(matches)} thrust curves; "
        "use csv_path or metadata_path."
    )


def _resource_key(value) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _mass_from_spacecraft(value, spacecraft) -> float:
    if value is not None:
        return value
    if spacecraft is not None and getattr(spacecraft, "mass", None) is not None:
        return _validate_positive(spacecraft.mass, "mass")
    if spacecraft is not None and getattr(spacecraft, "body", None) is not None:
        body = spacecraft.body
        if hasattr(body, "current_mass"):
            return _validate_positive(body.current_mass, "mass")
        if hasattr(body, "mass"):
            return _validate_positive(body.mass, "mass")
    raise ValueError("mass must be provided by the model, Spacecraft, or Spacecraft.body.")


def _default_maneuver_direction(frame: str) -> _np.ndarray:
    key = str(frame).strip().lower().replace("-", "_").replace(" ", "_")
    if key in {"ntw", "rtn", "rsw", "ric", "lvlh"}:
        return _np.array([0.0, 1.0, 0.0])
    return _np.array([1.0, 0.0, 0.0])


def _frame_vector_to_gcrf(frame: str, vector: ArrayLike, r, v, q) -> _np.ndarray:
    key = str(frame).strip().lower().replace("-", "_").replace(" ", "_")
    if key in {"gcrf", "eci", "inertial"}:
        return _as_vector3(vector, "vector")
    if key in {"body", "spacecraft", "sc"}:
        return _rotate_vector(q, vector)
    return frame_to_gcrf_matrix(frame, r=r, v=v, q=q) @ _as_vector3(vector, "vector")


def _ramped_thrust_profile(
    thrust: float,
    *,
    start: float,
    burn_time: float,
    rise_time: float,
    fall_time: float | None,
    smooth: bool,
):
    thrust = _validate_nonnegative(thrust, "thrust")
    start = float(start)
    burn_time = _validate_positive(burn_time, "burn_time")
    rise_time = _validate_nonnegative(rise_time, "rise_time")
    fall_time = rise_time if fall_time is None else _validate_nonnegative(fall_time, "fall_time")
    if rise_time + fall_time > burn_time:
        raise ValueError("rise_time + fall_time must not exceed burn_time.")
    stop = start + burn_time
    fall_start = stop - fall_time

    def blend(x):
        x = float(_np.clip(x, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x) if smooth else x

    def profile(t, *args):
        t = float(t)
        if t < start or t > stop:
            return 0.0
        if rise_time > 0.0 and t < start + rise_time:
            return thrust * blend((t - start) / rise_time)
        if fall_time > 0.0 and t > fall_start:
            return thrust * blend((stop - t) / fall_time)
        return thrust

    profile.time_breakpoints = _np.array([start, start + rise_time, fall_start, stop])
    return profile
