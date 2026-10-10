"""Finite-burn thrust profiles and maneuver accelerations."""

from __future__ import annotations

import csv as _csv
import json as _json
from io import StringIO as _StringIO
from pathlib import Path as _Path

import numpy as _np

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
    """

    def __init__(self, times: ArrayLike, thrusts: ArrayLike, *, fill_value: float = 0.0):
        self.times = _as_1d(times, "times")
        self.thrusts = _as_1d(thrusts, "thrusts")
        if self.times.size != self.thrusts.size or self.times.size < 2:
            raise ValueError("times and thrusts must have the same length >= 2.")
        if not _np.all(_np.diff(self.times) > 0.0):
            raise ValueError("times must be strictly increasing.")
        if _np.any(self.thrusts < 0.0):
            raise ValueError("thrusts must be non-negative.")
        self.fill_value = _validate_nonnegative(fill_value, "fill_value")

    def __call__(self, t, *args) -> float:
        return float(_np.interp(float(t), self.times, self.thrusts, left=self.fill_value, right=self.fill_value))

    @property
    def total_impulse(self) -> float:
        """Trapezoidal total impulse over the tabulated span in N s."""

        return float(_np.trapezoid(self.thrusts, self.times))


class SpacecraftManeuverAccel(SpacecraftAccel):
    """Finite thrust maneuver in a satellite-operations frame.

    ``thrust`` may be a scalar in newtons, a callable/profile, a
    :class:`ThrustCurve`, or a 3-vector thrust in the selected frame. ``frame``
    accepts the same frame names as ``ssatk.coordinates``: ``rtn``,
    ``lvlh``, ``ric``, ``vnb``, ``ntw``, ``body``, ``gcrf``/``inertial``, etc.
    The default ``frame="rtn", direction=[0, 1, 0]`` is transverse/prograde in
    the common RTN/RIC/LVLH operations convention. Use ``frame="ntw"`` for
    exact SSAPy-style ``[N, T, W]`` inputs.
    """

    def __init__(
        self,
        thrust,
        *,
        direction: ArrayLike | None = None,
        frame: str = "rtn",
        mass: float | None = None,
        isp: float | None = None,
        tank_name: str | None = None,
        start: float = -_np.inf,
        stop: float = _np.inf,
    ):
        self.thrust = thrust
        self.frame = frame
        self.direction = _default_maneuver_direction(frame) if direction is None else _unit_vector(direction, "direction")
        self.mass = None if mass is None else _validate_positive(mass, "mass")
        self.isp = None if isp is None else _validate_positive(isp, "isp")
        self.tank_name = None if tank_name is None else str(tank_name)
        self.start = float(start)
        self.stop = float(stop)
        if self.stop < self.start:
            raise ValueError("stop must be greater than or equal to start.")

    def acceleration(self, *, t, r, v, q, omega, spacecraft=None) -> _np.ndarray:
        thrust_frame = self.thrust_vector(t, r, v, q, omega, spacecraft)
        if not _np.any(thrust_frame):
            return _np.zeros(3)
        return _frame_vector_to_gcrf(self.frame, thrust_frame, r, v, q) / _mass_from_spacecraft(self.mass, spacecraft)

    def thrust_vector(self, t, r, v, q, omega, spacecraft=None) -> _np.ndarray:
        """Return thrust vector in the selected maneuver frame in newtons."""

        if not (self.start <= float(t) <= self.stop):
            return _np.zeros(3)
        if self.tank_name is not None and spacecraft is not None:
            tanks = tuple(getattr(getattr(spacecraft, "body", None), "tanks", ()))
            selected = [tank for tank in tanks if tank.name == self.tank_name]
            if len(selected) != 1:
                raise ValueError(f"expected exactly one tank named {self.tank_name!r}.")
            if selected[0].propellant_mass <= 0.0:
                return _np.zeros(3)
        value = _call_optional(self.thrust, t, r, v, q, omega, spacecraft) if callable(self.thrust) else self.thrust
        value = _np.asarray(value, dtype=float)
        if value.shape == ():
            thrust = _validate_nonnegative(float(value), "thrust")
            return self.direction * thrust
        if value.shape == (3,):
            if _np.any(~_np.isfinite(value)):
                raise ValueError("thrust vector must be finite.")
            return value
        raise ValueError("thrust must be a scalar profile or 3-vector.")

    def mass_flow_rate(self, *args, **kwargs) -> float:
        """Return positive propellant mass flow in kg/s if ``isp`` is set."""

        if self.isp is None:
            return 0.0
        spacecraft, t, r, v, q, omega = _parse_state_args(args, kwargs)
        if self.tank_name is not None and spacecraft is not None:
            tanks = tuple(getattr(getattr(spacecraft, "body", None), "tanks", ()))
            selected = [tank for tank in tanks if tank.name == self.tank_name]
            if len(selected) != 1:
                raise ValueError(f"expected exactly one tank named {self.tank_name!r}.")
            if selected[0].propellant_mass <= 0.0:
                return 0.0
        return float(_np.linalg.norm(self.thrust_vector(t, r, v, q, omega, spacecraft)) / (self.isp * STANDARD_GRAVITY))


def thrust_profile_constant(thrust: float, *, start: float = -_np.inf, stop: float = _np.inf):
    """Return a constant finite-burn thrust profile in newtons."""

    thrust = _validate_nonnegative(thrust, "thrust")
    start = float(start)
    stop = float(stop)
    if stop < start:
        raise ValueError("stop must be greater than or equal to start.")

    def profile(t, *args):
        return thrust if start <= float(t) <= stop else 0.0

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

    return profile


def load_thrust_curve_csv(
    path,
    *,
    time_column: str = "time_s",
    thrust_column: str = "thrust_n",
    time_scale: float = 1.0,
    thrust_scale: float = 1.0,
    fill_value: float = 0.0,
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
    return ThrustCurve(times, thrusts, fill_value=fill_value)


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
        return load_thrust_curve_data(resource_path, fill_value=fill_value)
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
    return ThrustCurve(times, thrusts, fill_value=fill_value)


def load_digitized_thrust_curve(identifier: str, **kwargs) -> ThrustCurve:
    """Load a digitized NASA NTRS thrust curve by NTRS ID or file stem."""

    return load_packaged_thrust_curve(identifier, collection="nasa_ntrs", **kwargs)


def integrated_thrust_impulse(profile, start: float, stop: float, *, samples: int = 1001) -> float:
    """Numerically integrate a thrust profile over ``[start, stop]`` in N s."""

    if samples < 2:
        raise ValueError("samples must be at least 2.")
    times = _np.linspace(float(start), float(stop), int(samples))
    thrusts = _np.array([float(profile(t)) for t in times], dtype=float)
    return float(_np.trapezoid(thrusts, times))


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
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return value


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

    return profile
