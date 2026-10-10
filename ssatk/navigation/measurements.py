"""Deterministic ground-station measurement models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ssapy import EarthObserver

from ..coordinates.earth_fixed import itrf_to_gcrf
from ..coordinates.satellite_frames import enu_to_ecef_matrix
from ..time_functions import to_gps

_MEASUREMENTS = frozenset({"range", "range_rate", "az_el", "ra_dec"})
_TWO_PI = 2.0 * np.pi


def _readonly_array(value, shape):
    array = np.array(value, dtype=float, copy=True)
    if array.shape != shape:
        raise ValueError(f"array must have shape {shape}; got {array.shape}.")
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class StationPrediction:
    """One deterministic station prediction and its state Jacobian."""

    time: float
    measurement: str
    value: float | np.ndarray
    jacobian: np.ndarray
    visible: bool
    elevation_rad: float

    def __post_init__(self):
        if self.measurement not in _MEASUREMENTS:
            raise ValueError(f"unsupported measurement {self.measurement!r}.")
        value = self.value if np.isscalar(self.value) else _readonly_array(self.value, (2,))
        object.__setattr__(self, "value", float(value) if np.isscalar(value) else value)
        rows = 1 if self.measurement in {"range", "range_rate"} else 2
        object.__setattr__(self, "jacobian", _readonly_array(self.jacobian, (rows, 6)))
        object.__setattr__(self, "time", float(self.time))
        object.__setattr__(self, "elevation_rad", float(self.elevation_rad))

    @property
    def kind(self) -> str:
        """Alias for the measurement type."""

        return self.measurement


@dataclass(frozen=True)
class GroundStationMeasurement:
    """EKF-compatible callable for one ground-station observable."""

    station: GroundStation
    time: object
    measurement: str = "range"

    def __post_init__(self):
        if not isinstance(self.station, GroundStation):
            raise TypeError("station must be a GroundStation.")
        if self.measurement not in _MEASUREMENTS:
            raise ValueError(f"unsupported measurement {self.measurement!r}.")

    @property
    def angle_indices(self) -> tuple[int, ...]:
        """Indices requiring circular innovation wrapping."""

        return (0,) if self.measurement in {"az_el", "ra_dec"} else ()

    def __call__(self, state) -> tuple[np.ndarray, np.ndarray]:
        prediction = self.station.predict(state, self.time, self.measurement)
        return np.atleast_1d(prediction.value).astype(float), prediction.jacobian


@dataclass(frozen=True)
class GroundStation:
    """Earth-fixed ground station using SSAPy's ``EarthObserver``.

    Longitude and latitude are geodetic degrees. State vectors are six-element
    GCRF ``[x, y, z, vx, vy, vz]`` arrays in metres and metres per second;
    ``time`` is numeric GPS seconds.
    """

    lon_deg: float
    lat_deg: float
    elevation_m: float = 0.0
    min_elevation_rad: float = 0.0
    fast: bool = False

    def __post_init__(self):
        for name in ("lon_deg", "lat_deg", "elevation_m", "min_elevation_rad"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value)
        if not -180.0 <= self.lon_deg <= 180.0:
            raise ValueError("lon_deg must be in [-180, 180].")
        if not -90.0 <= self.lat_deg <= 90.0:
            raise ValueError("lat_deg must be in [-90, 90].")
        if not -np.pi / 2 <= self.min_elevation_rad <= np.pi / 2:
            raise ValueError("min_elevation_rad must be in [-pi/2, pi/2].")

    def _observer(self) -> EarthObserver:
        return EarthObserver(
            lon=self.lon_deg,
            lat=self.lat_deg,
            elevation=self.elevation_m,
            fast=self.fast,
        )

    def _local_basis(self, time: float) -> tuple[np.ndarray, ...]:
        """Return GCRF (east, north, up) from geodetic ENU and ITRS."""

        enu_ecef = enu_to_ecef_matrix(self.lat_deg, self.lon_deg)
        basis = itrf_to_gcrf(enu_ecef.T, np.full(3, time, dtype=float))
        return basis[0], basis[1], basis[2]

    def predict(self, state, time, measurement: str = "range") -> StationPrediction:
        """Predict one supported observable and its analytic 1x6/2x6 Jacobian."""

        if measurement not in _MEASUREMENTS:
            raise ValueError(f"measurement must be one of {sorted(_MEASUREMENTS)}.")
        if isinstance(time, bool):
            raise TypeError("time must be numeric GPS seconds or scalar astropy Time.")
        if (getattr(time, "isscalar", True) is False) or np.asarray(time).ndim != 0:
            raise ValueError("time must be a scalar GPS time.")
        try:
            time_values = np.asarray(to_gps(time), dtype=float)
        except (TypeError, ValueError):
            raise TypeError("time must be numeric GPS seconds or scalar astropy Time.") from None
        if time_values.size != 1:
            raise ValueError("time must be a scalar GPS time.")
        time = float(time_values.reshape(-1)[0])
        if not np.isfinite(time):
            raise ValueError("time must be finite GPS seconds.")
        state = np.asarray(state, dtype=float)
        if state.shape != (6,) or not np.all(np.isfinite(state)):
            raise ValueError("state must be a finite six-element GCRF SI vector.")

        observer = self._observer()
        station_r, station_v = (np.asarray(item, dtype=float) for item in observer.getRV(time))
        if station_r.shape != (3,) or station_v.shape != (3,):
            raise ValueError("EarthObserver.getRV must return two three-element vectors.")
        dr = state[:3] - station_r
        dv = state[3:] - station_v
        rho = float(np.linalg.norm(dr))
        rho2 = rho * rho
        if rho <= np.finfo(float).eps:
            raise ValueError("target and station positions must be distinct.")
        u = dr / rho
        range_rate = float(np.dot(u, dv))
        position_jacobian = np.zeros((1, 6), dtype=float)
        position_jacobian[0, :3] = u
        range_rate_jacobian = np.zeros((1, 6), dtype=float)
        range_rate_jacobian[0, :3] = (dv - range_rate * u) / rho
        range_rate_jacobian[0, 3:] = u

        east, north, up = self._local_basis(time)
        local = np.array([np.dot(north, dr), np.dot(east, dr), np.dot(up, dr)])
        horizontal = float(np.hypot(local[0], local[1]))
        azimuth = float(np.mod(np.arctan2(local[1], local[0]), _TWO_PI))
        elevation = float(np.arctan2(local[2], horizontal))
        visible = elevation >= self.min_elevation_rad

        if measurement == "range":
            value, jacobian = rho, position_jacobian
        elif measurement == "range_rate":
            value, jacobian = range_rate, range_rate_jacobian
        elif measurement == "az_el":
            if horizontal <= np.finfo(float).eps:
                raise ValueError("azimuth is singular at the local zenith or nadir.")
            h2 = horizontal * horizontal
            q_jacobian = np.array(
                [[-local[1] / h2, local[0] / h2, 0.0],
                 [-local[2] * local[0] / (rho2 * horizontal),
                  -local[2] * local[1] / (rho2 * horizontal),
                  horizontal / rho2]],
            )
            value = np.array([azimuth, elevation])
            jacobian = np.zeros((2, 6), dtype=float)
            jacobian[:, :3] = q_jacobian @ np.array([north, east, up])
        else:
            xy2 = float(np.dot(dr[:2], dr[:2]))
            if xy2 <= np.finfo(float).eps:
                raise ValueError("right ascension is singular on the inertial z axis.")
            xy = np.sqrt(xy2)
            right_ascension = float(np.mod(np.arctan2(dr[1], dr[0]), _TWO_PI))
            declination = float(np.arctan2(dr[2], xy))
            value = np.array([right_ascension, declination])
            jacobian = np.zeros((2, 6), dtype=float)
            jacobian[0, :3] = [-dr[1] / xy2, dr[0] / xy2, 0.0]
            jacobian[1, :3] = [
                -dr[2] * dr[0] / (rho2 * xy),
                -dr[2] * dr[1] / (rho2 * xy),
                xy / rho2,
            ]

        return StationPrediction(time, measurement, value, jacobian, visible, elevation)


__all__ = ["GroundStation", "GroundStationMeasurement", "StationPrediction"]
