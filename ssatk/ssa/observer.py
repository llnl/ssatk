"""Ground-observer state and topocentric geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..coordinates.earth_fixed import itrf_to_gcrf
from ..coordinates.geodetic import llh_to_gcrf
from ..coordinates.satellite_frames import enu_to_ecef_matrix
from ..navigation.measurements import GroundStation
from ..time_functions import to_gps


def _finite_state(value, shape, name):
    array = np.array(value, dtype=float, copy=True)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    return array


@dataclass(frozen=True)
class ObserverStates:
    """Finite, strictly increasing GPS epochs and corresponding GCRF states."""

    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    enu_basis: np.ndarray | None = None

    def __post_init__(self):
        times = np.array(to_gps(self.times), dtype=float, copy=True)
        if times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times)):
            raise ValueError("times must be a non-empty finite one-dimensional GPS array.")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("times must be strictly increasing.")
        count = times.size
        positions = _finite_state(self.positions, (count, 3), "positions")
        velocities = _finite_state(self.velocities, (count, 3), "velocities")
        basis = None
        if self.enu_basis is not None:
            basis = _finite_state(self.enu_basis, (count, 3, 3), "enu_basis")
            if not np.allclose(np.einsum("nij,nkj->nik", basis, basis), np.eye(3), atol=2e-12, rtol=0.0):
                raise ValueError("enu_basis must contain orthonormal east/north/up rows.")
            if not np.all(np.linalg.det(basis) > 0.0):
                raise ValueError("enu_basis must be right-handed.")
        for value in (times, positions, velocities, basis):
            if value is not None:
                value.flags.writeable = False
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "velocities", velocities)
        object.__setattr__(self, "enu_basis", basis)


def from_ground_station(station, times):
    """Build observer states and GCRF ENU bases from a :class:`GroundStation`."""

    if not isinstance(station, GroundStation):
        raise TypeError("station must be a GroundStation.")
    times = np.atleast_1d(np.asarray(to_gps(times), dtype=float))
    if times.ndim != 1:
        raise ValueError("times must be one-dimensional.")
    states = [llh_to_gcrf(station.lon_deg, station.lat_deg, time, height=station.elevation_m)
              for time in times]
    positions = np.asarray([state[0] for state in states], dtype=float)
    velocities = np.asarray([state[1] for state in states], dtype=float)
    enu_ecef = enu_to_ecef_matrix(station.lat_deg, station.lon_deg)
    basis_rows = np.broadcast_to(enu_ecef.T, (times.size, 3, 3))
    enu_basis = itrf_to_gcrf(
        basis_rows.reshape(-1, 3), np.repeat(times, 3)
    ).reshape(times.size, 3, 3)
    return ObserverStates(times, positions, velocities, enu_basis)


def observer_metrics(positions, velocities, observer):
    """Return topocentric range, range rate, azimuth, elevation, and validity."""

    if not isinstance(observer, ObserverStates):
        raise TypeError("observer must be an ObserverStates instance.")
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    if positions.ndim < 2 or positions.shape[-1] != 3 or velocities.shape != positions.shape:
        raise ValueError("positions and velocities must have matching shape (..., 3).")
    if positions.shape[-2] != observer.times.size:
        raise ValueError("the observer time dimension must match positions.")
    delta = positions - observer.positions
    delta_v = velocities - observer.velocities
    raw_range_m = np.linalg.norm(delta, axis=-1)
    valid = (
        np.all(np.isfinite(delta), axis=-1)
        & np.all(np.isfinite(delta_v), axis=-1)
        & np.isfinite(raw_range_m)
        & (raw_range_m > 0.0)
    )
    range_m = np.where(valid, raw_range_m, np.nan)
    range_rate = np.full(range_m.shape, np.nan)
    np.divide(np.sum(delta * delta_v, axis=-1), range_m, out=range_rate, where=valid)
    azimuth = np.full(range_m.shape, np.nan)
    elevation = np.full(range_m.shape, np.nan)
    if observer.enu_basis is not None:
        local = np.einsum("tij,...tj->...ti", observer.enu_basis, delta)
        horizontal = np.hypot(local[..., 0], local[..., 1])
        angle_valid = valid & (horizontal > 0.0)
        azimuth[angle_valid] = np.mod(np.arctan2(local[..., 0], local[..., 1]), 2.0 * np.pi)[angle_valid]
        elevation[valid] = np.arctan2(local[..., 2], horizontal)[valid]
    return {
        "range_m": range_m,
        "range_rate_m_s": range_rate,
        "azimuth_rad": azimuth,
        "elevation_rad": elevation,
        "valid": valid,
    }


def earth_occulted(target_positions, observer_positions, radius_m=6378137.0):
    """Return spherical-Earth line-of-sight occultation flags.

    This is a geometric spherical model, not an ellipsoidal terrain or
    atmosphere visibility model.
    """

    target, observer = np.broadcast_arrays(
        np.asarray(target_positions, dtype=float),
        np.asarray(observer_positions, dtype=float),
    )
    if target.shape[-1:] != (3,):
        raise ValueError("target_positions and observer_positions must have shape (..., 3).")
    if not np.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError("radius_m must be positive and finite.")
    line = target - observer
    length2 = np.sum(line * line, axis=-1)
    parameter = np.divide(
        -np.sum(observer * line, axis=-1), length2,
        out=np.zeros_like(length2), where=length2 > 0.0,
    )
    closest = observer + np.clip(parameter, 0.0, 1.0)[..., None] * line
    return np.all(np.isfinite(target), axis=-1) & np.all(np.isfinite(observer), axis=-1) & (
        np.sum(closest * closest, axis=-1) < float(radius_m) ** 2
    )
