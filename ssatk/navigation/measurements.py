"""Deterministic ground-station measurement models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.time import Time
from ssapy import get_body

from ..constants import c
from ..coordinates.earth_fixed import itrf_to_gcrf
from ..coordinates.geodetic import llh_to_gcrf
from ..coordinates.satellite_frames import enu_to_ecef_matrix
from ..time_functions import to_gps
from ..propagators_orbit import propagate_orbit_state_with_stm

_RADIO_MEASUREMENTS = frozenset({"two_way_range", "two_way_doppler"})
_SCALAR_MEASUREMENTS = frozenset({"range", "range_rate"}) | _RADIO_MEASUREMENTS
_GEOMETRIC_MEASUREMENTS = frozenset({"range", "range_rate", "az_el", "ra_dec"})
_MEASUREMENTS = _GEOMETRIC_MEASUREMENTS | _RADIO_MEASUREMENTS
_RESERVED_PROPAGATION_KWARGS = frozenset({"times", "orbit0", "r0", "v0", "t0", "stm0", "dense_output"})
_TWO_PI = 2.0 * np.pi


def _readonly_array(value, shape):
    array = np.array(value, dtype=float, copy=True)
    if array.shape != shape:
        raise ValueError(f"array must have shape {shape}; got {array.shape}.")
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class RadioTrackingConfig:
    """Constant-frequency coherent two-way tracking, with SI hardware inputs.

    ``count_interval_s`` ends at the ground receive epoch. The turnaround
    ratio multiplies the uplink frequency. Transponder delay is a fixed GPS
    coordinate-time interval, not a modeled spacecraft proper-time delay.
    Atmospheric, relativistic, clock and frequency-ramp terms are excluded.
    """

    uplink_frequency_hz: float
    count_interval_s: float
    turnaround_ratio: float = 1.0
    transponder_delay_s: float = 0.0

    def __post_init__(self):
        for name in ("uplink_frequency_hz", "count_interval_s", "turnaround_ratio", "transponder_delay_s"):
            if isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f"{name} must be numeric, not boolean.")
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0 or (name != "transponder_delay_s" and value == 0):
                raise ValueError(f"{name} must be finite and {'nonnegative' if name == 'transponder_delay_s' else 'positive'}.")
            object.__setattr__(self, name, value)
        if not np.isfinite(self.uplink_frequency_hz * self.turnaround_ratio):
            raise ValueError("coherent reference frequency must be finite.")
        if not np.isfinite((self.uplink_frequency_hz * self.turnaround_ratio / c) / self.count_interval_s):
            raise ValueError("Doppler scaling must be finite.")


def _scalar_time(time, name="time") -> float:
    if isinstance(time, (bool, np.bool_)):
        raise TypeError(f"{name} must be numeric GPS seconds or scalar astropy Time.")
    if (getattr(time, "isscalar", True) is False) or np.asarray(time).ndim != 0:
        raise ValueError(f"{name} must be a scalar GPS time.")
    try:
        values = np.asarray(to_gps(time), dtype=float)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be numeric GPS seconds or scalar astropy Time.") from None
    if values.size != 1 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite scalar GPS seconds.")
    return float(values.reshape(-1)[0])


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
        rows = 1 if self.measurement in _SCALAR_MEASUREMENTS else 2
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
    radio: RadioTrackingConfig | None = None
    t0: object = None
    propagation_kwargs: dict | None = None

    def __post_init__(self):
        if not isinstance(self.station, GroundStation):
            raise TypeError("station must be a GroundStation.")
        if self.measurement not in _MEASUREMENTS:
            raise ValueError(f"unsupported measurement {self.measurement!r}.")
        if self.measurement in _RADIO_MEASUREMENTS:
            if not isinstance(self.radio, RadioTrackingConfig) or self.t0 is None:
                raise ValueError("radio measurements require RadioTrackingConfig and an explicit state epoch t0.")
            object.__setattr__(self, "t0", _scalar_time(self.t0, "t0"))
            kwargs = {} if self.propagation_kwargs is None else dict(self.propagation_kwargs)
            if _RESERVED_PROPAGATION_KWARGS & kwargs.keys():
                raise ValueError("propagation_kwargs cannot override state, epoch, STM or dense output.")
            object.__setattr__(self, "propagation_kwargs", kwargs)
        elif self.radio is not None:
            raise ValueError("radio configuration requires a radio measurement type.")

    @property
    def angle_indices(self) -> tuple[int, ...]:
        """Indices requiring circular innovation wrapping."""

        return (0,) if self.measurement in {"az_el", "ra_dec"} else ()

    def __call__(self, state) -> tuple[np.ndarray, np.ndarray]:
        if self.measurement in _RADIO_MEASUREMENTS:
            state = np.asarray(state, dtype=float)
            if state.shape != (6,) or not np.all(np.isfinite(state)):
                raise ValueError("state must be a finite six-element GCRF SI vector.")
            time = _scalar_time(self.time)
            trajectory = propagate_orbit_state_with_stm(
                r0=state[:3], v0=state[3:], t0=self.t0, times=[self.t0, time],
                dense_output=True, **self.propagation_kwargs,
            )
            prediction = self.station.predict_radio(trajectory, time, self.measurement, self.radio)
            return np.atleast_1d(prediction.value), prediction.jacobian
        prediction = self.station.predict(state, self.time, self.measurement)
        return np.atleast_1d(prediction.value).astype(float), prediction.jacobian


@dataclass(frozen=True)
class GroundStation:
    """Earth-fixed ground station using SSATK's single GCRF<->ITRF route.

    Longitude and latitude are geodetic degrees. State vectors are six-element
    GCRF ``[x, y, z, vx, vy, vz]`` arrays in metres and metres per second;
    ``time`` is numeric GPS seconds.

    The station's own GCRF position/velocity and its local ENU basis used to
    come from two different Earth-rotation models: ``ssapy.EarthObserver``
    (IAU 1976/1980) for the position, ``ssatk.coordinates.eci_to_ecf_matrix``
    (now IAU 2006/2000A) for the basis. Resolving a target's displacement
    against that basis then carried a systematic error of the same size as
    the gap between the two models (tests/test_navigation_measurements.py
    caught this as a ~6e-9 "up versus station radial direction" mismatch at
    1e-12 tolerance). Both now come from the same route
    (:func:`ssatk.coordinates.geodetic.llh_to_gcrf`).
    """

    lon_deg: float
    lat_deg: float
    elevation_m: float = 0.0
    min_elevation_rad: float = 0.0
    fast: bool = False  # unused: kept for constructor compatibility.

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

    def _station_rv(self, time: float) -> tuple[np.ndarray, np.ndarray]:
        """GCRF position and velocity of the station (SSATK's single route)."""

        r_gcrf, v_gcrf = llh_to_gcrf(self.lon_deg, self.lat_deg, time, height=self.elevation_m)
        return np.asarray(r_gcrf, dtype=float), np.asarray(v_gcrf, dtype=float)

    def _local_basis(self, time: float) -> tuple[np.ndarray, ...]:
        """Return GCRF (east, north, up) from geodetic ENU and ITRS."""

        enu_ecef = enu_to_ecef_matrix(self.lat_deg, self.lon_deg)
        basis = itrf_to_gcrf(enu_ecef.T, np.full(3, time, dtype=float))
        return basis[0], basis[1], basis[2]

    def predict(self, state, time, measurement: str = "range") -> StationPrediction:
        """Predict one supported observable and its analytic 1x6/2x6 Jacobian."""

        if measurement not in _GEOMETRIC_MEASUREMENTS:
            raise ValueError(f"measurement must be one of {sorted(_GEOMETRIC_MEASUREMENTS)}; radio observables use predict_radio.")
        time = _scalar_time(time)
        state = np.asarray(state, dtype=float)
        if state.shape != (6,) or not np.all(np.isfinite(state)):
            raise ValueError("state must be a finite six-element GCRF SI vector.")

        station_r, station_v = self._station_rv(time)
        if station_r.shape != (3,) or station_v.shape != (3,):
            raise ValueError("station position/velocity must be three-element vectors.")
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


    def _station_at_offset(self, time, offset_s):
        epoch = time + offset_s
        position, velocity = self._station_rv(epoch)
        position, velocity = np.asarray(position, dtype=float), np.asarray(velocity, dtype=float)
        if position.shape != (3,) or velocity.shape != (3,) or not np.all(np.isfinite(np.r_[position, velocity])):
            raise ValueError("station position/velocity must be finite three-element vectors.")
        # Restore the sub-ULP offset lost when forming an absolute GPS epoch.
        position = position + velocity * ((time - epoch) + offset_s)
        return np.r_[position, velocity], None

    def _two_way_path(self, trajectory, time, offset_s, radio, earth_state):
        origin, _ = earth_state(0.0)

        def barycentric(state, offset):
            position, velocity = earth_state(offset)
            return state + np.r_[position - origin, velocity]

        station_state, _ = self._station_at_offset(time, offset_s)
        station_state = barycentric(station_state, offset_s)

        def spacecraft(offset):
            state, stm = trajectory.at(time, offset_s=offset)
            return barycentric(state, offset), stm
        down_time, down, down_state, down_stm = _signal_leg(station_state[:3], offset_s, spacecraft)
        transmit_offset = offset_s - down_time
        receive_offset = transmit_offset - radio.transponder_delay_s
        receive_state, receive_stm = spacecraft(receive_offset)
        def ground(offset):
            state, _ = self._station_at_offset(time, offset)
            return barycentric(state, offset), None
        up_time, up, ground_state, _ = _signal_leg(receive_state[:3], receive_offset, ground)
        up_unit, down_unit = up / np.linalg.norm(up), down / np.linalg.norm(down)
        transmit_derivative = (down_unit @ down_stm[:3]) / (c - down_unit @ down_state[3:])
        ground_derivative = (
            (c - up_unit @ receive_state[3:]) * transmit_derivative - up_unit @ receive_stm[:3]
        ) / (c - up_unit @ ground_state[3:])
        jacobian = -0.5 * c * ground_derivative.reshape(1, 6)
        range_m = 0.5 * (np.linalg.norm(up) + np.linalg.norm(down) + c * radio.transponder_delay_s)
        elevations = []
        for vector, offset in ((-down, offset_s), (up, receive_offset - up_time)):
            _, _, vertical = self._local_basis(time + offset)
            elevations.append(float(np.arcsin(np.clip(vertical @ vector / np.linalg.norm(vector), -1, 1))))
        elevation = min(elevations)
        return range_m, jacobian, up, down, elevation

    def predict_radio(self, trajectory, time, measurement, radio) -> StationPrediction:
        """Predict coherent two-way range [m] or mean Doppler shift [Hz].

        ``trajectory`` is ``propagate_orbit_state_with_stm(dense_output=True)``
        and must cover both spacecraft events, including the beginning of a
        Doppler count. The Jacobian refers to its initial state (or ``stm0``
        reference), not the receive-epoch state. Range is half the complete
        round-trip path including fixed transponder delay. Doppler is received
        minus ``turnaround_ratio * uplink_frequency_hz``: recession is negative.

        Both endpoints include Earth's DE430 barycentric motion, reusing the
        planetary kernel already opened by SSAPy's Sun position model.

        """
        # shortcut: vacuum coordinate-time links only, add media and clock corrections before precision operational use.
        if measurement not in _RADIO_MEASUREMENTS or not isinstance(radio, RadioTrackingConfig):
            raise ValueError("predict_radio requires a radio measurement and RadioTrackingConfig.")
        time = _scalar_time(time)
        sun = get_body("sun")
        try:
            earth_state = lambda offset: _earth_state(time, offset, sun.position.kernel)
            end = self._two_way_path(trajectory, time, 0.0, radio, earth_state)
            start = (
                self._two_way_path(trajectory, time, -radio.count_interval_s, radio, earth_state)
                if measurement == "two_way_doppler" else None
            )
        finally:
            sun.position.close()
        value, jacobian, _, _, elevation = end
        if measurement == "two_way_doppler":
            # Difference distances through their vectors, avoiding cancellation
            # of large paths when the count interval or range rate is small.
            path_change = sum(
                (last - first) @ (last + first) / (np.linalg.norm(last) + np.linalg.norm(first))
                for last, first in zip(end[2:4], start[2:4])
            )
            scale = -(radio.turnaround_ratio * radio.uplink_frequency_hz / c) / radio.count_interval_s
            value = scale * path_change
            jacobian = 2.0 * scale * (end[1] - start[1])
            elevation = min(elevation, start[4])
        if not np.isfinite(value) or not np.all(np.isfinite(jacobian)):
            raise ValueError("radio prediction and Jacobian must be finite.")
        return StationPrediction(time, measurement, value, jacobian, elevation >= self.min_elevation_rad, elevation)


def _earth_state(time, offset_s, kernel):
    """Earth's barycentric SI state at a two-part GPS signal epoch."""
    epoch = Time(time, offset_s, format="gps", scale="tai").tdb
    emb_r, emb_v = kernel[0, 3].compute_and_differentiate(epoch.jd1, epoch.jd2)
    earth_r, earth_v = kernel[3, 399].compute_and_differentiate(epoch.jd1, epoch.jd2)
    return (emb_r + earth_r) * 1000, (emb_v + earth_v) * (1000 / 86400)


def _signal_leg(receiver, receive_offset, emitter):
    """Solve a vacuum leg backwards from its fixed receive event."""
    delay = 0.0
    for _ in range(20):
        state, stm = emitter(receive_offset - delay)
        state = np.asarray(state, dtype=float)
        if state.shape != (6,) or not np.all(np.isfinite(state)) or np.linalg.norm(state[3:]) >= c:
            raise ValueError("signal endpoints require finite subluminal GCRF SI states.")
        vector = receiver - state[:3]
        distance = float(np.linalg.norm(vector))
        if not np.isfinite(distance) or distance <= np.finfo(float).eps:
            raise ValueError("signal endpoints must be distinct and finite.")
        updated = distance / c
        if abs(updated - delay) <= 1e-12:
            return delay, vector, state, stm
        delay = updated
    raise RuntimeError("radio light-time solution did not converge in 20 iterations.")


__all__ = ["GroundStation", "GroundStationMeasurement", "RadioTrackingConfig", "StationPrediction"]
