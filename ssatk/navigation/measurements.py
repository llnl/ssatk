"""Deterministic ground-station measurement models."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from functools import cache

import numpy as np
from astropy.time import Time
from scipy.integrate import quad
from scipy.optimize import brentq
from ssapy import get_body

from ..constants import c, EARTH_MU
from ..coordinates.earth_fixed import itrf_to_gcrf
from ..coordinates.geodetic import llh_to_gcrf
from ..coordinates.satellite_frames import enu_to_ecef_matrix
from ..time_functions import to_gps
from ..propagators_orbit import propagate_orbit_state_with_stm

_RADIO_MEASUREMENTS = frozenset({
    "two_way_range", "two_way_doppler", "three_way_range", "three_way_doppler",
    "differenced_range", "differenced_doppler",
})
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
    """Coherent tracking hardware, ramps and relativistic corrections in SI.

    Ramp rows are [start GPS, stop GPS, frequency Hz, rate Hz/s], in the
    transmitting station's clock. No extrapolation or gap filling is allowed.
    The count is in receiving clock seconds. A separate transmitter supplies
    three-way links; differenced observables subtract the reference link.
    Proper delays use the geocentric weak-field d(proper time)/d(TT) by default,
    or an explicit calibrated ``transponder_clock(time, state)`` rate.
    """

    uplink_frequency_hz: float
    count_interval_s: float
    turnaround_ratio: float = 1.0
    transponder_delay_s: float = 0.0
    transmitter: GroundStation | None = None
    reference_receiver: GroundStation | None = None
    reference_radio: RadioTrackingConfig | None = None
    frequency_ramps: np.ndarray | None = None
    reference_frequency_hz: float | None = None
    transponder_delay_is_proper: bool = False
    transponder_clock: Callable | None = None
    shapiro_bodies: tuple[str, ...] = ()

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
        for name in ("transmitter", "reference_receiver"):
            if getattr(self, name) is not None and not isinstance(getattr(self, name), GroundStation):
                raise TypeError(f"{name} must be a GroundStation.")
        if self.reference_radio is not None:
            if not isinstance(self.reference_radio, RadioTrackingConfig) or self.reference_receiver is None:
                raise ValueError("reference_radio requires a reference_receiver and RadioTrackingConfig.")
            if self.reference_radio.reference_receiver is not None:
                raise ValueError("reference links cannot contain another reference link.")
        if self.reference_frequency_hz is not None:
            value = float(self.reference_frequency_hz)
            if not np.isfinite(value) or value <= 0:
                raise ValueError("reference_frequency_hz must be finite and positive.")
            object.__setattr__(self, "reference_frequency_hz", value)
        if self.frequency_ramps is not None:
            ramps = np.array(self.frequency_ramps, dtype=float, copy=True)
            if ramps.ndim != 2 or ramps.shape[1] != 4 or not len(ramps) or not np.all(np.isfinite(ramps)):
                raise ValueError("frequency_ramps must be nonempty finite [start, stop, frequency, rate] rows.")
            if np.any(ramps[:, 1] <= ramps[:, 0]) or np.any(ramps[1:, 0] != ramps[:-1, 1]):
                raise ValueError("frequency ramps must be chronological, contiguous and have positive durations.")
            ends = ramps[:, 2] + ramps[:, 3] * (ramps[:, 1] - ramps[:, 0])
            if np.any(ramps[:, 2] <= 0) or np.any(ends <= 0) or not np.all(np.isfinite(ends)):
                raise ValueError("ramp frequencies must remain finite and positive throughout every interval.")
            ramps.flags.writeable = False
            object.__setattr__(self, "frequency_ramps", ramps)
        if not isinstance(self.transponder_delay_is_proper, bool):
            raise TypeError("transponder_delay_is_proper must be boolean.")
        if self.transponder_clock is not None and (not callable(self.transponder_clock) or not self.transponder_delay_is_proper):
            raise ValueError("transponder_clock must be callable and requires a proper-time delay.")
        bodies = tuple(self.shapiro_bodies)
        if any(name not in {"earth", "sun"} for name in bodies) or len(set(bodies)) != len(bodies):
            raise ValueError("shapiro_bodies must contain distinct 'earth' or 'sun' names.")
        object.__setattr__(self, "shapiro_bodies", bodies)

    @property
    def reference_frequency(self):
        """Receiving reference oscillator frequency, in Hz."""
        return self.uplink_frequency_hz * self.turnaround_ratio if self.reference_frequency_hz is None else self.reference_frequency_hz

    def uplink_frequency(self, time, offset_s=0.0):
        """Frequency at a two-part transmitting-clock epoch, in Hz."""
        if not np.isfinite(time) or not np.isfinite(offset_s):
            raise ValueError("uplink epoch and offset must be finite GPS seconds.")
        if self.frequency_ramps is None:
            return self.uplink_frequency_hz
        starts = self.frequency_ramps[:, 0] - time
        stops = self.frequency_ramps[:, 1] - time
        if offset_s < starts[0] or offset_s > stops[-1]:
            raise ValueError("uplink epoch is outside the frequency-ramp coverage.")
        index = min(np.searchsorted(starts, offset_s, side="right") - 1, len(starts) - 1)
        row = self.frequency_ramps[index]
        return float(row[2] + row[3] * (offset_s - starts[index]))

    def uplink_phase_excess(self, time, start_s, stop_s):
        """Integral of uplink frequency minus the base carrier, in cycles."""
        if stop_s < start_s:
            raise ValueError("uplink phase interval must be chronological.")
        self.uplink_frequency(time, start_s)
        self.uplink_frequency(time, stop_s)
        if self.frequency_ramps is None:
            return 0.0
        total = 0.0
        for beginning, ending, frequency, rate in self.frequency_ramps:
            first, last = max(start_s, beginning - time), min(stop_s, ending - time)
            if last > first:
                midpoint = (first + last) / 2 - (beginning - time)
                total += (frequency - self.uplink_frequency_hz + rate * midpoint) * (last - first)
        return total


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
                r0=state[:3], v0=state[3:], t0=self.t0,
                times=[self.t0, max(time, *_radio_receive_epochs(self.station, time, self.radio))],
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
    clock_offset_s: float = 0.0
    clock_drift: float = 0.0
    clock_epoch: float = 0.0
    transmit_delay_s: float = 0.0
    receive_delay_s: float = 0.0
    meteorology: tuple[float, float, float] | None = None
    zenith_troposphere: tuple[float, float] | Callable | None = None
    troposphere_mapping: Callable | None = None
    slant_tec: float | Callable = 0.0
    media_delay: Callable | None = None

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
        for name in ("clock_offset_s", "clock_drift", "clock_epoch", "transmit_delay_s", "receive_delay_s"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or (name.endswith("delay_s") and value < 0):
                raise ValueError(f"{name} must be finite and equipment delays nonnegative.")
            object.__setattr__(self, name, value)
        if 1 + self.clock_drift <= 0:
            raise ValueError("station clock rate must be positive.")
        if self.meteorology is not None:
            weather = np.asarray(self.meteorology, dtype=float)
            if weather.shape != (3,) or not np.all(np.isfinite(weather)):
                raise ValueError("meteorology must be [pressure hPa, temperature K, water-vapour pressure hPa].")
            if weather[0] <= 0 or weather[1] <= 0 or weather[2] < 0 or weather[2] > weather[0]:
                raise ValueError("meteorology requires positive pressure/temperature and water pressure between zero and total pressure.")
            object.__setattr__(self, "meteorology", tuple(weather))
        if self.zenith_troposphere is not None:
            if self.meteorology is not None:
                raise ValueError("supply meteorology or calibrated zenith_troposphere, not both.")
            if not callable(self.zenith_troposphere):
                zenith = np.asarray(self.zenith_troposphere, dtype=float)
                if zenith.shape != (2,) or not np.all(np.isfinite(zenith)) or np.any(zenith < 0):
                    raise ValueError("zenith_troposphere must be nonnegative [dry m, wet m] or a callable.")
                object.__setattr__(self, "zenith_troposphere", tuple(zenith))
        if self.troposphere_mapping is not None and not callable(self.troposphere_mapping):
            raise TypeError("troposphere_mapping must be callable.")
        if not callable(self.slant_tec):
            tec = float(self.slant_tec)
            if not np.isfinite(tec) or tec < 0:
                raise ValueError("slant_tec must be nonnegative electrons/m² or a callable.")
            object.__setattr__(self, "slant_tec", tec)
        if self.media_delay is not None and not callable(self.media_delay):
            raise TypeError("media_delay must be callable.")

    def _clock_offset(self, time, offset_s):
        return self.clock_offset_s + self.clock_drift * ((time - self.clock_epoch) + offset_s)

    def atmospheric_delay(self, time, elevation_rad, frequency_hz, *, phase=False):
        """Slant excess path [m]: Saastamoinen weather and first-order plasma.

        Weather is explicit hPa/K/hPa; the plane-parallel mapping is restricted
        to elevations >= 10 degrees and station heights from -500 to 5000 m.
        Slant TEC is electrons/m², optionally ``slant_tec(time, elevation)``.
        ``media_delay(time, elevation, frequency, phase)`` supplies an additional
        calibrated slant path, e.g. a higher-fidelity mapping or solar plasma.
        Plasma delays code/group and advances carrier phase (opposite signs).
        """
        if not np.isfinite(frequency_hz) or frequency_hz <= 0 or not np.isfinite(elevation_rad):
            raise ValueError("media corrections require finite elevation and positive frequency.")
        delay = 0.0
        zenith = None
        if self.meteorology is not None:
            if not -500 <= self.elevation_m <= 5000:
                raise ValueError("Saastamoinen weather requires station height -500..5000 m.")
            pressure, temperature, water = self.meteorology
            denominator = 1 - .00266 * np.cos(2 * np.deg2rad(self.lat_deg)) - .00028 * self.elevation_m / 1000
            zenith = (.0022768 * pressure / denominator, .002277 * (1255 / temperature + .05) * water)
        elif self.zenith_troposphere is not None:
            zenith = self.zenith_troposphere(time) if callable(self.zenith_troposphere) else self.zenith_troposphere
        if zenith is not None:
            zenith = np.asarray(zenith, dtype=float)
            if zenith.shape != (2,) or not np.all(np.isfinite(zenith)) or np.any(zenith < 0):
                raise ValueError("zenith_troposphere must return nonnegative finite dry/wet delays.")
            if self.troposphere_mapping is None:
                if elevation_rad < np.deg2rad(10):
                    raise ValueError("plane-parallel troposphere mapping requires elevation >= 10 degrees.")
                mapping = np.full(2, 1 / np.sin(elevation_rad))
            else:
                mapping = np.asarray(self.troposphere_mapping(time, elevation_rad), dtype=float)
            if mapping.shape != (2,) or not np.all(np.isfinite(mapping)) or np.any(mapping <= 0):
                raise ValueError("troposphere_mapping must return finite positive dry/wet factors.")
            delay += float(zenith @ mapping)
        tec = self.slant_tec(time, elevation_rad) if callable(self.slant_tec) else self.slant_tec
        tec = float(tec)
        if not np.isfinite(tec) or tec < 0:
            raise ValueError("slant_tec must return finite nonnegative electrons/m².")
        delay += (-1 if phase else 1) * 40.3 * tec / frequency_hz**2
        if self.media_delay is not None:
            delay += float(self.media_delay(time, elevation_rad, frequency_hz, phase))
        if not np.isfinite(delay):
            raise ValueError("media excess path must be finite.")
        return float(delay)

    @property
    def _has_radio_corrections(self):
        return bool(self.clock_offset_s or self.clock_drift or self.transmit_delay_s or self.receive_delay_s
                    or self.meteorology is not None or self.zenith_troposphere is not None
                    or callable(self.slant_tec) or self.slant_tec or self.media_delay is not None)

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


    def _station_at_offset(self, time, offset_s, state=None):
        epoch = time + offset_s
        position, velocity = self._station_rv(epoch) if state is None else state
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
        """Predict coherent two-/three-way or differenced range [m]/Doppler [Hz].

        ``trajectory`` is ``propagate_orbit_state_with_stm(dense_output=True)``
        and must cover both spacecraft events, including the beginning of a
        Doppler count. The Jacobian refers to its initial state (or ``stm0``
        reference), not the receive-epoch state. Range is half the complete
        round-trip path including fixed transponder delay. Doppler is received
        minus ``turnaround_ratio * uplink_frequency_hz``: recession is negative.
        A separate transmitter, ramps, clock/media calibration, proper delays
        and Earth/Sun Shapiro terms are supplied through the existing config.
        Differenced predictions subtract the reference link in that config.

        Both endpoints include Earth's DE430 barycentric motion, reusing the
        planetary kernel already opened by SSAPy's Sun position model.

        """
        if measurement not in _RADIO_MEASUREMENTS or not isinstance(radio, RadioTrackingConfig):
            raise ValueError("predict_radio requires a radio measurement and RadioTrackingConfig.")
        time = _scalar_time(time)
        if measurement.startswith("three_way") and radio.transmitter is None:
            raise ValueError("three-way observations require an explicit transmitting station.")
        if measurement.startswith("differenced") and radio.reference_receiver is None:
            raise ValueError("differenced observations require a reference receiving station.")
        advanced = (measurement not in {"two_way_range", "two_way_doppler"} or radio.transmitter is not None
                    or radio.frequency_ramps is not None or radio.reference_frequency_hz is not None
                    or radio.transponder_delay_is_proper or radio.shapiro_bodies or self._has_radio_corrections)
        if advanced:
            return self._predict_advanced_radio(trajectory, time, measurement, radio)
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

    def _predict_advanced_radio(self, trajectory, time, measurement, radio):
        sun = get_body("sun")
        station_cache = {}
        try:
            earth_state = cache(lambda offset: _earth_state(time, offset, sun.position.kernel))
            solar_mu = sun.mu
            @cache
            def solar_position(offset):
                epoch = Time(time, offset, format="gps", scale="tai").tdb
                return sun.position.kernel[0, 10].compute(epoch.jd1, epoch.jd2) * 1000

            def evaluate(delta):
                main = self._radio_observable(trajectory, time, measurement, radio, earth_state, solar_position, solar_mu, delta, station_cache)
                if measurement.startswith("differenced"):
                    reference_radio = radio.reference_radio or radio
                    other = radio.reference_receiver._radio_observable(
                        trajectory, time, measurement, reference_radio, earth_state, solar_position, solar_mu, delta, station_cache,
                    )
                    return main[0] - other[0], min(main[1], other[1]), main[2] and other[2]
                return main

            value, elevation, visible = evaluate(np.zeros(6))
            jacobian = np.empty((1, 6))
            # Differentiate solved events along the existing STM, including all corrections.
            for column, step in enumerate((1., 1., 1., .001, .001, .001)):
                delta = np.zeros(6)
                delta[column] = step
                jacobian[0, column] = (evaluate(delta)[0] - evaluate(-delta)[0]) / (2 * step)
        finally:
            sun.position.close()
        if not np.isfinite(value) or not np.all(np.isfinite(jacobian)):
            raise ValueError("radio prediction and Jacobian must be finite.")
        return StationPrediction(time, measurement, value, jacobian, visible, elevation)

    def _radio_observable(self, trajectory, time, measurement, radio, earth_state, solar_position, solar_mu, delta, station_cache):
        transmitter = radio.transmitter or self
        receive_offset = -self._clock_offset(time, 0.0) / (1 + self.clock_drift)
        phase = measurement.endswith("doppler")
        end = self._radio_path(trajectory, time, receive_offset, radio, transmitter, earth_state, solar_position, solar_mu, delta, phase, station_cache)
        if not phase:
            clock_difference = self._clock_offset(time, receive_offset) - transmitter._clock_offset(time, end[1])
            value = .5 * (end[0] + c * clock_difference)
            return value, end[3], end[4]
        interval = radio.count_interval_s / (1 + self.clock_drift)
        start = self._radio_path(trajectory, time, receive_offset - interval, radio, transmitter, earth_state, solar_position, solar_mu, delta, phase, station_cache)
        geometric_change = sum(
            (last - first) @ (last + first) / (np.linalg.norm(last) + np.linalg.norm(first))
            for last, first in zip(end[2], start[2])
        )
        delay_change = end[5] - start[5]
        duration_error = ((transmitter.clock_drift - self.clock_drift) * interval
                          - (geometric_change / c + delay_change) * (1 + transmitter.clock_drift))
        first = start[1] + transmitter._clock_offset(time, start[1])
        last = end[1] + transmitter._clock_offset(time, end[1])
        excess = radio.uplink_phase_excess(time, first, last)
        value = ((radio.turnaround_ratio * (radio.uplink_frequency_hz * duration_error + excess)) / radio.count_interval_s
                 + (radio.uplink_frequency_hz * radio.turnaround_ratio - radio.reference_frequency))
        return value, min(end[3], start[3]), end[4] and start[4]

    def _radio_path(self, trajectory, time, receive_offset, radio, transmitter, earth_state, solar_position, solar_mu, delta, phase, station_cache):
        origin, _ = earth_state(0.0)
        def station_state(site, offset):
            key = ("state", id(site), time + offset)
            if key not in station_cache:
                station_cache[key] = site._station_rv(time + offset)
            return site._station_at_offset(time, offset, station_cache[key])
        def barycentric(state, offset):
            position, velocity = earth_state(offset)
            return state + np.r_[position - origin, velocity]

        def spacecraft(offset):
            state, stm = trajectory.at(time, offset_s=offset)
            if stm is None:
                raise ValueError("advanced radio observations require a dense trajectory with an STM.")
            return barycentric(state + stm @ delta, offset), stm

        antenna_receive = receive_offset - self.receive_delay_s
        ground_state, _ = station_state(self, antenna_receive)
        ground_state = barycentric(ground_state, antenna_receive)
        # The clock phase rate also sets the physical carrier used in plasma delay.
        carrier_clock_rate = 1 + transmitter.clock_drift
        frequency = radio.uplink_frequency_hz * carrier_clock_rate * radio.turnaround_ratio
        down_elevation = 0.0
        up_elevation = 0.0
        # Dispersive delays use the local received frequency; iterate frequency and events together.
        for _ in range(12):
            def correction(vector, state, emission, reception, station, station_offset, frequency_hz):
                key = ("vertical", id(station), time + station_offset)
                if key not in station_cache:
                    station_cache[key] = station._local_basis(time + station_offset)[2]
                vertical = station_cache[key]
                pointing = -vector if station is self and station_offset == antenna_receive else vector
                elevation = float(np.arcsin(np.clip(vertical @ pointing / np.linalg.norm(vector), -1, 1)))
                media = station.atmospheric_delay(time + station_offset, elevation, frequency_hz, phase=phase) / c
                gravity = 0.0
                for name in radio.shapiro_bodies:
                    if name == "earth":
                        center1 = earth_state(emission)[0] - origin
                        center2 = earth_state(reception)[0] - origin
                        mu = EARTH_MU
                    else:
                        center1, center2 = solar_position(emission) - origin, solar_position(reception) - origin
                        mu = solar_mu
                    receiver_position = state[:3] + vector
                    r1, r2 = np.linalg.norm(state[:3] - center1), np.linalg.norm(receiver_position - center2)
                    distance = np.linalg.norm(vector)
                    denominator = r1 + r2 - distance
                    if denominator <= 0:
                        raise ValueError("Shapiro path intersects the central singularity.")
                    gravity += 2 * mu / c**3 * np.log1p(2 * distance / denominator)
                return media + gravity, elevation

            def down_correction(vector, state, emission):
                return correction(vector, state, emission, antenna_receive, self, antenna_receive, frequency)[0]

            down_time, down, down_state, _ = _signal_leg(ground_state[:3], antenna_receive, spacecraft, down_correction)
            transmit_offset = antenna_receive - down_time
            delay = radio.transponder_delay_s
            if radio.transponder_delay_is_proper and delay:
                def rate(offset):
                    state, stm = trajectory.at(time, offset_s=offset)
                    state = state + stm @ delta
                    value = (radio.transponder_clock(time + offset, state) if radio.transponder_clock is not None
                             else 1 + 6.969290134e-10 - (EARTH_MU / np.linalg.norm(state[:3]) + np.dot(state[3:], state[3:]) / 2) / c**2)
                    if not np.isfinite(value) or not 0 < value <= 2:
                        raise ValueError("transponder proper-time rate must be finite and in (0, 2].")
                    return value
                upper = delay / rate(transmit_offset) * 2
                def residual(duration):
                    return quad(lambda elapsed: rate(transmit_offset - elapsed), 0, duration, epsabs=1e-14, epsrel=1e-12)[0] - delay
                while residual(upper) < 0:
                    upper *= 2
                    if not np.isfinite(upper):
                        raise RuntimeError("proper-time delay could not be bracketed.")
                delay = brentq(residual, 0, upper, xtol=1e-14)
            spacecraft_receive = transmit_offset - delay
            receive_state, _ = spacecraft(spacecraft_receive)

            def ground(offset):
                state, _ = station_state(transmitter, offset)
                return barycentric(state, offset), None

            def up_correction(vector, state, emission):
                clock_epoch = emission - transmitter.transmit_delay_s
                clock_epoch += transmitter._clock_offset(time, clock_epoch)
                uplink = carrier_clock_rate * radio.uplink_frequency(time, clock_epoch)
                return correction(vector, state, emission, spacecraft_receive, transmitter, emission, uplink)[0]

            up_time, up, transmit_state, _ = _signal_leg(receive_state[:3], spacecraft_receive, ground, up_correction)
            ground_transmit = spacecraft_receive - up_time
            electronics_transmit = ground_transmit - transmitter.transmit_delay_s
            uplink = carrier_clock_rate * radio.uplink_frequency(time, electronics_transmit + transmitter._clock_offset(time, electronics_transmit))
            down_unit = down / np.linalg.norm(down)
            up_unit = up / np.linalg.norm(up)
            shifted_frequency = (uplink * radio.turnaround_ratio
                                 * (c - up_unit @ receive_state[3:]) / (c - up_unit @ transmit_state[3:])
                                 * (c - down_unit @ ground_state[3:]) / (c - down_unit @ down_state[3:]))
            if abs(shifted_frequency - frequency) <= max(1e-6, abs(frequency) * 1e-13):
                break
            frequency = shifted_frequency
        else:
            raise RuntimeError("dispersive frequency/light-time solution did not converge.")
        down_delay, down_elevation = correction(down, down_state, transmit_offset, antenna_receive, self, antenna_receive, frequency)
        up_delay, up_elevation = correction(up, transmit_state, ground_transmit, spacecraft_receive, transmitter, ground_transmit, uplink)
        excess_delay = down_delay + up_delay + delay + self.receive_delay_s + transmitter.transmit_delay_s
        path_m = np.linalg.norm(up) + np.linalg.norm(down) + c * excess_delay
        return (path_m, electronics_transmit, (up, down), min(down_elevation, up_elevation),
                down_elevation >= self.min_elevation_rad and up_elevation >= transmitter.min_elevation_rad, excess_delay)


def _radio_receive_epochs(station, time, radio):
    stations = [station] if radio.reference_receiver is None else [station, radio.reference_receiver]
    return [np.nextafter(time - site._clock_offset(time, 0.) / (1 + site.clock_drift), np.inf) for site in stations]


def _earth_state(time, offset_s, kernel):
    """Earth's barycentric SI state at a two-part GPS signal epoch."""
    epoch = Time(time, offset_s, format="gps", scale="tai").tdb
    emb_r, emb_v = kernel[0, 3].compute_and_differentiate(epoch.jd1, epoch.jd2)
    earth_r, earth_v = kernel[3, 399].compute_and_differentiate(epoch.jd1, epoch.jd2)
    return (emb_r + earth_r) * 1000, (emb_v + earth_v) * (1000 / 86400)


def _signal_leg(receiver, receive_offset, emitter, correction=None):
    """Solve a leg backwards, with an optional excess propagation time [s]."""
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
        updated = distance / c + (0.0 if correction is None else correction(vector, state, receive_offset - delay))
        if not np.isfinite(updated) or updated <= 0:
            raise ValueError("signal propagation time must be finite and positive.")
        if abs(updated - delay) <= 1e-12:
            return delay, vector, state, stm
        delay = updated
    raise RuntimeError("radio light-time solution did not converge in 20 iterations.")


__all__ = ["GroundStation", "GroundStationMeasurement", "RadioTrackingConfig", "StationPrediction"]
