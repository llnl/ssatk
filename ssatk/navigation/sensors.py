"""Deterministic, testable navigation sensor primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .measurements import GroundStation

_SCALAR_MEASUREMENTS = frozenset({"range", "range_rate"})
_VECTOR_MEASUREMENTS = frozenset({"az_el", "ra_dec"})
_MEASUREMENTS = _SCALAR_MEASUREMENTS | _VECTOR_MEASUREMENTS


def _readonly(value, shape):
    array = np.array(value, dtype=float, copy=True)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"value must be finite and have shape {shape}.")
    array.flags.writeable = False
    return array


def _covariance(value, shape):
    covariance = _readonly(value, shape)
    if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-14):
        raise ValueError("covariance must be symmetric.")
    if np.any(np.linalg.eigvalsh(covariance) < -1e-14):
        raise ValueError("covariance must be positive semidefinite.")
    return covariance


@dataclass(frozen=True)
class SensorMeasurement:
    """One sensor sample, including validity and reported covariance."""

    time: float
    value: np.ndarray | None
    covariance: np.ndarray
    valid: bool = True


class CartesianSensor:
    """Cartesian sensor with fixed bias, Gaussian noise, and dropouts."""

    def __init__(
        self,
        covariance,
        *,
        bias=None,
        rng=None,
        dropout_probability: float = 0.0,
        valid_interval: tuple[float, float] | None = None,
    ):
        covariance = np.asarray(covariance, dtype=float)
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            raise ValueError("covariance must be square.")
        if not np.all(np.isfinite(covariance)) or np.any(np.linalg.eigvalsh(covariance) < 0.0):
            raise ValueError("covariance must be finite and positive semidefinite.")
        self.covariance = covariance
        self.bias = np.zeros(covariance.shape[0]) if bias is None else np.asarray(bias, dtype=float)
        if self.bias.shape != (covariance.shape[0],) or not np.all(np.isfinite(self.bias)):
            raise ValueError("bias must match covariance dimension and be finite.")
        if not 0.0 <= float(dropout_probability) <= 1.0:
            raise ValueError("dropout_probability must be between 0 and 1.")
        if valid_interval is not None and valid_interval[1] < valid_interval[0]:
            raise ValueError("valid_interval stop must be >= start.")
        self.rng = np.random.default_rng() if rng is None else rng
        self.dropout_probability = float(dropout_probability)
        self.valid_interval = valid_interval

    def measure(self, truth, time: float) -> SensorMeasurement:
        truth = np.asarray(truth, dtype=float)
        if truth.shape != self.bias.shape or not np.all(np.isfinite(truth)):
            raise ValueError("truth must match the sensor dimension and be finite.")
        time = float(time)
        in_interval = self.valid_interval is None or self.valid_interval[0] <= time <= self.valid_interval[1]
        valid = in_interval and float(self.rng.random()) >= self.dropout_probability
        if not valid:
            return SensorMeasurement(time, None, self.covariance.copy(), False)
        noise = self.rng.multivariate_normal(np.zeros(truth.size), self.covariance)
        return SensorMeasurement(time, truth + self.bias + noise, self.covariance.copy(), True)


@dataclass(frozen=True)
class StationObservation:
    """One noisy ground-station observation and its quality metadata."""

    time: float
    measurement: str
    value: float | np.ndarray | None
    covariance: np.ndarray
    bias: float | np.ndarray
    elevation_rad: float
    visible: bool
    valid: bool = True
    prediction: object | None = None

    def __post_init__(self):
        if self.measurement not in _MEASUREMENTS:
            raise ValueError(f"unsupported measurement {self.measurement!r}.")
        dimension = 1 if self.measurement in _SCALAR_MEASUREMENTS else 2
        if not np.isfinite(float(self.time)) or not np.isfinite(float(self.elevation_rad)):
            raise ValueError("time and elevation_rad must be finite.")
        covariance = _covariance(self.covariance, (dimension, dimension))
        if self.value is None:
            value = None
        elif dimension == 1 and np.isscalar(self.value):
            value = float(self.value)
            if not np.isfinite(value):
                raise ValueError("value must be finite.")
        else:
            value = _readonly(self.value, (dimension,))
        if dimension == 1 and np.isscalar(self.bias):
            bias = float(self.bias)
        else:
            bias = _readonly(self.bias, (dimension,))
        object.__setattr__(self, "time", float(self.time))
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "bias", bias)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "elevation_rad", float(self.elevation_rad))

    @property
    def jacobian(self):
        """State Jacobian from the deterministic prediction, if retained."""

        return None if self.prediction is None else self.prediction.jacobian

    @property
    def kind(self) -> str:
        """Alias for the measurement type."""

        return self.measurement

    def as_measurement(self) -> tuple[np.ndarray, np.ndarray]:
        """Return a bias-corrected EKF measurement and immutable covariance."""

        if not self.valid or self.value is None:
            raise ValueError("invalid station observations cannot update an EKF.")
        measurement = np.atleast_1d(np.asarray(self.value, dtype=float) - self.bias).copy()
        measurement.flags.writeable = False
        covariance = np.array(self.covariance, dtype=float, copy=True)
        covariance.flags.writeable = False
        return measurement, covariance


class GroundStationSensor:
    """Noisy, masked sensor wrapper around :class:`GroundStation`."""

    def __init__(
        self,
        station: GroundStation,
        covariance,
        *,
        measurement: str = "range",
        kind: str | None = None,
        bias=None,
        rng=None,
        dropout_probability: float = 0.0,
        min_elevation_rad: float | None = None,
        visibility_override: bool = False,
    ):
        if not isinstance(station, GroundStation):
            raise TypeError("station must be a GroundStation.")
        if kind is not None:
            if measurement != "range":
                raise ValueError("specify only one of measurement and kind.")
            measurement = kind
        if measurement not in _MEASUREMENTS:
            raise ValueError(f"measurement must be one of {sorted(_MEASUREMENTS)}.")
        dimension = 1 if measurement in _SCALAR_MEASUREMENTS else 2
        covariance = np.asarray(covariance, dtype=float)
        if dimension == 1 and covariance.ndim == 0:
            covariance = covariance.reshape(1, 1)
        covariance = _covariance(covariance, (dimension, dimension))
        if bias is None:
            bias = 0.0 if dimension == 1 else np.zeros(dimension)
        elif dimension == 1 and not np.isscalar(bias):
            raise ValueError("scalar measurement bias must be scalar.")
        elif dimension == 2:
            bias = _readonly(bias, (dimension,))
        if dimension == 1:
            bias = float(bias)
            if not np.isfinite(bias):
                raise ValueError("bias must be finite.")
        if not 0.0 <= float(dropout_probability) <= 1.0:
            raise ValueError("dropout_probability must be between 0 and 1.")
        if min_elevation_rad is not None:
            min_elevation_rad = float(min_elevation_rad)
            if not np.isfinite(min_elevation_rad) or not -np.pi / 2 <= min_elevation_rad <= np.pi / 2:
                raise ValueError("min_elevation_rad must be finite and in [-pi/2, pi/2].")
        if rng is not None and not isinstance(rng, (np.random.Generator, int, np.integer)):
            raise TypeError("rng must be a numpy Generator or integer seed.")
        self.station = station
        self.measurement = measurement
        self.covariance = covariance
        self.bias = bias
        self.rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        self.dropout_probability = float(dropout_probability)
        self.min_elevation_rad = min_elevation_rad
        self.visibility_override = bool(visibility_override)

    @property
    def kind(self) -> str:
        """Alias for the measurement type."""

        return self.measurement

    def measure(self, truth, time) -> StationObservation:
        """Return one noisy observation, or an invalid masked/dropout sample."""

        prediction = self.station.predict(truth, time, self.measurement)
        threshold = self.station.min_elevation_rad if self.min_elevation_rad is None else self.min_elevation_rad
        visible = prediction.elevation_rad >= threshold
        eligible = visible or self.visibility_override
        valid = eligible and float(self.rng.random()) >= self.dropout_probability
        if valid:
            dimension = 1 if self.measurement in _SCALAR_MEASUREMENTS else 2
            noise = self.rng.multivariate_normal(np.zeros(dimension), self.covariance)
            if dimension == 1:
                value = float(prediction.value) + float(self.bias) + float(noise[0])
            else:
                value = np.asarray(prediction.value) + self.bias + noise
                if self.measurement in {"az_el", "ra_dec"}:
                    value[0] = np.mod(value[0], 2.0 * np.pi)
        else:
            value = None
        return StationObservation(
            prediction.time,
            self.measurement,
            value,
            self.covariance,
            self.bias,
            prediction.elevation_rad,
            visible,
            valid,
            prediction,
        )


__all__ = ["CartesianSensor", "GroundStationSensor", "SensorMeasurement", "StationObservation"]
