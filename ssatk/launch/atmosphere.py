"""Bounded launch weather and a thermodynamically consistent table handoff."""

import numpy as np
from scipy.interpolate import interp1d

from ..coordinates._earth_rotation import gcrs_to_itrs_matrix_and_rate
from ..coordinates.satellite_frames import enu_to_ecef_matrix
from ..environment import _itrf_to_geodetic
from ..time_functions import to_gps

__all__ = ["LaunchWeatherProfile", "LaunchAtmosphere"]


class LaunchWeatherProfile:
    """Piecewise-linear pressure, temperature and ENU wind versus WGS84 height.

    Inputs are metres, Pa, K and m/s. Wind components are velocities *toward*
    east, north and up, not meteorological directions *from*. The profile is
    horizontally homogeneous only within its declared latitude/longitude box
    (degrees, without wrapping across the date line) and GPS time interval.
    ``source`` identifies the sounding, forecast or sampled upper model.

    One ideal gas has specific gas constant ``gas_constant`` (J/kg/K), heat
    capacity ratio ``gamma``, and Sutherland parameters ``(mu_ref, T_ref, S)``
    in Pa s, K and K. Density is p/(R T), sound speed sqrt(gamma R T), and
    viscosity mu_ref (T/T_ref)**1.5 (T_ref+S)/(T+S). These are declared model
    assumptions, not a reconciliation of independently measured densities.
    """

    def __init__(self, heights, pressure, temperature, wind_enu, *, source,
                 valid_time_gps, latitude_range, longitude_range,
                 gas_constant, gamma, sutherland):
        heights = np.array(heights, dtype=float, copy=True)
        pressure = np.asarray(pressure, dtype=float)
        temperature = np.asarray(temperature, dtype=float)
        wind = np.asarray(wind_enu, dtype=float)
        if (heights.ndim != 1 or heights.size < 2 or not np.isfinite(heights).all()
                or np.any(np.diff(heights) <= 0)):
            raise ValueError("heights must contain at least two finite increasing samples")
        for name, values in (("pressure", pressure), ("temperature", temperature)):
            if values.shape != heights.shape or not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError(f"{name} must contain one positive finite value per height")
        if wind.shape != (heights.size, 3) or not np.isfinite(wind).all():
            raise ValueError("wind_enu must have finite shape (number of heights, 3)")
        gas = np.asarray([gas_constant, gamma, *sutherland], dtype=float)
        if gas.shape != (5,) or not np.isfinite(gas).all() or np.any(gas <= 0) or gas[1] <= 1:
            raise ValueError("gas and Sutherland parameters must be positive finite; gamma must exceed one")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must identify the weather/model provenance")
        self.source = source
        self.gas = gas
        self.heights = heights
        self.bounds = {}
        for name, pair in (("time_gps", valid_time_gps), ("latitude", latitude_range),
                           ("longitude", longitude_range)):
            pair = np.array(pair, dtype=float, copy=True)
            if pair.shape != (2,) or not np.isfinite(pair).all() or pair[0] > pair[1]:
                raise ValueError(f"{name} bounds must be a finite ordered pair")
            self.bounds[name] = pair
        if (np.any(np.abs(self.bounds["latitude"]) > 90)
                or np.any(np.abs(self.bounds["longitude"]) > 180)):
            raise ValueError("latitude/longitude bounds must lie within +/-90 and +/-180 degrees")
        self._profile = interp1d(heights, np.column_stack((pressure, temperature, wind)),
                                axis=0, bounds_error=True)

    def primitives(self, height, time_gps, latitude, longitude):
        """Return [pressure, temperature, east/north/up wind] within coverage."""
        for name, value in (("time_gps", time_gps), ("latitude", latitude), ("longitude", longitude)):
            low, high = self.bounds[name]
            if not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} query {value} is outside [{low}, {high}]")
        if not np.isfinite(height):
            raise ValueError("height must be finite")
        return np.asarray(self._profile(height))


class LaunchAtmosphere:
    """Weather in GCRF, using the shared strict packaged-EOP rotation and rate.

    ``upper`` is a bounded profile sampled from the selected upper atmosphere.
    ``transition`` is an overlapping WGS84 height interval (m). Across it,
    linearly blend p, T and ENU wind, then derive rho, a and mu from the common
    gas model. The two profiles must use identical gas parameters; changing
    composition requires another explicitly derived thermophysical model.
    No upper profile means no transition or extrapolation beyond ``lower``.
    """

    def __init__(self, lower, *, upper=None, transition=None, eop=None, allow_predicted=False):
        if (upper is None) != (transition is None):
            raise ValueError("upper and transition must be supplied together")
        if upper is not None:
            transition = np.asarray(transition, dtype=float)
            if (transition.shape != (2,) or not np.isfinite(transition).all()
                    or transition[0] >= transition[1]):
                raise ValueError("transition must contain two finite increasing heights")
            for profile in (lower, upper):
                if transition[0] < profile.heights[0] or transition[1] > profile.heights[-1]:
                    raise ValueError("both profiles must cover the entire transition")
            if not np.array_equal(lower.gas, upper.gas):
                raise ValueError("transition profiles must use the same gas model")
        self.lower, self.upper, self.transition = lower, upper, transition
        self.eop, self.allow_predicted = eop, allow_predicted

    def properties(self, height, time_gps, latitude, longitude):
        """Sample thermophysical properties without performing a frame transform."""
        args = (height, time_gps, latitude, longitude)
        weight = 0.0
        if self.upper is None or height <= self.transition[0]:
            values = self.lower.primitives(*args)
            sources = [self.lower.source]
        elif height >= self.transition[1]:
            values = self.upper.primitives(*args)
            sources = [self.upper.source]
            weight = 1.0
        else:
            weight = (height - self.transition[0]) / np.diff(self.transition)[0]
            values = (1 - weight)*self.lower.primitives(*args) + weight*self.upper.primitives(*args)
            sources = [self.lower.source, self.upper.source]
        pressure, temperature = values[:2]
        gas_constant, gamma, mu_ref, temp_ref, sutherland = self.lower.gas
        result = {"height_m": float(height), "pressure_pa": float(pressure),
                "temperature_k": float(temperature), "density_kg_m3": float(pressure/(gas_constant*temperature)),
                "sound_speed_m_s": float(np.sqrt(gamma*gas_constant*temperature)),
                "dynamic_viscosity_pa_s": float(mu_ref*(temperature/temp_ref)**1.5
                                                *(temp_ref+sutherland)/(temperature+sutherland)),
                "wind_enu_m_s": values[2:], "sources": sources, "upper_weight": float(weight)}
        if not all(np.isfinite(result[key]) and result[key] > 0 for key in
                   ("pressure_pa", "temperature_k", "density_kg_m3", "sound_speed_m_s", "dynamic_viscosity_pa_s")):
            raise ValueError("gas model produced nonpositive or nonfinite properties")
        return result

    def sample(self, time, r_gcrf):
        """Return properties and total atmosphere velocity in GCRF m/s.

        Time is absolute GPS seconds or Astropy Time. Rotation is counted
        exactly once, via v_gcrf = R.T (wind_itrf - dR/dt r_gcrf).
        """
        gps = float(to_gps(time))
        r = np.asarray(r_gcrf, dtype=float)
        if r.shape != (3,) or not np.isfinite(r).all() or np.linalg.norm(r) == 0:
            raise ValueError("r_gcrf must be a finite nonzero 3-vector")
        rotation, rate = gcrs_to_itrs_matrix_and_rate(
            [gps], eop=self.eop, allow_predicted=self.allow_predicted)
        rotation, rate = rotation[0], rate[0]
        longitude, latitude, height = _itrf_to_geodetic(rotation @ r)
        result = self.properties(height, gps, latitude, longitude)
        wind_itrf = enu_to_ecef_matrix(latitude, longitude) @ result["wind_enu_m_s"]
        result["atmosphere_velocity_gcrf_m_s"] = rotation.T @ (wind_itrf - rate @ r)
        result.update(time_gps_s=gps, latitude_deg=latitude, longitude_deg=longitude)
        return result
