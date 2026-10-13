"""Analytic access geometry and direct-SGP4/independent-frame references."""

import numpy as np
import pytest
from astropy.coordinates import CartesianRepresentation, ITRS, TEME
from astropy.time import Time
import astropy.units as u
from scipy.integrate import quad
from sgp4.api import Satrec
from ssapy import Orbit, compute

from ssatk.compute.coverage import (
    access_statistics,
    coverage_grid,
    latitude_band_weights,
)
from ssatk.plots.coverage_analysis import coverage_propagator, gcrf_to_itrf


@pytest.mark.parametrize(
    "elevations, expected",
    [
        ([1, 1, 1, 1, 1], ([0.0, 120.0], 120.0, 0.0)),
        ([-1, 1, 1, -1, -1], ([15.0, 75.0], 60.0, 45.0)),
        ([-1, -1, -1, -1, -1], ([], 0.0, 120.0)),
    ],
)
def test_piecewise_linear_events_and_finite_window_budget(elevations, expected):
    """R1: literal piecewise-linear crossings; 1e-12 s/min arithmetic budget."""
    intervals, contact, gap = expected
    stats = access_statistics(elevations, np.arange(5) * 30.0)
    np.testing.assert_allclose(
        stats["intervals_seconds"].ravel(), intervals, rtol=0, atol=1e-12
    )
    assert stats["contact_seconds"] == pytest.approx(contact, abs=1e-12)
    assert stats["max_gap_min"] * 60 == pytest.approx(gap, abs=1e-12)
    assert stats["cum_contact_min_per_day"] == pytest.approx(
        contact / 120 * 1440, abs=1e-12
    )


def test_overhead_circular_pass_matches_analytic_horizon_angle():
    """R1: psi=acos(R/r*cos(epsilon))-epsilon, 0.003 s crossing budget.

    A 700-km equatorial circle is specified directly in Earth-fixed axes,
    at omega=2*pi/6000 rad/s. One-second samples interpolate its elevations;
    this test is geometric, not a claim about a real satellite trajectory.
    """
    radius = 6378137.0
    r = radius + 700e3
    omega = 2 * np.pi / 6000
    times = np.arange(6001.0)
    angle = omega * (times - 3000)
    elevation = np.degrees(
        np.arctan2(r * np.cos(angle) - radius, np.abs(r * np.sin(angle)))
    )
    stats = access_statistics(elevation, times, 5.0)
    psi = np.arccos(radius / r * np.cos(np.radians(5))) - np.radians(5)
    expected = np.array([[3000 - psi / omega, 3000 + psi / omega]])
    np.testing.assert_allclose(stats["intervals_seconds"], expected, rtol=0, atol=0.003)
    assert stats["n_passes"] == 1
    assert stats["contact_seconds"] == pytest.approx(2 * psi / omega, abs=0.006)


def test_wgs84_stationary_map_matches_ellipsoid_normals_and_exact_time_budget():
    """R1: gradient of x²/a²+y²/a²+z²/b² gives each ellipsoid normal.

    A fixed Earth-relative point tests GEO-style longitude dependence, no
    access, continuous contact, seam handling, and the exact 1440 min/day
    limit. Independent explicit WGS84 coordinates and normals classify all
    648 cells (no cells lie on the threshold).
    """
    a = 6378137.0
    f = 1 / 298.257223563
    b = a * (1 - f)
    e2 = f * (2 - f)
    point = np.array([42164000.0, 0, 0])
    result = coverage_grid(np.tile(point, (2, 1)), [0, 3 * 86400.0], n_lat=18, n_lon=36)
    lon, lat = np.meshgrid(
        np.radians(result["lon_grid"]), np.radians(result["lat_grid"])
    )
    n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    sites = np.stack(
        [
            n * np.cos(lat) * np.cos(lon),
            n * np.cos(lat) * np.sin(lon),
            n * (1 - e2) * np.sin(lat),
        ],
        axis=-1,
    )
    normal = sites / np.array([a * a, a * a, b * b])
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    rho = point - sites
    expected = np.einsum("ijk,ijk->ij", rho, normal) / np.linalg.norm(
        rho, axis=-1
    ) > np.sin(np.radians(5))
    np.testing.assert_array_equal(result["fraction"], expected)
    np.testing.assert_array_equal(result["cum_contact_min_per_day"], expected * 1440.0)
    np.testing.assert_array_equal(result["max_gap_min"], (~expected) * 4320.0)
    assert not expected[:, 0].any() and expected[:, 18].any()
    assert len(np.unique(result["lon_grid"] % 360)) == 36
    # Independent quadrature of the ellipsoid's surface-area element.
    edges = np.radians(result["lat_edges"])
    weights = np.array(
        [
            quad(
                lambda p: np.cos(p) / (1 - e2 * np.sin(p) ** 2) ** 2,
                lo,
                hi,
                epsabs=1e-13,
            )[0]
            for lo, hi in zip(edges[:-1], edges[1:])
        ]
    )
    assert result["global_mean"] == pytest.approx(
        np.sum(expected * weights[:, None]) / (weights.sum() * 36), abs=1e-13
    )


def test_latitude_area_weights_match_surface_integral():
    """R1: WGS84 surface integral, relative 1e-12; polar cells cannot weigh equally."""
    f = 1 / 298.257223563
    e2 = f * (2 - f)
    edges = np.linspace(-90, 90, 91)
    reference = np.array(
        [
            quad(
                lambda p: np.cos(p) / (1 - e2 * np.sin(p) ** 2) ** 2,
                lo,
                hi,
                epsabs=1e-13,
            )[0]
            for lo, hi in zip(np.radians(edges[:-1]), np.radians(edges[1:]))
        ]
    )
    actual = latitude_band_weights(edges)
    np.testing.assert_allclose(
        actual / actual.sum(), reference / reference.sum(), rtol=1e-12, atol=0
    )


def test_tle_access_trajectory_matches_direct_sgp4_and_astropy_itrf():
    """R2: archived HST TLE, direct python-sgp4 TEME -> Astropy ITRS, 0.1 m.

    Verifies SGP4 selection, original drag terms, units and frame convention
    over three days; it does not independently certify the TLE against flight
    tracking. The 0.1 m allowance covers the distinct TEME/GCRF frame chains.
    """
    lines = (
        "1 20580U 90037B   26187.43980126  .00006941  00000-0  21758-3 0  9997",
        "2 20580  28.4736 308.9720 0001834  18.1844 341.8817 15.30993245791525",
    )
    orbit = Orbit.fromTLETuple(lines)
    times = Time(orbit.t + np.linspace(0, 3 * 86400.0, 73), format="gps")
    error, position, _ = Satrec.twoline2rv(*lines).sgp4_array(
        times.utc.jd1, times.utc.jd2
    )
    assert not error.any()
    reference = (
        TEME(CartesianRepresentation(position.T * u.km), obstime=times)
        .transform_to(ITRS(obstime=times))
        .cartesian.xyz.to_value(u.m)
        .T
    )
    states, _ = compute.rv(orbit, times, propagator=coverage_propagator(orbit))
    actual = gcrf_to_itrf(states, times.gps)
    np.testing.assert_allclose(actual, reference, rtol=0, atol=0.1)
