"""Angular speed remains accurate when radial speed dominates."""

import numpy as np
import pytest

from ssatk.compute.proper_motions import proper_motion, proper_motion_ra_dec

# R3: 1 rad = 648000/pi arcsec; REBOUND's G = 1 time unit in au and Msun is
# yr2pi = 1/k day with the Gaussian constant k = 0.01720209895.
ARCSEC_PER_RAD = 648000.0 / np.pi
REBOUND_TIME_UNIT_S = 86400.0 / 0.01720209895
AU_M = 149_597_870_700.0


@pytest.mark.parametrize("radial_speed", [-7500.0, 7500.0])
@pytest.mark.parametrize("transverse_speed", [1e-6, 1e-4, 1.0])
def test_nearly_radial_motion_retains_transverse_component(radial_speed, transverse_speed):
    distance = 7.0e6
    actual = proper_motion(distance, 0.0, 0.0, radial_speed, transverse_speed, 0.0)
    expected = transverse_speed / distance * ARCSEC_PER_RAD
    assert actual == pytest.approx(expected, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("input_unit, divisor", [("si", 1.0), ("rebound", REBOUND_TIME_UNIT_S)])
def test_radial_motion_with_observer_offsets_and_units(input_unit, divisor):
    actual = proper_motion(
        7.0e6 + 100.0, 200.0, 300.0, 7500.0 + 40.0, 1e-4 - 50.0, 60.0,
        xe=100.0, ye=200.0, ze=300.0, vxe=40.0, vye=-50.0, vze=60.0,
        input_unit=input_unit,
    )
    # Account only for rounding already present in the supplied velocity.
    transverse = (1e-4 - 50.0) - (-50.0)
    assert actual == pytest.approx(transverse / 7.0e6 * ARCSEC_PER_RAD / divisor, rel=1e-12, abs=0.0)


def test_general_motion_matches_independent_tangent_components():
    rng = np.random.default_rng(21)
    for _ in range(100):
        ra, dec = rng.uniform(-np.pi, np.pi), rng.uniform(-1.4, 1.4)
        radial = np.array([np.cos(ra) * np.cos(dec), np.sin(ra) * np.cos(dec), np.sin(dec)])
        east = np.array([-np.sin(ra), np.cos(ra), 0.0])
        north = np.cross(radial, east)
        vr, ve, vn = rng.normal(size=3) * 1000.0
        distance = rng.uniform(7.0e6, 4.2e7)
        velocity = vr * radial + ve * east + vn * north
        actual = proper_motion(*(distance * radial), *velocity)
        assert actual == pytest.approx(np.hypot(ve, vn) / distance * ARCSEC_PER_RAD, rel=1e-12, abs=0.0)


def test_ra_dec_rates_match_astropy_in_si_and_rebound_units():
    # R2: astropy SphericalCosLatDifferential of the same Cartesian state gives
    # mu_alpha cos(dec) and mu_dec. The SI and REBOUND-unit inputs describe the
    # same motion, so both must match astropy in arcsec/s to 1e-9 relative.
    import astropy.units as u
    from astropy.coordinates import (
        CartesianDifferential,
        CartesianRepresentation,
        SphericalCosLatDifferential,
        SphericalRepresentation,
    )

    r = np.array([[1.2, -0.4, 0.3], [-2.5, 0.8, -1.1], [0.1, 0.2, 3.0]]) * AU_M
    v = np.array([[5.0e3, 12.0e3, -3.0e3], [-8.0e3, 2.0e3, 6.0e3], [1.0e3, -4.0e3, 0.5e3]])

    representation = CartesianRepresentation(r.T * u.m, differentials=CartesianDifferential(v.T * u.m / u.s))
    spherical = representation.represent_as(SphericalRepresentation, SphericalCosLatDifferential)
    rates = spherical.differentials["s"]
    expected_ra = rates.d_lon_coslat.to_value(u.arcsec / u.s, equivalencies=u.dimensionless_angles())
    expected_dec = rates.d_lat.to_value(u.arcsec / u.s)

    pmra, pmdec = proper_motion_ra_dec(r=r, v=v)
    np.testing.assert_allclose(pmra, expected_ra, rtol=1e-9, atol=0)
    np.testing.assert_allclose(pmdec, expected_dec, rtol=1e-9, atol=0)

    pmra_rb, pmdec_rb = proper_motion_ra_dec(r=r / AU_M, v=v * REBOUND_TIME_UNIT_S / AU_M, input_unit="rebound")
    np.testing.assert_allclose(pmra_rb, expected_ra, rtol=1e-9, atol=0)
    np.testing.assert_allclose(pmdec_rb, expected_dec, rtol=1e-9, atol=0)

    total_rb = [
        proper_motion(*(ri / AU_M), *(vi * REBOUND_TIME_UNIT_S / AU_M), input_unit="rebound")
        for ri, vi in zip(r, v)
    ]
    np.testing.assert_allclose(total_rb, np.hypot(expected_ra, expected_dec), rtol=1e-9, atol=0)
