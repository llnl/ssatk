
import numpy as np

from ssatk.constants import EARTH_MU
from ssatk.coordinates import equatorial_ecliptic
from ssatk.coordinates.cartesian import cart2sph_deg, cart_to_cyl
from ssatk.coordinates.satellite_frames import gcrf_to_ntw, ntw_to_gcrf, ntw_to_gcrf_matrix
from ssatk.coordinates.angle_units import (
    deg0to360,
    deg0to360array,
    deg90to90,
    deg90to90array,
    dms_to_deg,
    dms_to_rad,
    rad0to2pi,
)
from ssatk.orbital_mechanics import misc

eqecl = equatorial_ecliptic


def test_orbital_misc_formula_helpers():
    mu = EARTH_MU
    r = 7000e3
    v_circ = np.sqrt(mu / r)
    assert np.isclose(misc.escape_velocity(mu, r), np.sqrt(2) * v_circ)
    assert np.isclose(misc.circular_velocity(mu, r), v_circ)
    assert np.isclose(misc.vis_viva(mu, r, r), v_circ)
    assert np.isclose(misc.specific_orbital_energy(mu, r, v_circ), -mu / (2 * r))
    np.testing.assert_allclose(misc.specific_angular_momentum([r, 0, 0], [0, v_circ, 0]), [0, 0, r * v_circ])
    np.testing.assert_allclose(misc.eccentricity_vector(np.array([r, 0, 0]), np.array([0, v_circ, 0]), mu), [0, 0, 0], atol=1e-12)

    a, e, inc, raan, argp, nu, M = misc.orbital_elements_from_state(np.array([r, 0, 0]), np.array([0, v_circ, 0]), mu)
    assert np.isclose(a, r)
    assert e < 1e-12
    assert inc == 0.0
    assert raan == argp == nu == 0.0
    assert M is not None

    E = misc.kepler_E_from_M(0.5, 0.1)
    assert np.isclose(E - 0.1 * np.sin(E), 0.5)
    assert np.isclose(misc.kepler_E_from_M_from_nu(0.0, 0.1), 0.0)
    assert np.isclose(misc.orbital_period(r, mu), 2 * np.pi * np.sqrt(r**3 / mu))
    assert misc.orbital_period(-1.0, mu) is None
    dv1, dv2, total = misc.hohmann_transfer_delta_v(r, 2 * r, mu)
    assert np.isclose(total, dv1 + dv2)
    vals = misc.bi_elliptic_transfer_delta_v(r, 2 * r, 4 * r, mu)
    assert len(vals) == 4
    assert np.isclose(vals[-1], sum(vals[:3]))
    assert np.isclose(misc.plane_change_delta_v(10.0, 0.0, np.pi / 3), 10.0)
    assert np.isclose(misc.sphere_of_influence_radius(1.0, 1.0, 32.0), 1.0 * (1.0 / 32.0) ** (2.0 / 5.0))


def test_coordinate_conversion_helpers():
    radius, theta, z = cart_to_cyl(3.0, 4.0, 5.0)
    assert radius == 5.0
    assert np.isclose(theta, np.arctan2(4.0, 3.0))
    assert z == 5.0
    az, el, r = cart2sph_deg(0.0, 1.0, 1.0)
    assert np.isclose(az, 90.0)
    assert np.isclose(el, 45.0)
    assert np.isclose(r, np.sqrt(2.0))

    assert np.isclose(dms_to_rad("180d"), np.pi)
    assert dms_to_deg(["0d", "90d"]) == [0.0, 90.0]
    np.testing.assert_allclose(rad0to2pi([-np.pi, 3 * np.pi]), [np.pi, np.pi])
    assert deg0to360(-90) == 270.0
    assert deg0to360array([-1, 360]) == [359.0, 0.0]
    assert deg90to90(100) == -80.0
    assert deg90to90array([100, -100]) == [-80.0, 80.0]

    r_vec = np.array([1.0, 0.0, 0.0])
    v_vec = np.array([0.0, 1.0, 0.0])
    matrix = ntw_to_gcrf_matrix(r_vec, v_vec)
    np.testing.assert_allclose(matrix, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(ntw_to_gcrf([1, 2, 3], r_vec, v_vec), [1, 2, 3], atol=1e-12)
    np.testing.assert_allclose(gcrf_to_ntw([1, 2, 3], r_vec, v_vec), [1, 2, 3], atol=1e-12)

    assert equatorial_ecliptic.equatorial_to_ecliptic is eqecl.equatorial_to_ecliptic

    xq, yq, zq = eqecl.ecliptic_xyz_to_equatorial_xyz(1.0, 2.0, 3.0)
    xc, yc, zc = eqecl.equatorial_xyz_to_ecliptic_xyz(xq, yq, zq)
    np.testing.assert_allclose([xc, yc, zc], [1.0, 2.0, 3.0])
    lon, lat = eqecl.xyz_to_ecliptic(1.0, 0.0, 0.0, degrees=True)
    assert np.isclose(lon, 0.0)
    assert np.isclose(lat, 0.0)
    ra, dec = eqecl.xyz_to_equatorial(1.0, 0.0, 0.0, degrees=True)
    assert np.isclose(ra, 0.0)
    assert np.isclose(dec, 0.0)
    ra2, dec2 = eqecl.ecliptic_to_equatorial(*eqecl.equatorial_to_ecliptic(ra, dec, degrees=True), degrees=True)
    assert np.isfinite(ra2)
    assert np.isfinite(dec2)
