import numpy as np
from astropy.time import Time

from ssatk.coordinates.inertial import j2000_to_gcrf
from ssatk.coordinates.lunar import get_lunar_rv

MAS = np.radians(1.0 / 3.6e6)


def _rot(axis, angle):
    c, s = np.cos(angle), np.sin(angle)
    if axis == 1:
        return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])
    if axis == 2:
        return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def test_j2000_to_gcrf_is_the_iers_frame_bias():
    # R3: IERS Conventions 2010 eqs. 5.20-5.21, B = R1(-eta0) R2(xi0) R3(da0)
    # with xi0 = -16.6170 mas, eta0 = -6.8192 mas, da0 = -14.6 mas, and
    # r_GCRF = B^T r_J2000. Agreement to 1e-11 relative (0.4 mm at GEO), at
    # any epoch: the 23 mas rotation is time-independent.
    bias = _rot(1, 6.8192 * MAS) @ _rot(2, -16.6170 * MAS) @ _rot(3, -14.6 * MAS)
    r_j2000 = np.array([[7000e3, 0.0, 0.0], [0.0, 42164e3, 0.0], [1e6, 2e6, 6e6]])
    expected = (bias.T @ r_j2000.T).T
    for epoch in (None, "2000-01-01T12:00:00", 1.4e9):
        np.testing.assert_allclose(j2000_to_gcrf(r_j2000, epoch), expected, rtol=0, atol=1e-11 * 42164e3)


def test_lunar_velocity_matches_astropy_for_sparse_epochs():
    # R2: astropy get_body_barycentric_posvel (built-in ephemeris), Moon minus
    # Earth. Daily epochs must give instantaneous velocities, within 0.1 m/s
    # (ephemeris differences are 0.01 m/s); positions agree within 10 km.
    import astropy.units as u
    from astropy.coordinates import get_body_barycentric_posvel

    times = Time(["2026-10-08T00:00:00", "2026-10-09T00:00:00", "2026-10-10T00:00:00"], scale="utc")
    r, v = get_lunar_rv(times.gps)
    moon_r, moon_v = get_body_barycentric_posvel("moon", times)
    earth_r, earth_v = get_body_barycentric_posvel("earth", times)
    np.testing.assert_allclose(v, (moon_v - earth_v).xyz.to_value(u.m / u.s).T, rtol=0, atol=0.1)
    np.testing.assert_allclose(r, (moon_r - earth_r).xyz.to_value(u.m).T, rtol=0, atol=10e3)
