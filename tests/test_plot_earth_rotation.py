import numpy as np
import pytest
from astropy.time import Time

UTC_EPOCHS = ["2008-09-20T12:25:40", "2019-03-01T06:00:00", "2026-10-08T00:00:00"]


def _astropy_gcrs_to_itrs(r_m, t):
    import astropy.units as u
    from astropy.coordinates import GCRS, ITRS, CartesianRepresentation

    itrs = GCRS(CartesianRepresentation(*(r_m.T * u.m)), obstime=t).transform_to(ITRS(obstime=t))
    return np.stack([itrs.x.to_value(u.m), itrs.y.to_value(u.m), itrs.z.to_value(u.m)], axis=1)


def _angle_arcsec(a, b):
    cos = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))) * 3600.0


@pytest.mark.parametrize("utc", UTC_EPOCHS)
def test_gcrf_to_itrf_matrix_matches_astropy_gcrs_to_itrs(utc):
    # R2: eci_to_ecf_matrix (IAU 1976/1980 + GAST94, as SSAPy's groundTrack)
    # against astropy GCRS->ITRS (IAU 2006/2000A), 500 random directions.
    # Measured model difference: 0.021" (2000) to 0.050" (2026).
    # Tolerance 0.5 arcsec. The former GMST-only matrix was 1354" off in 2026.
    from ssatk.coordinates.frames import eci_to_ecf_matrix

    t = Time(utc, scale="utc")
    rng = np.random.default_rng(7)
    r = rng.normal(size=(500, 3))
    r *= 7.0e6 / np.linalg.norm(r, axis=1)[:, None]
    ours = r @ eci_to_ecf_matrix(float(t.gps)).T
    assert _angle_arcsec(ours, _astropy_gcrs_to_itrs(r, t)).max() < 0.5


@pytest.mark.parametrize("utc", UTC_EPOCHS)
def test_earth_texture_rotation_is_gcrf_right_ascension_of_greenwich(utc):
    # R2: the texture angle must be the GCRF right ascension of the ITRF
    # x-axis, taken from astropy ITRS->GCRS; tolerance 0.5 arcsec. The former
    # angle (GAST) is measured from the equinox of date and was 1242" off in
    # 2026 (38.4 km at the equator).
    import astropy.units as u
    from astropy.coordinates import GCRS, ITRS, CartesianRepresentation

    from ssatk.plots.scene_primitives import earth_rotation_deg_from_time

    t = Time(utc, scale="utc")
    greenwich = ITRS(CartesianRepresentation(1.0, 0.0, 0.0, unit=u.m), obstime=t).transform_to(
        GCRS(obstime=t)
    )
    expected = np.degrees(np.arctan2(greenwich.cartesian.y.value, greenwich.cartesian.x.value))
    angle = earth_rotation_deg_from_time(float(t.gps))
    assert abs(((angle - expected + 180.0) % 360.0) - 180.0) * 3600.0 < 0.5


@pytest.mark.parametrize("scale", ["utc", "tt", "tdb", "tai"])
def test_groundtrack_enhanced_itrf_matches_ssapy_ground_track_in_any_time_scale(scale):
    # R2: SSAPy groundTrack (cartesian ITRF) is the reference; tolerance 1 m
    # at 7000 km. The same instant passed in TT used to move points 32.2 km,
    # because a scalar Time was rebuilt from its ISO string as UTC.
    from ssapy.compute import groundTrack

    from ssatk.plots.groundtrack_enhanced import gcrf_to_itrf

    t = Time("2026-10-09T12:00:00", scale="utc")
    rng = np.random.default_rng(3)
    r_km = rng.normal(size=(50, 3))
    r_km *= 7000.0 / np.linalg.norm(r_km, axis=1)[:, None]
    ours_km = gcrf_to_itrf(r_km, getattr(t, scale))
    # One trajectory (N, 3) needs N times; with a scalar time groundTrack
    # silently returns only the first row.
    x, y, z = groundTrack(r_km * 1000.0, np.full(len(r_km), float(t.gps)), format="cartesian")
    reference_km = np.column_stack([np.ravel(x), np.ravel(y), np.ravel(z)]) / 1000.0
    np.testing.assert_allclose(ours_km, reference_km, rtol=0.0, atol=1.0e-3)


def test_groundtrack_enhanced_geodetic_matches_astropy_wgs84():
    # R2: astropy EarthLocation WGS84 geodetic conversion; tolerance 1e-9 deg
    # and 1 mm. The former spherical conversion was 0.19 deg off at 45 deg.
    import astropy.units as u
    from astropy.coordinates import EarthLocation

    from ssatk.plots.groundtrack_enhanced import ecef_to_geodetic

    lat0 = np.array([-89.0, -45.0, 0.0, 30.363, 45.0, 71.0])
    lon0 = np.array([-170.0, 10.0, 0.0, -97.979, 120.0, 145.0])
    h0 = np.array([0.0, 400.0, 35_786.0, 0.2, 550.0, 20_200.0])
    loc = EarthLocation.from_geodetic(lon0 * u.deg, lat0 * u.deg, h0 * u.km, ellipsoid="WGS84")
    r_km = np.column_stack([c.to_value(u.km) for c in loc.to_geocentric()])
    lat, lon, h = ecef_to_geodetic(r_km)
    np.testing.assert_allclose(lat, lat0, atol=1e-9)
    np.testing.assert_allclose(((lon - lon0 + 180.0) % 360.0) - 180.0, 0.0, atol=1e-9)
    np.testing.assert_allclose(h, h0, atol=1e-6)
