import numpy as np
import pytest
from astropy.time import Time

UTC_EPOCHS = ["2008-09-20T12:25:40", "2019-03-01T06:00:00", "2026-10-08T00:00:00"]


def _astropy_gcrs_to_itrs(r_m, t):
    import astropy.units as u
    from astropy.coordinates import GCRS, ITRS, CartesianRepresentation
    from astropy.utils import iers

    from ssatk.data import data_path

    # Feed Astropy the same packaged EOP snapshot SSATK uses, so the comparison
    # measures the transform rather than two different EOP tables.
    with data_path("environment/eop/finals2000A.all") as path:
        table = iers.IERS_A.open(str(path))
    with iers.conf.set_temp("auto_download", False), iers.earth_orientation_table.set(table):
        itrs = GCRS(CartesianRepresentation(*(r_m.T * u.m)), obstime=t).transform_to(ITRS(obstime=t))
    return np.stack([itrs.x.to_value(u.m), itrs.y.to_value(u.m), itrs.z.to_value(u.m)], axis=1)


def _angle_arcsec(a, b):
    # atan2(|a x b|, a.b) keeps precision for tiny angles; arccos of the dot
    # product bottoms out near 5 mas in float64.
    cross = np.linalg.norm(np.cross(a, b), axis=1)
    return np.degrees(np.arctan2(cross, np.sum(a * b, axis=1))) * 3600.0


@pytest.mark.parametrize("utc", UTC_EPOCHS)
def test_gcrf_to_itrf_matrix_matches_astropy_gcrs_to_itrs(utc):
    # R2: eci_to_ecf_matrix against astropy GCRS->ITRS with the same packaged
    # EOP table, 500 random directions. Both are IAU 2006/2000A; the measured
    # 0.30-0.64 mas difference is the IERS celestial-pole offsets (dX, dY)
    # that SSATK applies and astropy's built-in transform omits.
    # Tolerance 1 mas.
    from ssatk.coordinates.frames import eci_to_ecf_matrix

    t = Time(utc, scale="utc")
    rng = np.random.default_rng(7)
    r = rng.normal(size=(500, 3))
    r *= 7.0e6 / np.linalg.norm(r, axis=1)[:, None]
    ours = r @ eci_to_ecf_matrix(float(t.gps)).T
    assert _angle_arcsec(ours, _astropy_gcrs_to_itrs(r, t)).max() < 1.0e-3


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
def test_groundtrack_enhanced_itrf_matches_astropy_in_any_time_scale(scale):
    """R2: Astropy GCRS->ITRS, configured with SSATK's packaged
    finals2000A.all, as the reference; tolerance 2 cm at 7000 km.

    Previously compared against ``ssapy.compute.groundTrack`` (IAU
    1976/1980). ``groundtrack_enhanced.gcrf_to_itrf`` shares SSATK's single
    IAU 2006/2000A route, so that comparison measured a ~1.7 m model
    difference, not a bug; feeding Astropy the same packaged EOP data gives
    1.07 cm agreement (same across all four time scales below). The
    parametrization over time scales guards a past regression: a scalar
    ``Time`` passed in a non-UTC scale used to be rebuilt from its ISO string
    and silently reinterpreted as UTC, moving points by 32.2 km in TT.
    """
    from astropy.utils import iers

    from ssatk.data import data_path
    from ssatk.plots.groundtrack_enhanced import gcrf_to_itrf

    with data_path("environment/eop/finals2000A.all") as path:
        table = iers.IERS_A.open(str(path))

    t = Time("2026-10-09T12:00:00", scale="utc")
    rng = np.random.default_rng(3)
    r_km = rng.normal(size=(50, 3))
    r_km *= 7000.0 / np.linalg.norm(r_km, axis=1)[:, None]
    ours_km = gcrf_to_itrf(r_km, getattr(t, scale))

    with iers.conf.set_temp("auto_download", False), iers.earth_orientation_table.set(table):
        import astropy.units as u
        from astropy.coordinates import GCRS, ITRS, CartesianRepresentation

        itrs = GCRS(CartesianRepresentation(r_km.T * u.km), obstime=t).transform_to(ITRS(obstime=t))
        reference_km = itrs.cartesian.xyz.to_value(u.km).T

    np.testing.assert_allclose(ours_km, reference_km, rtol=0.0, atol=2.0e-5)


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
    np.testing.assert_allclose(lat, lat0, atol=1e-9, rtol=0)
    np.testing.assert_allclose(((lon - lon0 + 180.0) % 360.0) - 180.0, 0.0, atol=1e-9, rtol=0)
    np.testing.assert_allclose(h, h0, atol=1e-6, rtol=0)
