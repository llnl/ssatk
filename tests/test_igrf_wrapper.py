import numpy as np
import pytest

pytest.importorskip("ppigrf")


def test_igrf_magnetic_field_in_gcrf_matches_ppigrf_and_ignores_external_models():
    # R2: ppigrf.igrf at the geodetic point (east, north, up nT), rotated to
    # ITRS with the geodetic normal and to GCRS with astropy, agrees with the
    # GCRF wrapper to 0.1 % (ppigrf evaluates at the geodetic location; the
    # wrapper goes through astropy's frame chain). A module-level external
    # model must not change the result.
    import astropy.units as u
    import ppigrf
    from astropy.coordinates import GCRS, ITRS, CartesianRepresentation, EarthLocation
    from astropy.time import Time

    from ssatk import geomagnetics
    from ssatk.environment import igrf_magnetic_field

    t = Time("2026-10-08T00:00:00", scale="utc")
    lat, lon, height_km = 37.7, -121.7, 500.0
    location = EarthLocation.from_geodetic(lon * u.deg, lat * u.deg, height_km * u.km)
    r_gcrf = location.get_gcrs(t).cartesian.xyz.to_value(u.m)

    be, bn, bu = (np.ravel(c)[0] for c in ppigrf.igrf(lon, lat, height_km, t.to_datetime()))
    phi, lam = np.radians(lat), np.radians(lon)
    east = np.array([-np.sin(lam), np.cos(lam), 0.0])
    north = np.array([-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi)])
    up = np.array([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)])
    b_itrs = (be * east + bn * north + bu * up) * 1e-9
    basis = ITRS(CartesianRepresentation(np.eye(3).T * 1e7 * u.m), obstime=t).transform_to(GCRS(obstime=t))
    rotation = basis.cartesian.xyz.to_value(u.m) / 1e7  # columns are ITRS axes in GCRS
    expected = rotation @ b_itrs

    actual = igrf_magnetic_field(t, r_gcrf)
    assert np.linalg.norm(actual - expected) <= 1e-3 * np.linalg.norm(expected)

    previous = geomagnetics.set_external_model(lambda p: np.full_like(p, 1e4))
    try:
        np.testing.assert_allclose(igrf_magnetic_field(t, r_gcrf), actual, rtol=1e-12, atol=0)
    finally:
        geomagnetics.set_external_model(previous)
