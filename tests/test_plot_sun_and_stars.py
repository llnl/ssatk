"""Sun and star directions used by the plots, against astropy (R2)."""

from datetime import datetime, timezone

import numpy as np
import pytest
from astropy.time import Time

UTC_EPOCHS = ["2000-01-01T12:00:00", "2024-04-08T18:17:00", "2026-10-09T20:00:00"]


def _sep_arcsec(a, b):
    a = np.atleast_2d(a) / np.linalg.norm(np.atleast_2d(a), axis=1, keepdims=True)
    b = np.atleast_2d(b) / np.linalg.norm(np.atleast_2d(b), axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=1), -1.0, 1.0))) * 3600.0


@pytest.mark.parametrize("utc", UTC_EPOCHS)
def test_plot_sun_direction_matches_astropy_get_sun(utc):
    # R2: astropy get_sun (GCRS, apparent) is the reference. SSAPy's fast
    # sunPos is geometric, so ~20" of annual aberration is expected; measured
    # 10-37" over 2000-2040. Tolerance 60". The mean-longitude formula this
    # replaces was 1236" off on 2024-04-08 and 1333" on 2026-10-09.
    from astropy.coordinates import get_sun

    from ssatk.plots.eclipse_brightness_plot import sun_direction_eci
    from ssatk.plots.globe_orbit_daynight_plotly import sun_direction_eci as globe_sun

    t = Time(utc, scale="utc")
    reference = get_sun(t).cartesian.xyz.value
    for t_s, epoch_jd in ((0.0, t.jd), (3600.0, t.jd - 3600.0 / 86400.0)):
        got = sun_direction_eci(np.array([t_s]), epoch_jd=epoch_jd)[0]
        assert _sep_arcsec(got, reference)[0] < 60.0
    assert globe_sun is sun_direction_eci


def _bright_catalogue(when, frame):
    from ssatk.plots.starfield import _load_stars

    stars = _load_stars(mag_limit=2.0, when=when, frame=frame)
    if stars is None:
        pytest.skip("star catalogue not installed")
    return stars


def test_gcrf_stars_match_astropy_icrs_with_proper_motion():
    # R2: catalogue RA/Dec/proper motion propagated by astropy
    # SkyCoord.apply_space_motion (ICRS, no aberration). Tolerance 1".
    # Before this change frame="gcrf" applied precession: median 1142".
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    from ssatk.plots.starfield import _catalog_path, _apply_frame
    import pandas as pd

    when = datetime(2026, 10, 9, 20, 0, tzinfo=timezone.utc)
    df = pd.read_csv(_catalog_path())
    df = df[df["mag"] <= 2.0]
    ra_h, dec_d = df["ra"].to_numpy(float), df["dec"].to_numpy(float)
    pmra, pmdec = df["pmra"].to_numpy(float), df["pmdec"].to_numpy(float)
    ours = _apply_frame(ra_h, dec_d, pmra, pmdec, when, "gcrf")
    c = SkyCoord(ra=ra_h * 15 * u.deg, dec=dec_d * u.deg, distance=1e6 * u.pc,
                 pm_ra_cosdec=pmra * u.mas / u.yr, pm_dec=pmdec * u.mas / u.yr,
                 frame="icrs", obstime=Time("J2000")).apply_space_motion(new_obstime=Time(when))
    assert _sep_arcsec(ours, c.cartesian.xyz.value.T).max() < 1.0


def test_teme_stars_match_astropy_teme_rotation():
    # R2: astropy's GCRS->TEME rotation (applied to geocentric unit vectors, so
    # no aberration enters) rotates the GCRF star directions; tolerance 2".
    import astropy.units as u
    from astropy.coordinates import GCRS, TEME, CartesianRepresentation

    when = datetime(2026, 10, 9, 20, 0, tzinfo=timezone.utc)
    t = Time(when)
    gcrf = _bright_catalogue(when, "gcrf")["v"]
    teme = _bright_catalogue(when, "teme")["v"]
    basis = GCRS(CartesianRepresentation(np.eye(3).T * u.km), obstime=t).transform_to(TEME(obstime=t))
    rot = basis.cartesian.xyz.to_value(u.km)          # columns: images of x, y, z
    assert _sep_arcsec(teme, gcrf @ rot.T).max() < 2.0
