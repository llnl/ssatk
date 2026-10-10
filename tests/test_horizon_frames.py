"""Equatorial <-> horizontal transforms checked against exact cases and astropy."""

import numpy as np
import pytest

from ssatk.coordinates import local_equatorial as le

LATITUDE_DEG = 37.68  # Livermore


@pytest.mark.parametrize(
    "hour_angle, expected_azimuth",
    [(0.0, 180.0), (90.0, 270.0), (270.0, 90.0)],
)
def test_celestial_equator_cardinal_points(hour_angle, expected_azimuth):
    # R1: from a northern site, dec = 0 transits due south and sets due west
    # (HA = +90 deg) and rises due east (HA = -90 deg), on the horizon.
    azimuth, altitude = le.equatorial_to_horizontal(LATITUDE_DEG, 0.0, hour_angle=hour_angle)
    assert azimuth == pytest.approx(expected_azimuth, abs=1e-9)
    expected_altitude = 90.0 - LATITUDE_DEG if hour_angle == 0.0 else 0.0
    assert altitude == pytest.approx(expected_altitude, abs=1e-9)
    hour_angle_back, declination = le.horizontal_to_equatorial(LATITUDE_DEG, azimuth, altitude)
    assert np.mod(hour_angle_back - hour_angle + 180.0, 360.0) - 180.0 == pytest.approx(0.0, abs=1e-6)
    assert declination == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("latitude", [37.68, -33.87])
@pytest.mark.parametrize("hour_angle", [60.0, 300.0])
def test_horizontal_coordinates_match_astropy_altaz(latitude, hour_angle):
    # R2: astropy AltAz without refraction. The toolkit formula ignores
    # precession and nutation of date, so allow 0.5 deg.
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time

    time = Time("2025-03-20T06:00:00", scale="utc")
    longitude = -121.77
    location = EarthLocation(lat=latitude * u.deg, lon=longitude * u.deg)
    lst = time.sidereal_time("apparent", longitude=longitude * u.deg).deg
    ra = np.mod(lst - hour_angle, 360.0)
    reference = SkyCoord(ra=ra * u.deg, dec=20.0 * u.deg).transform_to(
        AltAz(obstime=time, location=location, pressure=0 * u.hPa))

    azimuth, altitude = le.equatorial_to_horizontal(latitude, 20.0, hour_angle=hour_angle)
    assert np.mod(azimuth - reference.az.deg + 180.0, 360.0) - 180.0 == pytest.approx(0.0, abs=0.5)
    assert altitude == pytest.approx(reference.alt.deg, abs=0.5)


def test_right_ascension_and_sidereal_time_path_matches_hour_angle_path():
    # R1: HA = LST - RA. RA 06:00:00 at LST 08:00:00 is HA = 2 h = 30 deg.
    assert le.hms_to_dd(le.rightascension2hourangle("06:00:00", "08:00:00")) == pytest.approx(30.0)
    assert le.hms_to_dd(le.rightascension2hourangle("22:00:00", "01:00:00")) == pytest.approx(45.0)
    via_ra = le.equatorial_to_horizontal(LATITUDE_DEG, 20.0, right_ascension="06:00:00", local_time="08:00:00")
    via_ha = le.equatorial_to_horizontal(LATITUDE_DEG, 20.0, hour_angle=30.0)
    np.testing.assert_allclose(via_ra, via_ha, atol=1e-9)
