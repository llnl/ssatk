import numpy as np
from astropy.time import Time


def test_low_precision_moon_meets_its_stated_accuracy():
    # R2: astropy get_body("moon") in the geocentric mean ecliptic of date,
    # sampled 200 times over 2000-2040. The series is held to its docstring:
    # 6 arcmin in longitude, 3.5 arcmin in latitude, 550 km in distance
    # (measured 5.1 arcmin, 2.9 arcmin, 494 km). Before the 2M' and 2F terms and
    # the sign of the 2D - M' - F term were fixed it was 22 arcmin, 8 arcmin,
    # and 1,112 km.
    import astropy.units as u
    from astropy.coordinates import GeocentricMeanEcliptic, get_body

    from ssatk.plots.solar_bodies import moon_geocentric_ecliptic

    worst_lon = worst_lat = worst_dist = 0.0
    for jd in np.linspace(Time("2000-01-01", scale="tt").jd, Time("2040-01-01", scale="tt").jd, 200):
        t = Time(jd, format="jd", scale="tt")
        x, y, z = moon_geocentric_ecliptic(jd)
        reference = get_body("moon", t).transform_to(GeocentricMeanEcliptic(equinox=t, obstime=t))
        distance = np.linalg.norm([x, y, z])
        d_lon = (np.degrees(np.arctan2(y, x)) - reference.lon.deg + 180.0) % 360.0 - 180.0
        d_lat = np.degrees(np.arcsin(z / distance)) - reference.lat.deg
        worst_lon = max(worst_lon, abs(d_lon) * 60.0)
        worst_lat = max(worst_lat, abs(d_lat) * 60.0)
        worst_dist = max(worst_dist, abs(distance - reference.distance.to_value(u.km)))
    assert worst_lon < 6.0
    assert worst_lat < 3.5
    assert worst_dist < 550.0
