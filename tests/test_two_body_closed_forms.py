import numpy as np
import pytest

from ssatk.constants import EARTH_MU, EARTH_RADIUS
from ssatk.coordinates.geodetic import lonlat_distance
from ssatk.orbital_mechanics import keplerian, vis_viva


@pytest.mark.parametrize("e", [0.0, 0.3, 0.7])
def test_vis_viva_matches_periapsis_and_apoapsis_speeds(e):
    # R1: at periapsis v^2 = mu (1 + e) / (a (1 - e)); at apoapsis
    # v^2 = mu (1 - e) / (a (1 + e)). Both exports agree, 1e-12 relative.
    a = 12_000e3
    rp, ra = a * (1 - e), a * (1 + e)
    vp = np.sqrt(EARTH_MU * (1 + e) / (a * (1 - e)))
    va = np.sqrt(EARTH_MU * (1 - e) / (a * (1 + e)))
    assert vis_viva(EARTH_MU, rp, a) == pytest.approx(vp, rel=1e-12)
    assert vis_viva(EARTH_MU, ra, a) == pytest.approx(va, rel=1e-12)
    assert keplerian.vis_viva(a=a, r=rp, mu=EARTH_MU) == pytest.approx(vp, rel=1e-12)


def test_lonlat_distance_is_the_haversine_great_circle():
    # R1: on a sphere of radius R, a quarter meridian is pi R / 2, a quarter of
    # the equator is pi R / 2, and 1 deg of arc is pi R / 180. Inputs in
    # radians; 1e-12 relative (1e-5 m on 10,000 km; an absolute 1e-9 m is below
    # float64 resolution at that size).
    quarter = np.pi * EARTH_RADIUS / 2
    assert lonlat_distance(0.0, np.pi / 2, 0.0, 0.0) == pytest.approx(quarter, rel=1e-12)
    assert lonlat_distance(0.0, 0.0, 0.0, np.pi / 2) == pytest.approx(quarter, rel=1e-12)
    lat = np.radians(37.7)
    expected = 2 * EARTH_RADIUS * np.arcsin(np.cos(lat) * np.sin(np.radians(1.0) / 2))
    assert lonlat_distance(lat, lat, 0.0, np.radians(1.0)) == pytest.approx(expected, rel=1e-12)
    assert lonlat_distance(np.radians(10.0), np.radians(11.0), 0.3, 0.3) == pytest.approx(
        np.pi * EARTH_RADIUS / 180, abs=1e-6
    )
