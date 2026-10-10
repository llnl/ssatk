import numpy as np
import pytest

from ssatk.constants import EARTH_MU, EARTH_RADIUS
from ssatk.launch.gravity_turn import accel_gravity_turn
from ssatk.launch.sites import launch_pads


def _site(lat_deg, lon_deg):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    return up, east, np.cross(up, east)


@pytest.mark.parametrize("name", ["Kennedy Space Center LC-39A", "Vandenberg SLC-4E", "Baikonur Cosmodrome Site 1/5"])
def test_gravity_turn_thrust_rises_vertically_and_levels_on_the_azimuth(name):
    # R1: at t = 0 the thrust is along the local vertical; at the end of the
    # turn it is horizontal at the azimuth clockwise from local north. Gravity
    # is -mu r / r^3. Exact to 1e-12 m/s^2.
    pad = launch_pads[name]
    up, east, north = _site(pad["latitude"], pad["longitude"])
    r = EARTH_RADIUS * up
    times = np.linspace(0.0, 100.0, 11)
    thrust = np.full(times.size, 20.0)
    gravity = -EARTH_MU * r / EARTH_RADIUS**3
    azimuth = np.radians(45.0)
    np.testing.assert_allclose(accel_gravity_turn(r, 0, times, thrust, 100.0, azimuth), gravity + 20.0 * up, atol=1e-12)
    level = np.cos(azimuth) * north + np.sin(azimuth) * east
    np.testing.assert_allclose(accel_gravity_turn(r, 10, times, thrust, 100.0, azimuth), gravity + 20.0 * level, atol=1e-12)
    half = (np.sin(np.pi / 4) * up + np.cos(np.pi / 4) * level)
    np.testing.assert_allclose(accel_gravity_turn(r, 5, times, thrust, 100.0, azimuth), gravity + 20.0 * half, atol=1e-12)
