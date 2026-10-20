import numpy as np
from astropy.time import Time
from ssapy import get_body

from ssatk.coordinates.lunar import gcrf_to_lunar_fixed


def test_earth_moon_rotating_frame_axes():
    # R1: in the Moon-centred Earth-Moon rotating frame Earth sits at
    # (-d, 0, 0), a point 1000 km beyond the Moon along the Earth-Moon line is
    # at (+1000 km, 0, 0), and a point 1000 km along the orbit normal is at
    # (0, 0, +1000 km). 1e-6 relative to d.
    t = Time(["2026-10-08T00:00:00"], scale="utc").gps
    moon = get_body("moon")
    r_moon = np.asarray(moon.position(t), dtype=float).reshape(3)
    v_moon = (np.asarray(moon.position(t + 5.0)).reshape(3) - np.asarray(moon.position(t - 5.0)).reshape(3)) / 10.0
    d = np.linalg.norm(r_moon)
    x_hat = r_moon / d
    z_hat = np.cross(r_moon, v_moon) / np.linalg.norm(np.cross(r_moon, v_moon))
    for point, expected in (
        (np.zeros(3), [-d, 0.0, 0.0]),
        (r_moon + 1000e3 * x_hat, [1000e3, 0.0, 0.0]),
        (r_moon + 1000e3 * z_hat, [0.0, 0.0, 1000e3]),
    ):
        got = np.asarray(gcrf_to_lunar_fixed(point.reshape(1, 3), t), dtype=float).reshape(3)
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6 * d)
