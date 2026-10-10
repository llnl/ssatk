"""Lambert solver checked against closed-form and independent-propagation oracles."""

import numpy as np
import pytest
from ssapy.orbit import Orbit

from ssatk.constants import EARTH_MU
from ssatk.orbital_mechanics import transfer_ssapy_function as tsf

RADIUS_M = 7000.0e3
MEAN_MOTION = np.sqrt(EARTH_MU / RADIUS_M**3)
CIRCULAR_SPEED = np.sqrt(EARTH_MU / RADIUS_M)


def _on_circle(theta):
    return RADIUS_M * np.array([np.cos(theta), np.sin(theta), 0.0])


@pytest.mark.parametrize(
    "prograde, sweep_deg, expected_direction",
    [(True, 60.0, [0.0, 1.0, 0.0]), (False, 300.0, [0.0, -1.0, 0.0])],
)
def test_lambert_recovers_the_circular_orbit_through_both_points(prograde, sweep_deg, expected_direction):
    # Two points 60 deg apart on a 7000 km circle, reached in the time the
    # circular orbit takes to sweep between them, have exactly one conic
    # solution: the circle itself, flown prograde or retrograde.
    tof = np.radians(sweep_deg) / MEAN_MOTION
    v1, v2 = tsf.solve_lambert(_on_circle(0.0), _on_circle(np.radians(60.0)), tof,
                               mu=EARTH_MU, prograde=prograde)

    np.testing.assert_allclose(v1, CIRCULAR_SPEED * np.array(expected_direction),
                               atol=1e-6 * CIRCULAR_SPEED)
    np.testing.assert_allclose(np.linalg.norm(v2), CIRCULAR_SPEED, rtol=1e-9)


def test_lambert_arc_reaches_target_under_independent_kepler_propagation():
    # A non-circular arc: SSAPy's Kepler propagation of (r1, v1) for the
    # requested time of flight must land on r2 (sub-metre over 2500 km).
    r1 = _on_circle(0.0)
    r2 = 1.3 * RADIUS_M * np.array([np.cos(2.0), np.sin(2.0), 0.15])
    tof = 2400.0
    v1, v2 = tsf.solve_lambert(r1, r2, tof, mu=EARTH_MU, prograde=True)

    arrival = Orbit(r=r1, v=v1, t=0.0, mu=EARTH_MU).at(tof)

    np.testing.assert_allclose(arrival.r, r2, atol=0.5)
    np.testing.assert_allclose(arrival.v, v2, atol=1e-3)
