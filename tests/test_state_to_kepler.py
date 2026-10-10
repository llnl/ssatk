import numpy as np
import pytest
import ssapy

from ssatk.constants import EARTH_MU
from ssatk.orbital_mechanics.keplerian import kepler_to_parametric, state_to_kepler

# (a [m], e, i, pa, raan, nu) with angles in degrees; ISS-like, Molniya-like,
# retrograde, and a near-polar GTO, chosen to put every angle in a different
# quadrant.
ELEMENT_SETS = [
    (6_778e3, 0.0005, 51.6, 30.0, 120.0, 75.0),
    (26_560e3, 0.72, 63.4, 270.0, 300.0, 200.0),
    (7_200e3, 0.05, 98.0, 160.0, 210.0, 330.0),
    (24_400e3, 0.73, 89.0, 100.0, 45.0, 150.0),
]


def _ssapy_orbit(elements):
    a, e, inc, pa, raan, nu = elements
    return ssapy.Orbit.fromKeplerianElements(
        a, e, np.radians(inc), np.radians(pa), np.radians(raan), np.radians(nu), t=1.4e9, mu=EARTH_MU
    )


def _angle_difference(x, y):
    return np.abs((np.asarray(x) - np.asarray(y) + np.pi) % (2 * np.pi) - np.pi)


@pytest.mark.parametrize("elements", ELEMENT_SETS)
def test_state_to_kepler_inverts_ssapy_elements(elements):
    # R2: SSAPy Orbit.fromKeplerianElements builds the state; state_to_kepler
    # must return the same a (1e-9 relative), e (1e-10), and angles (1e-9 rad).
    orbit = _ssapy_orbit(elements)
    a, e, inc, pa, raan, nu = state_to_kepler(orbit.r, orbit.v)
    assert a == pytest.approx(elements[0], rel=1e-9)
    assert e == pytest.approx(elements[1], abs=1e-10)
    expected_angles = np.radians(elements[2:])
    np.testing.assert_allclose(_angle_difference([inc, pa, raan, nu], expected_angles), 0.0, atol=1e-9)


def test_state_to_kepler_accepts_a_batch_of_states():
    # R2: an (N, 3) batch returns the per-state SSAPy elements (1e-9 rad).
    orbits = [_ssapy_orbit(elements) for elements in ELEMENT_SETS]
    r = np.array([orbit.r for orbit in orbits])
    v = np.array([orbit.v for orbit in orbits])
    a, e, inc, pa, raan, nu = state_to_kepler(r, v)
    expected = np.array([orbit.keplerianElements for orbit in orbits])
    np.testing.assert_allclose(a, expected[:, 0], rtol=1e-9)
    np.testing.assert_allclose(e, expected[:, 1], rtol=0, atol=1e-10)
    np.testing.assert_allclose(_angle_difference(np.column_stack([inc, pa, raan, nu]), expected[:, 2:]), 0.0, atol=1e-9)


@pytest.mark.parametrize("elements", ELEMENT_SETS)
def test_kepler_to_parametric_matches_ssapy_position(elements):
    # R2: the focus-centred position from the elements matches SSAPy's orbit
    # position to 1e-6 m.
    orbit = _ssapy_orbit(elements)
    a, e, inc, pa, raan, nu = elements
    position = np.array(kepler_to_parametric(a, e, inc, raan, pa, nu))
    np.testing.assert_allclose(position, orbit.r, rtol=0, atol=1e-6)
