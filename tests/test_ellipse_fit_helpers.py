import numpy as np
import pytest
import ssapy

from ssatk.constants import EARTH_MU
from ssatk.orbital_mechanics.ellipse_fit import ellipse_fit

P1 = np.array([7000e3, 0.0, 0.0])
P2 = np.array([0.0, 8000e3, 0.0])
DEPARTURE_GPS_S = 1.4e9


def test_default_fit_is_the_fundamental_ellipse():
    # R1: the minimum-eccentricity ellipse through two points with one focus at
    # the origin has a = (r1 + r2) / 2 and e = |r2 - r1| / c, c the chord.
    # For r1 = 7000 km, r2 = 8000 km at 90 deg: a = 7500 km, e = 0.0940721.
    # Held to 1e-8 relative.
    fit = ellipse_fit(P1, P2, n_pts=16)
    r1, r2 = np.linalg.norm(P1), np.linalg.norm(P2)
    chord = np.linalg.norm(P2 - P1)
    assert fit["a"] == pytest.approx(0.5 * (r1 + r2), rel=1e-8)
    assert fit["e"] == pytest.approx(abs(r2 - r1) / chord, rel=1e-8)


def test_fit_from_second_focus_matches_focal_definition():
    # R1: with foci at the origin and F2, 2a = |P1| + |P1 - F2| and
    # e = |F2| / (2a). F2 = (500, -500, 0) km puts P2 on the same ellipse.
    # Held to 1e-10 relative.
    focus = np.array([500e3, -500e3, 0.0])
    p2 = _point_on_ellipse(focus, P1, angle_rad=np.pi / 2)
    fit = ellipse_fit(P1, p2, n_pts=16, F2_m=focus)
    two_a = np.linalg.norm(P1) + np.linalg.norm(P1 - focus)
    assert fit["a"] == pytest.approx(0.5 * two_a, rel=1e-10)
    assert fit["e"] == pytest.approx(np.linalg.norm(focus) / two_a, rel=1e-10)
    np.testing.assert_allclose(fit["r"][-1], p2, rtol=0, atol=1e-3)


def test_second_focus_that_misses_p2_is_rejected():
    # R1: foci at the origin and (1000, 1000, 0) km give 2a = 13,083 km through
    # P1, but |P2| + |P2 - F2| = 15,071 km, so no such ellipse reaches P2. The
    # function used to return an arc ending 1,000 km from P2.
    with pytest.raises(ValueError, match="does not define an ellipse"):
        ellipse_fit(P1, P2, n_pts=16, F2_m=np.array([1000e3, 1000e3, 0.0]))


@pytest.mark.parametrize("kwargs", [{}, {"a_m": 10000e3}, {"e": 0.25}], ids=["min-e", "fixed-a", "fixed-e"])
def test_sampled_arc_is_a_two_body_trajectory_from_p1_to_p2(kwargs):
    # R2: SSAPy KeplerianPropagator from the first sample reproduces every
    # sampled state (1e-2 m, 1e-5 m/s), the arc ends at P2 (1e-2 m), and the
    # first sample satisfies vis-viva (1e-9 relative).
    fit = ellipse_fit(P1, P2, n_pts=16, time_of_departure=DEPARTURE_GPS_S, **kwargs)
    if "a_m" in kwargs:
        assert fit["a"] == pytest.approx(kwargs["a_m"], rel=1e-12)
    if "e" in kwargs:
        assert fit["e"] == pytest.approx(kwargs["e"], rel=1e-12)

    orbit = ssapy.Orbit(r=fit["r"][0], v=fit["v"][0], t=fit["t_abs"][0], mu=EARTH_MU)
    r_ref, v_ref = ssapy.rv(orbit, fit["t_abs"], propagator=ssapy.KeplerianPropagator())
    np.testing.assert_allclose(fit["r"], r_ref, rtol=0, atol=1e-2)
    np.testing.assert_allclose(fit["v"], v_ref, rtol=0, atol=1e-5)
    np.testing.assert_allclose(fit["r"][0], P1, rtol=0, atol=1e-2)
    np.testing.assert_allclose(fit["r"][-1], P2, rtol=0, atol=1e-2)
    assert fit["t_abs"][0] == pytest.approx(DEPARTURE_GPS_S, abs=1e-6)

    r0 = np.linalg.norm(fit["r"][0])
    assert np.dot(fit["v"][0], fit["v"][0]) == pytest.approx(EARTH_MU * (2.0 / r0 - 1.0 / fit["a"]), rel=1e-9)


def test_arrival_epoch_anchors_the_end_of_the_arc():
    # R2: with time_of_arrival set, SSAPy propagation from the first sample
    # reaches P2 at that epoch (1e-2 m).
    arrival = DEPARTURE_GPS_S + 5000.0
    fit = ellipse_fit(P1, P2, n_pts=12, time_of_arrival=arrival)
    assert fit["t_abs"][-1] == pytest.approx(arrival, abs=1e-6)
    orbit = ssapy.Orbit(r=fit["r"][0], v=fit["v"][0], t=fit["t_abs"][0], mu=EARTH_MU)
    r_end, _ = ssapy.rv(orbit, arrival, propagator=ssapy.KeplerianPropagator())
    np.testing.assert_allclose(np.squeeze(r_end), P2, rtol=0, atol=1e-2)


def _point_on_ellipse(focus, p1, *, angle_rad):
    """Point at in-plane polar angle angle_rad on the ellipse with foci 0 and focus through p1."""
    two_a = np.linalg.norm(p1) + np.linalg.norm(p1 - focus)
    direction = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0])
    # |r| + |r d - F| = 2a  ->  r = (4a^2 - |F|^2) / (2 (2a - d.F))
    radius = (two_a**2 - np.dot(focus, focus)) / (2.0 * (two_a - np.dot(direction, focus)))
    return radius * direction
