from types import SimpleNamespace

import numpy as np
import pytest

from ssatk.constants import EARTH_MU
from ssatk.ssa.conjunction import catalog_conjunction_screen

T_CROSS = 1000.0


def test_quadratic_between_sample_collision_is_not_screened_out():
    """R1: x=(t-1)² touches x=0 at t=1 s; endpoint chord stays at 1 m."""
    t = np.array([0., 2.])
    first = SimpleNamespace(t=t, r=np.zeros((2, 3)), v=np.zeros((2, 3)))
    second = SimpleNamespace(t=t, r=np.array([[1., 0, 0], [1., 0, 0]]),
                             v=np.array([[-2., 0, 0], [2., 0, 0]]))
    events = catalog_conjunction_screen({"a": first, "b": second}, .01)
    assert len(events) == 1
    assert events[0].closest_approach.tca == pytest.approx(1., abs=1e-6)  # s
    assert events[0].closest_approach.miss_distance == pytest.approx(0., abs=1e-10)  # m


def test_three_contacts_inside_one_cubic_sample_interval_remain_separate():
    """R1: a cubic with roots at 1/4, 1/2 and 3/4 has three distinct contacts."""
    roots = np.array([0.25, 0.5, 0.75])
    coefficients = np.polynomial.polynomial.polyfromroots(roots)
    derivative = np.polynomial.polynomial.polyder(coefficients)
    times = np.array([0.0, 1.0])
    first = SimpleNamespace(t=times, r=np.zeros((2, 3)), v=np.zeros((2, 3)))
    second = SimpleNamespace(
        t=times,
        r=np.array([[np.polynomial.polynomial.polyval(t, coefficients), 0, 0] for t in times]),
        v=np.array([[np.polynomial.polynomial.polyval(t, derivative), 0, 0] for t in times]),
    )

    events = catalog_conjunction_screen({"a": first, "b": second}, 1e-8)

    assert len(events) == 3
    np.testing.assert_allclose(
        [event.closest_approach.tca for event in events], roots, rtol=0, atol=1e-9
    )
    np.testing.assert_allclose(
        [event.closest_approach.miss_distance for event in events], 0, rtol=0, atol=1e-12
    )


def test_trajectories_touching_at_only_one_epoch_are_screened():
    """R1: two closed trajectory domains sharing t=1 s have a 2 m contact."""
    first = SimpleNamespace(
        t=np.array([0.0, 1.0]), r=np.zeros((2, 3)), v=np.zeros((2, 3))
    )
    second = SimpleNamespace(
        t=np.array([1.0, 2.0]),
        r=np.array([[2.0, 0, 0], [2.0, 0, 0]]),
        v=np.zeros((2, 3)),
    )

    events = catalog_conjunction_screen({"a": first, "b": second}, 2.0)

    assert len(events) == 1
    assert events[0].closest_approach.tca == pytest.approx(1.0, abs=0)  # s
    assert events[0].closest_approach.miss_distance == pytest.approx(2.0, abs=0)  # m


def _circular(radius, inclination_deg, t):
    # Circular orbit through the +x node at t = T_CROSS.
    n = np.sqrt(EARTH_MU / radius**3)
    u = n * (t - T_CROSS)
    inc = np.radians(inclination_deg)
    r = radius * np.column_stack([np.cos(u), np.sin(u) * np.cos(inc), np.sin(u) * np.sin(inc)])
    v = radius * n * np.column_stack([-np.sin(u), np.cos(u) * np.cos(inc), np.cos(u) * np.sin(inc)])
    return r, v


def test_catalog_screen_finds_the_node_crossing_and_ignores_distant_objects():
    # R1/R2: two circular orbits (7000 km equatorial; 7000.5 km at 60 deg) both
    # pass the +x node at t = 1000 s. The true closest approach, from the
    # analytic positions on a 1 ms grid, is 500 m near t = 1000 s; the screen of
    # 30 s samples must find it to 1 ms and 1 mm, and must not pair either with
    # a GEO object 35,000 km away.
    t = np.arange(0.0, 2000.0 + 1e-9, 30.0)
    tracks = {}
    for name, (radius, inclination) in {"A": (7000e3, 0.0), "B": (7000.5e3, 60.0), "GEO": (42164e3, 0.0)}.items():
        r, v = _circular(radius, inclination, t)
        tracks[name] = SimpleNamespace(t=t, r=r, v=v)

    events = catalog_conjunction_screen(tracks, threshold=10e3)
    assert [(e.object_id_a, e.object_id_b) for e in events] == [("A", "B")]

    fine = np.arange(T_CROSS - 5.0, T_CROSS + 5.0, 1e-3)
    ra, _ = _circular(7000e3, 0.0, fine)
    rb, _ = _circular(7000.5e3, 60.0, fine)
    separation = np.linalg.norm(rb - ra, axis=1)
    k = int(np.argmin(separation))
    approach = events[0].closest_approach
    assert approach.tca == pytest.approx(fine[k], abs=1e-3)
    assert approach.miss_distance == pytest.approx(separation[k], abs=1e-3)
    assert approach.miss_distance == pytest.approx(500.0, abs=1.0)
