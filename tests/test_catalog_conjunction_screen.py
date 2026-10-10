from types import SimpleNamespace

import numpy as np
import pytest

from ssatk.constants import EARTH_MU
from ssatk.ssa.conjunction import catalog_conjunction_screen

T_CROSS = 1000.0


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
