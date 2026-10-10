import numpy as np
import pytest
import ssapy

from ssatk.constants import EARTH_MU
from ssatk.orbital_mechanics.synthetic_orbit_population import synthetic_orbit_population


def test_population_spread_and_trajectories():
    # R1: perturbed elements are N(nominal, sigma), so the sample standard
    # deviation of a over 400 orbits is within 10 % of sigma_a (the 99.8 %
    # range of the chi distribution at n = 399 is +/- 8 %). Orbit 0 is the
    # nominal orbit. R2: each trajectory equals SSAPy's Kepler propagation of
    # its orbit (1e-6 m).
    orbits, r_list, v_list, t_list, mu = synthetic_orbit_population(M=401, N=5, dt=60.0, a_sigma=500.0, seed=4)
    a = np.array([o.a for o in orbits])
    assert orbits[0].a == pytest.approx(7_000e3, rel=1e-12)
    assert np.std(a[1:], ddof=1) == pytest.approx(500.0, rel=0.10)
    for k in (0, 7, 400):
        r_ref, _ = ssapy.rv(orbits[k], orbits[k].t + t_list[k], propagator=ssapy.KeplerianPropagator())
        np.testing.assert_allclose(r_list[k], r_ref, rtol=0, atol=1e-6)
    assert mu == EARTH_MU
