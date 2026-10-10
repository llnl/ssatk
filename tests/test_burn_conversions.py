"""Finite-burn and impulsive conversions against the closed-form post-burn orbit."""

import numpy as np
import pytest
import ssapy

from ssatk.constants import EARTH_MU
from ssatk.orbital_mechanics.burn_to_deltav import burn_to_deltav
from ssatk.orbital_mechanics.calculate_finite_burn_acceleration import calculate_finite_burn_acceleration
from ssatk.orbital_mechanics.deltav_to_burn import deltav_to_burn

RADIUS = 7000e3
V_CIRC = np.sqrt(EARTH_MU / RADIUS)
ORBIT = ssapy.Orbit(r=[RADIUS, 0.0, 0.0], v=[0.0, V_CIRC, 0.0], t=0.0, mu=EARTH_MU)
DELTA_V = 10.0


def _semi_major_axis(r, v):
    return 1.0 / (2.0 / np.linalg.norm(r) - np.dot(v, v) / EARTH_MU)


@pytest.mark.parametrize("convert", ["burn_to_deltav", "deltav_to_burn"])
@pytest.mark.parametrize("duration, samples", [(10.0, 201), (100.0, 2000)])
def test_tangential_burn_on_a_circular_orbit_matches_vis_viva(convert, duration, samples):
    # R1: a tangential delta-v on a circular orbit gives
    # a = 1 / (2/r - (v_c + dv)^2 / mu). The impulsive branch holds it to
    # 1e-12 relative and the 10-100 s continuous burn to 1e-8 (gravity losses
    # are second order in burn time); the impulse lands on the reported t_center.
    times = np.linspace(0.0, duration, samples)
    if convert == "burn_to_deltav":
        out = burn_to_deltav(ORBIT, times, [0.0, DELTA_V / duration, 0.0])
    else:
        out = deltav_to_burn(ORBIT, times, [0.0, DELTA_V, 0.0])
    expected = 1.0 / (2.0 / RADIUS - (V_CIRC + DELTA_V) ** 2 / EARTH_MU)
    assert _semi_major_axis(out["r_instantaneous"][-1], out["v_instantaneous"][-1]) == pytest.approx(expected, rel=1e-12)
    assert _semi_major_axis(out["r_continuous"][-1], out["v_continuous"][-1]) == pytest.approx(expected, rel=1e-8)
    np.testing.assert_allclose(np.linalg.norm(out["delta_v_gcrf"]), DELTA_V, rtol=1e-12)
    k = int(np.flatnonzero(times == out["t_center"])[0])
    jump = out["v_instantaneous"][k] - out["v_instantaneous"][k - 1]  # post-impulse state sits at index k
    assert np.linalg.norm(jump) > 0.9 * DELTA_V


def test_finite_burn_window_is_centred_and_delivers_the_delta_v():
    # R1: a constant acceleration a for dv/a seconds centred on t_imp delivers dv
    # along its direction. Exact.
    delta_v = np.array([3.0, -4.0, 12.0])
    a_vec, t_burn, t_start, t_end = calculate_finite_burn_acceleration(delta_v, 1000.0, 0.5)
    np.testing.assert_allclose(a_vec * t_burn, delta_v, rtol=1e-15)
    assert (t_burn, t_start, t_end) == (26.0, 987.0, 1013.0)
