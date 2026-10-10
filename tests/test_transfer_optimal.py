import numpy as np
import pytest

from ssatk.constants import EARTH_MU
from ssatk.orbital_mechanics.transfer_optimal_function import transfer_optimal


def _circular(radius, phase):
    r = radius * np.array([np.cos(phase), np.sin(phase), 0.0])
    v = np.sqrt(EARTH_MU / radius) * np.array([-np.sin(phase), np.cos(phase), 0.0])
    return r, v


def test_minimum_delta_v_search_recovers_the_hohmann_transfer():
    # R1: between coplanar circular orbits with r2/r1 < 11.94 the Hohmann
    # transfer is the minimum-delta-v two-impulse transfer. For 7000 -> 9000 km
    # it costs 887.562 m/s over a half period of the 8000 km transfer ellipse
    # (3560.5 s). The porkchop search plus Nelder-Mead polish finds it to 1e-5
    # in delta-v and 0.1 % in time of flight (measured 9e-8 and 1.2e-4).
    r1, v1 = _circular(7000e3, 0.0)
    r2, v2 = _circular(9000e3, 0.4)
    result = transfer_optimal(
        (r1, v1, 0.0), (r2, v2, 0.0), delta_v_mode="total", t_window=(0.0, 20000.0),
        tof_range=(1000.0, 6000.0), n_grid=(24, 24), polish=True, propagate=False, refine=False,
    )
    a = 8000e3
    hohmann = (np.sqrt(EARTH_MU * (2 / 7000e3 - 1 / a)) - np.sqrt(EARTH_MU / 7000e3)
               + np.sqrt(EARTH_MU / 9000e3) - np.sqrt(EARTH_MU * (2 / 9000e3 - 1 / a)))
    assert result["delta_v_total"] == pytest.approx(hohmann, rel=1e-5)
    assert result["tof"] == pytest.approx(np.pi * np.sqrt(a**3 / EARTH_MU), rel=1e-3)
