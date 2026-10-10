import numpy as np
import pytest
import ssapy

from ssatk.constants import EARTH_MU
from ssatk.orbital_mechanics.ellipse_inject import (
    ellipse_inject,
    ellipse_insertion,
    ellipse_intercept,
)

EPOCH_GPS_S = 1.4e9


def _circular_state(radius=7000e3, theta=0.0):
    r = radius * np.array([np.cos(theta), np.sin(theta), 0.0])
    v = np.sqrt(EARTH_MU / radius) * np.array([-np.sin(theta), np.cos(theta), 0.0])
    return r, v


def _propagate(r, v, tof):
    orbit = ssapy.Orbit(r=r, v=v, t=EPOCH_GPS_S, mu=EARTH_MU)
    r_end, v_end = ssapy.rv(orbit, EPOCH_GPS_S + tof, propagator=ssapy.KeplerianPropagator())
    return np.squeeze(r_end), np.squeeze(v_end)


def test_rendezvous_burns_fly_the_two_body_transfer_and_match_the_target():
    # R2: SSAPy KeplerianPropagator. The departure burn (v1 + dv1) must reach r2
    # after the requested time of flight (1e-2 m), and the arrival burn must
    # turn the propagated arrival velocity into v2 (1e-5 m/s). The reported
    # delta-v magnitudes equal those two vector differences (1e-6 m/s).
    r1, v1 = _circular_state(7000e3)
    r2, v2 = _circular_state(8000e3, theta=np.pi / 2.0)
    tof = ellipse_insertion(r1, v1, r2, v2, n_pts=32)["tof"]

    result = ellipse_inject(r1, v1, r2, v2, tof=tof, n_pts=32, arrival_mode="rendezvous")
    departure, arrival = result["burns"]

    assert result["tof"] == pytest.approx(tof, abs=1e-6)
    r_end, v_end = _propagate(r1, v1 + departure["delta_v"], result["tof"])
    np.testing.assert_allclose(r_end, r2, rtol=0, atol=1e-2)
    np.testing.assert_allclose(v_end + arrival["delta_v"], v2, rtol=0, atol=1e-5)
    np.testing.assert_allclose(
        result["delta_v_magnitudes"],
        [np.linalg.norm(departure["delta_v"]), np.linalg.norm(v2 - v_end)],
        rtol=0,
        atol=1e-6,
    )


def test_intercept_uses_one_burn_and_arrives_on_the_propagated_velocity():
    # R2: SSAPy KeplerianPropagator. One departure burn reaches r2 after the
    # requested time of flight (1e-2 m); the reported final velocity is the
    # propagated arrival velocity (1e-5 m/s), with no arrival burn.
    r1, v1 = _circular_state(7000e3)
    r2, _v2 = _circular_state(9000e3, theta=0.7)
    tof = ellipse_inject(r1, v1, r2, n_pts=16, arrival_mode="inject")["tof"]

    result = ellipse_intercept(r1, v1, r2, tof=tof, n_pts=16)

    assert len(result["burns"]) == 1
    r_end, v_end = _propagate(r1, v1 + result["burns"][0]["delta_v"], result["tof"])
    np.testing.assert_allclose(r_end, r2, rtol=0, atol=1e-2)
    np.testing.assert_allclose(result["final"]["r"], r2, rtol=0, atol=1e-2)
    np.testing.assert_allclose(result["final"]["v"], v_end, rtol=0, atol=1e-5)
