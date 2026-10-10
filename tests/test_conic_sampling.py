"""ellipse_from_rv and rv_to_ellipse sample the conic SSAPy propagates."""

import numpy as np
import pytest
import ssapy

from ssatk.constants import EARTH_MU
from ssatk.orbital_mechanics.ellipse_from_rv import ellipse_from_rv
from ssatk.orbital_mechanics.rv_to_ellipse import rv_to_ellipse

EPOCH = 1.4e9
ORBITS = {  # (a [m], e, i, pa, raan, nu) in degrees
    "leo-inclined": (9_000e3, 0.2, 51.6, 30.0, 120.0, 75.0),
    "molniya": (26_560e3, 0.72, 63.4, 270.0, 300.0, 200.0),
    "near-circular-retrograde": (7_200e3, 1e-4, 98.0, 160.0, 210.0, 330.0),
}


def _state(elements):
    a, e, inc, pa, raan, nu = elements
    orbit = ssapy.Orbit.fromKeplerianElements(
        a, e, np.radians(inc), np.radians(pa), np.radians(raan), np.radians(nu), t=EPOCH, mu=EARTH_MU
    )
    return orbit.r, orbit.v


def _propagated(r0, v0, t_rel):
    orbit = ssapy.Orbit(r=r0, v=v0, t=EPOCH, mu=EARTH_MU)
    return ssapy.rv(orbit, EPOCH + np.asarray(t_rel), propagator=ssapy.KeplerianPropagator())


@pytest.mark.parametrize("name", ORBITS)
def test_ellipse_from_rv_samples_lie_on_the_input_orbit(name):
    # R2: SSAPy KeplerianPropagator from the input state reaches each sample
    # at its t_rel offset from the input epoch, once the input's own time since
    # periapsis is removed (1e-3 m, 1e-6 m/s), and every sample lies in the
    # input orbit plane.
    r0, v0 = _state(ORBITS[name])
    out = ellipse_from_rv(r0, v0, num=64)
    h_hat = np.cross(r0, v0) / np.linalg.norm(np.cross(r0, v0))
    np.testing.assert_allclose(out["r"] @ h_hat, 0.0, atol=1e-3)

    # Samples start at periapsis; shift times so t = 0 is the input state.
    period = out["period"]
    a, e = out["a"], out["e"]
    nu0 = out["ta"]
    E0 = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(nu0 / 2))
    t_since_periapsis = ((E0 - e * np.sin(E0)) % (2 * np.pi)) / np.sqrt(EARTH_MU / a**3)
    r_ref, v_ref = _propagated(r0, v0, out["t_rel"] - t_since_periapsis + period)
    np.testing.assert_allclose(out["r"], r_ref, rtol=0, atol=1e-3)
    np.testing.assert_allclose(out["v"], v_ref, rtol=0, atol=1e-6)


@pytest.mark.parametrize("name", ORBITS)
def test_rv_to_ellipse_samples_follow_the_input_state_in_time(name):
    # R2: SSAPy KeplerianPropagator from the input state reaches each sample
    # at its t_rel (1e-3 m, 1e-6 m/s); the first sample is the input state.
    r0, v0 = _state(ORBITS[name])
    out = rv_to_ellipse(r0, v0, num=64)
    r_ref, v_ref = _propagated(r0, v0, out["t_rel"])
    np.testing.assert_allclose(out["r"], r_ref, rtol=0, atol=1e-3)
    np.testing.assert_allclose(out["v"], v_ref, rtol=0, atol=1e-6)


@pytest.mark.parametrize("gamma_deg", [20.0, -20.0], ids=["outbound", "inbound"])
def test_rv_to_ellipse_follows_a_hyperbolic_flyby(gamma_deg):
    # R2: SSAPy SciPyPropagator(AccelKepler) integrates the two-body ODE (SSAPy's
    # KeplerianPropagator mishandles hyperbolic orbits). An escape trajectory
    # (v = 1.2 v_esc at 10,000 km, +/-20 deg flight-path angle, periapsis
    # 9,090 km) is sampled out to 2 |r0|, through periapsis when inbound;
    # positions agree to 1 m.
    radius = 10_000e3
    speed = 1.2 * np.sqrt(2 * EARTH_MU / radius)
    gamma = np.radians(gamma_deg)
    r0 = np.array([radius, 0.0, 0.0])
    v0 = speed * np.array([np.sin(gamma), np.cos(gamma) * np.cos(0.3), np.cos(gamma) * np.sin(0.3)])
    out = rv_to_ellipse(r0, v0, num=32)
    orbit = ssapy.Orbit(r=r0, v=v0, t=EPOCH, mu=EARTH_MU)
    r_ref, _ = ssapy.rv(orbit, EPOCH + out["t_rel"], propagator=ssapy.SciPyPropagator(ssapy.AccelKepler()))
    np.testing.assert_allclose(out["r"], r_ref, rtol=0, atol=1.0)
    assert np.linalg.norm(out["r"][-1]) == pytest.approx(2 * radius, rel=1e-9)


def test_ellipse_from_rv_parabola_obeys_barkers_equation():
    # R1: on a parabola the time from periapsis to true anomaly 90 deg is
    # (1/2) sqrt(p^3 / mu) (1 + 1/3) = (2/3) sqrt(p^3 / mu). 1e-9 relative.
    rp = 7000e3
    r0 = np.array([rp, 0.0, 0.0])
    v0 = np.array([0.0, np.sqrt(2 * EARTH_MU / rp), 0.0])
    out = ellipse_from_rv(r0, v0, num=401, f_span=np.pi / 2)
    p = 2 * rp
    f = np.linspace(-np.pi / 2, np.pi / 2, 401)
    t_from_periapsis = out["t_rel"] - out["t_rel"][200]
    assert f[-1] == pytest.approx(np.pi / 2)
    assert t_from_periapsis[-1] == pytest.approx((2 / 3) * np.sqrt(p**3 / EARTH_MU), rel=1e-9)
