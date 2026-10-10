"""Thrust profiles and packaged thrust curves against closed forms and certified impulses."""

import numpy as np
import pytest

from ssatk.accelerations_6dof.thrust import (
    ThrustCurve,
    integrated_thrust_impulse,
    load_packaged_thrust_curve,
    thrust_profile_exponential,
    thrust_profile_pulsed,
    thrust_profile_smoothstep,
    thrust_profile_trapezoid,
)

THRUST_N = 400.0


@pytest.mark.parametrize("make", [thrust_profile_trapezoid, thrust_profile_smoothstep], ids=["linear", "smoothstep"])
def test_ramped_profiles_deliver_the_closed_form_impulse(make):
    # R1: both a linear ramp and a smoothstep ramp average half the thrust over
    # the ramp, so I = F (T - (t_rise + t_fall) / 2). 1e-6 relative with 200,001
    # trapezoid samples.
    profile = make(THRUST_N, start=5.0, burn_time=60.0, rise_time=4.0, fall_time=10.0)
    impulse = integrated_thrust_impulse(profile, 0.0, 70.0, samples=200_001)
    assert impulse == pytest.approx(THRUST_N * (60.0 - 7.0), rel=1e-6)
    assert profile(5.0 + 2.0) == pytest.approx(THRUST_N / 2.0, rel=1e-12)  # mid-ramp, both shapes


def test_exponential_profile_matches_first_order_rise_and_decay():
    # R1: F(1 - e^(-t/tau_r)) on [0, T] integrates to F (T - tau_r (1 - e^(-T/tau_r)));
    # the decay tail F(T) e^(-(t - T)/tau_d) adds F(T) tau_d. 1e-5 relative.
    tau_r, tau_d, burn = 2.0, 3.0, 20.0
    profile = thrust_profile_exponential(THRUST_N, start=0.0, stop=burn, rise_tau=tau_r, decay_tau=tau_d)
    at_stop = THRUST_N * (1 - np.exp(-burn / tau_r))
    expected = THRUST_N * (burn - tau_r * (1 - np.exp(-burn / tau_r))) + at_stop * tau_d
    impulse = integrated_thrust_impulse(profile, 0.0, burn + 40 * tau_d, samples=400_001)
    assert impulse == pytest.approx(expected, rel=1e-5)
    assert profile(tau_r) == pytest.approx(THRUST_N * (1 - np.exp(-1.0)), rel=1e-12)


def test_pulsed_profile_delivers_duty_cycle_fraction():
    # R1: over 10 whole periods a duty cycle d delivers d F t. 1e-4 relative.
    profile = thrust_profile_pulsed(THRUST_N, period=6.0, duty_cycle=0.25, start=0.0, stop=60.0)
    impulse = integrated_thrust_impulse(profile, 0.0, 60.0, samples=600_001)
    assert impulse == pytest.approx(0.25 * THRUST_N * 60.0, rel=1e-4)


def test_tabulated_curve_total_impulse_is_exact_for_piecewise_linear_thrust():
    # R1: a triangle of height F and base T has impulse F T / 2; linear
    # interpolation reproduces it exactly (1e-12 relative).
    curve = ThrustCurve([0.0, 3.0, 8.0], [0.0, THRUST_N, 0.0])
    assert curve.total_impulse == pytest.approx(THRUST_N * 8.0 / 2.0, rel=1e-12)
    assert curve(1.5) == pytest.approx(THRUST_N / 2.0, rel=1e-12)


@pytest.mark.parametrize(
    "simfile_id, certified_impulse_ns",
    [
        ("5f923f0a1bca5800041716ae", 8.96),   # AeroTech C3.4T
        ("5f923e731bca58000417164f", 27.24),  # Estes E12
        ("5f923edb1bca5800041716ab", 49.61),  # Estes F15
        ("5f9241441bca580004171726", 136.6),  # AeroTech G80T
    ],
)
def test_packaged_motor_curves_deliver_their_certified_total_impulse(simfile_id, certified_impulse_ns):
    # R3: NAR-certified total impulse as listed on ThrustCurve.org. The packaged
    # RockSim curves integrate to within 1 %.
    curve = load_packaged_thrust_curve(simfile_id, collection="thrustcurve_org_pd")
    assert curve.total_impulse == pytest.approx(certified_impulse_ns, rel=0.01)
