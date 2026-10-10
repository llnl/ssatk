"""Final-behavior parity checks against the installed SSAPy implementation."""

import numpy as np

from ssatk.accelerations_6dof import wrap_ssapy_acceleration
from ssatk.constants import EARTH_MU
from ssatk.propagators_6dof import propagate_6dof


def test_real_ssapy_accel_kepler_matches_ssatk_adapter():
    from ssapy.accel import AccelKepler

    r = np.array([7_000_000.0, -1_200_000.0, 800_000.0])
    v = np.array([1_200.0, 7_300.0, -500.0])
    ssapy_accel = AccelKepler(EARTH_MU)
    adapter = wrap_ssapy_acceleration(ssapy_accel)

    np.testing.assert_allclose(
        adapter(r, v, 123.0), ssapy_accel(r, v, 123.0), rtol=0.0, atol=1e-15
    )


def test_real_ssapy_accel_const_ntw_matches_ssatk_adapter():
    from ssapy.accel import AccelConstNTW

    r = np.array([7_000_000.0, 1_000_000.0, 400_000.0])
    v = np.array([-900.0, 7_400.0, 1_100.0])
    ssapy_accel = AccelConstNTW([2.5e-6, -1.25e-6, 3.0e-6])
    adapter = wrap_ssapy_acceleration(ssapy_accel)

    np.testing.assert_allclose(adapter(r, v, 123.0), ssapy_accel(r, v, 123.0))
    np.testing.assert_array_equal(adapter.time_breakpoints, ssapy_accel.time_breakpoints)


def test_real_ssapy_scipy_propagator_matches_ssatk_two_body():
    from ssapy import Orbit
    from ssapy.accel import AccelKepler
    from ssapy.propagator import SciPyPropagator

    r0 = np.array([7_000_000.0, 0.0, 0.0])
    v0 = np.array([0.0, 7_500.0, 0.0])
    times = np.linspace(0.0, 900.0, 10)
    # SSAPy and SSATK both use SI units for numerical state vectors.
    orbit = Orbit(r0, v0, 0.0, mu=EARTH_MU)
    expected = orbit.at(times, propagator=SciPyPropagator(
        AccelKepler(EARTH_MU), ode_kwargs={"rtol": 1e-11, "atol": 1e-9}
    ))
    actual = propagate_6dof(
        times=times, r0=r0, v0=v0, q0=[1.0, 0.0, 0.0, 0.0],
        omega0=[0.0, 0.0, 0.0], inertia=np.eye(3),
        rtol=1e-11, atol=1e-3,
    )

    np.testing.assert_allclose(actual.r, expected.r, rtol=2e-10, atol=2e-3)
    np.testing.assert_allclose(actual.v, expected.v, rtol=2e-10, atol=2e-6)
