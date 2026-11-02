"""The remaining adaptive propagators integrate in elapsed time too.

`propagate_orbit_state`, its STM variant, `propagate_6dof_variational` and
`propagate_6dof_extended` all handed SciPy an absolute span and an absolute
``t_eval``, so they carried the same ``min_step = 10 * ulp(t)`` floor that
`propagate_6dof` did.
"""

import numpy as np
import pytest

from ssatk.constants import EARTH_MU
from ssatk.propagators_orbit.high_accuracy import (
    propagate_orbit_state,
    propagate_orbit_state_with_stm,
)

EPOCHS = pytest.mark.parametrize(
    "epoch", [0.0, 1.0e8, 1.4e9, 3.8e9], ids=["t0", "gps1983", "gps2024", "gps2100"]
)

R0 = [7.0e6, 0.0, 0.0]
V0 = [0.0, float(np.sqrt(EARTH_MU / 7.0e6)), 0.0]


@pytest.mark.parametrize("kind", ["orbit", "stm", "sixdof"])
@pytest.mark.parametrize("epoch", [0.0, 1.4e9])
def test_interior_burn_matches_integrated_rectangular_pulse(kind, epoch):
    """R1: 1 m/s² during [50,51] gives 1 m/s and 49.5 m at t=100 s."""
    from ssatk.accelerations_6dof import make_finite_burn_acceleration, thrust_profile_constant
    from ssatk.propagators_6dof import propagate_6dof

    burn = make_finite_burn_acceleration(
        thrust_profile_constant(1, start=epoch + 50, stop=epoch + 51),
        mass=1, frame="gcrf", direction=[1, 0, 0],
    )
    function = {"orbit": propagate_orbit_state, "stm": propagate_orbit_state_with_stm,
                "sixdof": propagate_6dof}[kind]
    options = {"inertia": np.eye(3)} if kind == "sixdof" else {}
    result = function(r0=[1, 0, 0], v0=[0, 0, 0], t0=epoch,
                      times=epoch + np.array([0, 100]), mu=0, acceleration=burn, **options)
    np.testing.assert_allclose(result.v[-1], [1, 0, 0], atol=1e-6, rtol=0)  # m/s
    np.testing.assert_allclose(result.r[-1], [50.5, 0, 0], atol=1e-4, rtol=0)  # m


@pytest.mark.parametrize("kind", ["orbit", "sixdof"])
@pytest.mark.parametrize("bounds", [(1.0, 2.0), (-1.0, 0.0)])
def test_zero_duration_burn_at_propagation_endpoint_has_no_impulse(kind, bounds):
    """R1: a unit burn starting at final or ending at initial time has zero impulse."""
    from ssatk.accelerations_6dof import make_finite_burn_acceleration, thrust_profile_constant
    from ssatk.propagators_6dof import propagate_6dof

    burn = make_finite_burn_acceleration(
        thrust_profile_constant(1.0, start=bounds[0], stop=bounds[1]),
        mass=1.0, frame="gcrf", direction=[1, 0, 0],
    )
    function = {"orbit": propagate_orbit_state, "sixdof": propagate_6dof}[kind]
    options = {"inertia": np.eye(3)} if kind == "sixdof" else {}
    result = function(
        r0=[0, 0, 0], v0=[0, 0, 0], t0=0.0, times=[0.0, 1.0],
        mu=0.0, acceleration=burn, **options,
    )
    np.testing.assert_allclose(result.v[-1], [0, 0, 0], atol=1e-12, rtol=0)  # m/s
    np.testing.assert_allclose(result.r[-1], [0, 0, 0], atol=1e-12, rtol=0)  # m


@EPOCHS
def test_orbit_state_is_bit_identical_across_epochs(epoch):
    def run(reference):
        # Integer-second offsets are exactly representable at GPS magnitude.
        return propagate_orbit_state(
            r0=R0, v0=V0, t0=reference,
            times=reference + np.arange(0.0, 5880.0, 60.0), mu=EARTH_MU,
        )

    base = run(0.0)
    shifted = run(epoch)
    np.testing.assert_array_equal(shifted.r, base.r)
    np.testing.assert_array_equal(shifted.v, base.v)
    np.testing.assert_array_equal(shifted.t - epoch, base.t)


@EPOCHS
def test_stm_propagation_is_bit_identical_across_epochs(epoch):
    def run(reference):
        return propagate_orbit_state_with_stm(
            r0=R0, v0=V0, t0=reference,
            times=reference + np.arange(0.0, 1800.0, 60.0), mu=EARTH_MU,
        )

    base = run(0.0)
    shifted = run(epoch)
    np.testing.assert_array_equal(shifted.r, base.r)
    np.testing.assert_array_equal(shifted.stm, base.stm)
    np.testing.assert_array_equal(shifted.t - epoch, base.t)


@EPOCHS
def test_acceleration_models_still_receive_absolute_epochs(epoch):
    seen: list[float] = []

    def probe(r, v, t):
        seen.append(float(t))
        return np.zeros(3)

    propagate_orbit_state(
        r0=R0, v0=V0, t0=epoch, times=epoch + np.array([0.0, 10.0]),
        mu=EARTH_MU, acceleration=probe,
    )
    assert seen
    assert min(seen) >= epoch - 1.0e-6
    assert max(seen) <= epoch + 10.0 + 1.0e-6
