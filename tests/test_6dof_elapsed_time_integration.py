"""``propagate_6dof`` integrates in elapsed time, not absolute GPS seconds.

SciPy's Runge-Kutta steppers bound the minimum step at ``10 * ulp(t)``. On
absolute GPS seconds that is 2.4e-6 s at GPS 1.4e9 and 4.8e-6 s at GPS 3.8e9,
against roughly 5e-322 s at zero, so a discontinuity needing a finer step
aborted the run with "Required step size is less than spacing between
numbers". Referencing the integration to ``t0`` restores the resolution while
keeping every callback and every returned epoch absolute.
"""

import numpy as np
import pytest

from ssatk.accelerations_6dof.thrust import SpacecraftManeuverAccel
from ssatk.constants import EARTH_MU
from ssatk.propagators_6dof.sixdof import Spacecraft, propagate_6dof
from ssatk.satellites import SpacecraftBody, Tank

# GPS seconds. 1.4e9 is 2024-05-13; 3.8e9 is 2100-06-20.
EPOCHS = pytest.mark.parametrize(
    "epoch", [0.0, 1.0e8, 1.4e9, 3.8e9], ids=["t0", "gps1983", "gps2024", "gps2100"]
)

R0 = np.array([7.0e6, 0.0, 0.0])
V0 = np.array([0.0, np.sqrt(EARTH_MU / 7.0e6), 0.0])


def _depleting_burn(epoch):
    """A tank that empties mid-arc: the discontinuity that used to abort."""
    body = SpacecraftBody.box(name="bus", mass=10.0, size=(1.0, 1.0, 1.0)).with_tanks(
        Tank(propellant_mass=0.02, dry_mass=1.0, name="main", position_body=[2.0, 0.0, 0.0]),
        Tank(propellant_mass=0.5, dry_mass=1.0, name="aux", position_body=[0.0, 2.0, 0.0]),
    )
    spacecraft = Spacecraft(r=[7.0e6, 0.0, 0.0], v=[0.0, 0.0, 0.0], t=epoch, body=body)
    burn = SpacecraftManeuverAccel(
        10.0, frame="gcrf", direction=[1.0, 0.0, 0.0], isp=100.0, tank_name="main"
    )
    return spacecraft.propagate(
        times=epoch + np.linspace(0.0, 4.0, 9), models=[burn], mu=0.0
    )


@EPOCHS
def test_tank_depletion_resolves_at_any_epoch(epoch):
    """Raised RuntimeError at GPS 1.4e9 and 3.8e9 before this change."""
    trajectory = _depleting_burn(epoch)
    assert trajectory.mass[-1] == pytest.approx(
        trajectory.mass[0] - 0.02, rel=0.0, abs=1.0e-8
    )


@EPOCHS
def test_trajectory_is_bit_identical_across_epochs(epoch):
    """The strongest available statement: the epoch no longer perturbs the state.

    Before this change a one-revolution Keplerian arc differed by 1.8e-2 m at
    GPS 1.4e9 and 2.0e-2 m at GPS 3.8e9 against the same arc at zero.
    """
    def run(reference):
        # Integer-second offsets, so ``reference + offset`` is exact at GPS
        # magnitude and bit-identity is a fair thing to assert.
        return propagate_6dof(
            r0=R0, v0=V0, t0=reference,
            times=reference + np.arange(0.0, 5880.0, 60.0),
            inertia=np.eye(3), mu=EARTH_MU, omega0=[0.0, 0.0, 0.1],
        )

    base = run(0.0)
    shifted = run(epoch)
    np.testing.assert_array_equal(shifted.r, base.r)
    np.testing.assert_array_equal(shifted.v, base.v)
    np.testing.assert_array_equal(shifted.q, base.q)
    np.testing.assert_array_equal(shifted.t - epoch, base.t)


@EPOCHS
def test_force_models_still_receive_absolute_epochs(epoch):
    seen: list[float] = []

    def probe(t, _r, _v, _q, _omega):
        seen.append(float(t))
        return np.zeros(3)

    propagate_6dof(
        r0=R0, v0=V0, t0=epoch, times=[epoch, epoch + 10.0],
        inertia=np.eye(3), mu=0.0, acceleration=probe,
    )
    assert seen
    assert min(seen) >= epoch - 1.0e-6
    assert max(seen) <= epoch + 10.0 + 1.0e-6


@EPOCHS
def test_events_are_defined_and_reported_in_absolute_time(epoch):
    def crosses_quarter_metre(t, y):
        assert t >= epoch - 1.0e-6  # the callback sees absolute epochs
        return y[0] - 0.25

    crosses_quarter_metre.terminal = True
    crosses_quarter_metre.direction = 1

    trajectory = propagate_6dof(
        r0=[0.0, 0.0, 0.0], v0=[1.0, 0.0, 0.0], t0=epoch,
        times=[epoch, epoch + 1.0], inertia=np.eye(3), mu=0.0,
        events=crosses_quarter_metre, dense_output=True,
    )

    assert trajectory.status == 1
    assert trajectory.t[-1] == pytest.approx(epoch + 0.25, rel=0.0, abs=1.0e-6)
    assert trajectory.t_events[0][0] == pytest.approx(epoch + 0.25, rel=0.0, abs=1.0e-6)
    np.testing.assert_allclose(trajectory.r[-1], [0.25, 0.0, 0.0], atol=1.0e-9)
