"""Terminal endpoints at absolute GPS epochs."""

import numpy as np
import pytest

from ssatk.propagators_6dof import propagate_6dof

BASE = {
    "r0": [7.0e6, 0.0, 0.0],
    "v0": [0.0, 7546.0, 0.0],
    "q0": [1.0, 0.0, 0.0, 0.0],
    "omega0": [0.0, 0.0, 0.1],
    "mu": 0.0,
    "inertia": np.diag([4.0, 5.0, 6.0]),
}


def _terminal_at(root):
    def event(t, y):
        return t - root
    event.terminal = True
    return event


@pytest.mark.parametrize("epoch", [0.0, 1.4e9, 3.8e9], ids=["t0", "gps2024", "gps2100"])
def test_root_one_ulp_past_a_sample_collapses_onto_the_stopping_epoch(epoch):
    interior = epoch + 5.0
    root = np.nextafter(interior, np.inf)
    trajectory = propagate_6dof(
        times=np.array([epoch, interior, epoch + 10.0]), t0=epoch,
        events=(_terminal_at(root),), rtol=1e-12, atol=1e-14, **BASE)
    times = np.asarray(trajectory.t, dtype=float)
    assert trajectory.status == 1
    assert times.size == 2
    assert float(times[1]) - float(times[0]) > 1.0
    assert times[-1] == pytest.approx(root, abs=np.spacing(interior))
    assert np.all(np.diff(times) > 1.0e-6)


def test_requested_sample_just_before_the_root_survives_at_small_epochs():
    trajectory = propagate_6dof(
        times=np.array([0.0, 1.25 - 5.0e-7, 2.0]), t0=0.0,
        events=(_terminal_at(1.25),), rtol=1e-12, atol=1e-14, **BASE)
    times = np.asarray(trajectory.t, dtype=float)
    assert times.size == 3
    assert times[-1] == pytest.approx(1.25, abs=1.0e-9)
    assert times[-1] - times[-2] == pytest.approx(5.0e-7, rel=1e-6)
