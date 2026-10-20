"""Final-behavior validation of J2 secular nodal precession."""

import numpy as np

from ssatk.accelerations_6dof import SpacecraftAccelJ2
from ssatk.constants import EARTH_MU, EARTH_RADIUS, J2_wgs
from ssatk.orbital_mechanics import kepler_to_state
from ssatk.propagators_6dof import propagate_6dof_high_accuracy


def _raan(trajectory):
    h = np.cross(trajectory.r, trajectory.v)
    node = np.column_stack((-h[:, 1], h[:, 0]))
    return np.unwrap(np.arctan2(node[:, 1], node[:, 0]))


def test_j2_raan_precession_matches_first_order_rate_and_converges():
    a, eccentricity, inclination = 7_000_000.0, 0.01, np.deg2rad(63.0)
    r0, v0 = kepler_to_state(a, eccentricity, inclination, 0.2, 0.3, 0.1)
    period = 2.0 * np.pi * np.sqrt(a**3 / EARTH_MU)
    times = np.linspace(0.0, 20.0 * period, 401)
    rate = -1.5 * J2_wgs * np.sqrt(EARTH_MU / a**3)
    rate *= (EARTH_RADIUS / (a * (1.0 - eccentricity**2))) ** 2 * np.cos(inclination)
    expected = rate * times[-1]

    results = []
    for max_step in (120.0, 30.0):
        results.append(
            propagate_6dof_high_accuracy(
                r0=r0,
                v0=v0,
                times=times,
                inertia=np.eye(3),
                acceleration=SpacecraftAccelJ2(),
                max_step=max_step,
            )
        )

    errors = [abs(_raan(result)[-1] - _raan(result)[0] - expected) for result in results]
    assert errors[1] < 3.0e-4
    assert abs((_raan(results[0])[-1] - _raan(results[0])[0]) - (
        _raan(results[1])[-1] - _raan(results[1])[0]
    )) < 1.0e-9
