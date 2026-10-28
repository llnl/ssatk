"""State-provider references: analytic circular two-body motion in SI."""

import numpy as np

import ssatk
from ssatk.constants import EARTH_MU
from ssatk.ssa.trajectories import StateBatch, ArrayStateProvider, OrbitStateProvider


def test_chunked_orbits_against_circular_closed_form():
    radius = 7e6
    epoch = 1.4e9
    times = epoch + np.arange(7) * 30.0
    orbit = ssatk.Orbit.fromKeplerianElements(
        a=radius, e=0, i=0, pa=0, raan=0, trueAnomaly=0, t=epoch,
    )
    provider = OrbitStateProvider({"synthetic": orbit}, times,
                                  propagator=ssatk.KeplerianPropagator())
    chunks = list(provider.iter_chunks(2))
    angle = np.sqrt(EARTH_MU / radius**3) * (times - epoch)
    expected_r = radius * np.column_stack((np.cos(angle), np.sin(angle), np.zeros(7)))
    speed = np.sqrt(EARTH_MU / radius)
    expected_v = speed * np.column_stack((-np.sin(angle), np.cos(angle), np.zeros(7)))
    # Analytic circular orbit, 10 micrometres / 10 nanometres per second.
    np.testing.assert_allclose(np.concatenate([c.positions[0] for c in chunks]),
                               expected_r, rtol=0, atol=1e-5)
    np.testing.assert_allclose(np.concatenate([c.velocities[0] for c in chunks]),
                               expected_v, rtol=0, atol=1e-8)
    np.testing.assert_array_equal(np.concatenate([c.times for c in chunks]), times)


def test_precomputed_provider_retains_si_states_and_invalid_sample_mask():
    times = np.arange(5, dtype=float)
    r = np.stack((times, 2 * times, 3 * times), axis=-1)[None]
    v = np.broadcast_to([1., 2., 3.], r.shape).copy()
    r[0, 3] = np.nan
    batch = StateBatch(("A",), times, r, v)
    provider = ArrayStateProvider(batch)
    # Closed-form straight-line positions; exact 0 m tolerance including NaN.
    np.testing.assert_allclose(np.concatenate([c.positions for c in provider.iter_chunks(2)], axis=1),
                               r, rtol=0, atol=0, equal_nan=True)
    np.testing.assert_array_equal(batch.valid, [[True, True, True, False, True]])
    r[0, 0] = 99
    np.testing.assert_array_equal(batch.positions[0, 0], [0., 0., 0.])
