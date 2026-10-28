import numpy as np
import pytest

from ssatk.navigation import GroundStation
from ssatk.ssa.observer import (
    ObserverStates,
    earth_occulted,
    from_ground_station,
    observer_metrics,
)


def test_observer_metrics_matches_ground_station_si_and_enu():
    station = GroundStation(0.0, 0.0)
    times = np.array([0.0, 10.0])
    observer = from_ground_station(station, times)
    target = observer.positions + np.array([1_000.0, 2_000.0, 3_000.0])
    target_velocity = observer.velocities + np.array([4.0, -5.0, 6.0])
    metrics = observer_metrics(target, target_velocity, observer)
    for index, time in enumerate(times):
        prediction = station.predict(
            np.r_[target[index], target_velocity[index]], time, "range"
        )
        assert metrics["range_m"][index] == pytest.approx(prediction.value, abs=1e-8)
        assert metrics["range_rate_m_s"][index] == pytest.approx(
            station.predict(np.r_[target[index], target_velocity[index]], time, "range_rate").value,
            abs=1e-12,
        )
        np.testing.assert_allclose(
            [metrics["azimuth_rad"][index], metrics["elevation_rad"][index]],
            station.predict(np.r_[target[index], target_velocity[index]], time, "az_el").value,
            atol=1e-12,
            rtol=0,
        )


def test_observer_states_validate_time_and_spherical_occultation():
    with pytest.raises(ValueError, match="strictly increasing"):
        ObserverStates([1.0, 1.0], [[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]])
    assert earth_occulted([[-2.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]], radius_m=1.0)[0]
    assert not earth_occulted([[2.0, 2.0, 0.0]], [[2.0, 0.0, 0.0]], radius_m=1.0)[0]


def test_observer_states_copy_inputs_and_reject_reflected_enu():
    times = np.array([1.0, 2.0])
    positions = np.zeros((2, 3))
    velocities = np.zeros((2, 3))
    basis = np.broadcast_to(np.eye(3), (2, 3, 3)).copy()
    observer = ObserverStates(times, positions, velocities, basis)
    times[0] = 99.0
    positions[0, 0] = 99.0
    basis[0, 0, 0] = 99.0
    assert observer.times[0] == 1.0
    assert observer.positions[0, 0] == 0.0
    assert observer.enu_basis[0, 0, 0] == 1.0
    reflected = np.broadcast_to(np.eye(3), (2, 3, 3)).copy()
    reflected[:, 0, 0] = -1.0
    with pytest.raises(ValueError, match="right-handed"):
        ObserverStates([1.0, 2.0], positions, velocities, reflected)
