import numpy as np
import pytest

from ssatk.navigation import GroundStation
from ssatk.ssa.observer import (
    ObserverStates,
    earth_occulted,
    from_ground_station,
    observer_metrics,
)


def test_observer_metrics_parity_with_ground_station_si_and_enu():
    """C: parity with the existing GroundStation prediction model."""
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


def test_observer_states_and_metrics_match_astropy_gcrs_reference():
    """R2: Astropy EarthLocation/GCRS with the packaged IERS-A table.

    The established core Earth-rotation budgets are 0.02 m in position and
    2e-4 m/s in velocity for the measured dX/dY model difference. The 3e-9 rad
    basis/angle budget covers the same route difference; the observed maximum
    elevation residual is 2.109e-9 rad. Range and range-rate use independent
    closed forms in metres and metres per second.
    """
    from astropy import units as u
    from astropy.coordinates import CartesianRepresentation, EarthLocation, GCRS, ITRS
    from astropy.time import Time
    from astropy.utils import iers

    from ssatk.data import data_path

    station = GroundStation(-106.5, 35.1, elevation_m=1600.0)
    times = np.array([1.4e9, 1.4e9 + 10.0])
    observer = from_ground_station(station, times)
    epoch = Time(times, format="gps")
    location = EarthLocation.from_geodetic(
        station.lon_deg * u.deg,
        station.lat_deg * u.deg,
        station.elevation_m * u.m,
        ellipsoid="WGS84",
    )
    with data_path("environment/eop/finals2000A.all") as path:
        table = iers.IERS_A.open(str(path))
    with iers.conf.set_temp("auto_download", False), iers.earth_orientation_table.set(table):
        position, velocity = location.get_gcrs_posvel(epoch)
        np.testing.assert_allclose(
            observer.positions,
            position.xyz.to_value(u.m).T,
            rtol=0,
            atol=0.02,
        )
        np.testing.assert_allclose(
            observer.velocities,
            velocity.xyz.to_value(u.m / u.s).T,
            rtol=0,
            atol=2e-4,
        )

        offset = np.array([1_000.0, 2_000.0, 3_000.0])
        relative_velocity = np.array([4.0, -5.0, 6.0])
        metrics = observer_metrics(
            observer.positions + offset,
            observer.velocities + relative_velocity,
            observer,
        )
        latitude = np.radians(station.lat_deg)
        longitude = np.radians(station.lon_deg)
        enu_itrf = np.array([
            [-np.sin(longitude), np.cos(longitude), 0.0],
            [-np.sin(latitude) * np.cos(longitude),
             -np.sin(latitude) * np.sin(longitude), np.cos(latitude)],
            [np.cos(latitude) * np.cos(longitude),
             np.cos(latitude) * np.sin(longitude), np.sin(latitude)],
        ])
        reference_basis = []
        for time in epoch:
            itrs_basis = ITRS(
                CartesianRepresentation(*(enu_itrf.T * u.m)),
                obstime=time,
            )
            basis = itrs_basis.transform_to(GCRS(obstime=time)).cartesian.xyz
            reference_basis.append(basis.to_value(u.m).T)
        reference_basis = np.asarray(reference_basis)
        np.testing.assert_allclose(
            observer.enu_basis,
            reference_basis,
            rtol=0,
            atol=3e-9,
        )
        local = np.einsum(
            "tij,tj->ti", reference_basis, np.broadcast_to(offset, (times.size, 3))
        )
        reference_azimuth = np.mod(
            np.arctan2(local[:, 0], local[:, 1]), 2.0 * np.pi
        )
        reference_elevation = np.arctan2(
            local[:, 2], np.hypot(local[:, 0], local[:, 1])
        )
        np.testing.assert_allclose(
            metrics["azimuth_rad"],
            reference_azimuth,
            rtol=0,
            atol=3e-9,
        )
        np.testing.assert_allclose(
            metrics["elevation_rad"],
            reference_elevation,
            rtol=0,
            atol=3e-9,
        )
        expected_range = np.linalg.norm(offset)
        expected_rate = np.dot(offset, relative_velocity) / expected_range
        np.testing.assert_allclose(
            metrics["range_m"],
            np.full(times.shape, expected_range),
            rtol=0,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            metrics["range_rate_m_s"],
            np.full(times.shape, expected_rate),
            rtol=0,
            atol=1e-12,
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
