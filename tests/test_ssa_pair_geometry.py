import numpy as np

from ssatk.coordinates.satellite_frames import rtn_to_gcrf_matrix
from ssatk.ssa.pair_geometry import pair_metrics


def test_pair_metrics_circular_rtn_and_sky_geometry():
    result = pair_metrics(
        [7_000_000.0, 0.0, 0.0], [0.0, 7_500.0, 0.0],
        [7_000_000.0, 1_000.0, 2_000.0], [0.0, 7_500.0, 10.0],
        [0.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(result["distance_m"], np.sqrt(5_000_000.0), rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(result["relative_speed_m_s"], 10.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        result["relative_position_rtn_m"], [0.0, 1_000.0, 2_000.0], rtol=0.0, atol=1e-9
    )
    np.testing.assert_allclose(
        result["relative_velocity_rtn_m_s"], [0.0, 0.0, 10.0], rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        result["velocity_cosine"], 7_500.0 / np.sqrt(7_500.0**2 + 10.0**2),
        rtol=0.0, atol=1e-14,
    )
    expected_angle = np.arctan2(np.sqrt(5_000_000.0), 7_000_000.0)
    np.testing.assert_allclose(result["angular_separation_rad"], expected_angle, rtol=0.0, atol=1e-15)


def test_pair_metrics_broadcasts_and_isolates_degenerate_quantities():
    result = pair_metrics(
        np.zeros((2, 1, 3)), np.ones(3), np.ones((1, 3)), [0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0],
    )
    assert result["distance_m"].shape == (2, 1)
    assert np.isfinite(result["distance_m"]).all()
    assert np.isfinite(result["relative_speed_m_s"]).all()
    assert np.isnan(result["relative_position_rtn_m"]).all()
    assert np.isnan(result["along_track_angle_rad"]).all()


def test_pair_metrics_vectorized_rtn_matches_public_frame_helper():
    position = np.array([7_000_000.0, 0.0, 0.0])
    velocity = np.array([0.0, 7_500.0, 0.0])
    delta = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = pair_metrics(
        position, velocity, position + delta, np.broadcast_to(velocity, delta.shape),
        np.zeros(3),
    )
    expected = delta @ rtn_to_gcrf_matrix(position, velocity)
    np.testing.assert_allclose(result["relative_position_rtn_m"], expected, rtol=0.0, atol=1e-9)


def test_pair_metrics_scalar_and_moving_observer_apparent_direction():
    result = pair_metrics(
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0], [0.0, 2.0, 0.0],
    )
    assert result["distance_m"].shape == ()
    assert result["along_track_angle_rad"].shape == ()
    np.testing.assert_allclose(result["along_track_angle_rad"], -np.pi / 4, rtol=0.0, atol=1e-15)


def test_pair_metrics_velocity_cosine_and_small_sky_angle_closed_forms():
    small_angle = 1.0e-10
    result = pair_metrics(
        [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [np.cos(small_angle), np.sin(small_angle), 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(result["velocity_cosine"], 0.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        result["angular_separation_rad"], small_angle, atol=1e-20, rtol=0.0
    )


def test_pair_metrics_broadcasts_observer_velocity_and_rejects_radial_tangent():
    result = pair_metrics(
        np.array([[[1.0, 0.0, 0.0]]]),
        np.array([[[0.0, 1.0, 0.0]]]),
        np.array([[[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]]),
        np.zeros((1, 2, 3)),
        np.zeros(3),
        np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    assert result["distance_m"].shape == (1, 2)
    assert np.isnan(result["along_track_angle_rad"]).all()
