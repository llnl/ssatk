import numpy as np
import pytest

from ssatk.coordinates.satellite_frames import (
    body_to_gcrf_matrix,
    ecef_to_enu_matrix,
    ecef_to_ned_matrix,
    ecef_to_sez_matrix,
    enu_to_ecef_matrix,
    frame_to_gcrf_matrix,
    gcrf_to_body_matrix,
    gcrf_to_frame_matrix,
    nadir_velocity_to_gcrf_matrix,
    ned_to_ecef_matrix,
    ntw_to_gcrf_matrix,
    rtn_to_gcrf_matrix,
    sez_to_ecef_matrix,
    transform_from_gcrf,
    transform_to_gcrf,
    vnb_to_gcrf_matrix,
)


def test_orbit_relative_frames_match_standard_circular_case():
    r = np.array([7000.0, 0.0, 0.0])
    v = np.array([0.0, 7.5, 0.0])

    np.testing.assert_allclose(frame_to_gcrf_matrix("gcrf"), np.eye(3))
    np.testing.assert_allclose(frame_to_gcrf_matrix("eci"), np.eye(3))
    np.testing.assert_allclose(frame_to_gcrf_matrix("ntw", r=r, v=v), np.eye(3))
    np.testing.assert_allclose(frame_to_gcrf_matrix("rtn", r=r, v=v), np.eye(3))
    np.testing.assert_allclose(frame_to_gcrf_matrix("rsw", r=r, v=v), np.eye(3))
    np.testing.assert_allclose(frame_to_gcrf_matrix("lvlh", r=r, v=v), np.eye(3))
    np.testing.assert_allclose(frame_to_gcrf_matrix("ric", r=r, v=v), np.eye(3))

    np.testing.assert_allclose(
        frame_to_gcrf_matrix("ntw", r=r, v=v),
        ntw_to_gcrf_matrix(r, v),
    )
    np.testing.assert_allclose(transform_to_gcrf([1.0, 2.0, 3.0], "ntw", r=r, v=v), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(transform_from_gcrf([1.0, 2.0, 3.0], "rtn", r=r, v=v), [1.0, 2.0, 3.0])


def test_ntw_and_rtn_are_distinct_for_radial_velocity():
    r = np.array([7000.0, 0.0, 0.0])
    v = np.array([1.0, 7.5, 0.0])

    ntw = frame_to_gcrf_matrix("ntw", r=r, v=v)
    rtn = rtn_to_gcrf_matrix(r, v)

    assert not np.allclose(ntw, rtn)
    np.testing.assert_allclose(ntw.T @ ntw, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(rtn.T @ rtn, np.eye(3), atol=1e-12)


def test_vnb_and_nadir_velocity_frames_are_right_handed():
    r = np.array([7000.0, 0.0, 0.0])
    v = np.array([0.0, 7.5, 0.0])

    vnb = vnb_to_gcrf_matrix(r, v)
    np.testing.assert_allclose(vnb[:, 0], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(vnb[:, 1], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(vnb[:, 2], [1.0, 0.0, 0.0])
    assert np.linalg.det(vnb) == pytest.approx(1.0)

    nadir = nadir_velocity_to_gcrf_matrix(r, v)
    np.testing.assert_allclose(nadir[:, 0], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(nadir[:, 1], [0.0, 0.0, -1.0])
    np.testing.assert_allclose(nadir[:, 2], [-1.0, 0.0, 0.0])
    assert np.linalg.det(nadir) == pytest.approx(1.0)


def test_body_quaternion_frame_matches_dynamics_convention():
    q_z90 = [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]
    matrix = body_to_gcrf_matrix(q_z90)

    np.testing.assert_allclose(matrix @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(gcrf_to_body_matrix(q_z90) @ [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(gcrf_to_frame_matrix("body", q=q_z90), matrix.T)


def test_local_tangent_frames_at_greenwich_equator():
    enu = enu_to_ecef_matrix(0.0, 0.0)
    np.testing.assert_allclose(enu[:, 0], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(enu[:, 1], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(enu[:, 2], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(ecef_to_enu_matrix(0.0, 0.0), enu.T)

    ned = ned_to_ecef_matrix(0.0, 0.0)
    np.testing.assert_allclose(ned[:, 0], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(ned[:, 1], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(ned[:, 2], [-1.0, 0.0, 0.0])
    np.testing.assert_allclose(ecef_to_ned_matrix(0.0, 0.0), ned.T)

    sez = sez_to_ecef_matrix(0.0, 0.0)
    np.testing.assert_allclose(sez[:, 0], [0.0, 0.0, -1.0])
    np.testing.assert_allclose(sez[:, 1], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(sez[:, 2], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(ecef_to_sez_matrix(0.0, 0.0), sez.T)
