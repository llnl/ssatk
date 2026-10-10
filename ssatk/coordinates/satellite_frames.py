"""Common satellite-operations coordinate frame matrices.

All matrices returned here have columns equal to the frame unit axes expressed
in GCRF/inertial coordinates. Therefore ``matrix @ vector_in_frame`` rotates a
vector into GCRF, and ``matrix.T @ vector_gcrf`` rotates a GCRF vector into the
requested frame.
"""

from __future__ import annotations

import numpy as np


SATELLITE_FRAME_ALIASES = {
    "gcrf": "gcrf",
    "eci": "gcrf",
    "inertial": "gcrf",
    "ntw": "ntw",
    "rtn": "rtn",
    "rsw": "rtn",
    "ric": "rtn",
    "lvlh": "rtn",
    "vnb": "vnb",
    "body": "body",
    "spacecraft": "body",
    "sc": "body",
    "nadir": "nadir_velocity",
    "nadir_velocity": "nadir_velocity",
    "velocity_nadir": "nadir_velocity",
    "enu": "enu",
    "ned": "ned",
    "sez": "sez",
    "los": "los",
    "target": "los",
    "sun": "sun",
    "sun_pointing": "sun",
}


def frame_to_gcrf_matrix(
    frame: str,
    *,
    r=None,
    v=None,
    q=None,
    lat=None,
    lon=None,
    degrees: bool = True,
    origin=None,
    target=None,
    up_hint=(0.0, 0.0, 1.0),
) -> np.ndarray:
    """Return a 3x3 matrix rotating ``frame`` components into GCRF/ECI.

    Supported frames are:

    - ``gcrf``/``eci``/``inertial``: identity.
    - ``ntw``: SSAPy ``[N, T, W]``; ``T`` follows velocity, ``W = r × v``.
    - ``rtn``/``rsw``/``ric``/``lvlh``: radial, transverse, orbit-normal.
    - ``vnb``: velocity, orbit-normal, binormal.
    - ``body``: quaternion ``q=[w, x, y, z]`` body-to-GCRF.
    - ``nadir_velocity``: +X velocity, +Y opposite orbit normal, +Z nadir.
    - ``enu``/``ned``/``sez``: local tangent frames at geodetic ``lat``/``lon``.
    - ``los``/``target``/``sun``: +X points from ``origin`` to ``target`` and
      ``up_hint`` defines the projected +Z direction.
    """

    name = _canonical_frame(frame)
    if name == "gcrf":
        return np.eye(3)
    if name == "ntw":
        return ntw_to_gcrf_matrix(_vector3(r, "r"), _vector3(v, "v"))
    if name == "rtn":
        return rtn_to_gcrf_matrix(r, v)
    if name == "vnb":
        return vnb_to_gcrf_matrix(r, v)
    if name == "body":
        return body_to_gcrf_matrix(q)
    if name == "nadir_velocity":
        return nadir_velocity_to_gcrf_matrix(r, v)
    if name == "enu":
        return enu_to_ecef_matrix(lat, lon, degrees=degrees)
    if name == "ned":
        return ned_to_ecef_matrix(lat, lon, degrees=degrees)
    if name == "sez":
        return sez_to_ecef_matrix(lat, lon, degrees=degrees)
    if name in {"los", "sun"}:
        return los_to_gcrf_matrix(origin, target, up_hint=up_hint)
    raise ValueError(f"unsupported satellite frame {frame!r}")


def gcrf_to_frame_matrix(frame: str, **kwargs) -> np.ndarray:
    """Return a 3x3 matrix rotating GCRF/ECI components into ``frame``."""

    return frame_to_gcrf_matrix(frame, **kwargs).T


def transform_to_gcrf(vector, frame: str, **kwargs) -> np.ndarray:
    """Rotate a vector from a satellite operations frame into GCRF/ECI."""

    return frame_to_gcrf_matrix(frame, **kwargs) @ _vector3(vector, "vector")


def transform_from_gcrf(vector, frame: str, **kwargs) -> np.ndarray:
    """Rotate a vector from GCRF/ECI into a satellite operations frame."""

    return gcrf_to_frame_matrix(frame, **kwargs) @ _vector3(vector, "vector")


def ntw_to_gcrf_matrix(r, v):
    """Return SSAPy NTW-to-GCRF matrix with columns ``[N, T, W]``."""

    r = _vector3(r, "r")
    v = _vector3(v, "v")
    e_t = _unit(v, "v")
    e_w = _unit(np.cross(r, v), "r cross v")
    e_n = np.cross(e_t, e_w)
    return np.column_stack((e_n, e_t, e_w))


def ntw_to_gcrf(delta_v_ntw, r_center, v_center):
    """Rotate an SSAPy ``[N, T, W]`` vector into GCRF."""

    return ntw_to_gcrf_matrix(r_center, v_center) @ _vector3(delta_v_ntw, "delta_v_ntw")


def gcrf_to_ntw(delta_v_gcrf, r_center, v_center):
    """Rotate a GCRF vector into SSAPy ``[N, T, W]`` coordinates."""

    return ntw_to_gcrf_matrix(r_center, v_center).T @ _vector3(delta_v_gcrf, "delta_v_gcrf")


def rtn_to_gcrf_matrix(r, v) -> np.ndarray:
    """Return RTN/RSW/RIC/LVLH-to-GCRF matrix with columns ``[R, T, N]``."""

    r_hat = _unit(_vector3(r, "r"), "r")
    h_hat = _unit(np.cross(r, v), "r cross v")
    t_hat = np.cross(h_hat, r_hat)
    return np.column_stack((r_hat, t_hat, h_hat))


def vnb_to_gcrf_matrix(r, v) -> np.ndarray:
    """Return VNB-to-GCRF matrix with columns ``[V, N, B]``."""

    v_hat = _unit(_vector3(v, "v"), "v")
    n_hat = _unit(np.cross(r, v), "r cross v")
    b_hat = np.cross(v_hat, n_hat)
    return np.column_stack((v_hat, n_hat, b_hat))


def nadir_velocity_to_gcrf_matrix(r, v) -> np.ndarray:
    """Return body-like nadir-pointing matrix: +X velocity, +Y -normal, +Z nadir."""

    r_hat = _unit(_vector3(r, "r"), "r")
    v_hat = _unit(_vector3(v, "v"), "v")
    h_hat = _unit(np.cross(r, v), "r cross v")
    return np.column_stack((v_hat, -h_hat, -r_hat))


def body_to_gcrf_matrix(q) -> np.ndarray:
    """Return quaternion body-to-GCRF direction cosine matrix.

    The quaternion convention is ``[w, x, y, z]`` and matches
    :mod:`ssatk.propagators_6dof`.
    """

    w, x, y, z = _unit(_vector4(q, "q"), "q")
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def gcrf_to_body_matrix(q) -> np.ndarray:
    """Return GCRF-to-body direction cosine matrix for ``q=[w, x, y, z]``."""

    return body_to_gcrf_matrix(q).T


def enu_to_ecef_matrix(lat, lon, *, degrees: bool = True) -> np.ndarray:
    """Return ENU-to-ECEF matrix with columns ``[east, north, up]``."""

    lat, lon = _lat_lon(lat, lon, degrees)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    east = np.array([-sin_lon, cos_lon, 0.0])
    north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
    up = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])
    return np.column_stack((east, north, up))


def ecef_to_enu_matrix(lat, lon, *, degrees: bool = True) -> np.ndarray:
    """Return ECEF-to-ENU matrix."""

    return enu_to_ecef_matrix(lat, lon, degrees=degrees).T


def ned_to_ecef_matrix(lat, lon, *, degrees: bool = True) -> np.ndarray:
    """Return NED-to-ECEF matrix with columns ``[north, east, down]``."""

    enu = enu_to_ecef_matrix(lat, lon, degrees=degrees)
    return np.column_stack((enu[:, 1], enu[:, 0], -enu[:, 2]))


def ecef_to_ned_matrix(lat, lon, *, degrees: bool = True) -> np.ndarray:
    """Return ECEF-to-NED matrix."""

    return ned_to_ecef_matrix(lat, lon, degrees=degrees).T


def sez_to_ecef_matrix(lat, lon, *, degrees: bool = True) -> np.ndarray:
    """Return SEZ-to-ECEF matrix with columns ``[south, east, zenith]``."""

    enu = enu_to_ecef_matrix(lat, lon, degrees=degrees)
    return np.column_stack((-enu[:, 1], enu[:, 0], enu[:, 2]))


def ecef_to_sez_matrix(lat, lon, *, degrees: bool = True) -> np.ndarray:
    """Return ECEF-to-SEZ matrix."""

    return sez_to_ecef_matrix(lat, lon, degrees=degrees).T


def los_to_gcrf_matrix(origin, target, *, up_hint=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Return line-of-sight frame matrix with +X to target and +Z near up_hint."""

    x_hat = _unit(_vector3(target, "target") - _vector3(origin, "origin"), "target - origin")
    z_seed = _vector3(up_hint, "up_hint")
    z_hat = z_seed - np.dot(z_seed, x_hat) * x_hat
    if np.linalg.norm(z_hat) <= 1e-15:
        z_hat = _fallback_perpendicular(x_hat)
    z_hat = _unit(z_hat, "projected up_hint")
    y_hat = np.cross(z_hat, x_hat)
    return np.column_stack((x_hat, y_hat, z_hat))


def _canonical_frame(frame: str) -> str:
    key = str(frame).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in SATELLITE_FRAME_ALIASES:
        raise ValueError(f"unsupported satellite frame {frame!r}")
    return SATELLITE_FRAME_ALIASES[key]


def _lat_lon(lat, lon, degrees: bool) -> tuple[float, float]:
    if lat is None or lon is None:
        raise ValueError("lat and lon are required for local tangent frames")
    lat = float(lat)
    lon = float(lon)
    if degrees:
        lat = np.radians(lat)
        lon = np.radians(lon)
    return lat, lon


def _vector3(value, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector")
    return vector


def _vector4(value, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (4,):
        raise ValueError(f"{name} must be a 4-vector")
    return vector


def _unit(vector, name: str) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm <= 1e-15:
        raise ValueError(f"{name} must be non-zero")
    return np.asarray(vector, dtype=float) / norm


def _fallback_perpendicular(vector: np.ndarray) -> np.ndarray:
    axis = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(vector, axis)) > 0.9:
        axis = np.array([0.0, 1.0, 0.0])
    return axis - np.dot(axis, vector) * vector
