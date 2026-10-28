"""Bounded pair geometry in an A-centered GCRF frame."""

from __future__ import annotations

import numpy as np

def _state_array(value, name):
    value = np.asarray(value, dtype=float)
    if value.ndim == 0 or value.shape[-1:] != (3,):
        raise ValueError(f"{name} must have shape (..., 3).")
    return value


def _broadcast_states(*values):
    arrays = [_state_array(value, name) for value, name in values]
    try:
        shape = np.broadcast_shapes(*(array.shape[:-1] for array in arrays))
    except ValueError as exc:
        raise ValueError("state arrays must have broadcast-compatible leading shapes.") from exc
    return [np.broadcast_to(array, shape + (3,)) for array in arrays], shape


def _nan_array(shape, components=()):
    return np.full(shape + components, np.nan, dtype=float)


def _safe_unit(vector):
    norm = np.linalg.norm(vector, axis=-1)
    valid = np.all(np.isfinite(vector), axis=-1) & np.isfinite(norm) & (norm > 0.0)
    unit = _nan_array(vector.shape[:-1], (3,))
    np.divide(vector, norm[..., None], out=unit, where=valid[..., None])
    return unit, norm, valid


def pair_metrics(
    position_a,
    velocity_a,
    position_b,
    velocity_b,
    observer_position,
    observer_velocity=None,
):
    """Return broadcast pair and apparent-sky geometry in SI GCRF coordinates.

    The RTN frame is centered on A and has columns ``[R, T, N]``.  A
    singular A state, zero-length line of sight, or zero apparent velocity
    makes only the affected derived quantities NaN; finite scalar separation
    and relative speed metrics remain available. Cross-track sky angle uses
    ``atan2(cross, hypot(radial, along))``.
    """

    state_values = [
        (position_a, "position_a"),
        (velocity_a, "velocity_a"),
        (position_b, "position_b"),
        (velocity_b, "velocity_b"),
        (observer_position, "observer_position"),
    ]
    if observer_velocity is not None:
        state_values.append((observer_velocity, "observer_velocity"))
    states, shape = _broadcast_states(*state_values)
    r_a, v_a, r_b, v_b, r_observer = states[:5]
    if observer_velocity is None:
        v_observer = np.zeros(shape + (3,), dtype=float)
    else:
        v_observer = states[5]

    dr = r_b - r_a
    dv = v_b - v_a
    distance_m = np.linalg.norm(dr, axis=-1)
    relative_speed_m_s = np.linalg.norm(dv, axis=-1)
    distance_m = np.where(np.isfinite(distance_m), distance_m, np.nan)
    relative_speed_m_s = np.where(np.isfinite(relative_speed_m_s), relative_speed_m_s, np.nan)
    velocity_a_hat, _, velocity_a_valid = _safe_unit(v_a)
    velocity_b_hat, _, velocity_b_valid = _safe_unit(v_b)
    cosine = _nan_array(shape)
    cosine_valid = velocity_a_valid & velocity_b_valid
    cosine[cosine_valid] = np.einsum(
        "...i,...i->...", velocity_a_hat, velocity_b_hat
    )[cosine_valid]
    cosine[cosine_valid] = np.clip(cosine[cosine_valid], -1.0, 1.0)

    relative_position_rtn = _nan_array(shape, (3,))
    relative_velocity_rtn = _nan_array(shape, (3,))
    r_norm = np.linalg.norm(r_a, axis=-1)
    r_hat = np.divide(r_a, r_norm[..., None], out=_nan_array(shape, (3,)), where=np.isfinite(r_norm[..., None]) & (r_norm[..., None] > 0.0))
    h = np.cross(r_a, v_a)
    h_norm = np.linalg.norm(h, axis=-1)
    n_hat = np.divide(h, h_norm[..., None], out=_nan_array(shape, (3,)), where=np.isfinite(h_norm[..., None]) & (h_norm[..., None] > 0.0))
    t_hat = np.cross(n_hat, r_hat)
    frame_valid = np.all(np.isfinite(r_hat * n_hat * t_hat), axis=-1)
    position_valid = frame_valid & np.all(np.isfinite(dr), axis=-1)
    velocity_valid = frame_valid & np.all(np.isfinite(dv), axis=-1)
    relative_position_rtn[position_valid] = np.stack(
        [np.sum(dr * r_hat, axis=-1), np.sum(dr * t_hat, axis=-1), np.sum(dr * n_hat, axis=-1)],
        axis=-1,
    )[position_valid]
    relative_velocity_rtn[velocity_valid] = np.stack(
        [np.sum(dv * r_hat, axis=-1), np.sum(dv * t_hat, axis=-1), np.sum(dv * n_hat, axis=-1)],
        axis=-1,
    )[velocity_valid]

    los_a = r_a - r_observer
    los_b = r_b - r_observer
    u_a, _, los_a_valid = _safe_unit(los_a)
    u_b, _, los_b_valid = _safe_unit(los_b)
    sky_valid = los_a_valid & los_b_valid
    angular_separation_rad = _nan_array(shape)
    angular_separation_rad[sky_valid] = np.arctan2(
        np.linalg.norm(np.cross(los_a, los_b), axis=-1)[sky_valid],
        np.einsum("...i,...i->...", los_a, los_b)[sky_valid],
    )

    apparent_velocity = v_a - v_observer
    apparent_tangent = apparent_velocity - np.sum(apparent_velocity * u_a, axis=-1)[..., None] * u_a
    apparent_speed = np.linalg.norm(apparent_velocity, axis=-1)
    tangent_norm = np.linalg.norm(apparent_tangent, axis=-1)
    along_hat, _, along_valid = _safe_unit(apparent_tangent)
    along_valid &= np.isfinite(apparent_speed) & (tangent_norm > np.finfo(float).eps * apparent_speed)
    cross_hat = np.cross(u_a, along_hat)
    sky_angle_valid = sky_valid & along_valid
    along_track_angle_rad = _nan_array(shape)
    cross_track_angle_rad = _nan_array(shape)
    along_track_angle_rad[sky_angle_valid] = np.arctan2(
        np.sum(u_b * along_hat, axis=-1)[sky_angle_valid],
        np.sum(u_b * u_a, axis=-1)[sky_angle_valid],
    )
    radial_component = np.sum(u_b * u_a, axis=-1)
    along_component = np.sum(u_b * along_hat, axis=-1)
    cross_track_angle_rad[sky_angle_valid] = np.arctan2(
        np.sum(u_b * cross_hat, axis=-1)[sky_angle_valid],
        np.hypot(radial_component, along_component)[sky_angle_valid],
    )

    return {
        "distance_m": distance_m,
        "relative_speed_m_s": relative_speed_m_s,
        "velocity_cosine": cosine,
        "angular_separation_rad": angular_separation_rad,
        "relative_position_rtn_m": relative_position_rtn,
        "relative_velocity_rtn_m_s": relative_velocity_rtn,
        "along_track_angle_rad": along_track_angle_rad,
        "cross_track_angle_rad": cross_track_angle_rad,
    }
