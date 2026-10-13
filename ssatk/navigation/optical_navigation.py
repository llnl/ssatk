"""Reusable star/landmark navigation and Cartesian orbit-filter integration.

Inputs are calibrated bearings, not pixel images. Star references and landmark
positions must describe the observation epoch in geocentric GCRF-aligned axes.
Stars determine attitude; identified finite-distance landmarks determine
position. Velocity requires a prior and a time sequence with a dynamics model.
"""

from dataclasses import dataclass

import numpy as np

from .ekf import CartesianMeasurement, CartesianOrbitEKF, EKFState
from .landmarks import position_from_landmark_bearings
from .star_tracker import _unit_rows, fit_star_attitude, match_star_directions


def _epoch(time):
    value = np.asarray(time.gps if hasattr(time, "gps") else time, dtype=float)
    if isinstance(time, (bool, np.bool_)) or value.ndim or not np.isfinite(value):
        raise ValueError("time must be a finite scalar GPS epoch or scalar Astropy Time")
    return float(value)


def _rotation(value, name):
    value = np.asarray(value, dtype=float)
    if (value.shape != (3, 3) or not np.isfinite(value).all()
            or not np.allclose(value.T @ value, np.eye(3), atol=1e-10, rtol=0)
            or not np.isclose(np.linalg.det(value), 1., atol=1e-10, rtol=0)):
        raise ValueError(f"{name} must be a proper 3x3 rotation")
    return value


def _separation(a, b):
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.einsum("ij,ij->i", a, b))


@dataclass(frozen=True)
class OpticalNavigationFix:
    """One epoch's geometric position/attitude fix and angular diagnostics.

    ``time`` is GPS seconds; ``position_gcrf_m`` is in metres. The matrix maps
    star-camera column vectors into GCRF-aligned axes. Indices address the
    supplied detection/reference arrays, not catalogue IDs. ``fit_mask`` and
    ``star_residual_rad`` follow matched-star order, including withheld stars.
    No velocity or measurement covariance is inferred from a single frame.
    """

    time: float
    position_gcrf_m: np.ndarray
    camera_to_gcrf: np.ndarray
    star_indices: np.ndarray
    catalogue_indices: np.ndarray
    fit_mask: np.ndarray
    star_residual_rad: np.ndarray
    landmark_residual_rad: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "time", _epoch(self.time))
        position = np.asarray(self.position_gcrf_m, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("position_gcrf_m must be a finite three-vector")
        rotation = _rotation(self.camera_to_gcrf, "camera_to_gcrf")
        n = np.asarray(self.star_indices).size
        arrays = dict(position_gcrf_m=position, camera_to_gcrf=rotation)
        for name in ("star_indices", "catalogue_indices"):
            indices = np.asarray(getattr(self, name))
            if (indices.shape != (n,) or indices.dtype.kind not in "iu" or n < 2
                    or np.any(indices < 0) or len(np.unique(indices)) != n):
                raise ValueError(f"{name} must contain at least two distinct nonnegative integer indices")
            arrays[name] = indices
        mask = np.asarray(self.fit_mask)
        if mask.shape != (n,) or mask.dtype.kind != "b" or mask.sum() < 2:
            raise ValueError("fit_mask must select at least two matched stars")
        arrays["fit_mask"] = mask
        for name in ("star_residual_rad", "landmark_residual_rad"):
            residual = np.asarray(getattr(self, name), dtype=float)
            if (residual.ndim != 1 or not np.isfinite(residual).all()
                    or np.any(residual < 0) or np.any(residual > np.pi)
                    or (name == "star_residual_rad" and residual.size != n)
                    or (name == "landmark_residual_rad" and residual.size < 2)):
                raise ValueError(f"{name} must contain finite angular separations in [0, pi]")
            arrays[name] = residual
        for name, value in arrays.items():
            value = np.array(value, copy=True)
            value.flags.writeable = False
            object.__setattr__(self, name, value)


def solve_optical_navigation(
    camera_star_bearings,
    reference_star_directions,
    landmark_positions_gcrf_m,
    camera_landmark_bearings,
    *,
    time,
    prior_camera_to_gcrf,
    match_gate_rad,
    fit_star_mask=None,
    landmark_camera_to_star_camera=None,
) -> OpticalNavigationFix:
    """Match stars, solve attitude, and triangulate a geometric position fix.

    All bearings point outward from the observer. Reference star directions
    should already include the caller's epoch/observer corrections, e.g. via
    ``catalogue_directions``. Landmarks are identified finite-distance points
    in the same inertial frame and epoch, in metres; Earth-fixed coordinates
    must be transformed first. No simulated identities or truth state enter.

    ``fit_star_mask`` is a boolean mask over *input detections*. False entries
    may be matched for held-out residuals but do not fit attitude. By default
    all matches fit. The optional calibration maps landmark-camera column
    vectors to star-camera axes; identity means bearings already share axes.
    Matching is tracking mode and requires a coarse attitude prior.
    """
    time = _epoch(time)
    stars = _unit_rows(camera_star_bearings, "camera_star_bearings")
    reference = _unit_rows(reference_star_directions, "reference_star_directions")
    landmarks = np.asarray(landmark_positions_gcrf_m, dtype=float)
    bearings = _unit_rows(camera_landmark_bearings, "camera_landmark_bearings")
    calibration = np.eye(3) if landmark_camera_to_star_camera is None else _rotation(
        landmark_camera_to_star_camera, "landmark_camera_to_star_camera")
    if fit_star_mask is None:
        mask = np.ones(len(stars), dtype=bool)
    else:
        mask = np.asarray(fit_star_mask)
        if mask.shape != (len(stars),) or mask.dtype.kind != "b":
            raise ValueError("fit_star_mask must be a boolean mask over input detections")
    detected, found = match_star_directions(stars, reference, prior_camera_to_gcrf, match_gate_rad)
    fit_mask = mask[detected]
    attitude = fit_star_attitude(stars[detected[fit_mask]], reference[found[fit_mask]])
    inertial_bearings = bearings @ calibration.T @ attitude.T
    position = position_from_landmark_bearings(landmarks, inertial_bearings)
    predicted = _unit_rows(landmarks - position, "landmark offsets")
    return OpticalNavigationFix(
        time, position, attitude, detected, found, fit_mask,
        _separation(stars[detected] @ attitude.T, reference[found]),
        _separation(inertial_bearings, predicted),
    )


def update_orbit_from_optical_fix(
    orbit_filter: CartesianOrbitEKF,
    fix: OpticalNavigationFix,
    position_covariance_m2,
    *,
    propagation_kwargs=None,
) -> EKFState:
    """Fuse a position fix into a six-state GCRF orbit estimate for guidance.

    Propagate with ``CartesianOrbitEKF.predict_orbit`` to the fix epoch, then
    update position. The returned ``EKFState.x`` is [r (m), v (m/s)] at GPS
    ``time``; covariance follows the same state order. Successive fixes and
    dynamics constrain velocity; one image does not measure it directly.

    The required 3x3 covariance is positive definite in GCRF metres squared.
    It must include attitude, camera calibration and landmark uncertainty,
    not just star-fit residuals. The filter assumes independent measurement
    errors across updates. The caller must assess residuals and observability;
    this adapter does not model shared biases or select station-keeping burns.
    ``propagation_kwargs`` accepts predict_orbit's force/noise settings except
    ``time``. Failed input, prediction or update leaves the filter unchanged.
    """
    if not isinstance(orbit_filter, CartesianOrbitEKF) or orbit_filter.state.x.shape != (6,):
        raise TypeError("orbit_filter must be a six-state CartesianOrbitEKF")
    if not isinstance(fix, OpticalNavigationFix):
        raise TypeError("fix must be an OpticalNavigationFix")
    current_time = _epoch(orbit_filter.state.time)
    if fix.time < current_time:
        raise ValueError("optical fix precedes the current filter epoch")
    covariance = np.asarray(position_covariance_m2, dtype=float)
    if (covariance.shape != (3, 3) or not np.isfinite(covariance).all()
            or not np.allclose(covariance, covariance.T, rtol=0, atol=1e-12)):
        raise ValueError("position covariance must be a finite symmetric positive-definite 3x3 matrix")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("position covariance must be positive-definite") from exc
    kwargs = {} if propagation_kwargs is None else dict(propagation_kwargs)
    if "time" in kwargs:
        raise ValueError("propagation_kwargs cannot override the fix time")
    working = CartesianOrbitEKF(EKFState(orbit_filter.state.x, orbit_filter.state.covariance, current_time))
    if fix.time > current_time:
        working.predict_orbit(time=fix.time, **kwargs)
    updated = working.update(fix.position_gcrf_m, CartesianMeasurement((0, 1, 2)), covariance, time=fix.time)
    orbit_filter.state = updated
    return updated


__all__ = ["OpticalNavigationFix", "solve_optical_navigation", "update_orbit_from_optical_fix"]
