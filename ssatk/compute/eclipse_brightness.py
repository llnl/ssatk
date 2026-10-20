"""Finite-Sun illumination and Moon-shadow footprints on the WGS-84 Earth.

Distances are kilometres. Ephemerides and frame transforms come from
:mod:`ssatk.compute.eclipse_runtime`; the disc and ray geometry from
:mod:`ssatk.compute.eclipse_core`.
"""
from __future__ import annotations


import numpy as np

from ssatk.compute.eclipse_core import (
    finite_source_irradiance,
    finite_source_visibility,
    ray_ellipsoid_intersections as _core_ray_ellipsoid_intersections,
    ray_sphere_intersections as _core_ray_sphere_intersections,
)
from ssatk.compute.eclipse_core import LimbDarkeningLaw, resolve_limb_darkening
from ssatk.compute.eclipse_reference_events import AU_KM, EARTH_AXES_KM, R_SUN_KM, RE_KM


def _sun_position(sun_hat, D_km, sun_position_km):
    if sun_position_km is not None:
        return np.asarray(sun_position_km, dtype=float)
    if sun_hat is None:
        raise TypeError("Provide sun_hat or sun_position_km")
    sh = np.asarray(sun_hat, dtype=float)
    norm = np.linalg.norm(sh, axis=-1, keepdims=True)
    if np.any(norm <= np.finfo(float).tiny):
        raise ValueError("sun_hat must be nonzero")
    # A unit vector is scaled to D_km; a full position vector passes through.
    return sh / norm * np.asarray(D_km, dtype=float)[..., None] if np.nanmedian(norm) < 10.0 else sh


def illumination_fraction(r_eval_km, sun_hat=None, R_body_km=None, R_sun_km=R_SUN_KM, D_km=AU_KM,
                          *, sun_position_km=None):
    """Uniform-disc visible fraction of the Sun from ``r_eval_km`` past a body at the origin."""
    if R_body_km is None:
        raise TypeError("R_body_km is required")
    source = _sun_position(sun_hat, D_km, sun_position_km)
    return finite_source_visibility(r_eval_km, source, R_body_km, R_sun_km)


def irradiance_fraction(r_eval_km, sun_hat=None, R_body_km=None, R_sun_km=R_SUN_KM, D_km=AU_KM,
                        *, sun_position_km=None, photometry: str | LimbDarkeningLaw = "quadratic-visible",
                        quadrature_order: int = 48):
    """Limb-darkened visible solar-flux fraction; see :func:`illumination_fraction`."""
    if R_body_km is None:
        raise TypeError("R_body_km is required")
    source = _sun_position(sun_hat, D_km, sun_position_km)
    return finite_source_irradiance(
        r_eval_km, source, R_body_km, R_sun_km,
        law=resolve_limb_darkening(photometry), quadrature_order=quadrature_order,
    )


def ray_sphere_intersections(origin, direction, center, radius_km):
    """Sorted non-negative ray parameters where the ray meets a sphere."""
    roots = np.asarray(_core_ray_sphere_intersections(origin, direction, radius_km, center=center)).reshape(-1)
    return np.sort(roots[np.isfinite(roots) & (roots >= 0.0)])


def ray_ellipsoid_intersections(origin, direction, center, axes_km=EARTH_AXES_KM):
    """Sorted non-negative ray parameters where the ray meets an axis-aligned ellipsoid."""
    roots = np.asarray(_core_ray_ellipsoid_intersections(origin, direction, axes_km, center=center)).reshape(-1)
    return np.sort(roots[np.isfinite(roots) & (roots >= 0.0)])


def shadow_axis_surface_point(occluder_position_km, sun_position_km, target_center_km=np.zeros(3),
                              target_radius_km=RE_KM, target_axes_km=None):
    """Nearest target-surface hit of the Sun-occluder axis, or ``None``."""
    occ = np.asarray(occluder_position_km, dtype=float)
    target = np.asarray(target_center_km, dtype=float)
    d = occ - np.asarray(sun_position_km, dtype=float)
    d /= np.linalg.norm(d)
    roots = (ray_ellipsoid_intersections(occ, d, target, target_axes_km)
             if target_axes_km is not None
             else ray_sphere_intersections(occ, d, target, target_radius_km))
    return None if roots.size == 0 else occ + roots[0] * d


def ellipsoid_surface_in_direction(direction, axes_km=EARTH_AXES_KM):
    """Point where a ray from the center along ``direction`` meets the ellipsoid."""
    d = np.asarray(direction, dtype=float)
    norm = np.sqrt(np.sum((d / np.asarray(axes_km, dtype=float)) ** 2, axis=-1, keepdims=True))
    return d / np.maximum(norm, np.finfo(float).eps)
