"""Exact impact-plane finite-Sun geometry for the validated eclipses.

This module supplies the readable local geometry that the interactive dashboard
previously lacked.  The key distinction is that the plotted plane is the
*impact plane*: it contains the Sun--occluder shadow axis and the target centre.
A north-referenced plane generally does not contain the target centre when the
shadow axis has both north/south and east/west miss components, so projecting
that geometry into a north-only cross-section hides part of the real impact
parameter and makes otherwise correct tangent rays look displaced.

Coordinates used here
---------------------
+X
    Downstream, from the Sun through the occluder toward the target.

+Z
    From the shadow axis toward the target centre at the target-centre plane.

Y
    Normal to the impact plane.  Source, occluder and target centres have
    numerically zero Y in the exact peak-event cross-section.

The local origin is the point on the shadow axis closest to the target centre.
Therefore the shadow axis is Z=0 and the target centre is at X=0, Z=b, where b
is the full two-dimensional axis miss (impact parameter), not only its north
component.
"""
from __future__ import annotations

import math

import numpy as np

from ssatk.compute.eclipse_reference_events import (
    EARTH_AXES_KM,
    R_MOON_MEAN_KM,
    solar_central_line_wgs84,
    _unit,
)
from ssatk.compute.eclipse_raytrace import (
    ReferenceRayBundle,
)


COLORS = {
    "background": "#05070b",
    "panel": "#080d15",
    "grid": "#2a3444",
    "text": "#e8edf4",
    "muted": "#aab7c7",
    "sunlight": "#ffe36e",
    "axis": "#d66cff",
    "umbra": "#ff6c5c",
    "penumbra": "#8bc8ff",
    "earth": "#63adff",
    "earth_fill": "#193c61",
    "moon": "#dedede",
    "moon_fill": "#4e5158",
    "optical": "#f4f0ff",
    "impact": "#66e0c2",
    "target_plane": "#71839a",
}


def impact_plane_basis(bundle: ReferenceRayBundle) -> tuple[np.ndarray, np.ndarray, float]:
    """Return origin, right-handed basis and full target-axis miss.

    The origin is the closest point on the shadow axis to the target centre.
    The positive local Z direction points from the axis toward the target
    centre.  This makes the impact parameter explicit and keeps all three
    body centres in the local X-Z plane.
    """
    axis = _unit(bundle.axis_hat)
    occluder = (
        np.asarray(bundle.moon_center_km, dtype=float)
        if bundle.mode == "solar"
        else np.asarray(bundle.earth_center_km, dtype=float)
    )
    target = (
        np.asarray(bundle.earth_center_km, dtype=float)
        if bundle.mode == "solar"
        else np.asarray(bundle.moon_center_km, dtype=float)
    )
    axial_distance = float(np.dot(target - occluder, axis))
    origin = occluder + axial_distance * axis
    miss = target - origin
    impact = float(np.linalg.norm(miss))
    if impact > 1.0e-10:
        z_axis = miss / impact
    else:
        north = np.array([0.0, 0.0, 1.0])
        z_axis = north - axis * float(np.dot(north, axis))
        if np.linalg.norm(z_axis) < 1.0e-10:
            north = np.array([0.0, 1.0, 0.0])
            z_axis = north - axis * float(np.dot(north, axis))
        z_axis = _unit(z_axis)
    y_axis = _unit(np.cross(z_axis, axis))
    z_axis = _unit(np.cross(axis, y_axis))
    basis = np.stack([axis, y_axis, z_axis], axis=1)
    return origin, basis, impact


def _first_sphere_hit(start, direction, center, radius) -> np.ndarray | None:
    direction = _unit(direction)
    offset = np.asarray(start, dtype=float) - np.asarray(center, dtype=float)
    b = float(np.dot(offset, direction))
    c = float(np.dot(offset, offset) - float(radius) ** 2)
    disc = b * b - c
    if disc < 0.0:
        return None
    root = math.sqrt(max(disc, 0.0))
    for distance in sorted((-b - root, -b + root)):
        if distance > 1.0e-7:
            return np.asarray(start, dtype=float) + distance * direction
    return None


def shadow_axis_world(bundle: ReferenceRayBundle, jd_value: float) -> np.ndarray:
    """Blocked-light centreline, with no segment through the occluder."""
    axis = _unit(bundle.axis_hat)
    if bundle.mode == "solar":
        occluder = np.asarray(bundle.moon_center_km, dtype=float)
        start = occluder + axis * float(bundle.solid_occluder_radius_km)
        center = solar_central_line_wgs84(float(jd_value))
        if center is not None:
            end = np.asarray(center[2], dtype=float)
        else:
            distance = max(
                float(np.dot(bundle.earth_center_km - start, axis)), 0.0
            )
            end = start + distance * axis
    else:
        occluder = np.asarray(bundle.earth_center_km, dtype=float)
        # Start on the actual downstream WGS-84 surface in the ray
        # direction, not at the equatorial radius regardless of latitude.
        directional_radius = 1.0 / math.sqrt(
            float(np.sum((axis / EARTH_AXES_KM) ** 2))
        )
        start = occluder + axis * directional_radius
        end = _first_sphere_hit(
            start, axis, bundle.moon_center_km, R_MOON_MEAN_KM
        )
        if end is None:
            distance = max(
                float(np.dot(bundle.moon_center_km - start, axis)), 0.0
            )
            end = start + distance * axis
    return np.vstack([start, end])


__all__ = [
    "impact_plane_basis",
    "shadow_axis_world",
]
