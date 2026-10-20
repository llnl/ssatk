"""Integrated physical Sun-Earth-Moon eclipse animation.

This module renders a single Plotly 3-D scene with two precision-safe display
modes driven by one physical state:

* ``Earth-Moon ray trace`` uses a floating Earth-centred origin in kilometres.
  Earth rotates in GCRF, the Moon follows the validated event ephemeris, and
  finite-Sun rays are clipped only at the local display boundary or the first
  opaque body surface.
* ``Heliocentric true scale`` uses astronomical units.  The Sun is at the
  origin, Earth moves along its real heliocentric event arc, the Moon follows
  Earth, and the complete source-to-target ray paths are shown without moving
  the Sun closer or enlarging the physical bodies.  Optional locator markers
  remain explicitly non-physical screen-space aids.

The split is a floating-origin rendering strategy, not a change to the
physics.  Browser WebGL normally stores vertex coordinates as float32; at
1 AU this gives roughly 10--20 km coordinate granularity, too coarse for a
kilometre-level lunar tangent inspection.  Both modes are generated from the
same double-precision ray bundle and are synchronized frame by frame.
"""
from __future__ import annotations

from dataclasses import dataclass


import numpy as np

from ssatk.compute.eclipse_runtime import gcrf_to_itrf_km, itrf_to_gcrf_km, ssapy_ephemeris_positions_km
from ssatk.compute.eclipse_state import event_positions_gcrf
from ssatk.compute.eclipse_local_finite_geometry import impact_plane_basis, shadow_axis_world
from ssatk.compute.eclipse_reference_events import (
    AU_KM,
    RE_KM,
    ReferenceDefinition,
    ReferenceEvent,
    _unit,
)
from ssatk.compute.eclipse_raytrace import (
    ReferenceRayBundle,
    tangent_cross_section_paths,
    trace_event_rays,
    trace_reference_rays,
)


COLORS = {
    "bg": "#02040a",
    "grid": "#263246",
    "text": "#eef3fa",
    "muted": "#9fb0c4",
    "sun": "#ffe36e",
    "axis": "#d86cff",
    "umbra": "#ff6a5d",
    "penumbra": "#79c9ff",
    "orbit": "#6da6ff",
    "moon_orbit": "#aab8c8",
    "spin": "#63e0c4",
    "greenwich": "#55f0da",
    "footprint": "#ffd36d",
}

QUALITY = {
    "balanced": {
        "animation_body": (45, 90),
        "animation_atmosphere": (29, 56),
        "animation_shadow": 61,
        "peak_body": (121, 240),
        "peak_atmosphere": (61, 120),
        "peak_shadow": 241,
    },
    "high": {
        "animation_body": (61, 120),
        "animation_atmosphere": (37, 72),
        "animation_shadow": 91,
        "peak_body": (181, 360),
        "peak_atmosphere": (91, 180),
        "peak_shadow": 301,
    },
    "ultra": {
        "animation_body": (81, 160),
        "animation_atmosphere": (49, 96),
        "animation_shadow": 121,
        "peak_body": (241, 480),
        "peak_atmosphere": (121, 240),
        "peak_shadow": 361,
    },
}


@dataclass(frozen=True)
class InertialFrameState:
    event: ReferenceEvent
    jd_utc: float
    bundle_native: ReferenceRayBundle
    sun_gcrf_km: np.ndarray
    earth_gcrf_km: np.ndarray
    moon_gcrf_km: np.ndarray
    sun_heliocentric_km: np.ndarray
    earth_heliocentric_km: np.ndarray
    moon_heliocentric_km: np.ndarray
    central_gcrf_km: np.ndarray
    shadow_axis_gcrf_km: np.ndarray
    umbra_gcrf_km: tuple[np.ndarray, np.ndarray]
    penumbra_gcrf_km: tuple[np.ndarray, np.ndarray]
    umbra_bundle_gcrf_km: tuple[np.ndarray, ...]
    penumbra_bundle_gcrf_km: tuple[np.ndarray, ...]
    target_hits: dict[str, tuple[bool, bool]]


@dataclass(frozen=True)
class SceneGeometry:
    event: ReferenceEvent
    basis_world_from_scene: np.ndarray
    reference_cross_axis_gcrf: np.ndarray
    orbit_au: np.ndarray
    event_earth_au: np.ndarray
    event_moon_au: np.ndarray
    event_moon_local_km: np.ndarray
    local_xrange_km: tuple[float, float]
    local_transverse_km: float
    earth_peak_local_km: np.ndarray
    moon_peak_local_km: np.ndarray


def _quality(name: str, *, animated: bool) -> tuple[tuple[int, int], tuple[int, int], int]:
    key = str(name).lower().strip()
    if key not in QUALITY:
        raise ValueError("quality must be 'balanced', 'high', or 'ultra'")
    profile = QUALITY[key]
    prefix = "animation" if animated else "peak"
    return (
        tuple(profile[f"{prefix}_body"]),
        tuple(profile[f"{prefix}_atmosphere"]),
        int(profile[f"{prefix}_shadow"]),
    )


def _rotation_from_to(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Proper column-vector rotation that maps ``source`` to ``target``."""
    a = _unit(source)
    b = _unit(target)
    cross = np.cross(a, b)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if sine < 1.0e-14:
        if cosine > 0.0:
            return np.eye(3)
        ref = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
        axis = _unit(np.cross(a, ref))
        # 180-degree Rodrigues rotation.
        return -np.eye(3) + 2.0*np.outer(axis, axis)
    axis = cross/sine
    k = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + sine*k + (1.0-cosine)*(k@k)


def _event_basis(event: ReferenceEvent) -> np.ndarray:
    """Fixed right-handed basis: +X from Sun to Earth at greatest eclipse."""
    peak = _inertial_centers(event, event.greatest_jd)
    sun = peak[0]
    x_axis = _unit(-sun)  # heliocentric Sun -> Earth
    north = np.array([0.0, 0.0, 1.0])
    z_axis = north-x_axis*float(np.dot(north, x_axis))
    if np.linalg.norm(z_axis) < 1.0e-12:
        north = np.array([0.0, 1.0, 0.0])
        z_axis = north-x_axis*float(np.dot(north, x_axis))
    z_axis = _unit(z_axis)
    y_axis = _unit(np.cross(z_axis, x_axis))
    z_axis = _unit(np.cross(x_axis, y_axis))
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def _native_to_gcrf(points: np.ndarray, definition_or_event, jd_value: float) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    definition = (definition_or_event.definition
                  if isinstance(definition_or_event, ReferenceEvent)
                  else definition_or_event)
    if definition.mode == "solar":
        if isinstance(definition_or_event, ReferenceEvent):
            return np.asarray(
                itrf_to_gcrf_km(values, float(jd_value)),
                dtype=float,
            )
        # A bare reference definition is deterministic and must not change
        # when optional Toolkit/Astropy packages happen to be installed.
        return np.asarray(
            itrf_to_gcrf_km(values, float(jd_value)),
            dtype=float,
        )
    return values.copy()


def _gcrf_to_native(vector: np.ndarray, definition_or_event, jd_value: float) -> np.ndarray:
    values = np.asarray(vector, dtype=float)
    definition = (definition_or_event.definition
                  if isinstance(definition_or_event, ReferenceEvent)
                  else definition_or_event)
    if definition.mode == "solar":
        if isinstance(definition_or_event, ReferenceEvent):
            return np.asarray(
                gcrf_to_itrf_km(values, float(jd_value)),
                dtype=float,
            )
        return np.asarray(
            gcrf_to_itrf_km(values, float(jd_value)),
            dtype=float,
        )
    return values.copy()


def _inertial_centers(
    definition_or_event: ReferenceDefinition | ReferenceEvent,
    jd_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Geocentric GCRF Sun and Moon centres for one event instant."""
    if isinstance(definition_or_event, ReferenceEvent):
        return event_positions_gcrf(definition_or_event, float(jd_value))
    bundle = trace_reference_rays(definition_or_event, float(jd_value), n_azimuth=8)
    sun = _native_to_gcrf(bundle.sun_center_km, definition_or_event, jd_value)
    moon = _native_to_gcrf(bundle.moon_center_km, definition_or_event, jd_value)
    return sun.reshape(3), moon.reshape(3)


def _reference_cross_axis_gcrf(event: ReferenceEvent) -> np.ndarray:
    peak_bundle = trace_event_rays(event, event.greatest_jd, n_azimuth=16)
    _, basis_native, _ = impact_plane_basis(peak_bundle)
    cross_native = basis_native[:, 2]
    if str(peak_bundle.frame).upper().startswith("GCRF"):
        return _unit(cross_native)
    return _unit(_native_to_gcrf(cross_native, event, event.greatest_jd))


def _build_inertial_state(
    event: ReferenceEvent,
    jd_value: float,
    reference_cross_axis_gcrf: np.ndarray,
    *,
    n_azimuth: int = 24,
) -> InertialFrameState:
    bundle = trace_event_rays(event, float(jd_value), n_azimuth=max(8, int(n_azimuth)))
    bundle_is_gcrf = str(bundle.frame).upper().startswith("GCRF")

    def to_gcrf(values):
        if bundle_is_gcrf:
            return np.asarray(values, dtype=float).copy()
        return _native_to_gcrf(values, event, jd_value)

    sun = to_gcrf(bundle.sun_center_km).reshape(3)
    moon = to_gcrf(bundle.moon_center_km).reshape(3)
    earth = np.zeros(3)

    cross_native = (
        np.asarray(reference_cross_axis_gcrf, dtype=float)
        if bundle_is_gcrf
        else _gcrf_to_native(reference_cross_axis_gcrf, event, jd_value)
    )
    umbra_u, umbra_l, _ = tangent_cross_section_paths(bundle, "umbra", cross_native)
    pen_u, pen_l, _ = tangent_cross_section_paths(bundle, "penumbra", cross_native)

    central = to_gcrf(bundle.central.points_km)
    axis_native = shadow_axis_world(bundle, float(jd_value))
    axis = to_gcrf(axis_native)
    umbra = (
        to_gcrf(umbra_u.points_km),
        to_gcrf(umbra_l.points_km),
    )
    penumbra = (
        to_gcrf(pen_u.points_km),
        to_gcrf(pen_l.points_km),
    )
    umbra_bundle = tuple(
        to_gcrf(path.points_km)
        for path in bundle.umbra
    )
    penumbra_bundle = tuple(
        to_gcrf(path.points_km)
        for path in bundle.penumbra
    )

    # Heliocentric translation: Earth = -Sun_geocentric, Moon = Earth + Moon_geocentric.
    sun_h = np.zeros(3)
    earth_h = -sun
    moon_h = earth_h+moon
    return InertialFrameState(
        event=event,
        jd_utc=float(jd_value),
        bundle_native=bundle,
        sun_gcrf_km=sun,
        earth_gcrf_km=earth,
        moon_gcrf_km=moon,
        sun_heliocentric_km=sun_h,
        earth_heliocentric_km=earth_h,
        moon_heliocentric_km=moon_h,
        central_gcrf_km=central,
        shadow_axis_gcrf_km=axis,
        umbra_gcrf_km=umbra,
        penumbra_gcrf_km=penumbra,
        umbra_bundle_gcrf_km=umbra_bundle,
        penumbra_bundle_gcrf_km=penumbra_bundle,
        target_hits={
            "umbra": (bool(umbra_u.target_hit), bool(umbra_l.target_hit)),
            "penumbra": (bool(pen_u.target_hit), bool(pen_l.target_hit)),
        },
    )


def _scene_coordinates(points_world_km: np.ndarray, origin_world_km: np.ndarray,
                       basis: np.ndarray, scale_km: float = 1.0) -> np.ndarray:
    values = np.asarray(points_world_km, dtype=float)
    return ((values-np.asarray(origin_world_km, dtype=float)) @ np.asarray(basis, dtype=float))/float(scale_km)


def _clip_segment_x(a: np.ndarray, b: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = b-a
    if abs(float(d[0])) < 1.0e-15:
        if xmin <= float(a[0]) <= xmax:
            return np.vstack([a, b])
        return np.empty((0, 3), dtype=float)
    t0 = (xmin-float(a[0]))/float(d[0])
    t1 = (xmax-float(a[0]))/float(d[0])
    lo = max(0.0, min(t0, t1))
    hi = min(1.0, max(t0, t1))
    if hi < lo:
        return np.empty((0, 3), dtype=float)
    return np.vstack([a+lo*d, a+hi*d])


def _clip_polyline_x(points: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    values = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(values) < 2:
        return np.empty((0, 3), dtype=float)
    result: list[np.ndarray] = []
    for index, (a, b) in enumerate(zip(values[:-1], values[1:])):
        segment = _clip_segment_x(a, b, xmin, xmax)
        if not len(segment):
            continue
        if result and np.linalg.norm(result[-1][-1]-segment[0]) < 1.0e-7:
            result.append(segment[1:])
        else:
            if result:
                result.append(np.full((1, 3), np.nan))
            result.append(segment)
    return np.vstack(result) if result else np.empty((0, 3), dtype=float)


def _aligned_earth_orbit(event: ReferenceEvent, basis: np.ndarray) -> np.ndarray:
    """Earth's heliocentric orbit over the year centred on the event (SSAPy DE430)."""
    year = np.linspace(event.greatest_jd-182.625, event.greatest_jd+182.625, 721)
    _, sun = ssapy_ephemeris_positions_km(year)
    return _scene_coordinates(-sun, np.zeros(3), basis, AU_KM)


def _build_scene_geometry(event: ReferenceEvent) -> SceneGeometry:
    basis = _event_basis(event)
    cross = _reference_cross_axis_gcrf(event)
    orbit = _aligned_earth_orbit(event, basis)
    earth_event = []
    moon_event = []
    moon_local = []
    for jd_value in event.jd:
        sun, moon = _inertial_centers(event, float(jd_value))
        earth_h = -sun
        moon_h = earth_h+moon
        earth_event.append(_scene_coordinates(earth_h.reshape(1, 3), np.zeros(3), basis, AU_KM)[0])
        moon_event.append(_scene_coordinates(moon_h.reshape(1, 3), np.zeros(3), basis, AU_KM)[0])
        moon_local.append(_scene_coordinates(moon.reshape(1, 3), np.zeros(3), basis, 1.0)[0])
    earth_event = np.asarray(earth_event)
    moon_event = np.asarray(moon_event)
    moon_local = np.asarray(moon_local)
    peak_index = event.peak_index
    earth_peak = np.zeros(3)
    moon_peak = moon_local[peak_index]
    xmin = float(min(np.min(moon_local[:, 0]), 0.0)-42_000.0)
    xmax = float(max(np.max(moon_local[:, 0]), 0.0)+42_000.0)
    transverse = float(max(
        4.0*RE_KM,
        np.max(np.abs(moon_local[:, 1]))+22_000.0,
        np.max(np.abs(moon_local[:, 2]))+22_000.0,
    ))
    return SceneGeometry(
        event=event,
        basis_world_from_scene=basis,
        reference_cross_axis_gcrf=cross,
        orbit_au=orbit,
        event_earth_au=earth_event,
        event_moon_au=moon_event,
        event_moon_local_km=moon_local,
        local_xrange_km=(xmin, xmax),
        local_transverse_km=transverse,
        earth_peak_local_km=earth_peak,
        moon_peak_local_km=moon_peak,
    )


