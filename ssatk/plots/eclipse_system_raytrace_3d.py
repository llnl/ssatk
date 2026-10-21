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
from pathlib import Path

from ssatk.io.eclipse_asset_resolver import resolve_image
from typing import Iterable
import copy
import math

import numpy as np
import plotly.graph_objects as go

from ssatk.compute.eclipse_brightness import (
    ephemeris_positions,
    earth_cone_boundary_gcrf,
    shadow_axis_surface_point,
    gcrf_to_itrf_km,
    itrf_to_gcrf_km,
)
from ssatk.compute.eclipse_state import event_gcrf_to_itrf_km, event_itrf_to_gcrf_km, event_positions_gcrf
from ssatk.plots.globe_orbit_daynight_plotly import (
    _earth_atmosphere_trace,
    _earth_eclipse_shadow_trace,
    _eclipse_earth_mesh as _earth_mesh,
    _sun_sphere_traces,
)
from ssatk.compute.eclipse_local_finite_geometry import impact_plane_basis, shadow_axis_world
from ssatk.plots.moon_render import _eclipse_moon_mesh_plotly as moon_mesh_plotly
from ssatk.compute.eclipse_reference_events import (
    AU_KM,
    EARTH_AXES_KM,
    LUNAR_2025,
    RE_KM,
    RP_KM,
    R_MOON_MEAN_KM,
    R_SUN_KM,
    SOLAR_2024,
    ReferenceDefinition,
    ReferenceEvent,
    build_reference_event,
    contact_aware_jd,
    itrf_surface_point,
    jd_to_datetime,
    solar_central_line_wgs84,
    _unit,
)
from ssatk.compute.eclipse_raytrace import (
    LUNAR_DANJON_EARTH_RADIUS_KM,
    ReferenceRayBundle,
    RayPath,
    bundle_penetrations,
    solar_footprint_segments,
    tangent_cross_section_paths,
    trace_event_rays,
    trace_reference_rays,
)

ASSET_DIR = Path(__file__).resolve().parent

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
                event_itrf_to_gcrf_km(definition_or_event, values, float(jd_value)),
                dtype=float,
            )
        from ssatk.compute.eclipse_reference_events import reference_itrf_to_gcrf_km

        return reference_itrf_to_gcrf_km(values, float(jd_value))
    return values.copy()


def _gcrf_to_native(vector: np.ndarray, definition_or_event, jd_value: float) -> np.ndarray:
    values = np.asarray(vector, dtype=float)
    definition = (definition_or_event.definition
                  if isinstance(definition_or_event, ReferenceEvent)
                  else definition_or_event)
    if definition.mode == "solar":
        if isinstance(definition_or_event, ReferenceEvent):
            return np.asarray(
                event_gcrf_to_itrf_km(definition_or_event, values, float(jd_value)),
                dtype=float,
            )
        from ssatk.compute.eclipse_reference_events import reference_gcrf_to_itrf_km

        return reference_gcrf_to_itrf_km(values, float(jd_value))
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


def _transform_xyz_trace(trace, origin_world_km: np.ndarray, basis: np.ndarray,
                         scale_km: float = 1.0):
    x = np.asarray(trace.x, dtype=float)
    y = np.asarray(trace.y, dtype=float)
    z = np.asarray(trace.z, dtype=float)
    shape = x.shape
    points = np.column_stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)])
    local = _scene_coordinates(points, origin_world_km, basis, scale_km=scale_km)
    trace.x = local[:, 0].reshape(shape)
    trace.y = local[:, 1].reshape(shape)
    trace.z = local[:, 2].reshape(shape)
    return trace


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




def _join_polylines(lines: Iterable[np.ndarray]) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for line in lines:
        values = np.asarray(line, dtype=float).reshape(-1, 3)
        if not len(values):
            continue
        if pieces:
            pieces.append(np.full((1, 3), np.nan))
        pieces.append(values)
    return np.vstack(pieces) if pieces else np.empty((0, 3), dtype=float)

def _scatter3d(points, *, name: str, color: str, width: float = 4.0,
               dash: str | None = None, showlegend: bool = True,
               legendgroup: str | None = None, visible=True,
               mode: str = "lines", marker: dict | None = None,
               text: Iterable[str] | None = None, hovertemplate: str | None = None):
    values = np.asarray(points, dtype=float).reshape(-1, 3)
    line = dict(color=color, width=width)
    if dash is not None:
        line["dash"] = dash
    return go.Scatter3d(
        x=values[:, 0] if len(values) else [],
        y=values[:, 1] if len(values) else [],
        z=values[:, 2] if len(values) else [],
        mode=mode,
        line=line if "lines" in mode else None,
        marker=marker,
        text=list(text) if text is not None else None,
        textposition="top center",
        name=name,
        showlegend=showlegend,
        legendgroup=legendgroup,
        visible=visible,
        connectgaps=False,
        hoverinfo="skip" if hovertemplate is None else None,
        hovertemplate=hovertemplate,
    )


def _earth_reference_lines(jd_value: float, basis: np.ndarray, *, event: ReferenceEvent | None = None,
                           heliocentric_center_km=None, scale_km: float = 1.0,
                           lift_km: float = 22.0):
    lat = np.linspace(-89.8, 89.8, 181)
    prime_itrf = np.asarray([itrf_surface_point(v, 0.0, lift_km) for v in lat])
    if event is None:
        prime_gcrf = itrf_to_gcrf_km(
            prime_itrf, float(jd_value), prefer_toolkit=False, prefer_astropy=False
        )
    else:
        prime_gcrf = event_itrf_to_gcrf_km(event, prime_itrf, float(jd_value))
    greenwich_itrf = itrf_surface_point(0.0, 0.0, lift_km+20.0)
    if event is None:
        greenwich_gcrf = itrf_to_gcrf_km(
            greenwich_itrf, float(jd_value), prefer_toolkit=False, prefer_astropy=False
        )
    else:
        greenwich_gcrf = event_itrf_to_gcrf_km(event, greenwich_itrf, float(jd_value))
    if heliocentric_center_km is not None:
        center = np.asarray(heliocentric_center_km, dtype=float)
        prime_gcrf = prime_gcrf+center
        greenwich_gcrf = greenwich_gcrf+center
        origin = np.zeros(3)
    else:
        origin = np.zeros(3)
    prime = _scene_coordinates(prime_gcrf, origin, basis, scale_km)
    mark = _scene_coordinates(np.asarray(greenwich_gcrf).reshape(1, 3), origin, basis, scale_km)
    return prime, mark


def _spin_axis(center_world_km: np.ndarray, basis: np.ndarray, *, scale_km: float,
               half_length_km: float) -> np.ndarray:
    center = np.asarray(center_world_km, dtype=float)
    points = np.vstack([center+np.array([0.0, 0.0, -half_length_km]),
                        center+np.array([0.0, 0.0, half_length_km])])
    return _scene_coordinates(points, np.zeros(3), basis, scale_km)


def _aligned_earth_orbit(event: ReferenceEvent, basis: np.ndarray) -> np.ndarray:
    year = np.linspace(event.greatest_jd-182.625, event.greatest_jd+182.625, 721)
    source = str(event.metadata.get("state_source", "reference"))
    ephem_backend = "ssapy" if source in {"ssapy", "ssapy-core"} else "auto"
    ephem = ephemeris_positions(year, backend=ephem_backend)
    analytic_earth = -ephem.sun_gcrf_km
    peak_analytic = -ephemeris_positions([event.greatest_jd], backend=ephem_backend).sun_gcrf_km[0]
    peak_reference_sun, _ = _inertial_centers(event, event.greatest_jd)
    peak_reference = -peak_reference_sun
    if source in {"ssapy", "ssapy-core"}:
        return _scene_coordinates(analytic_earth, np.zeros(3), basis, AU_KM)
    rotation = _rotation_from_to(peak_analytic, peak_reference)
    scale = float(np.linalg.norm(peak_reference)/np.linalg.norm(peak_analytic))
    aligned = (analytic_earth@rotation.T)*scale
    return _scene_coordinates(aligned, np.zeros(3), basis, AU_KM)


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


def _surface_footprint_trace(state: InertialFrameState, geometry: SceneGeometry,
                             *, family: str, color: str, width: float, dash: str | None):
    if state.event.mode != "solar":
        return _scatter3d(np.empty((0, 3)), name=f"Current {family} footprint", color=color,
                          width=width, dash=dash, showlegend=False)
    source = str(state.event.metadata.get("state_source", "reference"))
    if source in {"ssapy", "ssapy-core"}:
        boundary = earth_cone_boundary_gcrf(
            state.moon_gcrf_km,
            state.sun_gcrf_km,
            state.jd_utc,
            kind="umbra" if family == "umbra" else "penumbra",
            n_azimuth=240 if family == "umbra" else 180,
            event=state.event,
        )
        segments_gcrf = [boundary]
    else:
        segments_native = solar_footprint_segments(
            state.jd_utc, family=family, n_azimuth=240 if family == "umbra" else 180,
        )
        segments_gcrf = [
            _native_to_gcrf(segment, state.event, state.jd_utc)
            for segment in segments_native
        ]
    pieces: list[np.ndarray] = []
    for gcrf in segments_gcrf:
        local = _scene_coordinates(gcrf, np.zeros(3), geometry.basis_world_from_scene, 1.0)
        # Lift a few kilometres to avoid depth fighting while preserving direction.
        radius = np.linalg.norm(local, axis=1)
        local = local*((radius+12.0)/np.maximum(radius, 1.0))[:, None]
        if pieces:
            pieces.append(np.full((1, 3), np.nan))
        pieces.append(local)
    values = np.vstack(pieces) if pieces else np.empty((0, 3))
    return _scatter3d(
        values, name=f"Current {family} footprint", color=color, width=width,
        dash=dash, showlegend=(family == "umbra"), legendgroup="surface-intersections",
        visible=True,
    )


def _event_earth_shadow_axis_surface_point(
    event: ReferenceEvent,
    moon_gcrf_km: np.ndarray,
    sun_gcrf_km: np.ndarray,
    jd_utc: float,
) -> np.ndarray | None:
    """Solve the WGS-84 shadow-axis hit through the event-owned frame path."""
    moon_sun_itrf = np.asarray(event_gcrf_to_itrf_km(
        event,
        np.asarray([moon_gcrf_km, sun_gcrf_km], dtype=float),
        np.asarray([jd_utc, jd_utc], dtype=float),
    ), dtype=float)
    patch_itrf = shadow_axis_surface_point(
        moon_sun_itrf[0], moon_sun_itrf[1], target_axes_km=EARTH_AXES_KM,
    )
    if patch_itrf is None:
        return None
    return np.asarray(event_itrf_to_gcrf_km(event, patch_itrf, float(jd_utc)), dtype=float)


def _local_body_traces(state: InertialFrameState, geometry: SceneGeometry,
                       *, body_resolution: tuple[int, int], atmosphere_resolution: tuple[int, int],
                       shadow_resolution: int, peak_quality: bool):
    n_lat, n_lon = body_resolution
    from ssatk.coordinates.eclipse_lunar_attitude import moon_attitude_for_event
    a_lat, a_lon = atmosphere_resolution
    basis = geometry.basis_world_from_scene
    sun_hat = _unit(state.sun_gcrf_km)
    view_local = np.array([-1.35, -1.45, 0.72]) if state.event.mode == "solar" else np.array([0.20, -1.80, 0.78])
    view_world = _unit(basis@view_local)

    earth = _earth_mesh(
        sun_hat,
        n_lat=n_lat,
        n_lon=n_lon,
        radius_scale=1.0,
        center=(0.0, 0.0, 0.0),
        shadow_body_center_km=None,
        shadow_body_radius_km=None,
        time_jd=float(state.jd_utc),
        sun_position_km=state.sun_gcrf_km,
        physical_center_km=(0.0, 0.0, 0.0),
        night_floor=0.012,
        texture_path=resolve_image("earth_albedo").path,
        exposure=1.18,
        view_hat=view_world,
        specular_strength=0.34,
    )
    earth.name = "Earth - rotating WGS-84 body"
    earth.showlegend = True
    earth.legendgroup = "bodies"
    _transform_xyz_trace(earth, np.zeros(3), basis, 1.0)

    traces: list[go.BaseTraceType] = [earth]
    if state.event.mode == "solar":
        # The event's immutable backend owns both transformations.
        patch_gcrf = _event_earth_shadow_axis_surface_point(
            state.event, state.moon_gcrf_km, state.sun_gcrf_km, state.jd_utc,
        )
        shadow = _earth_eclipse_shadow_trace(
            sun_hat,
            state.moon_gcrf_km,
            state.bundle_native.umbra_optical_radius_km,
            sun_position_km=state.sun_gcrf_km,
            physical_center_km=(0.0, 0.0, 0.0),
            center=(0.0, 0.0, 0.0),
            time_jd=float(state.jd_utc),
            patch_center_gcrf_km=patch_gcrf,
            resolution=int(shadow_resolution),
            nudge=1.0009,
            night_floor=0.012,
        )
        shadow.name = "Resolved finite-disc shadow on Earth"
        shadow.showlegend = True
        shadow.legendgroup = "surface-intersections"
        _transform_xyz_trace(shadow, np.zeros(3), basis, 1.0)
        traces.append(shadow)
    else:
        # Stable placeholder keeps animation trace order identical.
        traces.append(go.Mesh3d(x=[], y=[], z=[], i=[], j=[], k=[], name="Solar surface shadow",
                                showlegend=False, hoverinfo="skip"))

    atmosphere = _earth_atmosphere_trace(
        center=(0.0, 0.0, 0.0),
        radius_scale=1.0,
        sun_hat=sun_hat,
        view_hat=view_world,
        time_jd=float(state.jd_utc),
        n_lat=a_lat,
        n_lon=a_lon,
        altitude_km=105.0,
        max_alpha=0.32,
    )
    atmosphere.name = "Atmosphere"
    _transform_xyz_trace(atmosphere, np.zeros(3), basis, 1.0)
    traces.append(atmosphere)

    moon = moon_mesh_plotly(
        state.moon_gcrf_km,
        R_MOON_MEAN_KM,
        sun_hat=sun_hat,
        real_center_km=state.moon_gcrf_km,
        mode=state.event.mode,
        n_lat=n_lat,
        n_lon=n_lon,
        real_sun_position_km=state.sun_gcrf_km,
        ambient_floor=0.006,
        texture_path=resolve_image("moon_albedo").path,
        view_hat=view_world,
        exposure=1.16,
        eclipse_occluder_radius_km=LUNAR_DANJON_EARTH_RADIUS_KM,
        body_to_gcrf=moon_attitude_for_event(state.event, state.jd_utc).body_to_gcrf,
    )
    moon.name = "Moon - synchronously oriented textured body"
    moon.showlegend = True
    moon.legendgroup = "bodies"
    _transform_xyz_trace(moon, np.zeros(3), basis, 1.0)
    traces.append(moon)

    # Earth spin/rotation cues.
    prime, greenwich = _earth_reference_lines(state.jd_utc, basis, event=state.event, scale_km=1.0)
    traces.append(_scatter3d(
        prime, name="Rotating Greenwich meridian", color=COLORS["greenwich"], width=3.0,
        showlegend=True, legendgroup="rotation", visible="legendonly",
    ))
    traces.append(_scatter3d(
        greenwich, name="Greenwich surface marker", color=COLORS["greenwich"], width=0.0,
        mode="markers", marker=dict(size=4.5, color=COLORS["greenwich"],
                                    line=dict(color="white", width=0.5)),
        showlegend=False, legendgroup="rotation", visible="legendonly",
    ))
    spin = _spin_axis(np.zeros(3), basis, scale_km=1.0, half_length_km=1.45*RE_KM)
    traces.append(_scatter3d(
        spin, name="Earth spin axis", color=COLORS["spin"], width=3.0, dash="dash",
        showlegend=True, legendgroup="rotation", visible="legendonly",
    ))

    xmin, xmax = geometry.local_xrange_km
    central = _scene_coordinates(state.central_gcrf_km, np.zeros(3), basis, 1.0)
    central = _clip_polyline_x(central, xmin, xmax)
    axis = _scene_coordinates(state.shadow_axis_gcrf_km, np.zeros(3), basis, 1.0)
    axis = _clip_polyline_x(axis, xmin, xmax)
    traces.extend([
        _scatter3d(central, name="Direct sunlight - stops at first opaque surface",
                   color=COLORS["sun"], width=6.0, showlegend=True,
                   legendgroup="finite-sun-rays"),
        _scatter3d(axis, name="Blocked-light shadow axis", color=COLORS["axis"], width=4.0,
                   dash="dash", showlegend=True, legendgroup="finite-sun-rays"),
    ])

    for family, pair, color, dash in (
        ("Umbra / antumbra", state.umbra_gcrf_km, COLORS["umbra"], None),
        ("Penumbra", state.penumbra_gcrf_km, COLORS["penumbra"], "dot"),
    ):
        for sign_name, points in zip(("upper", "lower"), pair):
            local = _scene_coordinates(points, np.zeros(3), basis, 1.0)
            local = _clip_polyline_x(local, xmin, xmax)
            traces.append(_scatter3d(
                local, name=f"{family} tangent - {sign_name}", color=color, width=3.6,
                dash=dash, showlegend=(sign_name == "upper"),
                legendgroup="finite-sun-rays",
            ))

    # Complete sampled 3-D tangent rings. These are optional inspection
    # layers: one Scatter3d per family with NaN separators avoids hundreds of
    # independent traces while retaining the true three-dimensional bundle.
    for family, bundle_lines, color, dash in (
        ("Full umbra / antumbra ray bundle", state.umbra_bundle_gcrf_km, COLORS["umbra"], None),
        ("Full penumbra ray bundle", state.penumbra_bundle_gcrf_km, COLORS["penumbra"], "dot"),
    ):
        clipped = []
        for points in bundle_lines:
            local = _scene_coordinates(points, np.zeros(3), basis, 1.0)
            segment = _clip_polyline_x(local, xmin, xmax)
            if len(segment):
                clipped.append(segment)
        traces.append(_scatter3d(
            _join_polylines(clipped), name=family, color=color, width=1.6, dash=dash,
            showlegend=True, legendgroup="full-ray-bundles", visible="legendonly",
        ))

    # Surface intersections from the same tangent construction.
    traces.append(_surface_footprint_trace(
        state, geometry, family="umbra", color=COLORS["footprint"], width=5.0, dash=None,
    ))
    traces.append(_surface_footprint_trace(
        state, geometry, family="penumbra", color=COLORS["penumbra"], width=2.0, dash="dot",
    ))

    return traces


def _simple_sphere(center, radius, *, color: str, name: str, opacity: float = 1.0,
                   n_lat: int = 18, n_lon: int = 36, showlegend=False):
    lat = np.linspace(-0.5*math.pi, 0.5*math.pi, n_lat)
    lon = np.linspace(0.0, 2.0*math.pi, n_lon)
    lon_g, lat_g = np.meshgrid(lon, lat)
    x = center[0]+radius*np.cos(lat_g)*np.cos(lon_g)
    y = center[1]+radius*np.cos(lat_g)*np.sin(lon_g)
    z = center[2]+radius*np.sin(lat_g)
    return go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0.0, color], [1.0, color]],
        showscale=False, opacity=opacity,
        lighting=dict(ambient=0.72, diffuse=0.28, specular=0.0),
        name=name, showlegend=showlegend, hovertemplate=f"{name}<extra></extra>",
    )


def _heliocentric_dynamic_traces(state: InertialFrameState, geometry: SceneGeometry):
    basis = geometry.basis_world_from_scene
    earth = _scene_coordinates(state.earth_heliocentric_km.reshape(1, 3), np.zeros(3), basis, AU_KM)[0]
    moon = _scene_coordinates(state.moon_heliocentric_km.reshape(1, 3), np.zeros(3), basis, AU_KM)[0]
    earth_radius = RE_KM/AU_KM
    moon_radius = R_MOON_MEAN_KM/AU_KM

    traces: list[go.BaseTraceType] = [
        _simple_sphere(earth, earth_radius, color="#4e91ff", name="Earth - true radius", n_lat=14, n_lon=28),
        _simple_sphere(moon, moon_radius, color="#d8d8d8", name="Moon - true radius", n_lat=12, n_lon=24),
        _scatter3d(
            np.vstack([earth, moon]), name="Earth and Moon locators - screen-space aid",
            color="#ffffff", width=0.0, mode="markers+text",
            marker=dict(size=[8, 5], color=["#4e91ff", "#d8d8d8"],
                        line=dict(color="white", width=0.8)),
            text=["Earth locator", "Moon locator"], showlegend=True,
            legendgroup="heliocentric-locators",
            hovertemplate="%{text}<extra></extra>",
        ),
    ]

    # Heliocentric full source-to-target paths.  Translate geocentric ray
    # points by the current Earth heliocentric position before rotating to the
    # event basis and scaling to AU.
    def helio(points_gcrf):
        values = state.earth_heliocentric_km+np.asarray(points_gcrf, dtype=float)
        return _scene_coordinates(values, np.zeros(3), basis, AU_KM)

    traces.extend([
        _scatter3d(helio(state.central_gcrf_km), name="Direct sunlight - full 1 AU path",
                   color=COLORS["sun"], width=5.0, showlegend=False),
        _scatter3d(helio(state.shadow_axis_gcrf_km), name="Blocked-light axis - heliocentric",
                   color=COLORS["axis"], width=3.0, dash="dash", showlegend=False),
    ])
    for family, pair, color, dash in (
        ("Umbra", state.umbra_gcrf_km, COLORS["umbra"], None),
        ("Penumbra", state.penumbra_gcrf_km, COLORS["penumbra"], "dot"),
    ):
        for points in pair:
            traces.append(_scatter3d(
                helio(points), name=f"{family} tangent - heliocentric",
                color=color, width=2.8, dash=dash, showlegend=False,
            ))

    for family, bundle_lines, color, dash in (
        ("Full umbra / antumbra bundle - heliocentric", state.umbra_bundle_gcrf_km, COLORS["umbra"], None),
        ("Full penumbra bundle - heliocentric", state.penumbra_bundle_gcrf_km, COLORS["penumbra"], "dot"),
    ):
        lines = [helio(points) for points in bundle_lines]
        traces.append(_scatter3d(
            _join_polylines(lines), name=family, color=color, width=1.2, dash=dash,
            showlegend=False, visible=True,
        ))

    prime, greenwich = _earth_reference_lines(
        state.jd_utc, basis, event=state.event, heliocentric_center_km=state.earth_heliocentric_km,
        scale_km=AU_KM,
    )
    traces.append(_scatter3d(
        prime, name="Rotating Greenwich meridian - heliocentric", color=COLORS["greenwich"],
        width=2.5, showlegend=False, visible=True,
    ))
    traces.append(_scatter3d(
        greenwich, name="Greenwich marker - heliocentric", color=COLORS["greenwich"],
        width=0.0, mode="markers", marker=dict(size=3.5, color=COLORS["greenwich"]),
        showlegend=False,
    ))
    return traces


def _sun_traces_static():
    traces = _sun_sphere_traces(
        np.zeros(3), R_SUN_KM/AU_KM, n=48, view_hat=np.array([1.0, -0.5, 0.25]), glow=True,
    )
    for index, trace in enumerate(traces):
        trace.name = "Sun - true radius" if index == 0 else "Solar glow"
        trace.showlegend = index == 0
        trace.legendgroup = "heliocentric-bodies"
        trace.visible = False
    return traces


def _static_traces(geometry: SceneGeometry):
    event = geometry.event
    traces: list[go.BaseTraceType] = []

    orbit = _scatter3d(
        geometry.orbit_au, name="Earth heliocentric orbit - one year", color=COLORS["orbit"],
        width=2.2, showlegend=True, legendgroup="heliocentric-paths", visible=False,
    )
    traces.append(orbit)
    traces.append(_scatter3d(
        geometry.event_earth_au, name="Earth motion during eclipse", color="#ffffff",
        width=5.0, showlegend=True, legendgroup="heliocentric-paths", visible=False,
    ))
    traces.append(_scatter3d(
        geometry.event_moon_au, name="Moon heliocentric motion during eclipse", color="#c5ceda",
        width=2.2, showlegend=True, legendgroup="heliocentric-paths", visible=False,
    ))

    # Moon event arc in the local Earth-centred frame.
    traces.append(_scatter3d(
        geometry.event_moon_local_km, name="Moon path during eclipse", color=COLORS["moon_orbit"],
        width=2.5, dash="dot", showlegend=True, legendgroup="orbital-motion", visible=True,
    ))
    traces.extend(_sun_traces_static())
    return traces


def _trace_update(trace: go.BaseTraceType) -> go.BaseTraceType:
    """Frame payload without visibility/legend fields so view mode persists."""
    data = copy.deepcopy(trace.to_plotly_json())
    for key in (
        "visible", "showlegend", "name", "legendgroup", "legendgrouptitle",
        "hoverinfo", "hovertemplate", "scene", "legend",
    ):
        data.pop(key, None)
    trace_type = data.pop("type", None)
    if trace_type == "mesh3d":
        return go.Mesh3d(**data)
    if trace_type == "surface":
        return go.Surface(**data)
    if trace_type == "scatter3d":
        return go.Scatter3d(**data)
    raise TypeError(f"Unsupported animated trace type {trace_type!r}")


def _current_phase(event: ReferenceEvent, jd_value: float) -> str:
    ordered = sorted(event.contacts_jd.items(), key=lambda item: item[1])
    names = [name for name, _ in ordered]
    values = [value for _, value in ordered]
    if jd_value <= values[0]:
        return names[0]
    if jd_value >= values[-1]:
        return names[-1]
    exact = min(ordered, key=lambda item: abs(item[1]-jd_value))
    if abs(exact[1]-jd_value) < 0.2/86400.0:
        return exact[0]
    for (left_name, left), (right_name, right) in zip(ordered[:-1], ordered[1:]):
        if left < jd_value < right:
            return f"{left_name} -> {right_name}"
    return "event"


def _frame_dynamic_traces(state: InertialFrameState, geometry: SceneGeometry,
                          *, body_resolution: tuple[int, int], atmosphere_resolution: tuple[int, int],
                          shadow_resolution: int, peak_quality: bool):
    local = _local_body_traces(
        state, geometry,
        body_resolution=body_resolution,
        atmosphere_resolution=atmosphere_resolution,
        shadow_resolution=shadow_resolution,
        peak_quality=peak_quality,
    )
    helio = _heliocentric_dynamic_traces(state, geometry)
    for trace in helio:
        trace.visible = False
    return local+helio


def _view_visibility(static_count: int, local_count: int, helio_count: int,
                     *, mode: str, total_count: int) -> list[bool | str]:
    visible: list[bool | str] = [False]*total_count
    # Static traces: Earth orbit, event Earth, event Moon, local Moon path,
    # then Sun surfaces.
    if mode == "local":
        visible[3] = True
        for index in range(static_count, static_count+local_count):
            visible[index] = True
        # Preserve legend-only rotation cues.
        for index in range(static_count, static_count+local_count):
            name = ""
            # Name access is handled by caller when constructing final array.
    elif mode == "helio":
        for index in range(3):
            visible[index] = True
        for index in range(4, static_count):
            visible[index] = True
        for index in range(static_count+local_count, total_count):
            visible[index] = True
    else:
        raise ValueError("mode must be local or helio")
    return visible


def _apply_legendonly_defaults(visibility: list[bool | str], traces: list[go.BaseTraceType], mode: str):
    if mode == "local":
        for index, trace in enumerate(traces):
            if getattr(trace, "name", "") in {
                "Rotating Greenwich meridian", "Greenwich surface marker", "Earth spin axis",
                "Full umbra / antumbra ray bundle", "Full penumbra ray bundle",
            }:
                visibility[index] = "legendonly"
    return visibility


def _camera_layout(geometry: SceneGeometry, mode: str, focus: str = "system") -> dict:
    if mode == "helio":
        earth_peak = geometry.event_earth_au[geometry.event.peak_index]
        moon_peak = geometry.event_moon_au[geometry.event.peak_index]
        if focus == "orbit":
            span = 1.18
            ranges = ([-span, span], [-span, span], [-0.50, 0.50])
            camera = dict(eye=dict(x=1.55, y=-1.65, z=0.82), up=dict(x=0, y=0, z=1),
                          projection=dict(type="perspective"))
            aspect = "data"
        elif focus == "event":
            xmin = -0.045
            xmax = 1.055
            transverse = 0.020
            ranges = ([xmin, xmax], [-transverse, transverse], [-transverse, transverse])
            camera = dict(eye=dict(x=1.45, y=-1.42, z=0.60), up=dict(x=0, y=0, z=1),
                          projection=dict(type="perspective"))
            aspect = "data"
        else:
            center = 0.5*(earth_peak+moon_peak)
            half = 0.0040
            ranges = ([center[0]-half, center[0]+half],
                      [center[1]-half, center[1]+half],
                      [center[2]-half, center[2]+half])
            camera = dict(eye=dict(x=1.35, y=-1.55, z=0.70), up=dict(x=0, y=0, z=1),
                          projection=dict(type="perspective"))
            aspect = "cube"
        return {
            "scene.xaxis.range": ranges[0],
            "scene.yaxis.range": ranges[1],
            "scene.zaxis.range": ranges[2],
            "scene.xaxis.title.text": "Heliocentric X [AU] - Sun to Earth at greatest",
            "scene.yaxis.title.text": "Heliocentric Y [AU]",
            "scene.zaxis.title.text": "Heliocentric Z [AU]",
            "scene.aspectmode": aspect,
            "scene.camera": camera,
        }

    earth = geometry.earth_peak_local_km
    moon = geometry.moon_peak_local_km
    if focus == "earth":
        half = 15_500.0
        center = earth
        camera = dict(eye=dict(x=-1.35, y=-1.55, z=0.72), up=dict(x=0, y=0, z=1),
                      projection=dict(type="perspective"))
        aspect = "cube"
    elif focus == "moon":
        half = 8_000.0
        center = moon
        camera = dict(eye=dict(x=-1.20, y=-1.60, z=0.75), up=dict(x=0, y=0, z=1),
                      projection=dict(type="perspective"))
        aspect = "cube"
    elif focus == "optics":
        center = 0.5*(earth+moon)
        half_x = 0.58*(geometry.local_xrange_km[1]-geometry.local_xrange_km[0])
        half = geometry.local_transverse_km
        return {
            "scene.xaxis.range": [center[0]-half_x, center[0]+half_x],
            "scene.yaxis.range": [-half, half],
            "scene.zaxis.range": [-half, half],
            "scene.xaxis.title.text": "Earth-centred inertial X [km] - Sun to Earth at greatest",
            "scene.yaxis.title.text": "Earth-centred inertial Y [km]",
            "scene.zaxis.title.text": "Earth-centred inertial Z [km]",
            "scene.aspectmode": "data",
            "scene.camera": dict(eye=dict(x=0.03, y=-2.35, z=0.18), up=dict(x=0, y=0, z=1),
                                 projection=dict(type="orthographic")),
        }
    else:
        return {
            "scene.xaxis.range": list(geometry.local_xrange_km),
            "scene.yaxis.range": [-geometry.local_transverse_km, geometry.local_transverse_km],
            "scene.zaxis.range": [-geometry.local_transverse_km, geometry.local_transverse_km],
            "scene.xaxis.title.text": "Earth-centred inertial X [km] - Sun to Earth at greatest",
            "scene.yaxis.title.text": "Earth-centred inertial Y [km]",
            "scene.zaxis.title.text": "Earth-centred inertial Z [km]",
            "scene.aspectmode": "data",
            "scene.camera": dict(
                eye=(dict(x=-1.35, y=-1.45, z=0.72) if geometry.event.mode == "solar"
                     else dict(x=0.20, y=-1.80, z=0.78)),
                up=dict(x=0, y=0, z=1), projection=dict(type="perspective"),
            ),
        }
    return {
        "scene.xaxis.range": [center[0]-half, center[0]+half],
        "scene.yaxis.range": [center[1]-half, center[1]+half],
        "scene.zaxis.range": [center[2]-half, center[2]+half],
        "scene.xaxis.title.text": "Earth-centred inertial X [km] - Sun to Earth at greatest",
        "scene.yaxis.title.text": "Earth-centred inertial Y [km]",
        "scene.zaxis.title.text": "Earth-centred inertial Z [km]",
        "scene.aspectmode": aspect,
        "scene.camera": camera,
    }


def generate_system_animation(
    kind: str | ReferenceDefinition = "solar",
    output_path: str | Path = "eclipse_system_animation.html",
    *,
    n_frames: int = 31,
    quality: str = "high",
    include_plotlyjs: bool = True,
) -> str:
    """Generate one synchronized physical-system and local-ray animation."""
    event = build_reference_event(kind, n_frames=max(int(n_frames), 17))
    geometry = _build_scene_geometry(event)
    body_resolution, atmosphere_resolution, shadow_resolution = _quality(quality, animated=True)

    states = [
        _build_inertial_state(
            event, float(jd_value), geometry.reference_cross_axis_gcrf,
            n_azimuth=24,
        )
        for jd_value in event.jd
    ]
    static = _static_traces(geometry)
    dynamic_first = _frame_dynamic_traces(
        states[0], geometry,
        body_resolution=body_resolution,
        atmosphere_resolution=atmosphere_resolution,
        shadow_resolution=shadow_resolution,
        peak_quality=False,
    )
    traces = static+dynamic_first
    static_count = len(static)
    local_count = len(_local_body_traces(
        states[0], geometry,
        body_resolution=body_resolution,
        atmosphere_resolution=atmosphere_resolution,
        shadow_resolution=shadow_resolution,
        peak_quality=False,
    ))
    helio_count = len(dynamic_first)-local_count
    dynamic_indices = list(range(static_count, len(traces)))

    # Set default visibility for the local physical view.
    local_visibility = _view_visibility(
        static_count, local_count, helio_count, mode="local", total_count=len(traces),
    )
    local_visibility = _apply_legendonly_defaults(local_visibility, traces, "local")
    for trace, visible in zip(traces, local_visibility):
        trace.visible = visible

    frames = []
    for index, state in enumerate(states):
        dynamic = _frame_dynamic_traces(
            state, geometry,
            body_resolution=body_resolution,
            atmosphere_resolution=atmosphere_resolution,
            shadow_resolution=shadow_resolution,
            peak_quality=False,
        )
        frame_data = [_trace_update(trace) for trace in dynamic]
        phase = _current_phase(event, state.jd_utc)
        timestamp = jd_to_datetime(state.jd_utc).strftime("%Y-%m-%d %H:%M:%S")
        frames.append(go.Frame(
            name=str(index),
            data=frame_data,
            traces=dynamic_indices,
            layout=go.Layout(
                title=dict(
                    text=(f"{event.definition.title} - physical 3-D system and finite-Sun ray trace"
                          f"<br><sub>{timestamp} UTC | {phase} | Earth rotation, Earth heliocentric motion, and lunar motion animated</sub>"),
                ),
            ),
        ))

    fig = go.Figure(data=traces, frames=frames)
    helio_visibility = _view_visibility(
        static_count, local_count, helio_count, mode="helio", total_count=len(traces),
    )

    def button(label, mode, focus):
        visibility = local_visibility if mode == "local" else helio_visibility
        layout = _camera_layout(geometry, mode, focus)
        return dict(label=label, method="update", args=[{"visible": visibility}, layout])

    view_buttons = [
        button("Earth-Moon physical system", "local", "system"),
        button("Optics side - orthographic", "local", "optics"),
        button("Earth close-up - rotation and shadow", "local", "earth"),
        button("Moon close-up", "local", "moon"),
        button("Heliocentric Sun-Earth-Moon", "helio", "event"),
        button("Full Earth orbit around Sun", "helio", "orbit"),
    ]

    first_dt = jd_to_datetime(float(event.jd[0]))
    final_dt = jd_to_datetime(float(event.jd[-1]))
    peak_state = states[event.peak_index]
    earth_displacement = float(np.linalg.norm(states[-1].earth_heliocentric_km-states[0].earth_heliocentric_km))
    moon_displacement = float(np.linalg.norm(states[-1].moon_gcrf_km-states[0].moon_gcrf_km))
    event_hours = float((event.jd[-1]-event.jd[0])*24.0)
    earth_rotation_deg = event_hours*360.0/23.9344696

    initial_camera = _camera_layout(geometry, "local", "system")
    fig.update_layout(
        title=dict(
            text=(f"{event.definition.title} - physical 3-D system and finite-Sun ray trace"
                  f"<br><sub>{first_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC | first event contact | "
                  "yellow direct light stops at the occluder; magenta begins behind it</sub>"),
            x=0.5, y=0.985, yanchor="top", font=dict(size=22),
        ),
        scene=dict(
            xaxis=dict(
                range=initial_camera["scene.xaxis.range"],
                title=initial_camera["scene.xaxis.title.text"],
                gridcolor=COLORS["grid"], zerolinecolor="#6e7f98",
                backgroundcolor=COLORS["bg"], showspikes=False,
            ),
            yaxis=dict(
                range=initial_camera["scene.yaxis.range"],
                title=initial_camera["scene.yaxis.title.text"],
                gridcolor=COLORS["grid"], zerolinecolor="#6e7f98",
                backgroundcolor=COLORS["bg"], showspikes=False,
            ),
            zaxis=dict(
                range=initial_camera["scene.zaxis.range"],
                title=initial_camera["scene.zaxis.title.text"],
                gridcolor=COLORS["grid"], zerolinecolor="#6e7f98",
                backgroundcolor=COLORS["bg"], showspikes=False,
            ),
            aspectmode=initial_camera["scene.aspectmode"],
            camera=initial_camera["scene.camera"],
            bgcolor=COLORS["bg"], dragmode="orbit",
            uirevision=f"system-raytrace-{event.definition.key}",
        ),
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"], family="Arial, sans-serif", size=12),
        height=1040,
        margin=dict(l=10, r=330, t=112, b=72),
        legend=dict(
            title=dict(text="Physical scene key"),
            x=1.005, y=0.94, xanchor="left", yanchor="top",
            bgcolor="rgba(7,12,20,0.97)", bordercolor="#3b4a60", borderwidth=1,
            font=dict(size=10.2), tracegroupgap=5, groupclick="toggleitem",
        ),
        updatemenus=[
            dict(
                type="dropdown", direction="down", x=1.005, xanchor="left",
                y=1.03, yanchor="top", buttons=view_buttons,
                bgcolor="#1b2738", bordercolor="#74849a", font=dict(color="white"),
            ),
            dict(
                type="buttons", direction="left", showactive=False,
                x=0.02, xanchor="left", y=0.04, yanchor="bottom",
                bgcolor="#1b2738", bordercolor="#74849a", font=dict(color="white"),
                buttons=[
                    dict(label="Play", method="animate", args=[
                        None,
                        dict(frame=dict(duration=240, redraw=True),
                             transition=dict(duration=0), fromcurrent=True, mode="immediate"),
                    ]),
                    dict(label="Pause", method="animate", args=[
                        [None],
                        dict(frame=dict(duration=0, redraw=False),
                             transition=dict(duration=0), mode="immediate"),
                    ]),
                ],
            ),
        ],
        sliders=[dict(
            active=0, x=0.15, len=0.76, y=0.025, xanchor="left", yanchor="bottom",
            currentvalue=dict(prefix="UTC: ", font=dict(size=13, color="white")),
            font=dict(size=9, color="white"),
            steps=[
                dict(
                    label=jd_to_datetime(float(state.jd_utc)).strftime("%H:%M"),
                    method="animate",
                    args=[[str(index)], dict(frame=dict(duration=0, redraw=True),
                                              transition=dict(duration=0), mode="immediate")],
                )
                for index, state in enumerate(states)
            ],
        )],
        annotations=[
            dict(
                x=1.005, y=0.27, xref="paper", yref="paper", xanchor="left", yanchor="top",
                showarrow=False, align="left", width=292,
                bgcolor="rgba(7,12,20,0.97)", bordercolor="#3b4a60", borderwidth=1,
                borderpad=7, font=dict(size=10, color=COLORS["muted"]),
                text=(
                    "<b>One physical state, two render origins</b><br>"
                    "Local mode: Earth-centred GCRF kilometres for exact body and tangent detail.<br>"
                    "Heliocentric mode: true AU distances with the Sun at the origin.<br>"
                    "No body enlargement, no distance compression, and no ray continues through an opaque body.<br><br>"
                    f"Event interval: {first_dt.strftime('%H:%M:%S')} to {final_dt.strftime('%H:%M:%S')} UTC "
                    f"({event_hours:.3f} h)<br>"
                    f"Earth heliocentric displacement: {earth_displacement:,.0f} km<br>"
                    f"Earth rotation during interval: {earth_rotation_deg:.2f} deg<br>"
                    f"Moon geocentric displacement: {moon_displacement:,.0f} km<br>"
                    f"Peak Sun-Earth distance: {np.linalg.norm(peak_state.sun_gcrf_km)/1e6:.3f} million km<br>"
                    f"Peak Earth-Moon distance: {np.linalg.norm(peak_state.moon_gcrf_km):,.0f} km"
                ),
            ),
        ],
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        output, include_plotlyjs=include_plotlyjs, full_html=True,
        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
            "plotGlPixelRatio": 2.0,
            "toImageButtonOptions": {"format": "png", "scale": 2, "filename": output.stem},
        },
    )
    return str(output)


def generate_system_peak_view(
    kind: str | ReferenceDefinition = "solar",
    output_path: str | Path = "eclipse_system_peak.html",
    *,
    quality: str = "ultra",
    include_plotlyjs: bool = True,
) -> str:
    """Generate a high-resolution greatest-eclipse inspection scene."""
    definition = (kind if isinstance(kind, ReferenceDefinition)
                  else (SOLAR_2024 if str(kind).lower().startswith("solar") else LUNAR_2025))
    event = build_reference_event(definition, jd=[definition.greatest_jd])
    geometry = _build_scene_geometry(event)
    body_resolution, atmosphere_resolution, shadow_resolution = _quality(quality, animated=False)
    state = _build_inertial_state(event, event.greatest_jd, geometry.reference_cross_axis_gcrf, n_azimuth=48)
    static = _static_traces(geometry)
    dynamic = _frame_dynamic_traces(
        state, geometry,
        body_resolution=body_resolution,
        atmosphere_resolution=atmosphere_resolution,
        shadow_resolution=shadow_resolution,
        peak_quality=True,
    )
    traces = static+dynamic
    static_count = len(static)
    local_count = len(_local_body_traces(
        state, geometry,
        body_resolution=body_resolution,
        atmosphere_resolution=atmosphere_resolution,
        shadow_resolution=shadow_resolution,
        peak_quality=True,
    ))
    helio_count = len(dynamic)-local_count
    local_visibility = _view_visibility(static_count, local_count, helio_count,
                                        mode="local", total_count=len(traces))
    local_visibility = _apply_legendonly_defaults(local_visibility, traces, "local")
    helio_visibility = _view_visibility(static_count, local_count, helio_count,
                                        mode="helio", total_count=len(traces))
    for trace, visible in zip(traces, local_visibility):
        trace.visible = visible
    fig = go.Figure(data=traces)

    def button(label, mode, focus):
        visibility = local_visibility if mode == "local" else helio_visibility
        return dict(label=label, method="update",
                    args=[{"visible": visibility}, _camera_layout(geometry, mode, focus)])

    initial = _camera_layout(geometry, "local", "earth" if event.mode == "solar" else "moon")
    dt = jd_to_datetime(event.greatest_jd)
    fig.update_layout(
        title=dict(
            text=(f"{event.definition.title} - high-fidelity physical system at greatest eclipse"
                  f"<br><sub>{dt.strftime('%Y-%m-%d %H:%M:%S')} UTC; finite-Sun rays, true radii, true separation, rotating Earth orientation</sub>"),
            x=0.5, y=0.985, font=dict(size=22),
        ),
        scene=dict(
            xaxis=dict(range=initial["scene.xaxis.range"], title=initial["scene.xaxis.title.text"],
                       gridcolor=COLORS["grid"], backgroundcolor=COLORS["bg"]),
            yaxis=dict(range=initial["scene.yaxis.range"], title=initial["scene.yaxis.title.text"],
                       gridcolor=COLORS["grid"], backgroundcolor=COLORS["bg"]),
            zaxis=dict(range=initial["scene.zaxis.range"], title=initial["scene.zaxis.title.text"],
                       gridcolor=COLORS["grid"], backgroundcolor=COLORS["bg"]),
            aspectmode=initial["scene.aspectmode"], camera=initial["scene.camera"],
            bgcolor=COLORS["bg"], dragmode="orbit",
        ),
        paper_bgcolor=COLORS["bg"], font=dict(color="white"), height=1080,
        margin=dict(l=10, r=325, t=110, b=55),
        legend=dict(x=1.005, y=0.94, bgcolor="rgba(7,12,20,0.97)",
                    bordercolor="#3b4a60", borderwidth=1, font=dict(size=10.2)),
        updatemenus=[dict(
            type="dropdown", x=1.005, y=1.03, xanchor="left", yanchor="top",
            bgcolor="#1b2738", bordercolor="#74849a", font=dict(color="white"),
            buttons=[
                button("Target close-up", "local", "earth" if event.mode == "solar" else "moon"),
                button("Earth-Moon physical system", "local", "system"),
                button("Optics side - orthographic", "local", "optics"),
                button("Earth close-up", "local", "earth"),
                button("Moon close-up", "local", "moon"),
                button("Heliocentric Sun-Earth-Moon", "helio", "event"),
                button("Full Earth orbit around Sun", "helio", "orbit"),
            ],
        )],
        annotations=[dict(
            x=1.005, y=0.24, xref="paper", yref="paper", xanchor="left", yanchor="top",
            showarrow=False, align="left", width=285,
            bgcolor="rgba(7,12,20,0.97)", bordercolor="#3b4a60", borderwidth=1,
            borderpad=7, font=dict(size=10, color=COLORS["muted"]),
            text=(f"<b>Mesh and geometry</b><br>Earth/Moon: {body_resolution[0]} x {body_resolution[1]}<br>"
                  f"Atmosphere: {atmosphere_resolution[0]} x {atmosphere_resolution[1]}<br>"
                  f"Solar shadow patch: {shadow_resolution} x {shadow_resolution}<br>"
                  f"Sun-Earth: {np.linalg.norm(state.sun_gcrf_km)/1e6:.3f} million km<br>"
                  f"Earth-Moon: {np.linalg.norm(state.moon_gcrf_km):,.0f} km<br>"
                  "Local and heliocentric modes use the same double-precision physical state."),
        )],
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output, include_plotlyjs=include_plotlyjs, full_html=True,
                   config={"displaylogo": False, "responsive": True, "scrollZoom": True,
                           "plotGlPixelRatio": 2.0})
    return str(output)


def validate_system_animation(kind: str | ReferenceDefinition = "solar", *, n_frames: int = 25) -> dict:
    """Numerical audit for the synchronized physical-system scene."""
    event = build_reference_event(kind, n_frames=max(17, int(n_frames)))
    geometry = _build_scene_geometry(event)
    states = [
        _build_inertial_state(event, float(jd_value), geometry.reference_cross_axis_gcrf, n_azimuth=16)
        for jd_value in event.jd
    ]
    penetrations = {"earth": 0, "moon": 0}
    for state in states:
        counts = bundle_penetrations(state.bundle_native)
        penetrations["earth"] += counts["earth"]
        penetrations["moon"] += counts["moon"]
    earth_move = float(np.linalg.norm(states[-1].earth_heliocentric_km-states[0].earth_heliocentric_km))
    moon_move = float(np.linalg.norm(states[-1].moon_gcrf_km-states[0].moon_gcrf_km))
    event_hours = float((event.jd[-1]-event.jd[0])*24.0)
    return {
        "event": event.definition.key,
        "frame_count": len(states),
        "event_duration_hours": event_hours,
        "earth_heliocentric_displacement_km": earth_move,
        "earth_rotation_degrees": event_hours*360.0/23.9344696,
        "moon_geocentric_displacement_km": moon_move,
        "sun_earth_distance_min_km": float(min(np.linalg.norm(s.sun_gcrf_km) for s in states)),
        "sun_earth_distance_max_km": float(max(np.linalg.norm(s.sun_gcrf_km) for s in states)),
        "earth_moon_distance_min_km": float(min(np.linalg.norm(s.moon_gcrf_km) for s in states)),
        "earth_moon_distance_max_km": float(max(np.linalg.norm(s.moon_gcrf_km) for s in states)),
        "ray_penetrations": penetrations,
        "local_coordinate_max_abs_km": float(np.max(np.abs(geometry.event_moon_local_km))),
        "heliocentric_coordinate_max_abs_au": float(np.max(np.abs(geometry.orbit_au))),
    }
