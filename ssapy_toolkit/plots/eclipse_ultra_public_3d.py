"""Public-facing ultra-resolution eclipse scenes.

The V13 renderer keeps the validated finite-Sun geometry while correcting the
WebGL and lighting choices that made earlier "ultra" views unstable:

* local Earth-Moon and heliocentric Sun-Earth-Moon views are separate Plotly
  scenes, so one-AU coordinates never share a WebGL buffer with kilometre-scale
  lunar tangencies;
* WGS-84 Earth and spherical Sun/Moon meshes are split into compact indexed
  draw calls, avoiding 16-bit WebGL index overflow without resampling or
  deforming any surface;
* day/night incidence and finite-disc eclipse attenuation are computed once in
  linear light and sent to ambient-only WebGL meshes, preventing a second
  camera-dependent material pass from distorting the terminator;
* the exact Sun-occluder-target impact plane is used for the public ray view;
* direct light and blocked-light geometry are separate and are clipped at the
  first opaque surface;
* the public default shows a small deterministic sample of real
  photosphere-to-surface paths, while exact tangent families and optical
  volumes remain optional diagnostics.

The packaged reference products use the NASA/GSFC validated event definitions.
When LLNL SSAPy and SSAPy-Toolkit are installed, the existing adapters are used
for ephemeris diagnostics and GCRF/ITRF orientation transforms.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ssapy_toolkit.io.eclipse_asset_resolver import resolve_image
from typing import Iterable, Sequence
import base64
import copy
import html
import json
import math

import numpy as np

from ssapy_toolkit.compute.eclipse_photometry import quadrature_sample_weight, resolve_limb_darkening
from ssapy_toolkit.compute.eclipse_core import sample_spherical_photosphere
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs

from ssapy_toolkit.compute.eclipse_brightness import (
    available_backends,
    ephemeris_positions,
    ellipsoid_surface_in_direction,
    gcrf_to_itrf_km,
    itrf_to_gcrf_km,
    ray_ellipsoid_intersections,
    ray_sphere_intersections,
)
from ssapy_toolkit.compute.eclipse_state import build_event, event_gcrf_to_itrf_km, event_itrf_to_gcrf_km, resample_event
from ssapy_toolkit.plots.globe_orbit_daynight_plotly import (
    _earth_eclipse_shadow_trace,
    _earth_surface_data,
    _wgs84_vertices,
)
from ssapy_toolkit.compute.eclipse_local_finite_geometry import impact_plane_basis, shadow_axis_world
from ssapy_toolkit.plots.moon_render import _moon_surface_data, _moon_unit_mesh
from ssapy_toolkit.plots.eclipse_plotly_mesh import equal_unit_aspect, indexed_mesh_chunks, rgb_strings
from ssapy_toolkit.plots.eclipse_system_raytrace_3d import (
    ASSET_DIR,
    COLORS,
    _build_inertial_state,
    _build_scene_geometry,
    _native_to_gcrf,
    _scene_coordinates,
    _surface_footprint_trace,
    _transform_xyz_trace,
)
from ssapy_toolkit.compute.eclipse_reference_events import (
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
    SOLAR_GREATEST_SITE_LAT_DEG,
    SOLAR_GREATEST_SITE_LON_EAST_DEG,
    build_reference_event,
    jd_to_datetime,
    solar_central_line_wgs84,
    itrf_surface_point,
    _unit,
)
from ssapy_toolkit.compute.eclipse_raytrace import (
    LUNAR_DANJON_EARTH_RADIUS_KM,
    bundle_penetrations,
    solar_footprint_segments,
    tangent_residuals,
)


PUBLIC_COLORS = {
    "background": "#02040a",
    "panel": "#07101c",
    "grid": "#233044",
    "text": "#f1f6fc",
    "muted": "#9fb0c4",
    "sun": "#ffe56d",
    "sun_glow": "rgba(255,227,109,0.16)",
    "blocked": "#e27aff",
    "blocked_glow": "rgba(226,122,255,0.16)",
    "umbra": "#ff665c",
    "umbra_glow": "rgba(255,102,92,0.17)",
    "penumbra": "#6fc8ff",
    "penumbra_glow": "rgba(111,200,255,0.14)",
    "footprint": "#ffd36d",
    "totality_path": "#ffb547",
}


@dataclass(frozen=True)
class PublicPeakState:
    event: ReferenceEvent
    state: object
    jd_utc: float
    basis_world_from_scene: np.ndarray
    earth_local_km: np.ndarray
    moon_local_km: np.ndarray
    sun_local_km: np.ndarray
    axis_impact_km: float
    ssapy_diagnostics: dict[str, object]


@dataclass(frozen=True)
class PublicQuality:
    body_lat: int
    body_lon: int
    atmosphere_lat: int
    atmosphere_lon: int
    shadow_resolution: int
    ray_azimuth: int


QUALITY = {
    "motion": PublicQuality(91, 180, 41, 80, 161, 48),
    "motion_high": PublicQuality(121, 240, 61, 120, 201, 64),
    "high": PublicQuality(181, 360, 81, 160, 301, 72),
    "ultra": PublicQuality(241, 480, 101, 200, 401, 96),
    "cinema": PublicQuality(301, 600, 121, 240, 501, 128),
}


def _quality(name: str) -> PublicQuality:
    key = str(name).strip().lower()
    if key not in QUALITY:
        raise ValueError("quality must be 'motion', 'motion_high', 'high', 'ultra', or 'cinema'")
    return QUALITY[key]


def _orthonormalize_basis(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(3, 3)
    x = _unit(values[:, 0])
    z = values[:, 2]-x*float(np.dot(x, values[:, 2]))
    if np.linalg.norm(z) < 1.0e-12:
        z = np.array([0.0, 0.0, 1.0])-x*x[2]
    z = _unit(z)
    y = _unit(np.cross(z, x))
    z = _unit(np.cross(x, y))
    return np.stack([x, y, z], axis=1)


def _impact_basis_gcrf(event: ReferenceEvent, bundle_native) -> tuple[np.ndarray, float]:
    _, native_basis, impact = impact_plane_basis(bundle_native)
    if str(bundle_native.frame).upper().startswith("GCRF"):
        return _orthonormalize_basis(native_basis), float(impact)
    columns = [
        _native_to_gcrf(native_basis[:, index], event, event.greatest_jd).reshape(3)
        for index in range(3)
    ]
    return _orthonormalize_basis(np.stack(columns, axis=1)), float(impact)


def _ssapy_audit(event: ReferenceEvent, reference_sun: np.ndarray,
                 reference_moon: np.ndarray) -> dict[str, object]:
    backends = available_backends()
    capabilities = event.metadata.get("runtime_capabilities", {})
    result: dict[str, object] = {
        "llnl_ssapy_available": bool(
            capabilities.get("llnl_ssapy_importable", backends["llnl_ssapy"])
        ),
        "ssapy_toolkit_available": bool(
            capabilities.get("ssapy_toolkit_importable", backends["ssapy_toolkit"])
        ),
        "reference_backend": str(event.backend),
        "requested_backend": str(event.metadata.get("backend_requested", "reference")),
        "selected_backend": str(event.metadata.get("state_source", "reference")),
        "frame_backend": str(event.metadata.get("frame_backend", event.frame)),
        "fallback_reason": str(event.metadata.get("backend_fallback_reason", "")),
        "ephemeris_backend": str(event.metadata.get("ephemeris_backend", "not executed")),
    }
    if result["selected_backend"] in {"ssapy", "ssapy-core"}:
        result.update({
            "ssapy_sun_reference_delta_km": float(
                event.metadata.get("ssapy_sun_reference_max_km", 0.0)
            ),
            "ssapy_moon_reference_delta_km": float(
                event.metadata.get("ssapy_moon_reference_max_km", 0.0)
            ),
        })
        return result
    if backends["llnl_ssapy"]:
        try:
            ephem = ephemeris_positions([event.greatest_jd], backend="ssapy")
            result.update({
                "ephemeris_backend": ephem.backend,
                "ssapy_sun_reference_delta_km": float(
                    np.linalg.norm(ephem.sun_gcrf_km[0]-reference_sun)
                ),
                "ssapy_moon_reference_delta_km": float(
                    np.linalg.norm(ephem.moon_gcrf_km[0]-reference_moon)
                ),
            })
        except Exception as exc:  # optional dependency must never break rendering
            result["ephemeris_backend"] = f"failed: {type(exc).__name__}: {exc}"
    return result


def build_public_peak_state(kind: str | ReferenceDefinition | ReferenceEvent = "solar",
                            *, ray_azimuth: int = 96,
                            backend: str = "auto",
                            solar_scope: str = "global",
                            observer: object | None = None) -> PublicPeakState:
    if isinstance(kind, ReferenceEvent):
        definition = kind.definition
        scaffold = kind
        event = resample_event(kind, [kind.greatest_jd])
    else:
        definition = (
            kind if isinstance(kind, ReferenceDefinition)
            else (SOLAR_2024 if str(kind).lower().startswith("solar") else LUNAR_2025)
        )
        scaffold = build_event(
            definition, backend=backend, n_frames=25,
            solar_scope=solar_scope, observer=observer,
        )
        event = build_event(
            definition,
            backend=str(scaffold.metadata.get("state_source", backend)),
            jd=[scaffold.greatest_jd],
            solar_scope=solar_scope,
            observer=observer or scaffold.metadata.get("observer_model"),
        )
    legacy_geometry = _build_scene_geometry(event)
    state = _build_inertial_state(
        event,
        event.greatest_jd,
        legacy_geometry.reference_cross_axis_gcrf,
        n_azimuth=max(24, int(ray_azimuth)),
    )
    basis, impact = _impact_basis_gcrf(event, state.bundle_native)
    earth = np.zeros(3)
    moon = _scene_coordinates(state.moon_gcrf_km.reshape(1, 3), np.zeros(3), basis, 1.0)[0]
    sun = _scene_coordinates(state.sun_gcrf_km.reshape(1, 3), np.zeros(3), basis, 1.0)[0]
    audit = _ssapy_audit(event, state.sun_gcrf_km, state.moon_gcrf_km)
    return PublicPeakState(
        event=event,
        state=state,
        jd_utc=float(event.greatest_jd),
        basis_world_from_scene=basis,
        earth_local_km=earth,
        moon_local_km=moon,
        sun_local_km=sun,
        axis_impact_km=impact,
        ssapy_diagnostics=audit,
    )


def build_public_frame_state(
    event: ReferenceEvent,
    jd_utc: float,
    *,
    basis_world_from_scene: np.ndarray | None = None,
    reference_cross_axis_gcrf: np.ndarray | None = None,
    ray_azimuth: int = 48,
    run_ssapy_audit: bool = False,
) -> PublicPeakState:
    """Build one UTC state in a fixed greatest-eclipse display frame.

    The physical state, Earth orientation, Moon position, common tangents,
    target intersections, and surface illumination are recomputed at
    ``jd_utc``.  The display basis is intentionally fixed at greatest eclipse
    so an animation shows actual motion instead of a camera frame that rotates
    with the ray bundle.
    """
    if basis_world_from_scene is None or reference_cross_axis_gcrf is None:
        peak_event = build_event(
            event.definition,
            backend=str(event.metadata.get("state_source", "reference")),
            jd=[event.greatest_jd],
            solar_scope=str(event.metadata.get("solar_scope", "global")),
            observer=event.metadata.get("observer_model"),
        )
        legacy_peak = _build_scene_geometry(peak_event)
        peak_state = _build_inertial_state(
            peak_event,
            peak_event.greatest_jd,
            legacy_peak.reference_cross_axis_gcrf,
            n_azimuth=max(24, int(ray_azimuth)),
        )
        basis_world_from_scene, _ = _impact_basis_gcrf(
            peak_event, peak_state.bundle_native
        )
        reference_cross_axis_gcrf = legacy_peak.reference_cross_axis_gcrf

    state = _build_inertial_state(
        event,
        float(jd_utc),
        np.asarray(reference_cross_axis_gcrf, dtype=float),
        n_azimuth=max(24, int(ray_azimuth)),
    )
    basis = np.asarray(basis_world_from_scene, dtype=float)
    earth = np.zeros(3, dtype=float)
    moon = _scene_coordinates(
        state.moon_gcrf_km.reshape(1, 3), earth, basis, 1.0
    )[0]
    sun = _scene_coordinates(
        state.sun_gcrf_km.reshape(1, 3), earth, basis, 1.0
    )[0]
    _, _, impact = impact_plane_basis(state.bundle_native)
    audit = (
        _ssapy_audit(event, state.sun_gcrf_km, state.moon_gcrf_km)
        if run_ssapy_audit
        else {
            "llnl_ssapy_available": bool(available_backends()["llnl_ssapy"]),
            "ssapy_toolkit_available": bool(available_backends()["ssapy_toolkit"]),
            "reference_backend": str(event.backend),
            "ephemeris_backend": "not executed for animation frame",
        }
    )
    return PublicPeakState(
        event=event,
        state=state,
        jd_utc=float(jd_utc),
        basis_world_from_scene=basis,
        earth_local_km=earth,
        moon_local_km=moon,
        sun_local_km=sun,
        axis_impact_km=float(impact),
        ssapy_diagnostics=audit,
    )


def _rgb_strings(rgb: np.ndarray) -> list[str]:
    values = np.rint(np.clip(np.asarray(rgb, dtype=float), 0.0, 1.0)*255.0).astype(np.uint8)
    return [f"rgb({r},{g},{b})" for r, g, b in values]


def _scene_light_position(peak: PublicPeakState, center_local: np.ndarray) -> dict[str, float]:
    # Plotly constrains each light-position component to +/-100,000 and
    # interprets the vector as a distant scene light.  Use the physical Sun
    # direction but not the body's translated scene coordinate.
    direction = _unit(peak.sun_local_km-center_local)
    point = direction*90_000.0
    return {"x": float(point[0]), "y": float(point[1]), "z": float(point[2])}


def _public_earth_mesh(peak: PublicPeakState, quality: PublicQuality) -> list[go.Mesh3d]:
    """Render the exact WGS-84 body with the eclipse baked into irradiance.

    The V12 public view sent the unshaded albedo map to Plotly and placed a
    second, very large shadow patch just above it.  On some WebGL paths the
    >65k-vertex overlay folded or detached from the globe, while Plotly's
    material light double-shaded the surface.  V13 computes day/night and the
    finite-Sun Moon occultation once in linear light, then draws only the
    physical ellipsoid.  The indexed surface is split into safe WebGL chunks;
    its coordinates are never rescaled or deformed.
    """
    state = peak.state
    shadow_center = state.moon_gcrf_km if peak.event.mode == "solar" else None
    shadow_radius = R_MOON_MEAN_KM if peak.event.mode == "solar" else None
    data = _earth_surface_data(
        _unit(state.sun_gcrf_km),
        n_lat=quality.body_lat,
        n_lon=quality.body_lon,
        radius_scale=1.0,
        center=(0.0, 0.0, 0.0),
        shadow_body_center_km=shadow_center,
        shadow_body_radius_km=shadow_radius,
        time_jd=float(peak.jd_utc),
        sun_position_km=state.sun_gcrf_km,
        physical_center_km=(0.0, 0.0, 0.0),
        night_floor=0.0015,
        texture_path=resolve_image("earth_albedo").path,
        exposure=1.08,
        view_hat=_unit(state.sun_gcrf_km),
        specular_strength=0.0,
    )
    vertices = _scene_coordinates(
        data.display_vertices, np.zeros(3), peak.basis_world_from_scene, 1.0,
    )
    return indexed_mesh_chunks(
        vertices,
        data.faces,
        vertexcolor=rgb_strings(data.shaded_rgb_srgb),
        name="Earth — WGS-84, finite-Sun shaded",
        showlegend=True,
        legendgroup="body-earth",
        legendgrouptitle=dict(text="Bodies"),
        hovertemplate="Earth — WGS-84 ellipsoid; finite-Sun surface irradiance<extra></extra>",
        flatshading=False,
        # Surface colour already contains the physical Sun incidence and
        # eclipse visibility.  Ambient-only Plotly lighting prevents a second
        # camera/GPU-dependent shading pass from changing the terminator.
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0,
                      roughness=1.0, fresnel=0.0),
    )


def _public_moon_mesh(peak: PublicPeakState, quality: PublicQuality) -> list[go.Mesh3d]:
    """Render a true-radius spherical Moon with physical finite-Sun shading."""
    from ssapy_toolkit.coordinates.eclipse_lunar_attitude import moon_attitude_for_event

    state = peak.state
    data = _moon_surface_data(
        center=state.moon_gcrf_km,
        radius=R_MOON_MEAN_KM,
        sun_hat=_unit(state.sun_gcrf_km),
        real_center_km=state.moon_gcrf_km,
        mode=peak.event.mode,
        n_lat=quality.body_lat,
        n_lon=quality.body_lon,
        real_sun_position_km=state.sun_gcrf_km,
        ambient_floor=0.002,
        texture_path=resolve_image("moon_albedo").path,
        view_hat=-_unit(state.moon_gcrf_km),
        exposure=1.10,
        relief_exaggeration=0.0,
        eclipse_occluder_radius_km=LUNAR_DANJON_EARTH_RADIUS_KM,
        body_to_gcrf=moon_attitude_for_event(peak.event, peak.jd_utc).body_to_gcrf,
    )
    vertices = _scene_coordinates(
        data.display_vertices, np.zeros(3), peak.basis_world_from_scene, 1.0,
    )
    return indexed_mesh_chunks(
        vertices,
        data.faces,
        vertexcolor=rgb_strings(data.shaded_rgb_srgb),
        name="Moon — spherical photomosaic, finite-Sun shaded",
        showlegend=True,
        legendgroup="body-moon",
        hovertemplate="Moon — mean solid radius 1,737.4 km<extra></extra>",
        flatshading=False,
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0,
                      roughness=1.0, fresnel=0.0),
    )


def _atmosphere_shell(peak: PublicPeakState, quality: PublicQuality) -> list[go.Mesh3d]:
    """Thin, chunked WGS-84 atmosphere shell.

    The shell is intentionally subtle and hidden by default in the public
    scientific view.  It can be enabled from the legend without becoming a
    second opaque-looking globe or exposing a high-index WebGL mesh.
    """
    vertices, _, faces, *_ = _wgs84_vertices(
        quality.atmosphere_lat, quality.atmosphere_lon, altitude_km=92.0,
    )
    data = _earth_surface_data(
        _unit(peak.state.sun_gcrf_km),
        n_lat=quality.atmosphere_lat,
        n_lon=quality.atmosphere_lon,
        time_jd=float(peak.jd_utc),
        sun_position_km=peak.state.sun_gcrf_km,
        texture_path=resolve_image("earth_albedo").path,
        exposure=1.0,
        specular_strength=0.0,
    )
    oriented = vertices @ data.rotation_rows
    local = _scene_coordinates(oriented, np.zeros(3), peak.basis_world_from_scene, 1.0)
    return indexed_mesh_chunks(
        local,
        faces,
        color="#3f8cff",
        name="Atmospheric shell (optional)",
        showlegend=True,
        legendgroup="body-atmosphere",
        hoverinfo="skip",
        flatshading=False,
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0,
                      roughness=1.0, fresnel=0.0),
        opacity=0.045,
        visible="legendonly",
    )


def _ray_trace(points: np.ndarray, *, name: str, color: str, glow: str,
               width: float, dash: str | None = None, showlegend: bool = True,
               legendgroup: str = "rays", visible=True,
               legend_title: str | None = None,
               add_glow: bool = False) -> list[go.Scatter3d]:
    """Draw a precision-safe representative ray.

    Thick neon halos made the V12 paths appear wider than the Moon and could
    hide the exact surface stop.  Scientific rays are now thin by default;
    an optional restrained halo is available only for presentation views.
    """
    values = np.asarray(points, dtype=float).reshape(-1, 3)
    core_line = dict(color=color, width=width)
    if dash is not None:
        core_line["dash"] = dash
    traces: list[go.Scatter3d] = []
    if add_glow:
        glow_line = dict(color=glow, width=max(width*1.8, width+2.5))
        if dash is not None:
            glow_line["dash"] = dash
        traces.append(go.Scatter3d(
            x=values[:, 0], y=values[:, 1], z=values[:, 2], mode="lines",
            line=glow_line, name=f"{name} glow", showlegend=False,
            legendgroup=legendgroup, visible=visible, hoverinfo="skip",
            connectgaps=False,
        ))
    traces.append(go.Scatter3d(
        x=values[:, 0], y=values[:, 1], z=values[:, 2], mode="lines",
        line=core_line, name=name, showlegend=showlegend,
        legendgroup=legendgroup,
        legendgrouptitle=(dict(text=legend_title) if legend_title else None),
        visible=visible, connectgaps=False,
        hovertemplate=f"{html.escape(name)}<extra></extra>",
    ))
    return traces


def _distributed_surface_indices(points: np.ndarray, normals: np.ndarray,
                                 sun_hat: np.ndarray, n_samples: int,
                                 *, preferred: int | None = None,
                                 center: np.ndarray | None = None) -> list[int]:
    """Deterministic farthest-point samples on the illuminated hemisphere.

    ``points`` may be absolute GCRF positions.  Sampling their normalized
    absolute coordinates works for Earth at the geocentric origin but
    collapses every lunar vertex into almost the same direction because the
    400,000 km Moon-centre translation dominates its 1,737 km radius.  V13
    therefore samples body-relative directions explicitly.
    """
    pts = np.asarray(points, dtype=float)
    nrm = np.asarray(normals, dtype=float)
    mu = nrm @ _unit(sun_hat)
    candidates = np.where(mu > 0.08)[0]
    if len(candidates) == 0:
        candidates = np.arange(len(pts))
    origin = (np.zeros(3, dtype=float) if center is None
              else np.asarray(center, dtype=float).reshape(3))
    relative = pts[candidates]-origin
    unit = relative / np.maximum(np.linalg.norm(relative, axis=1, keepdims=True), 1e-12)
    selected_local: list[int] = []
    if preferred is not None:
        where = np.where(candidates == int(preferred))[0]
        if len(where):
            selected_local.append(int(where[0]))
    if not selected_local:
        selected_local.append(int(np.argmax(mu[candidates])))
    min_d2 = np.sum((unit-unit[selected_local[0]])**2, axis=1)
    while len(selected_local) < min(int(n_samples), len(candidates)):
        score = min_d2 + 0.08*np.clip(mu[candidates], 0.0, 1.0)
        score[selected_local] = -np.inf
        best = float(np.max(score))
        # Symmetric surface samples often have mathematically identical
        # scores.  Subtracting a large translated body centre can perturb
        # those ties at the 1e-11 level, causing ray identities to swap.
        # Choose the lowest original vertex index within a tight tolerance
        # of the optimum so sampling remains stable under translation and
        # across animation frames.
        tied = np.where(score >= best-1.0e-9*max(1.0, abs(best)))[0]
        nxt = int(tied[np.argmin(candidates[tied])])
        selected_local.append(nxt)
        min_d2 = np.minimum(min_d2, np.sum((unit-unit[nxt])**2, axis=1))
    return [int(candidates[index]) for index in selected_local]


@dataclass(frozen=True)
class PhotosphericPathSample:
    """One equal-area sample of the visible solar photosphere."""

    points_gcrf_km: np.ndarray
    blocked: bool
    weight: float
    disk_radius_fraction: float
    endpoint_body: str


def _fibonacci_photosphere_disk(n_samples: int) -> np.ndarray:
    """Deterministic equal-area samples over the apparent solar disk.

    The first sample is the disk center; the remaining points follow a Vogel
    spiral and approach the limb.  Unlike the former target-first construction,
    this spans the complete 1,391,400 km solar diameter instead of a tiny
    photospheric patch.
    """
    n = max(1, int(n_samples))
    if n == 1:
        return np.zeros((1, 2), dtype=float)
    points = np.zeros((n, 2), dtype=float)
    golden = math.pi*(3.0-math.sqrt(5.0))
    for index in range(1, n):
        radius = math.sqrt((index-0.5)/(n-0.5))
        angle = index*golden
        points[index] = [radius*math.cos(angle), radius*math.sin(angle)]
    return points


def _disk_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = _unit(axis)
    ref = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.88 else np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(ref, axis))
    v = _unit(np.cross(axis, u))
    return u, v


def photospheric_target_gcrf(peak: PublicPeakState) -> tuple[np.ndarray, str]:
    """Return the physical point whose finite-Sun integral the rays explain."""
    if peak.event.mode == "lunar":
        return np.asarray(peak.state.moon_gcrf_km, dtype=float), "Moon centerline"

    scope = str(peak.event.metadata.get("solar_scope", "global"))
    if scope == "local":
        lat = float(peak.event.metadata.get("observer_lat_deg", SOLAR_GREATEST_SITE_LAT_DEG))
        lon = float(peak.event.metadata.get("observer_lon_east_deg", SOLAR_GREATEST_SITE_LON_EAST_DEG))
        observer_itrf = itrf_surface_point(lat, lon, 0.0)
        observer_gcrf = event_itrf_to_gcrf_km(peak.event, observer_itrf, peak.jd_utc)
        return np.asarray(observer_gcrf, dtype=float), "NASA greatest-eclipse observer"

    axis_path = np.asarray(peak.state.shadow_axis_gcrf_km, dtype=float).reshape(-1, 3)
    if len(axis_path) >= 2:
        endpoint = axis_path[-1]
        radius = float(np.linalg.norm(endpoint))
        if RP_KM*0.95 <= radius <= RE_KM*1.05:
            return endpoint, "instantaneous WGS-84 shadow-axis intercept"
    return np.zeros(3, dtype=float), "Earth center"


def _first_earth_hit_for_event(
    peak: PublicPeakState, source: np.ndarray, direction: np.ndarray,
) -> tuple[float, np.ndarray] | None:
    # Transform the origin and direction independently.  Subtracting two
    # transformed ~1-AU positions separated by only one kilometre loses ray
    # direction precision through catastrophic cancellation and can move the
    # apparent WGS-84 endpoint by decimetres.  Terrestrial frame transforms
    # are rotations about the common geocentre, so a direction vector can be
    # rotated directly.
    source_itrf = np.asarray(
        event_gcrf_to_itrf_km(peak.event, source, peak.jd_utc), dtype=float,
    )
    direction_itrf = _unit(np.asarray(
        event_gcrf_to_itrf_km(peak.event, direction, peak.jd_utc), dtype=float,
    ))
    roots = ray_ellipsoid_intersections(
        source_itrf, direction_itrf, np.zeros(3), EARTH_AXES_KM,
    )
    roots = np.asarray(roots, dtype=float)
    roots = roots[roots > 1.0e-7]
    if not len(roots):
        return None
    distance = float(roots[0])
    endpoint_itrf = source_itrf+distance*direction_itrf
    endpoint = np.asarray(
        event_itrf_to_gcrf_km(peak.event, endpoint_itrf, peak.jd_utc),
        dtype=float,
    )
    return distance, endpoint


def sampled_photospheric_path_records(
    peak: PublicPeakState, *, n_samples: int = 13,
    photometry: str = "quadratic-visible",
) -> list[PhotosphericPathSample]:
    """Trace equal-area full-photosphere samples to one physical target point.

    Each source lies on the real visible solar hemisphere.  The ray is clipped
    at the first positive solid Moon or WGS-84 Earth intersection.  The weight
    applies a simple visible-continuum limb-darkening law and is normalized so
    the sample weights sum to one.  These rays are an explanatory quadrature of
    the same finite-disc geometry used by the surface shader.
    """
    state = peak.state
    sun = np.asarray(state.sun_gcrf_km, dtype=float)
    moon = np.asarray(state.moon_gcrf_km, dtype=float)
    central_target, _ = photospheric_target_gcrf(peak)
    source_axis = _unit(central_target-sun)
    source_u, source_v = _disk_basis(source_axis)
    core_samples = sample_spherical_photosphere(
        sun, R_SUN_KM, central_target, count=n_samples, law=photometry,
    )
    disk = np.asarray(core_samples.disk_xy, dtype=float)

    # The solar case illustrates the global irradiance field: one ray targets
    # the eclipse point and the remaining samples land across the Sun-facing
    # WGS-84 hemisphere.  The lunar case samples the Moon's illuminated disk.
    # This keeps the full-disk source quadrature honest while also showing that
    # a solar eclipse darkens only a narrow region of an otherwise illuminated
    # Earth.
    targets: list[np.ndarray] = []
    if peak.event.mode == "solar":
        earth_to_sun = _unit(sun)
        target_u, target_v = _disk_basis(earth_to_sun)
        for index, (x, y) in enumerate(disk):
            if index == 0:
                targets.append(np.asarray(central_target, dtype=float))
                continue
            # Keep the explanatory target samples away from the exact limb,
            # where small numerical changes can switch visibility abruptly.
            xt, yt = 0.86*float(x), 0.86*float(y)
            mu_t = math.sqrt(max(0.0, 1.0-xt*xt-yt*yt))
            direction_gcrf = _unit(mu_t*earth_to_sun+xt*target_u+yt*target_v)
            direction_itrf = event_gcrf_to_itrf_km(peak.event, direction_gcrf, peak.jd_utc)
            surface_itrf = ellipsoid_surface_in_direction(direction_itrf, EARTH_AXES_KM)
            targets.append(np.asarray(
                event_itrf_to_gcrf_km(peak.event, surface_itrf, peak.jd_utc),
                dtype=float,
            ))
    else:
        moon_to_sun = _unit(sun-moon)
        target_u, target_v = _disk_basis(moon_to_sun)
        for x, y in disk:
            xt, yt = 0.90*float(x), 0.90*float(y)
            mu_t = math.sqrt(max(0.0, 1.0-xt*xt-yt*yt))
            surface_hat = _unit(mu_t*moon_to_sun+xt*target_u+yt*target_v)
            targets.append(moon+R_MOON_MEAN_KM*surface_hat)

    records: list[PhotosphericPathSample] = []
    raw_weights: list[float] = []

    for (x, y), target in zip(disk, targets):
        rho2 = float(x*x+y*y)
        mu = math.sqrt(max(0.0, 1.0-rho2))
        source = np.asarray(core_samples.points[len(records)], dtype=float)
        direction = _unit(target-source)

        moon_roots = np.asarray(
            ray_sphere_intersections(source, direction, moon, R_MOON_MEAN_KM),
            dtype=float,
        )
        moon_roots = moon_roots[moon_roots > 1.0e-7]
        moon_hit = None
        if len(moon_roots):
            distance = float(moon_roots[0])
            moon_hit = (distance, source+distance*direction)

        earth_hit = _first_earth_hit_for_event(peak, source, direction)
        candidates: list[tuple[float, np.ndarray, str]] = []
        if moon_hit is not None:
            candidates.append((moon_hit[0], moon_hit[1], "Moon"))
        if earth_hit is not None:
            candidates.append((earth_hit[0], earth_hit[1], "Earth"))
        if candidates:
            distance, endpoint, body = min(candidates, key=lambda item: item[0])
        else:
            distance = float(np.linalg.norm(target-source))
            endpoint, body = target, "target point"

        occluder = "Moon" if peak.event.mode == "solar" else "Earth"
        blocked = body == occluder
        # Use the same configurable centre-to-limb law as the body shader.
        # Source geometry and first-surface clipping remain exact.
        raw_weight = float(core_samples.weights[len(records)])
        raw_weights.append(raw_weight)
        records.append(PhotosphericPathSample(
            points_gcrf_km=np.vstack([source, endpoint]),
            blocked=bool(blocked),
            weight=float(raw_weight),
            disk_radius_fraction=math.sqrt(rho2),
            endpoint_body=body,
        ))

    total = float(sum(raw_weights)) or 1.0
    return [
        PhotosphericPathSample(
            points_gcrf_km=record.points_gcrf_km,
            blocked=record.blocked,
            weight=record.weight/total,
            disk_radius_fraction=record.disk_radius_fraction,
            endpoint_body=record.endpoint_body,
        )
        for record in records
    ]


def _sampled_photospheric_paths(
    peak: PublicPeakState, *, n_samples: int = 13,
    photometry: str = "quadratic-visible",
) -> list[tuple[np.ndarray, bool]]:
    """Backward-compatible path/blocked view of the full-disk samples."""
    return [
        (record.points_gcrf_km, record.blocked)
        for record in sampled_photospheric_path_records(peak, n_samples=n_samples, photometry=photometry)
    ]


def _sampled_irradiance_traces(peak: PublicPeakState,
                               bounds: Sequence[tuple[float, float]],
                               *, n_samples: int = 13) -> list[go.Scatter3d]:
    traces: list[go.Scatter3d] = []
    for index, (path_gcrf, blocked) in enumerate(
        _sampled_photospheric_paths(peak, n_samples=n_samples)
    ):
        local = _scene_coordinates(
            path_gcrf, np.zeros(3), peak.basis_world_from_scene, 1.0,
        )
        clipped = _clip_segment_box(local, bounds)
        if len(clipped) < 2:
            clipped = np.empty((0, 3), dtype=float)
        traces.append(go.Scatter3d(
            x=clipped[:, 0], y=clipped[:, 1], z=clipped[:, 2],
            mode="lines",
            line=dict(
                color=("rgba(255,219,99,0.78)" if blocked
                       else "rgba(255,238,155,0.46)"),
                width=(1.7 if blocked else 1.25),
            ),
            name="Sampled finite-Sun irradiance paths",
            showlegend=(index == 0),
            legendgroup="irradiance",
            legendgrouptitle=(dict(text="Physical illumination") if index == 0 else None),
            hovertemplate=(
                "Photospheric path — clipped at occluder<extra></extra>"
                if blocked else
                "Photospheric path — reaches target surface<extra></extra>"
            ),
            connectgaps=False,
        ))
    return traces


def _clip_segment_box(points: np.ndarray, bounds: Sequence[tuple[float, float]]) -> np.ndarray:
    """Clip each polyline segment to an axis-aligned box (Liang-Barsky)."""
    values = np.asarray(points, dtype=float).reshape(-1, 3)
    pieces: list[np.ndarray] = []
    for a, b in zip(values[:-1], values[1:]):
        d = b-a
        t0, t1 = 0.0, 1.0
        valid = True
        for axis, (lower, upper) in enumerate(bounds):
            if abs(float(d[axis])) < 1.0e-15:
                if float(a[axis]) < lower or float(a[axis]) > upper:
                    valid = False
                    break
                continue
            q0 = (lower-float(a[axis]))/float(d[axis])
            q1 = (upper-float(a[axis]))/float(d[axis])
            enter, leave = min(q0, q1), max(q0, q1)
            t0, t1 = max(t0, enter), min(t1, leave)
            if t1 < t0:
                valid = False
                break
        if not valid:
            continue
        segment = np.vstack([a+t0*d, a+t1*d])
        if pieces and np.linalg.norm(pieces[-1][-1]-segment[0]) < 1.0e-6:
            pieces.append(segment[1:])
        else:
            if pieces:
                pieces.append(np.full((1, 3), np.nan))
            pieces.append(segment)
    return np.vstack(pieces) if pieces else np.empty((0, 3), dtype=float)


def _local_bounds(peak: PublicPeakState) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    x_values = [0.0, float(peak.moon_local_km[0])]
    xmin = min(x_values)-32_000.0
    xmax = max(x_values)+32_000.0
    transverse = max(
        4.2*RE_KM,
        abs(float(peak.moon_local_km[1]))+18_000.0,
        abs(float(peak.moon_local_km[2]))+18_000.0,
    )
    return (xmin, xmax), (-transverse, transverse), (-transverse, transverse)


def _local_ray_paths(peak: PublicPeakState, bounds):
    state = peak.state
    basis = peak.basis_world_from_scene

    def local(points):
        return _scene_coordinates(points, np.zeros(3), basis, 1.0)

    central = _clip_segment_box(local(state.central_gcrf_km), bounds)
    blocked = _clip_segment_box(local(state.shadow_axis_gcrf_km), bounds)
    umbra = tuple(_clip_segment_box(local(points), bounds) for points in state.umbra_gcrf_km)
    penumbra = tuple(_clip_segment_box(local(points), bounds) for points in state.penumbra_gcrf_km)
    return central, blocked, umbra, penumbra


def _volume_mesh(peak: PublicPeakState, family: str, *, color: str,
                 opacity: float, visible=True) -> go.Mesh3d:
    lines = (
        peak.state.umbra_bundle_gcrf_km if family == "umbra"
        else peak.state.penumbra_bundle_gcrf_km
    )
    basis = peak.basis_world_from_scene
    tangent_ring = []
    endpoint_ring = []
    for line in lines:
        local = _scene_coordinates(line, np.zeros(3), basis, 1.0)
        tangent_ring.append(local[1])
        endpoint_ring.append(local[-1])
    tangent_ring = np.asarray(tangent_ring, dtype=float)
    endpoint_ring = np.asarray(endpoint_ring, dtype=float)
    vertices = np.vstack([tangent_ring, endpoint_ring])
    n = len(tangent_ring)
    i: list[int] = []
    j: list[int] = []
    k: list[int] = []
    for index in range(n):
        nxt = (index+1) % n
        i.extend([index, index])
        j.extend([n+index, n+nxt])
        k.extend([n+nxt, nxt])
    label = "Umbra / antumbra optical volume" if family == "umbra" else "Penumbral optical volume"
    return go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=i, j=j, k=k, color=color, opacity=opacity,
        flatshading=False,
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0,
                      roughness=1.0, fresnel=0.0),
        name=label, legendgroup="volumes",
        legendgrouptitle=dict(text="Shadow envelopes"),
        showlegend=True, visible=visible, hoverinfo="skip",
    )


def _current_surface_traces(peak: PublicPeakState, quality: PublicQuality) -> list[go.BaseTraceType]:
    """Optional geometric boundaries drawn over already-shaded bodies.

    Surface darkness is no longer a separate floating mesh.  These traces are
    thin diagnostic outlines only and start hidden so the public default is a
    clean illuminated Earth/Moon rather than a forest of neon geometry.
    """
    if peak.event.mode != "solar":
        traces: list[go.BaseTraceType] = []
        for family, lines, color, dash, width in (
            ("Umbra at Moon plane", peak.state.umbra_bundle_gcrf_km,
             PUBLIC_COLORS["umbra"], None, 2.6),
            ("Penumbra at Moon plane", peak.state.penumbra_bundle_gcrf_km,
             PUBLIC_COLORS["penumbra"], "dot", 2.0),
        ):
            points = np.asarray([
                _scene_coordinates(line[-1:].copy(), np.zeros(3),
                                   peak.basis_world_from_scene, 1.0)[0]
                for line in lines
            ])
            points = np.vstack([points, points[0]])
            traces.extend(_ray_trace(
                points, name=family, color=color, glow=color,
                width=width, dash=dash, showlegend=True,
                legendgroup="surface", visible="legendonly",
                legend_title="Optional boundaries",
                add_glow=False,
            ))
        return traces

    traces: list[go.BaseTraceType] = []
    for family, color, width, dash in (
        ("umbra", PUBLIC_COLORS["footprint"], 3.5, None),
        ("penumbra", PUBLIC_COLORS["penumbra"], 2.0, "dot"),
    ):
        pieces: list[np.ndarray] = []
        for segment_native in solar_footprint_segments(
            peak.jd_utc,
            family=family,
            n_azimuth=360 if family == "umbra" else 240,
        ):
            gcrf = _native_to_gcrf(segment_native, peak.event, peak.jd_utc)
            local = _scene_coordinates(gcrf, np.zeros(3),
                                       peak.basis_world_from_scene, 1.0)
            # A tiny normal offset avoids z-fighting only; it does not alter
            # the physical footprint radius or create a second shadow layer.
            radius = np.linalg.norm(local, axis=1)
            local = local * ((radius + 2.0) / np.maximum(radius, 1.0))[:, None]
            if pieces:
                pieces.append(np.full((1, 3), np.nan))
            pieces.append(local)
        points = np.vstack(pieces) if pieces else np.empty((0, 3))
        traces.extend(_ray_trace(
            points,
            name=("Instantaneous totality boundary" if family == "umbra"
                  else "Instantaneous partial-eclipse boundary"),
            color=color, glow=color, width=width, dash=dash,
            showlegend=True, legendgroup="surface",
            visible="legendonly", legend_title="Optional boundaries",
            add_glow=False,
        ))
    return traces


def _body_labels(peak: PublicPeakState) -> go.Scatter3d:
    earth = peak.earth_local_km
    moon = peak.moon_local_km
    points = np.vstack([
        earth+np.array([0.0, 0.0, 1.28*RE_KM]),
        moon+np.array([0.0, 0.0, 1.55*R_MOON_MEAN_KM]),
    ])
    return go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode="text", text=["Earth", "Moon"], textposition="top center",
        textfont=dict(size=16, color="white"), showlegend=False,
        hoverinfo="skip", name="Body labels",
    )


def _direction_cones(peak: PublicPeakState, paths) -> list[go.Cone]:
    central, blocked, umbra, penumbra = paths
    cones: list[go.Cone] = []
    entries = [
        (central, PUBLIC_COLORS["sun"], "Incoming sunlight direction"),
        (blocked, PUBLIC_COLORS["blocked"], "Blocked-light direction"),
        (umbra[0], PUBLIC_COLORS["umbra"], "Umbral boundary direction"),
        (penumbra[0], PUBLIC_COLORS["penumbra"], "Penumbral boundary direction"),
    ]
    for values, color, name in entries:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.all(np.isfinite(finite), axis=1)]
        if len(finite) < 2:
            continue
        index = max(0, len(finite)//2-1)
        start, end = finite[index], finite[index+1]
        direction = _unit(end-start)
        cones.append(go.Cone(
            x=[start[0]], y=[start[1]], z=[start[2]],
            u=[direction[0]], v=[direction[1]], w=[direction[2]],
            anchor="tail", sizemode="absolute", sizeref=3300.0,
            colorscale=[[0.0, color], [1.0, color]], showscale=False,
            name=name, showlegend=False, hoverinfo="skip",
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0),
        ))
    return cones


def _sample_polyline(points: np.ndarray, fraction: float) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    values = values[np.all(np.isfinite(values), axis=1)]
    if len(values) == 0:
        return np.full(3, np.nan)
    if len(values) == 1:
        return values[0]
    lengths = np.linalg.norm(np.diff(values, axis=0), axis=1)
    total = float(np.sum(lengths))
    if total <= 0:
        return values[0]
    distance = (float(fraction) % 1.0)*total
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    index = int(np.clip(np.searchsorted(cumulative, distance, side="right")-1, 0, len(lengths)-1))
    local = (distance-cumulative[index])/max(lengths[index], 1.0e-12)
    return values[index]+local*(values[index+1]-values[index])


def _packet_frames(paths: Sequence[np.ndarray], packet_trace_index: int,
                   *, n_frames: int = 36) -> list[go.Frame]:
    colors = [PUBLIC_COLORS["sun"], PUBLIC_COLORS["blocked"],
              PUBLIC_COLORS["umbra"], PUBLIC_COLORS["umbra"],
              PUBLIC_COLORS["penumbra"], PUBLIC_COLORS["penumbra"]]
    frames: list[go.Frame] = []
    offsets = np.linspace(0.0, 1.0, len(paths), endpoint=False)
    for index, phase in enumerate(np.linspace(0.0, 1.0, n_frames, endpoint=False)):
        points = np.asarray([
            _sample_polyline(path, phase+offset)
            for path, offset in zip(paths, offsets)
        ])
        frames.append(go.Frame(
            name=f"packet-{index}", traces=[packet_trace_index],
            data=[go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                mode="markers", marker=dict(size=5.5, color=colors,
                                             line=dict(color="white", width=0.7)),
            )],
        ))
    return frames


def _camera_update(peak: PublicPeakState, focus: str) -> dict[str, object]:
    """Camera/range update with equal physical units on all three axes."""
    earth = peak.earth_local_km
    moon = peak.moon_local_km
    target = earth if peak.event.mode == "solar" else moon
    occluder = moon if peak.event.mode == "solar" else earth
    target_radius = RE_KM if peak.event.mode == "solar" else R_MOON_MEAN_KM
    occ_radius = R_MOON_MEAN_KM if peak.event.mode == "solar" else RE_KM
    bounds = _local_bounds(peak)

    if focus == "target":
        half = 3.0 * target_radius
        center = target
        eye = dict(x=-1.58, y=-1.72, z=0.86)
        projection = "perspective"
        ranges = [[center[i]-half, center[i]+half] for i in range(3)]
    elif focus == "occluder":
        half = 3.3 * occ_radius
        center = occluder
        eye = dict(x=1.45, y=-1.75, z=0.82)
        projection = "perspective"
        ranges = [[center[i]-half, center[i]+half] for i in range(3)]
    elif focus == "optics":
        ranges = [list(bounds[0]), list(bounds[1]), list(bounds[2])]
        eye = dict(x=0.05, y=-2.55, z=0.20)
        projection = "orthographic"
    elif focus == "shadow":
        center = 0.58*target + 0.42*occluder
        half_x = 0.34*abs(float(moon[0])) + 32_000.0
        half_t = max(5.2*RE_KM, abs(float(moon[2])) + 25_000.0)
        ranges = [[center[0]-half_x, center[0]+half_x],
                  [-half_t, half_t], [-half_t, half_t]]
        eye = dict(x=-1.25, y=-1.90, z=0.68)
        projection = "perspective"
    else:
        ranges = [list(bounds[0]), list(bounds[1]), list(bounds[2])]
        eye = dict(x=-1.35, y=-1.62, z=0.74)
        projection = "perspective"

    return {
        "scene.xaxis.range": ranges[0],
        "scene.yaxis.range": ranges[1],
        "scene.zaxis.range": ranges[2],
        "scene.aspectmode": "manual",
        "scene.aspectratio": equal_unit_aspect(ranges),
        "scene.camera": dict(
            eye=eye, up=dict(x=0.0, y=0.0, z=1.0),
            projection=dict(type=projection),
        ),
    }


def _local_public_traces(
    peak: PublicPeakState,
    quality: PublicQuality,
    *,
    bounds: Sequence[tuple[float, float]] | None = None,
    include_labels: bool = True,
    include_direction_cones: bool = False,
) -> tuple[list[go.BaseTraceType], tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...]]]:
    """Return the local physical scene in a stable, browser-safe order.

    The public default emphasizes the physically shaded bodies.  One thin
    axial path is shown to establish the Sun direction; blocked-axis,
    tangent, footprint, atmosphere, and optical-volume diagnostics remain
    available from the legend but do not obscure the eclipse.
    """
    bounds = tuple(bounds or _local_bounds(peak))
    paths = _local_ray_paths(peak, bounds)
    central, blocked, umbra, penumbra = paths

    traces: list[go.BaseTraceType] = []
    traces.extend(_public_earth_mesh(peak, quality))
    traces.extend(_public_moon_mesh(peak, quality))
    traces.extend(_atmosphere_shell(peak, quality))
    traces.append(_volume_mesh(
        peak, "penumbra", color="#4a83aa", opacity=0.030,
        visible="legendonly",
    ))
    traces.append(_volume_mesh(
        peak, "umbra", color="#5b1014", opacity=0.075,
        visible="legendonly",
    ))
    traces.extend(_current_surface_traces(peak, quality))
    traces.extend(_sampled_irradiance_traces(peak, bounds, n_samples=13))

    traces.extend(_ray_trace(
        central,
        name="Representative axial sunlight — first opaque-surface stop",
        color=PUBLIC_COLORS["sun"], glow=PUBLIC_COLORS["sun_glow"],
        width=2.4, showlegend=True, legendgroup="rays", visible="legendonly",
        legend_title="Optional optical diagnostics", add_glow=False,
    ))
    traces.extend(_ray_trace(
        blocked,
        name="Blocked-light axis (not transmitted light)",
        color=PUBLIC_COLORS["blocked"], glow=PUBLIC_COLORS["blocked_glow"],
        width=2.0, dash="dash", showlegend=True, legendgroup="rays",
        visible="legendonly", add_glow=False,
    ))
    for label, pair, color, dash in (
        ("Umbra / antumbra tangent", umbra, PUBLIC_COLORS["umbra"], None),
        ("Penumbra tangent", penumbra, PUBLIC_COLORS["penumbra"], "dot"),
    ):
        for index, path in enumerate(pair):
            traces.extend(_ray_trace(
                path, name=f"{label} — {'upper' if index == 0 else 'lower'}",
                color=color, glow=color, width=1.8, dash=dash,
                showlegend=(index == 0), legendgroup="rays",
                visible="legendonly", add_glow=False,
            ))
    if include_direction_cones:
        for trace in _direction_cones(peak, paths):
            trace.visible = "legendonly"
            traces.append(trace)
    if include_labels:
        traces.append(_body_labels(peak))
    return traces, paths


def build_local_public_figure(
    peak: PublicPeakState,
    *,
    quality: str = "ultra",
    animate_light_packets: bool = False,
) -> go.Figure:
    """Build the greatest-eclipse public local view.

    ``animate_light_packets`` is retained only as an optional teaching aid.
    It defaults to ``False`` because moving markers along fixed rays can be
    mistaken for the physical motion of the eclipse.  The public event
    animation uses independently recomputed UTC states instead.
    """
    q = _quality(quality)
    bounds = _local_bounds(peak)
    traces, paths = _local_public_traces(peak, q, bounds=bounds)
    central, blocked, umbra, penumbra = paths

    packet_index: int | None = None
    packet_paths = [central, blocked, umbra[0], umbra[1], penumbra[0], penumbra[1]]
    if animate_light_packets:
        packet_points = np.asarray([
            _sample_polyline(path, offset)
            for path, offset in zip(packet_paths, np.linspace(0, 1, 6, endpoint=False))
        ])
        packet_index = len(traces)
        traces.append(go.Scatter3d(
            x=packet_points[:, 0], y=packet_points[:, 1], z=packet_points[:, 2],
            mode="markers", marker=dict(
                size=5.5,
                color=[PUBLIC_COLORS["sun"], PUBLIC_COLORS["blocked"],
                       PUBLIC_COLORS["umbra"], PUBLIC_COLORS["umbra"],
                       PUBLIC_COLORS["penumbra"], PUBLIC_COLORS["penumbra"]],
                line=dict(color="white", width=0.7),
            ),
            name="Illustrative path markers", showlegend=False, hoverinfo="skip",
        ))

    fig = go.Figure(data=traces)
    if animate_light_packets and packet_index is not None:
        fig.frames = _packet_frames(packet_paths, packet_index)
    default_focus = "target"
    initial = _camera_update(peak, default_focus)
    target_name = "Earth" if peak.event.mode == "solar" else "Moon"
    occ_name = "Moon" if peak.event.mode == "solar" else "Earth"

    def camera_button(label: str, focus: str):
        return dict(label=label, method="relayout", args=[_camera_update(peak, focus)])

    menus = [
        dict(
            type="dropdown", direction="down", x=1.005, y=1.04,
            xanchor="left", yanchor="top", bgcolor="#18263a",
            bordercolor="#71849d", font=dict(color="white", size=12),
            buttons=[
                camera_button(f"{target_name} eclipse close-up", "target"),
                camera_button(f"{occ_name} tangent close-up", "occluder"),
                camera_button("Shadow envelope", "shadow"),
                camera_button("Full Earth–Moon system", "system"),
                camera_button("Orthographic optics side", "optics"),
            ],
        ),
        dict(
            type="buttons", direction="left", showactive=True,
            x=0.015, y=0.025, xanchor="left", yanchor="bottom",
            bgcolor="#18263a", bordercolor="#71849d",
            font=dict(color="white", size=11),
            buttons=[
                dict(label="Clean view", method="relayout", args=[{
                    "scene.xaxis.visible": False,
                    "scene.yaxis.visible": False,
                    "scene.zaxis.visible": False,
                }]),
                dict(label="Scientific axes", method="relayout", args=[{
                    "scene.xaxis.visible": True,
                    "scene.yaxis.visible": True,
                    "scene.zaxis.visible": True,
                    "scene.xaxis.gridcolor": PUBLIC_COLORS["grid"],
                    "scene.yaxis.gridcolor": PUBLIC_COLORS["grid"],
                    "scene.zaxis.gridcolor": PUBLIC_COLORS["grid"],
                }]),
            ],
        ),
    ]
    if animate_light_packets:
        menus.insert(1, dict(
            type="buttons", direction="left", showactive=False,
            x=0.015, y=0.075, xanchor="left", yanchor="bottom",
            bgcolor="#18263a", bordercolor="#71849d",
            font=dict(color="white", size=12),
            buttons=[
                dict(label="▶ Illustrate paths", method="animate", args=[
                    None,
                    dict(frame=dict(duration=90, redraw=False),
                         transition=dict(duration=0), fromcurrent=True,
                         mode="immediate"),
                ]),
                dict(label="❚❚ Pause", method="animate", args=[
                    [None],
                    dict(frame=dict(duration=0, redraw=False),
                         transition=dict(duration=0), mode="immediate"),
                ]),
            ],
        ))

    fig.update_layout(
        title=dict(
            text=(f"{peak.event.definition.title} — ultra-resolution finite-Sun ray trace"
                  f"<br><sub>Exact impact plane; direct light stops at {occ_name}; shadow reaches {target_name}</sub>"),
            x=0.5, y=0.985, font=dict(size=23, color=PUBLIC_COLORS["text"]),
        ),
        scene=dict(
            xaxis=dict(range=initial["scene.xaxis.range"], visible=False,
                       title="Impact-plane X [km] — Sun to target"),
            yaxis=dict(range=initial["scene.yaxis.range"], visible=False,
                       title="Out-of-plane Y [km]"),
            zaxis=dict(range=initial["scene.zaxis.range"], visible=False,
                       title="Impact offset Z [km]"),
            aspectmode=initial["scene.aspectmode"], aspectratio=initial["scene.aspectratio"],
            camera=initial["scene.camera"],
            bgcolor=PUBLIC_COLORS["background"], dragmode="orbit",
            uirevision=f"{peak.event.definition.key}-public-v19-2",
        ),
        paper_bgcolor=PUBLIC_COLORS["background"],
        plot_bgcolor=PUBLIC_COLORS["background"],
        font=dict(color=PUBLIC_COLORS["text"], family="Arial, sans-serif"),
        height=960,
        margin=dict(l=8, r=305, t=100, b=65),
        legend=dict(
            x=1.005, y=0.98, xanchor="left", yanchor="top",
            bgcolor="rgba(7,16,28,0.96)", bordercolor="#42546b", borderwidth=1,
            font=dict(size=10.5), groupclick="togglegroup", itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        updatemenus=menus,
        annotations=[dict(
            x=1.005, y=0.19, xref="paper", yref="paper",
            xanchor="left", yanchor="top", showarrow=False,
            width=275, align="left", borderpad=8,
            bgcolor="rgba(7,16,28,0.96)", bordercolor="#42546b", borderwidth=1,
            font=dict(size=10.2, color=PUBLIC_COLORS["muted"]),
            text=(
                "<b>What is physically scaled here</b><br>"
                "Earth and Moon radii: true<br>Earth–Moon separation: true<br>"
                "Ray tangencies and first-surface stops: true<br>"
                "Sun: off-screen at its real ephemeris distance<br><br>"
                f"Axis miss at target: {peak.axis_impact_km:,.3f} km<br>"
                f"Sun–Earth: {np.linalg.norm(peak.state.sun_gcrf_km)/1e6:.3f} million km<br>"
                f"Earth–Moon: {np.linalg.norm(peak.state.moon_gcrf_km):,.3f} km"
            ),
        )],
    )
    return fig


def _indexed_sphere_traces(
    center: np.ndarray,
    radius: float,
    *,
    name: str,
    color: str | None = None,
    vertexcolor: Sequence[str] | None = None,
    n_lat: int = 33,
    n_lon: int = 64,
    legendgroup: str,
    legend_title: str | None = None,
    showlegend: bool = True,
) -> list[go.Mesh3d]:
    """Exact sphere with one vertex per pole and a closed longitude seam."""
    directions, faces, _, _, _ = _moon_unit_mesh(n_lat, n_lon)
    vertices = np.asarray(center, dtype=float).reshape(3)+float(radius)*directions
    return indexed_mesh_chunks(
        vertices, faces, color=color, vertexcolor=vertexcolor, name=name,
        showlegend=showlegend, legendgroup=legendgroup,
        legendgrouptitle=(dict(text=legend_title) if legend_title else None),
        hovertemplate=f"{html.escape(name)}<extra></extra>",
        flatshading=False,
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0,
                      roughness=1.0, fresnel=0.0),
    )


def _sun_surface(radius_au: float) -> list[go.Mesh3d]:
    """Self-luminous, true-radius Sun without transparent shell artifacts."""
    directions, faces, _, lat_v, lon_v = _moon_unit_mesh(65, 128)
    lat = np.radians(lat_v)
    lon = np.radians(lon_v)
    noise = (
        0.64+0.14*np.sin(13.0*lon+4.0*np.sin(3.0*lat))
        +0.09*np.sin(27.0*lon-7.0*lat)
        +0.06*np.cos(17.0*lat)
    )
    noise = np.clip(noise, 0.25, 1.0)
    stops = np.array([0.25, 0.50, 0.72, 1.00])
    palette = np.array([
        [0.478, 0.125, 0.000],
        [0.827, 0.294, 0.031],
        [1.000, 0.678, 0.125],
        [1.000, 0.973, 0.835],
    ])
    rgb = np.column_stack([
        np.interp(noise, stops, palette[:, channel]) for channel in range(3)
    ])
    # A single opaque photosphere is more faithful and more robust than the
    # two overlapping transparent spheres used previously, which could sort
    # incorrectly and make the Sun look lobed or hollow in WebGL.
    return indexed_mesh_chunks(
        directions*float(radius_au), faces, vertexcolor=rgb_strings(rgb),
        name="Sun — true photospheric radius", showlegend=True,
        legendgroup="true-sun", legendgrouptitle=dict(text="Physical bodies"),
        hovertemplate="Sun — true photospheric radius<extra></extra>",
        flatshading=False,
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0,
                      roughness=1.0, fresnel=0.0),
    )


def _true_scale_ray(points_gcrf: np.ndarray, peak: PublicPeakState) -> np.ndarray:
    # Heliocentric Sun is the origin; geocentric ray coordinates translate by
    # the current heliocentric Earth position (-Sun_geocentric).
    earth_helio = -peak.state.sun_gcrf_km
    values = earth_helio+np.asarray(points_gcrf, dtype=float)
    return _scene_coordinates(values, np.zeros(3), peak.basis_world_from_scene, AU_KM)


def build_true_scale_figure(peak: PublicPeakState) -> go.Figure:
    traces: list[go.BaseTraceType] = []
    traces.extend(_sun_surface(R_SUN_KM/AU_KM))

    earth_helio = _scene_coordinates(
        (-peak.state.sun_gcrf_km).reshape(1, 3), np.zeros(3), peak.basis_world_from_scene, AU_KM,
    )[0]
    moon_helio = _scene_coordinates(
        (-peak.state.sun_gcrf_km+peak.state.moon_gcrf_km).reshape(1, 3),
        np.zeros(3), peak.basis_world_from_scene, AU_KM,
    )[0]

    traces.extend(_indexed_sphere_traces(
        earth_helio, RE_KM/AU_KM, color="#3f86e8",
        name="Earth — true radius", n_lat=25, n_lon=48,
        legendgroup="true-earth", showlegend=True,
    ))
    traces.extend(_indexed_sphere_traces(
        moon_helio, R_MOON_MEAN_KM/AU_KM, color="#c9c9c9",
        name="Moon — true radius", n_lat=19, n_lon=36,
        legendgroup="true-moon", showlegend=True,
    ))
    traces.append(go.Scatter3d(
        x=[earth_helio[0], moon_helio[0]],
        y=[earth_helio[1], moon_helio[1]],
        z=[earth_helio[2], moon_helio[2]],
        mode="markers+text", text=["Earth locator", "Moon locator"],
        textposition="top center",
        marker=dict(size=[9, 6], color=["#3f86e8", "#c9c9c9"],
                    line=dict(color="white", width=0.8)),
        name="Locator markers — not body size", showlegend=True,
        legendgroup="locators",
        legendgrouptitle=dict(text="Screen-space locators"),
        hovertemplate="%{text}<extra></extra>",
    ))

    for index, (path_gcrf, blocked) in enumerate(_sampled_photospheric_paths(peak, n_samples=13)):
        path_au = _true_scale_ray(path_gcrf, peak)
        traces.append(go.Scatter3d(
            x=path_au[:, 0], y=path_au[:, 1], z=path_au[:, 2], mode="lines",
            line=dict(
                color=("rgba(255,219,99,0.78)" if blocked
                       else "rgba(255,238,155,0.46)"),
                width=(2.2 if blocked else 1.4),
            ),
            name="Sampled finite-Sun irradiance paths",
            showlegend=(index == 0), legendgroup="true-scale-irradiance",
            legendgrouptitle=(dict(text="Physical illumination") if index == 0 else None),
            hovertemplate=(
                "Photosphere to first occluder surface<extra></extra>"
                if blocked else "Photosphere to target surface<extra></extra>"
            ),
        ))

    for name, points, color, glow, dash, showlegend in (
        ("Direct light — full source path", peak.state.central_gcrf_km,
         PUBLIC_COLORS["sun"], PUBLIC_COLORS["sun_glow"], None, True),
        ("Blocked-light axis", peak.state.shadow_axis_gcrf_km,
         PUBLIC_COLORS["blocked"], PUBLIC_COLORS["blocked_glow"], "dash", True),
        ("Umbra tangent — upper", peak.state.umbra_gcrf_km[0],
         PUBLIC_COLORS["umbra"], PUBLIC_COLORS["umbra_glow"], None, True),
        ("Umbra tangent — lower", peak.state.umbra_gcrf_km[1],
         PUBLIC_COLORS["umbra"], PUBLIC_COLORS["umbra_glow"], None, False),
        ("Penumbra tangent — upper", peak.state.penumbra_gcrf_km[0],
         PUBLIC_COLORS["penumbra"], PUBLIC_COLORS["penumbra_glow"], "dot", True),
        ("Penumbra tangent — lower", peak.state.penumbra_gcrf_km[1],
         PUBLIC_COLORS["penumbra"], PUBLIC_COLORS["penumbra_glow"], "dot", False),
    ):
        traces.extend(_ray_trace(
            _true_scale_ray(points, peak), name=name, color=color, glow=glow,
            width=3.0, dash=dash, showlegend=showlegend,
            legendgroup="true-scale-rays", visible="legendonly",
            legend_title="Optional boundary diagnostics",
        ))

    fig = go.Figure(data=traces)
    transverse = 0.014
    fig.update_layout(
        title=dict(
            text=(f"{peak.event.definition.title} — true Sun–Earth–Moon scale"
                  "<br><sub>All physical spheres use their real radii; locator markers are screen-space aids</sub>"),
            x=0.5, y=0.985, font=dict(size=22, color="white"),
        ),
        scene=dict(
            xaxis=dict(range=[-0.025, 1.035], title="Impact-plane X [AU]",
                       gridcolor=PUBLIC_COLORS["grid"], backgroundcolor=PUBLIC_COLORS["background"]),
            yaxis=dict(range=[-transverse, transverse], title="Y [AU]",
                       gridcolor=PUBLIC_COLORS["grid"], backgroundcolor=PUBLIC_COLORS["background"]),
            zaxis=dict(range=[-transverse, transverse], title="Impact offset Z [AU]",
                       gridcolor=PUBLIC_COLORS["grid"], backgroundcolor=PUBLIC_COLORS["background"]),
            aspectmode="manual",
            aspectratio=equal_unit_aspect([[-0.025, 1.035], [-transverse, transverse], [-transverse, transverse]]),
            bgcolor=PUBLIC_COLORS["background"], dragmode="orbit",
            camera=dict(eye=dict(x=1.35, y=-1.50, z=0.58), up=dict(x=0, y=0, z=1),
                        projection=dict(type="perspective")),
            uirevision=f"{peak.event.definition.key}-true-scale-v19-2",
        ),
        paper_bgcolor=PUBLIC_COLORS["background"],
        font=dict(color="white", family="Arial, sans-serif"),
        height=900, margin=dict(l=8, r=310, t=95, b=45),
        legend=dict(x=1.005, y=0.98, bgcolor="rgba(7,16,28,0.96)",
                    bordercolor="#42546b", borderwidth=1, font=dict(size=10.5)),
        annotations=[dict(
            x=1.005, y=0.20, xref="paper", yref="paper", xanchor="left", yanchor="top",
            showarrow=False, align="left", width=280, borderpad=8,
            bgcolor="rgba(7,16,28,0.96)", bordercolor="#42546b", borderwidth=1,
            font=dict(size=10.3, color=PUBLIC_COLORS["muted"]),
            text=(
                "<b>Scale note</b><br>"
                "The solar photosphere is visible at true radius. Earth and Moon are physically present at true radii, "
                "but are sub-pixel at this one-AU view; the labelled locator markers are deliberately not to scale.<br><br>"
                f"Sun–Earth: {np.linalg.norm(peak.state.sun_gcrf_km):,.0f} km<br>"
                f"Earth–Moon: {np.linalg.norm(peak.state.moon_gcrf_km):,.0f} km"
            ),
        )],
    )
    return fig


def _circle_trace(center_x: float, center_z: float, radius: float, *, name: str,
                  color: str, width: float, dash: str | None = None,
                  fill: str | None = None, opacity: float = 1.0) -> go.Scatter:
    angle = np.linspace(0.0, 2.0*math.pi, 361)
    line = dict(color=color, width=width)
    if dash:
        line["dash"] = dash
    return go.Scatter(
        x=center_x+radius*np.cos(angle),
        y=center_z+radius*np.sin(angle),
        mode="lines", line=line, fill=("toself" if fill else None),
        fillcolor=fill, opacity=opacity, name=name, hoverinfo="skip",
    )


def build_cross_section_figure(peak: PublicPeakState) -> go.Figure:
    # Use the same impact-plane coordinates as the 3-D scene, projected to X/Z.
    bounds = _local_bounds(peak)
    paths = _local_ray_paths(peak, bounds)
    central, blocked, umbra, penumbra = paths
    fig = go.Figure()
    earth = peak.earth_local_km
    moon = peak.moon_local_km
    fig.add_trace(_circle_trace(float(earth[0]), float(earth[2]), RE_KM,
                                name="Earth — WGS-84 equatorial radius outline",
                                color="#4d8edc", width=2.0, fill="rgba(39,93,158,0.25)"))
    fig.add_trace(_circle_trace(float(moon[0]), float(moon[2]), R_MOON_MEAN_KM,
                                name="Moon — mean solid radius", color="#d0d0d0",
                                width=2.0, fill="rgba(190,190,190,0.22)"))
    for name, values, color, dash, width in (
        ("Direct photospheric light", central, PUBLIC_COLORS["sun"], None, 3.5),
        ("Blocked-light axis", blocked, PUBLIC_COLORS["blocked"], "dash", 2.6),
        ("Umbra tangent — upper", umbra[0], PUBLIC_COLORS["umbra"], None, 2.2),
        ("Umbra tangent — lower", umbra[1], PUBLIC_COLORS["umbra"], None, 2.2),
        ("Penumbra tangent — upper", penumbra[0], PUBLIC_COLORS["penumbra"], "dot", 2.0),
        ("Penumbra tangent — lower", penumbra[1], PUBLIC_COLORS["penumbra"], "dot", 2.0),
    ):
        fig.add_trace(go.Scatter(
            x=values[:, 0], y=values[:, 2], mode="lines",
            line=dict(color=color, width=width, dash=dash), name=name,
            connectgaps=False, hovertemplate=f"{html.escape(name)}<extra></extra>",
        ))
    fig.add_hline(y=0.0, line=dict(color="rgba(226,122,255,0.45)", width=1.2, dash="dash"))
    fig.update_layout(
        title=dict(text="Exact finite-Sun impact-plane cross-section", x=0.5, font=dict(size=20)),
        xaxis=dict(title="Downstream X [km] — Sun → occluder → target",
                   gridcolor="#28364a", zeroline=False),
        yaxis=dict(title="Impact offset Z [km]", scaleanchor="x", scaleratio=1,
                   gridcolor="#28364a", zeroline=False),
        paper_bgcolor=PUBLIC_COLORS["background"], plot_bgcolor="#07101c",
        font=dict(color="white", family="Arial, sans-serif"),
        height=720, margin=dict(l=70, r=285, t=80, b=65),
        legend=dict(x=1.01, y=1.0, bgcolor="rgba(7,16,28,0.96)",
                    bordercolor="#42546b", borderwidth=1, font=dict(size=10)),
    )
    return fig


def _asset_data_uri(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _backend_card(peak: PublicPeakState) -> str:
    audit = peak.ssapy_diagnostics
    selected = str(audit.get("selected_backend", "reference"))
    ssapy_status = (
        "driving this state" if selected in {"ssapy", "ssapy-core"}
        else "available and audited" if audit.get("llnl_ssapy_available")
        else "adapter included; package not installed in this build environment"
    )
    toolkit_status = (
        "driving GCRF↔ITRF transforms" if selected == "ssapy"
        else "available for GCRF↔ITRF transforms" if audit.get("ssapy_toolkit_available")
        else "adapter included; deterministic transform fallback used here"
    )
    comparison = ""
    if "ssapy_sun_reference_delta_km" in audit:
        comparison = (
            f"<br>SSAPy/reference Sun delta: {float(audit['ssapy_sun_reference_delta_km']):,.3f} km"
            f"<br>SSAPy/reference Moon delta: {float(audit['ssapy_moon_reference_delta_km']):,.3f} km"
        )
    return (
        "<b>State provenance</b><br>"
        f"Requested mode: {html.escape(str(audit.get('requested_backend', 'reference')))}<br>"
        f"Selected mode: {html.escape(selected)}<br>"
        f"llnl-ssapy: {html.escape(ssapy_status)}<br>"
        f"ssapy-toolkit: {html.escape(toolkit_status)}<br>"
        f"Ephemeris: {html.escape(str(audit.get('ephemeris_backend')))}<br>"
        f"Frame path: {html.escape(str(audit.get('frame_backend')))}<br>"
        f"Rendered event state: {html.escape(str(audit.get('reference_backend')))}"
        + (
            f"<br>Fallback: {html.escape(str(audit.get('fallback_reason')))}"
            if audit.get("fallback_reason") else ""
        )
        + comparison
    )


def generate_public_ultra_view(
    kind: str | ReferenceDefinition | ReferenceEvent = "solar",
    output_path: str | Path = "eclipse_greatest_public_ultra.html",
    *,
    quality: str = "ultra",
    backend: str = "auto",
    solar_scope: str = "global",
    observer: object | None = None,
) -> str:
    """Generate a self-contained, public-facing greatest-eclipse dashboard."""
    q = _quality(quality)
    peak = build_public_peak_state(
        kind, ray_azimuth=q.ray_azimuth, backend=backend,
        solar_scope=solar_scope, observer=observer,
    )
    local = build_local_public_figure(peak, quality=quality)
    true_scale = build_true_scale_figure(peak)
    cross = build_cross_section_figure(peak)

    local_div = pio.to_html(local, include_plotlyjs=False, full_html=False,
                            div_id="local-plot", config={
                                "displaylogo": False, "responsive": True,
                                "scrollZoom": True, "plotGlPixelRatio": 2.0,
                                "toImageButtonOptions": {"format": "png", "scale": 2},
                            })
    true_div = pio.to_html(true_scale, include_plotlyjs=False, full_html=False,
                           div_id="true-scale-plot", config={
                               "displaylogo": False, "responsive": True,
                               "scrollZoom": True, "plotGlPixelRatio": 2.0,
                           })
    cross_div = pio.to_html(cross, include_plotlyjs=False, full_html=False,
                            div_id="cross-section-plot", config={
                                "displaylogo": False, "responsive": True,
                            })
    dt = jd_to_datetime(peak.jd_utc)
    target = "Earth" if peak.event.mode == "solar" else "Moon"
    occluder = "Moon" if peak.event.mode == "solar" else "Earth"
    backend_card = _backend_card(peak)
    penetrations = bundle_penetrations(peak.state.bundle_native)
    residuals = tangent_residuals(peak.state.bundle_native)

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(peak.event.definition.title)} — SSAPy eclipse scientific Plotly view V22.2</title>
<script>{get_plotlyjs()}</script>
<style>
:root {{ color-scheme: dark; --bg:#02040a; --panel:#07101c; --line:#42546b; --text:#f1f6fc; --muted:#9fb0c4; --accent:#6fc8ff; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:radial-gradient(circle at 35% 10%,#0b1827 0,#02040a 44%,#010207 100%); color:var(--text); font-family:Inter,Segoe UI,Arial,sans-serif; }}
header {{ padding:20px 26px 15px; border-bottom:1px solid var(--line); background:rgba(2,4,10,.93); position:sticky; top:0; z-index:10; backdrop-filter:blur(8px); }}
h1 {{ margin:0; font-size:clamp(22px,2.6vw,36px); font-weight:650; letter-spacing:.01em; }}
.subtitle {{ color:var(--muted); margin-top:7px; line-height:1.45; }}
.badges {{ display:flex; gap:9px; flex-wrap:wrap; margin-top:12px; }}
.badge {{ border:1px solid #53677f; background:#0d1a2a; padding:6px 9px; border-radius:999px; font-size:12px; }}
nav {{ display:flex; gap:8px; padding:12px 20px 0; flex-wrap:wrap; }}
.tab-button {{ border:1px solid #53677f; background:#0c1725; color:var(--text); padding:9px 13px; border-radius:8px 8px 0 0; cursor:pointer; font-weight:600; }}
.tab-button.active {{ background:#18314b; border-color:#78b8ee; }}
.tab {{ display:none; padding:0 12px 14px; }}
.tab.active {{ display:block; }}
.plot-shell {{ border:1px solid var(--line); border-radius:10px; overflow:hidden; background:var(--panel); box-shadow:0 16px 60px rgba(0,0,0,.35); }}
.guide {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:12px; margin:14px 12px 24px; }}
.card {{ background:rgba(7,16,28,.94); border:1px solid var(--line); border-radius:9px; padding:13px 15px; line-height:1.45; color:var(--muted); }}
.card b {{ color:var(--text); }}
footer {{ padding:18px 26px 28px; color:var(--muted); border-top:1px solid var(--line); font-size:13px; line-height:1.5; }}
@media (max-width:800px) {{ .tab {{ padding:0 4px 10px; }} header {{ position:static; }} }}
</style>
</head>
<body>
<header>
<h1>{html.escape(peak.event.definition.title)} — V22.2 scientific Plotly finite-Sun inspection</h1>
<div class="subtitle">Greatest eclipse: {dt.strftime('%Y-%m-%d %H:%M:%S')} UTC. The default view uses exact WGS-84/spherical body geometry, true Earth–Moon separation, and finite-Sun surface irradiance computed before WebGL rendering.</div>
<div class="badges">
<span class="badge">{html.escape(str(peak.event.metadata.get('state_source', 'reference')))} state mode</span>
<span class="badge">NASA/GSFC contact scaffold</span>
<span class="badge">WGS-84 Earth</span>
<span class="badge">finite solar photosphere</span>
<span class="badge">explicit backend provenance</span>
<span class="badge">SSAPy / SSAPy-Toolkit ready — selected backend shown below</span>
</div>
</header>
<nav>
<button class="tab-button active" data-tab="local">Public eclipse view</button>
<button class="tab-button" data-tab="true">True Sun distance</button>
<button class="tab-button" data-tab="cross">Scientific cross-section</button>
</nav>
<section id="tab-local" class="tab active"><div class="plot-shell">{local_div}</div></section>
<section id="tab-true" class="tab"><div class="plot-shell">{true_div}</div></section>
<section id="tab-cross" class="tab"><div class="plot-shell">{cross_div}</div></section>
<div class="guide">
<div class="card"><b>Physical illumination</b><br>The visible yellow paths are 13 deterministic samples from the real solar photosphere. Each stops at the first opaque Moon/Earth intersection or reaches the actual target surface. The continuous finite-disc Sun visibility is evaluated at every body vertex; the samples are only a readable visualization of that field.</div>
<div class="card"><b>Optional optical diagnostics</b><br>Magenta is blocked-light geometry, not transmitted sunlight. Red and blue are exact common tangents from the finite solar limb. They and the translucent envelopes start hidden so the public view is not obscured by construction lines.</div>
<div class="card"><b>high-fidelity validated 3-D view</b><br><b>WebGL-safe exact bodies:</b><br>Earth is the WGS-84 ellipsoid; Sun and Moon are exact spheres. Large meshes are divided into small indexed draw calls without changing coordinates. No distance compression or body enlargement is used. Local and true-distance scenes use separate buffers with equal physical axis units.</div>
<div class="card">{backend_card}</div>
<div class="card"><b>Numerical audit</b><br>Earth penetrations: {penetrations['earth']}<br>Moon penetrations: {penetrations['moon']}<br>Maximum tangent radius residual: {max(residuals['umbra_radius_error_km'], residuals['penumbra_radius_error_km']):.3e} km<br>Maximum tangent orthogonality residual: {max(residuals['umbra_orthogonality_km'], residuals['penumbra_orthogonality_km']):.3e} km</div>
</div>
<footer>Appearance uses the packaged SSAPy Earth and Moon mosaics. Eclipse darkness is part of the same physically shaded body surface, not a second floating overlay. Lunar totality colour is an illustrative Danjon-style atmospheric cue; the geometric shadow and contact construction are the validated quantities. No cloud layer is shown because the package does not contain a date-matched global cloud analysis.</footer>
<script>
const buttons=[...document.querySelectorAll('.tab-button')];
const tabs=[...document.querySelectorAll('.tab')];
buttons.forEach(btn=>btn.addEventListener('click',()=>{{
  buttons.forEach(x=>x.classList.remove('active'));
  tabs.forEach(x=>x.classList.remove('active'));
  btn.classList.add('active');
  const target=document.getElementById('tab-'+btn.dataset.tab);
  target.classList.add('active');
  target.querySelectorAll('.js-plotly-plot').forEach(div=>Plotly.Plots.resize(div));
}}));
</script>
</body>
</html>"""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return str(output)


def public_metrics(kind: str | ReferenceDefinition | ReferenceEvent = "solar", *, quality: str = "ultra",
                   backend: str = "auto") -> dict[str, object]:
    q = _quality(quality)
    peak = build_public_peak_state(
        kind, ray_azimuth=q.ray_azimuth, backend=backend,
    )
    penetrations = bundle_penetrations(peak.state.bundle_native)
    residuals = tangent_residuals(peak.state.bundle_native)
    bounds = _local_bounds(peak)
    central, blocked, umbra, penumbra = _local_ray_paths(peak, bounds)
    earth_chunks = _public_earth_mesh(peak, q)
    moon_chunks = _public_moon_mesh(peak, q)
    earth_vertices = np.vstack([
        np.column_stack([trace.x, trace.y, trace.z]) for trace in earth_chunks
    ])
    moon_vertices = np.vstack([
        np.column_stack([trace.x, trace.y, trace.z]) for trace in moon_chunks
    ])
    earth_radii = np.linalg.norm(earth_vertices-peak.earth_local_km, axis=1)
    moon_radii = np.linalg.norm(moon_vertices-peak.moon_local_km, axis=1)
    sampled_paths = _sampled_photospheric_paths(peak, n_samples=13)
    return {
        "event": peak.event.definition.key,
        "backend_requested": backend,
        "backend_selected": peak.event.metadata.get("state_source", "reference"),
        "quality": quality,
        "body_resolution": [q.body_lat, q.body_lon],
        "atmosphere_resolution": [q.atmosphere_lat, q.atmosphere_lon],
        "shadow_resolution": q.shadow_resolution,
        "ray_azimuth_count": q.ray_azimuth,
        "earth_mesh_chunks": len(earth_chunks),
        "moon_mesh_chunks": len(moon_chunks),
        "maximum_chunk_vertices": int(max(
            max(len(trace.x) for trace in earth_chunks),
            max(len(trace.x) for trace in moon_chunks),
        )),
        "earth_radius_range_km": [float(np.min(earth_radii)), float(np.max(earth_radii))],
        "moon_radius_range_km": [float(np.min(moon_radii)), float(np.max(moon_radii))],
        "sampled_photospheric_paths": len(sampled_paths),
        "sampled_blocked_paths": int(sum(blocked for _, blocked in sampled_paths)),
        "sampled_target_reaching_paths": int(sum(not blocked for _, blocked in sampled_paths)),
        "sun_earth_distance_km": float(np.linalg.norm(peak.state.sun_gcrf_km)),
        "earth_moon_distance_km": float(np.linalg.norm(peak.state.moon_gcrf_km)),
        "target_axis_miss_km": peak.axis_impact_km,
        "ray_penetrations": penetrations,
        "tangent_residuals": residuals,
        "local_bounds_km": [list(pair) for pair in bounds],
        "central_ray_vertices": int(len(central)),
        "blocked_axis_vertices": int(len(blocked)),
        "umbra_target_hits": list(peak.state.target_hits["umbra"]),
        "penumbra_target_hits": list(peak.state.target_hits["penumbra"]),
        "ssapy_diagnostics": peak.ssapy_diagnostics,
    }
