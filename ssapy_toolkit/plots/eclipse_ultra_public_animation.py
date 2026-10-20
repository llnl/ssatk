"""Contact-to-contact scientific Plotly eclipse animation for V22.2.

Every animation state is rebuilt from the validated event geometry.  The
Earth orientation, Moon centre and synchronous attitude, finite-Sun common
tangents, first-surface ray stops, optical volumes, and target-surface shadow
are therefore functions of UTC rather than decorative motion applied to a
frozen greatest-eclipse scene.

The kilometre-scale eclipse scene, the one-AU true-scale scene, and the exact
impact-plane cross-section are separate Plotly figures with a shared HTML
transport.  This preserves WebGL precision while keeping all three views on
the same contact-aware timeline.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence
import html
import json
import math

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs

from ssapy_toolkit.compute.eclipse_brightness import itrf_to_gcrf_km
from ssapy_toolkit.compute.eclipse_state import build_event, event_itrf_to_gcrf_km, resample_event
from ssapy_toolkit.plots.eclipse_plotly_mesh import equal_unit_aspect
from ssapy_toolkit.plots.eclipse_system_raytrace_3d import _build_scene_geometry, _build_inertial_state, _scene_coordinates
from ssapy_toolkit.plots.eclipse_ultra_public_3d import (
    PUBLIC_COLORS,
    PublicPeakState,
    PublicQuality,
    _backend_card,
    _camera_update,
    _impact_basis_gcrf,
    _local_public_traces,
    _quality,
    build_cross_section_figure,
    build_local_public_figure,
    build_public_frame_state,
    build_true_scale_figure,
)
from ssapy_toolkit.compute.eclipse_reference_events import (
    EARTH_AXES_KM,
    LUNAR_2025,
    R_MOON_MEAN_KM,
    SOLAR_2024,
    ReferenceDefinition,
    ReferenceEvent,
    build_reference_event,
    build_solar_local_event,
    jd_to_datetime,
    solar_local_contacts,
)
from ssapy_toolkit.compute.eclipse_raytrace import bundle_penetrations, tangent_residuals


@dataclass(frozen=True)
class PublicAnimation:
    event: ReferenceEvent
    states: tuple[PublicPeakState, ...]
    greatest_state: PublicPeakState
    bounds_km: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    frame_names: tuple[str, ...]
    frame_labels: tuple[str, ...]
    phase_labels: tuple[str, ...]


def _event_for_animation(
    kind: str | ReferenceDefinition | ReferenceEvent,
    *,
    n_frames: int,
    solar_scope: str = "local",
    backend: str = "auto",
    observer: object | None = None,
    observer_contacts: dict[str, float] | None = None,
) -> ReferenceEvent:
    if isinstance(kind, ReferenceEvent):
        event = kind
        definition = event.definition
        requested = max(25, int(n_frames))
        if requested != len(event.jd):
            contacts = sorted(float(v) for v in event.contacts_jd.values())
            start, stop = contacts[0], contacts[-1]
            u = np.linspace(0.0, 1.0, requested)
            base = start + (stop-start)*(0.5-0.5*np.cos(np.pi*u))
            offsets = event.greatest_jd + np.array([-900,-600,-300,-120,-60,-20,0,20,60,120,300,600,900])/86400.0
            exact = np.concatenate([np.asarray(contacts), offsets])
            exact = exact[(exact >= start) & (exact <= stop)]
            keep = np.ones(base.shape, dtype=bool)
            for value in exact:
                keep &= np.abs(base-value) > 0.05/86400.0
            event = resample_event(event, np.sort(np.unique(np.concatenate([base[keep], exact]))))
    else:
        if isinstance(kind, ReferenceDefinition):
            definition = kind
        else:
            definition = SOLAR_2024 if str(kind).lower().startswith("solar") else LUNAR_2025
        event = build_event(
            definition,
            backend=backend,
            n_frames=max(25, int(n_frames)),
            solar_scope=solar_scope,
            observer=observer,
        )
    if definition.mode == "solar" and str(solar_scope).lower() == "local" and observer_contacts:
        from ssapy_toolkit.coordinates.eclipse_observer_geometry import ObserverConfig, contact_aware_observer_jd
        site = ObserverConfig.from_value(observer or event.metadata.get("observer_model"))
        if site is None:
            raise ValueError("observer_contacts require a canonical ObserverConfig")
        times = contact_aware_observer_jd(
            event, site, n_frames=max(25, int(n_frames)), contacts=observer_contacts,
        )
        event = build_event(
            definition, backend=backend, jd=times, solar_scope="local", observer=site,
        )
        metadata = dict(event.metadata)
        metadata.update({
            "observer_contacts_jd": dict(observer_contacts),
            "local_max_jd": float(observer_contacts.get("MAX", event.greatest_jd)),
            "observer_contact_backend": "selected ephemeris/frame backend + optional topographic lunar limb",
        })
        event = replace(event, metadata=metadata)
    return event


def _phase_label(event: ReferenceEvent, jd: float) -> str:
    """Contact-aware label supporting total, annular, partial, and penumbral events."""
    value = float(jd)
    tol = 0.75 / 86400.0
    contacts = dict(event.contacts_jd)
    for name, contact in sorted(contacts.items(), key=lambda item: float(item[1])):
        if abs(value - float(contact)) <= tol:
            return str(name)
    if abs(value - float(event.greatest_jd)) <= tol:
        return "MAX"

    if event.mode == "solar":
        local = str(event.metadata.get("solar_scope", "global")) == "local"
        first_name, last_name = (("C1", "C4") if local else ("P1", "P4"))
        first = float(contacts.get(first_name, min(contacts.values())))
        last = float(contacts.get(last_name, max(contacts.values())))
        ingress_central = contacts.get("C2" if local else "U1")
        egress_central = contacts.get("C3" if local else "U4")
        if value < first:
            return f"before {first_name}"
        if ingress_central is not None and egress_central is not None:
            if value < float(ingress_central):
                return "partial ingress"
            if value < event.greatest_jd:
                return "central phase — ingress"
            if value < float(egress_central):
                return "central phase — egress"
            if value < last:
                return "partial egress"
        else:
            if value < event.greatest_jd:
                return "partial ingress"
            if value < last:
                return "partial egress"
        return f"after {last_name}"

    p1 = float(contacts.get("P1", min(contacts.values())))
    p4 = float(contacts.get("P4", max(contacts.values())))
    u1, u4 = contacts.get("U1"), contacts.get("U4")
    u2, u3 = contacts.get("U2"), contacts.get("U3")
    if value < p1:
        return "before P1"
    if u1 is None or u4 is None:
        return "penumbral ingress" if value < event.greatest_jd else (
            "penumbral egress" if value < p4 else "after P4"
        )
    if value < float(u1):
        return "penumbral ingress"
    if u2 is None or u3 is None:
        if value < event.greatest_jd:
            return "partial umbral ingress"
        if value < float(u4):
            return "partial umbral egress"
        if value < p4:
            return "penumbral egress"
        return "after P4"
    if value < float(u2):
        return "partial umbral ingress"
    if value < event.greatest_jd:
        return "totality — ingress"
    if value < float(u3):
        return "totality — egress"
    if value < float(u4):
        return "partial umbral egress"
    if value < p4:
        return "penumbral egress"
    return "after P4"


def _animation_bounds(states: Sequence[PublicPeakState]) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    centers = np.asarray([state.moon_local_km for state in states], dtype=float)
    x_min = min(0.0, float(np.min(centers[:, 0])))-36_000.0
    x_max = max(0.0, float(np.max(centers[:, 0])))+36_000.0
    transverse = max(
        4.6*float(EARTH_AXES_KM[0]),
        float(np.max(np.abs(centers[:, 1])))+22_000.0,
        float(np.max(np.abs(centers[:, 2])))+22_000.0,
    )
    return (x_min, x_max), (-transverse, transverse), (-transverse, transverse)


def build_public_animation(
    kind: str | ReferenceDefinition | ReferenceEvent = "solar",
    *,
    n_frames: int = 39,
    solar_scope: str = "local",
    ray_azimuth: int = 48,
    backend: str = "auto",
    observer: object | None = None,
    observer_contacts: dict[str, float] | None = None,
) -> PublicAnimation:
    event = _event_for_animation(
        kind, n_frames=n_frames, solar_scope=solar_scope, backend=backend,
        observer=observer, observer_contacts=observer_contacts,
    )
    peak_event = (
        resample_event(event, [event.greatest_jd])
        if bool(event.metadata.get("dynamic_event", False))
        else build_event(
            event.definition,
            backend=str(event.metadata.get("state_source", "reference")),
            jd=[event.greatest_jd],
            solar_scope=solar_scope,
            observer=observer or event.metadata.get("observer_model"),
        )
    )
    geometry = _build_scene_geometry(peak_event)
    peak_raw = _build_inertial_state(
        peak_event,
        peak_event.greatest_jd,
        geometry.reference_cross_axis_gcrf,
        n_azimuth=max(24, int(ray_azimuth)),
    )
    basis, _ = _impact_basis_gcrf(peak_event, peak_raw.bundle_native)
    states = tuple(
        build_public_frame_state(
            event,
            float(jd),
            basis_world_from_scene=basis,
            reference_cross_axis_gcrf=geometry.reference_cross_axis_gcrf,
            ray_azimuth=max(24, int(ray_azimuth)),
            run_ssapy_audit=False,
        )
        for jd in event.jd
    )
    peak_index = int(np.argmin(np.abs(event.jd-event.greatest_jd)))
    greatest = states[peak_index]
    names = tuple(f"utc-{index:03d}" for index in range(len(states)))
    labels = tuple(jd_to_datetime(state.jd_utc).strftime("%H:%M:%S UTC") for state in states)
    phases = tuple(_phase_label(event, state.jd_utc) for state in states)
    return PublicAnimation(
        event=event,
        states=states,
        greatest_state=greatest,
        bounds_km=_animation_bounds(states),
        frame_names=names,
        frame_labels=labels,
        phase_labels=phases,
    )


def _prime_meridian_trace(state: PublicPeakState, *, showlegend: bool = True) -> go.Scatter3d:
    lat = np.linspace(-90.0, 90.0, 181)
    phi = np.radians(lat)
    a, _, b = EARTH_AXES_KM
    denom = np.sqrt((a*np.cos(phi))**2+(b*np.sin(phi))**2)
    x = a*a*np.cos(phi)/denom
    z = b*b*np.sin(phi)/denom
    itrf = np.column_stack([x, np.zeros_like(x), z])
    gcrf = np.asarray(event_itrf_to_gcrf_km(state.event, itrf, state.jd_utc), dtype=float)
    local = _scene_coordinates(gcrf, np.zeros(3), state.basis_world_from_scene, 1.0)
    local *= 1.0014
    return go.Scatter3d(
        x=local[:, 0], y=local[:, 1], z=local[:, 2], mode="lines",
        line=dict(color="rgba(255,255,255,0.72)", width=2.0, dash="dot"),
        name="Greenwich meridian — Earth rotation reference",
        legendgroup="orientation", legendgrouptitle=dict(text="Body motion"),
        showlegend=showlegend, visible="legendonly", hoverinfo="skip",
    )


def _dynamic_update(trace: go.BaseTraceType) -> go.BaseTraceType:
    """Keep only fields that change between UTC states.

    Triangle topology, line styling, legend metadata, and hover templates live
    in the initial traces.  Omitting them from frames substantially reduces the
    self-contained HTML size without changing the rendered geometry.
    """
    if isinstance(trace, go.Mesh3d):
        kwargs = dict(x=trace.x, y=trace.y, z=trace.z)
        if trace.vertexcolor is not None:
            kwargs["vertexcolor"] = trace.vertexcolor
        if trace.facecolor is not None:
            kwargs["facecolor"] = trace.facecolor
        if trace.intensity is not None:
            kwargs["intensity"] = trace.intensity
        if trace.lightposition is not None:
            kwargs["lightposition"] = trace.lightposition
        return go.Mesh3d(**kwargs)
    if isinstance(trace, go.Scatter3d):
        kwargs = dict(x=trace.x, y=trace.y, z=trace.z)
        if trace.text is not None:
            kwargs["text"] = trace.text
        return go.Scatter3d(**kwargs)
    if isinstance(trace, go.Cone):
        return go.Cone(x=trace.x, y=trace.y, z=trace.z,
                       u=trace.u, v=trace.v, w=trace.w)
    if isinstance(trace, go.Surface):
        kwargs = dict(x=trace.x, y=trace.y, z=trace.z)
        if trace.surfacecolor is not None:
            kwargs["surfacecolor"] = trace.surfacecolor
        return go.Surface(**kwargs)
    if isinstance(trace, go.Scatter):
        return go.Scatter(x=trace.x, y=trace.y)
    raise TypeError(f"Unsupported dynamic trace type: {type(trace).__name__}")


def _animation_camera_update(animation: PublicAnimation, focus: str) -> dict[str, object]:
    """Event-wide camera ranges with an equal physical unit scale."""
    states = animation.states
    mode = animation.event.mode
    target_centers = np.asarray([
        state.earth_local_km if mode == "solar" else state.moon_local_km
        for state in states
    ])
    occ_centers = np.asarray([
        state.moon_local_km if mode == "solar" else state.earth_local_km
        for state in states
    ])
    target_radius = float(EARTH_AXES_KM[0]) if mode == "solar" else R_MOON_MEAN_KM
    occ_radius = R_MOON_MEAN_KM if mode == "solar" else float(EARTH_AXES_KM[0])

    def ranges_for(points: np.ndarray, margin: float) -> list[list[float]]:
        lo = np.min(points, axis=0) - margin
        hi = np.max(points, axis=0) + margin
        # Equal close-up ranges keep spherical bodies visibly spherical while
        # still containing the complete trajectory.
        center = 0.5*(lo+hi)
        half = max(float(np.max(0.5*(hi-lo))), margin)
        return [[float(center[i]-half), float(center[i]+half)] for i in range(3)]

    if focus == "target":
        ranges = ranges_for(target_centers, 3.15*target_radius)
        eye = dict(x=-1.58, y=-1.72, z=0.86)
        projection = "perspective"
    elif focus == "occluder":
        ranges = ranges_for(occ_centers, 3.45*occ_radius)
        eye = dict(x=1.45, y=-1.75, z=0.82)
        projection = "perspective"
    elif focus == "optics":
        ranges = [list(pair) for pair in animation.bounds_km]
        eye = dict(x=0.05, y=-2.55, z=0.20)
        projection = "orthographic"
    elif focus == "shadow":
        combined = 0.58*target_centers + 0.42*occ_centers
        margin = max(5.2*float(EARTH_AXES_KM[0]),
                     0.12*abs(animation.bounds_km[0][1]-animation.bounds_km[0][0]))
        ranges = ranges_for(combined, margin)
        eye = dict(x=-1.25, y=-1.90, z=0.68)
        projection = "perspective"
    else:
        ranges = [list(pair) for pair in animation.bounds_km]
        eye = dict(x=-1.35, y=-1.62, z=0.74)
        projection = "perspective"
    return {
        "scene.xaxis.range": ranges[0],
        "scene.yaxis.range": ranges[1],
        "scene.zaxis.range": ranges[2],
        "scene.aspectmode": "manual",
        "scene.aspectratio": equal_unit_aspect(ranges),
        "scene.camera": dict(eye=eye, up=dict(x=0.0, y=0.0, z=1.0),
                             projection=dict(type=projection)),
    }


def _local_animation_figure(animation: PublicAnimation, *, quality: str) -> go.Figure:
    q = _quality(quality)
    first = animation.states[0]
    traces, _ = _local_public_traces(first, q, bounds=animation.bounds_km)
    meridian_index = len(traces)
    traces.append(_prime_meridian_trace(first))
    dynamic_count = len(traces)

    moon_path = np.asarray([state.moon_local_km for state in animation.states])
    traces.append(go.Scatter3d(
        x=moon_path[:, 0], y=moon_path[:, 1], z=moon_path[:, 2], mode="lines",
        line=dict(color="rgba(200,214,230,0.65)", width=2.5),
        name="Moon centre trajectory during event", legendgroup="orientation",
        showlegend=True, visible="legendonly", hoverinfo="skip",
    ))

    fig = go.Figure(data=traces)
    frame_data = []
    for name, state in zip(animation.frame_names, animation.states):
        current, _ = _local_public_traces(state, q, bounds=animation.bounds_km)
        current.append(_prime_meridian_trace(state, showlegend=False))
        if len(current) != dynamic_count:
            raise RuntimeError("Local animation trace order changed between UTC states")
        frame_data.append(go.Frame(
            name=name,
            traces=list(range(dynamic_count)),
            data=[_dynamic_update(trace) for trace in current],
        ))
    fig.frames = tuple(frame_data)

    peak = animation.greatest_state
    target_name = "Earth" if animation.event.mode == "solar" else "Moon"
    occ_name = "Moon" if animation.event.mode == "solar" else "Earth"
    initial = _animation_camera_update(animation, "target")

    def camera_button(label: str, focus: str):
        return dict(label=label, method="relayout",
                    args=[_animation_camera_update(animation, focus)])

    fig.update_layout(
        title=dict(
            text=(f"{animation.event.definition.title} — physical UTC animation"
                  f"<br><sub>Earth rotation, Moon motion, finite-Sun tangents, and surface shadow are rebuilt every state</sub>"),
            x=0.5, y=0.985, font=dict(size=22, color=PUBLIC_COLORS["text"]),
        ),
        scene=dict(
            xaxis=dict(range=list(animation.bounds_km[0]), visible=False,
                       title="Impact-frame X [km]"),
            yaxis=dict(range=list(animation.bounds_km[1]), visible=False,
                       title="Out-of-plane Y [km]"),
            zaxis=dict(range=list(animation.bounds_km[2]), visible=False,
                       title="Impact offset Z [km]"),
            aspectmode=initial["scene.aspectmode"], aspectratio=initial["scene.aspectratio"],
            camera=initial["scene.camera"],
            bgcolor=PUBLIC_COLORS["background"], dragmode="orbit",
            uirevision=f"{animation.event.definition.key}-motion-v19-2",
        ),
        paper_bgcolor=PUBLIC_COLORS["background"],
        font=dict(color=PUBLIC_COLORS["text"], family="Arial, sans-serif"),
        height=900, margin=dict(l=6, r=315, t=94, b=38),
        legend=dict(x=1.005, y=0.98, bgcolor="rgba(7,16,28,0.96)",
                    bordercolor="#42546b", borderwidth=1, font=dict(size=10.2),
                    groupclick="togglegroup", itemclick="toggle"),
        updatemenus=[
            dict(type="dropdown", direction="down", x=1.005, y=1.04,
                 xanchor="left", yanchor="top", bgcolor="#18263a",
                 bordercolor="#71849d", font=dict(color="white", size=12),
                 buttons=[
                     camera_button(f"{target_name} eclipse close-up", "target"),
                     camera_button(f"{occ_name} tangent close-up", "occluder"),
                     camera_button("Shadow envelope", "shadow"),
                     camera_button("Full Earth–Moon system", "system"),
                     camera_button("Orthographic optics side", "optics"),
                 ]),
            dict(type="buttons", direction="left", x=0.012, y=0.018,
                 xanchor="left", yanchor="bottom", showactive=True,
                 bgcolor="#18263a", bordercolor="#71849d",
                 font=dict(color="white", size=11), buttons=[
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
                 ]),
        ],
        annotations=[dict(
            x=1.005, y=0.18, xref="paper", yref="paper", xanchor="left",
            yanchor="top", showarrow=False, width=285, align="left",
            borderpad=8, bgcolor="rgba(7,16,28,0.96)",
            bordercolor="#42546b", borderwidth=1,
            font=dict(size=10.1, color=PUBLIC_COLORS["muted"]),
            text=(
                "<b>Physical display rules</b><br>"
                "No boosted body radii<br>No Earth–Moon distance compression<br>"
                "Sun remains off-screen at real distance<br>"
                "Yellow ends at first opaque surface<br>"
                "Magenta is blocked-light geometry<br><br>"
                f"Frames: {len(animation.states)}<br>"
                f"Earth–Moon at maximum: {np.linalg.norm(peak.state.moon_gcrf_km):,.1f} km"
            ),
        )],
    )
    return fig


def _true_scale_animation_figure(animation: PublicAnimation) -> go.Figure:
    first_fig = build_true_scale_figure(animation.states[0])
    traces = list(first_fig.data)
    # Only the heliocentric Sun is static. Earlier code assumed that the
    # first three traces were concentric Sun/glow surfaces; after replacing
    # those artifacts with one exact photosphere, that hard-coded slice also
    # froze the Earth and Moon at frame zero. Select by semantic legend group
    # so every moving body, locator, and photospheric path updates each UTC
    # state regardless of mesh chunk count.
    dynamic_indices = [
        index for index, trace in enumerate(traces)
        if getattr(trace, "legendgroup", None) != "true-sun"
    ]

    earth_path = np.asarray([
        _scene_coordinates(state.state.earth_heliocentric_km.reshape(1, 3),
                           np.zeros(3), state.basis_world_from_scene,
                           149_597_870.7)[0]
        for state in animation.states
    ])
    moon_path = np.asarray([
        _scene_coordinates(state.state.moon_heliocentric_km.reshape(1, 3),
                           np.zeros(3), state.basis_world_from_scene,
                           149_597_870.7)[0]
        for state in animation.states
    ])
    traces.extend([
        go.Scatter3d(x=earth_path[:, 0], y=earth_path[:, 1], z=earth_path[:, 2],
                     mode="lines", line=dict(color="#4f9cff", width=4),
                     name="Earth event arc", legendgroup="event-paths", showlegend=True),
        go.Scatter3d(x=moon_path[:, 0], y=moon_path[:, 1], z=moon_path[:, 2],
                     mode="lines", line=dict(color="#d4d4d4", width=2),
                     name="Moon event track", legendgroup="event-paths", showlegend=True),
    ])
    fig = go.Figure(data=traces)
    frames = []
    for name, state in zip(animation.frame_names, animation.states):
        current = list(build_true_scale_figure(state).data)
        frames.append(go.Frame(
            name=name,
            traces=dynamic_indices,
            data=[_dynamic_update(current[index]) for index in dynamic_indices],
        ))
    fig.frames = tuple(frames)
    layout = first_fig.layout.to_plotly_json()
    layout["title"] = dict(
        text=(f"{animation.event.definition.title} — true heliocentric scale"
              "<br><sub>Real solar radius, real Sun distance, and the event-length Earth/Moon trajectories</sub>"),
        x=0.5, y=0.985, font=dict(size=22, color="white"),
    )
    layout["height"] = 860
    fig.update_layout(**layout)
    return fig


def _cross_section_animation_figure(animation: PublicAnimation) -> go.Figure:
    first_fig = build_cross_section_figure(animation.states[0])
    fig = go.Figure(data=list(first_fig.data))
    indices = list(range(len(fig.data)))
    frames = []
    for name, state in zip(animation.frame_names, animation.states):
        current = list(build_cross_section_figure(state).data)
        if len(current) != len(indices):
            raise RuntimeError("Cross-section trace order changed between UTC states")
        frames.append(go.Frame(
            name=name, traces=indices,
            data=[_dynamic_update(trace) for trace in current],
        ))
    fig.frames = tuple(frames)
    layout = first_fig.layout.to_plotly_json()
    layout["title"] = dict(
        text=("Exact finite-Sun impact-plane cross-section"
              "<br><sub>The same UTC state as the local and one-AU 3-D views</sub>"),
        x=0.5, font=dict(size=20, color="white"),
    )
    layout.setdefault("xaxis", {})["range"] = list(animation.bounds_km[0])
    layout.setdefault("yaxis", {})["range"] = list(animation.bounds_km[2])
    fig.update_layout(**layout)
    return fig


def _plot_div(fig: go.Figure, div_id: str, *, webgl: bool = True) -> str:
    config = {
        "displaylogo": False,
        "responsive": True,
        "scrollZoom": True,
        "toImageButtonOptions": {"format": "png", "scale": 2},
    }
    if webgl:
        config["plotGlPixelRatio"] = 2.0
    return pio.to_html(fig, include_plotlyjs=False, full_html=False,
                       div_id=div_id, config=config)


def generate_public_ultra_animation(
    kind: str | ReferenceDefinition | ReferenceEvent = "solar",
    output_path: str | Path = "eclipse_public_physical_animation_v22_2.html",
    *,
    n_frames: int = 39,
    quality: str = "motion",
    solar_scope: str = "local",
    backend: str = "auto",
) -> str:
    """Write a self-contained, contact-aware public eclipse animation."""
    q = _quality(quality)
    animation = build_public_animation(
        kind, n_frames=n_frames, solar_scope=solar_scope,
        ray_azimuth=q.ray_azimuth, backend=backend,
    )
    local = _local_animation_figure(animation, quality=quality)
    true_scale = _true_scale_animation_figure(animation)
    cross = _cross_section_animation_figure(animation)
    local_div = _plot_div(local, "local-motion-plot")
    true_div = _plot_div(true_scale, "true-motion-plot")
    cross_div = _plot_div(cross, "cross-motion-plot", webgl=False)

    info = [
        {
            "name": name,
            "utc": label,
            "phase": phase,
            "jd": float(state.jd_utc),
            "earthMoonKm": float(np.linalg.norm(state.state.moon_gcrf_km)),
            "sunEarthKm": float(np.linalg.norm(state.state.sun_gcrf_km)),
            "axisMissKm": float(state.axis_impact_km),
        }
        for name, label, phase, state in zip(
            animation.frame_names, animation.frame_labels,
            animation.phase_labels, animation.states
        )
    ]
    target = "Earth" if animation.event.mode == "solar" else "Moon"
    occluder = "Moon" if animation.event.mode == "solar" else "Earth"
    peak = animation.greatest_state
    backend_card = _backend_card(peak)
    start = jd_to_datetime(animation.states[0].jd_utc)
    stop = jd_to_datetime(animation.states[-1].jd_utc)

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(animation.event.definition.title)} — scientific Plotly animation V22.2</title>
<script>{get_plotlyjs()}</script>
<style>
:root{{--bg:#02040a;--panel:#07101c;--line:#42546b;--text:#f1f6fc;--muted:#9fb0c4;--accent:#6fc8ff}}
*{{box-sizing:border-box}} body{{margin:0;background:radial-gradient(circle at 38% 8%,#0b1827 0,#02040a 46%,#010207 100%);color:var(--text);font-family:Inter,Segoe UI,Arial,sans-serif}}
header{{padding:18px 24px 14px;border-bottom:1px solid var(--line);background:rgba(2,4,10,.94)}}
h1{{margin:0;font-size:clamp(22px,2.5vw,35px);font-weight:660}} .subtitle{{color:var(--muted);margin-top:7px;line-height:1.45}}
.badges{{display:flex;gap:8px;flex-wrap:wrap;margin-top:11px}} .badge{{border:1px solid #53677f;background:#0d1a2a;padding:5px 9px;border-radius:999px;font-size:12px}}
.transport{{position:sticky;top:0;z-index:20;display:grid;grid-template-columns:auto auto minmax(180px,1fr) auto;gap:10px;align-items:center;padding:10px 18px;background:rgba(4,9,16,.96);border-bottom:1px solid var(--line);backdrop-filter:blur(8px)}}
.transport button{{background:#18314b;color:white;border:1px solid #78b8ee;border-radius:7px;padding:8px 12px;font-weight:650;cursor:pointer}}
.transport input{{width:100%}} .readout{{min-width:265px;text-align:right;font-variant-numeric:tabular-nums}} .readout b{{color:#fff}} .phase{{color:#ffd36d}}
nav{{display:flex;gap:8px;padding:12px 18px 0;flex-wrap:wrap}} .tab-button{{border:1px solid #53677f;background:#0c1725;color:var(--text);padding:9px 13px;border-radius:8px 8px 0 0;cursor:pointer;font-weight:600}} .tab-button.active{{background:#18314b;border-color:#78b8ee}}
.tab{{display:none;padding:0 10px 13px}} .tab.active{{display:block}} .plot-shell{{border:1px solid var(--line);border-radius:10px;overflow:hidden;background:var(--panel);box-shadow:0 16px 60px rgba(0,0,0,.35)}}
.guide{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin:14px 12px 24px}} .card{{background:rgba(7,16,28,.94);border:1px solid var(--line);border-radius:9px;padding:13px 15px;line-height:1.45;color:var(--muted)}} .card b{{color:var(--text)}}
footer{{padding:17px 24px 26px;color:var(--muted);border-top:1px solid var(--line);font-size:13px;line-height:1.5}}
@media(max-width:840px){{.transport{{grid-template-columns:auto auto 1fr}}.readout{{grid-column:1/-1;text-align:left}}}}
</style></head>
<body>
<header><h1>{html.escape(animation.event.definition.title)} — V22.2 scientific Plotly contact-to-contact illumination</h1>
<div class="subtitle">{start.strftime('%Y-%m-%d %H:%M:%S')} to {stop.strftime('%H:%M:%S')} UTC. Every state recomputes Earth rotation, Moon motion, exact photosphere-to-surface paths, first opaque intersections, and finite-disc eclipse shading.</div>
<div class="badges"><span class="badge">{html.escape(str(animation.event.metadata.get('state_source', 'reference')))} state mode</span><span class="badge">NASA/GSFC contact scaffold</span><span class="badge">WGS-84 Earth</span><span class="badge">finite solar photosphere</span><span class="badge">true Earth–Moon scale</span></div></header>
<div class="transport"><button id="play">▶ Play</button><button id="pause">❚❚ Pause</button><input id="timeline" type="range" min="0" max="{len(info)-1}" value="0" step="1"><div class="readout"><b id="utc"></b><br><span class="phase" id="phase"></span></div></div>
<nav><button class="tab-button active" data-tab="local">Earth–Moon eclipse system</button><button class="tab-button" data-tab="true">True Sun distance</button><button class="tab-button" data-tab="cross">Exact optical cross-section</button></nav>
<section id="tab-local" class="tab active"><div class="plot-shell">{local_div}</div></section>
<section id="tab-true" class="tab"><div class="plot-shell">{true_div}</div></section>
<section id="tab-cross" class="tab"><div class="plot-shell">{cross_div}</div></section>
<div class="guide"><div class="card"><b>What moves</b><br>The WGS-84 Earth rotates beneath the inertial geometry; its texture and Greenwich meridian move together. The Moon follows the validated event track and remains synchronously Earth-facing. Body shading and the representative photospheric paths are solved again at every UTC state.</div>
<div class="card"><b>Physical light paths</b><br>The visible yellow lines are deterministic samples from the real finite solar photosphere. Each stops at the first opaque {occluder}/target surface. Full Sun visibility is evaluated over every rendered surface vertex, so the shading is continuous rather than limited to these sample lines.</div>
<div class="card"><b>Optional diagnostics</b><br>Magenta is blocked-light geometry, not transmitted sunlight. Red and blue are exact umbral/antumbral and penumbral tangents. They and the optical volumes start hidden to keep the public animation readable.</div>
<div class="card"><b>Precision architecture</b><br>Earth is WGS-84; Sun and Moon are exact spheres. Mesh chunks remain below WebGL index limits. The local scene uses uncompressed kilometres and the true-distance scene uses AU, both with equal physical axis units.</div>
<div class="card">{backend_card}</div></div>
<footer>The NASA/GSFC contacts provide the public event timeline. SSAPy and SSAPy-Toolkit are supported as explicit runtime state drivers; the state-provenance card identifies whether they actually drove this file or whether the validated reference reconstruction was selected. Lunar totality colour is an illustrative Danjon-style appearance cue.</footer>
<script>
const frames={json.dumps(info, separators=(',',':'))}; const plots=['local-motion-plot','true-motion-plot','cross-motion-plot']; let timer=null; let index=0;
function updateReadout(){{const f=frames[index];document.getElementById('utc').textContent=f.utc;document.getElementById('phase').textContent=f.phase+' · Earth–Moon '+f.earthMoonKm.toLocaleString(undefined,{{maximumFractionDigits:0}})+' km';document.getElementById('timeline').value=index;}}
function show(i){{index=Math.max(0,Math.min(frames.length-1,Number(i)));const name=frames[index].name;plots.forEach(id=>{{const div=document.getElementById(id);if(div)Plotly.animate(div,[name],{{mode:'immediate',frame:{{duration:0,redraw:true}},transition:{{duration:0}}}});}});updateReadout();}}
document.getElementById('timeline').addEventListener('input',e=>{{if(timer){{clearInterval(timer);timer=null}}show(e.target.value)}});
document.getElementById('play').addEventListener('click',()=>{{if(timer)return;timer=setInterval(()=>{{if(index>=frames.length-1)index=-1;show(index+1)}},180)}});
document.getElementById('pause').addEventListener('click',()=>{{if(timer)clearInterval(timer);timer=null}});
const buttons=[...document.querySelectorAll('.tab-button')],tabs=[...document.querySelectorAll('.tab')];buttons.forEach(btn=>btn.addEventListener('click',()=>{{buttons.forEach(x=>x.classList.remove('active'));tabs.forEach(x=>x.classList.remove('active'));btn.classList.add('active');const target=document.getElementById('tab-'+btn.dataset.tab);target.classList.add('active');target.querySelectorAll('.js-plotly-plot').forEach(div=>Plotly.Plots.resize(div));}}));
updateReadout();
</script></body></html>"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return str(output)


def public_animation_metrics(
    kind: str | ReferenceDefinition | ReferenceEvent = "solar",
    *,
    n_frames: int = 39,
    solar_scope: str = "local",
    quality: str = "motion",
    backend: str = "auto",
) -> dict[str, object]:
    q = _quality(quality)
    animation = build_public_animation(kind, n_frames=n_frames,
                                       solar_scope=solar_scope,
                                       ray_azimuth=q.ray_azimuth,
                                       backend=backend)
    penetrations = {"earth": 0, "moon": 0}
    max_residual = 0.0
    moon_positions = []
    meridian_points = []
    for state in animation.states:
        result = bundle_penetrations(state.state.bundle_native)
        penetrations["earth"] += int(result["earth"])
        penetrations["moon"] += int(result["moon"])
        residual = tangent_residuals(state.state.bundle_native)
        max_residual = max(max_residual, *(float(v) for v in residual.values()))
        moon_positions.append(state.state.moon_gcrf_km)
        meridian = _prime_meridian_trace(state, showlegend=False)
        mid = len(meridian.x)//2
        meridian_points.append([meridian.x[mid], meridian.y[mid], meridian.z[mid]])
    moon_positions = np.asarray(moon_positions, dtype=float)
    meridian_points = np.asarray(meridian_points, dtype=float)
    earth_rotation_angle = math.degrees(math.acos(np.clip(
        np.dot(meridian_points[0]/np.linalg.norm(meridian_points[0]),
               meridian_points[-1]/np.linalg.norm(meridian_points[-1])), -1.0, 1.0)))
    return {
        "event": animation.event.definition.key,
        "backend_requested": backend,
        "backend_selected": animation.event.metadata.get("state_source", "reference"),
        "scope": animation.event.metadata.get("animation_scope", "global P1-P4"),
        "frame_count": len(animation.states),
        "quality": quality,
        "body_resolution": [q.body_lat, q.body_lon],
        "shadow_resolution": q.shadow_resolution,
        "ray_azimuth_count": q.ray_azimuth,
        "start_utc": jd_to_datetime(animation.states[0].jd_utc).isoformat(),
        "end_utc": jd_to_datetime(animation.states[-1].jd_utc).isoformat(),
        "earth_rotation_reference_angle_deg": earth_rotation_angle,
        "moon_geocentric_displacement_km": float(np.linalg.norm(moon_positions[-1]-moon_positions[0])),
        "sun_earth_distance_range_km": [
            float(min(np.linalg.norm(state.state.sun_gcrf_km) for state in animation.states)),
            float(max(np.linalg.norm(state.state.sun_gcrf_km) for state in animation.states)),
        ],
        "earth_moon_distance_range_km": [
            float(min(np.linalg.norm(state.state.moon_gcrf_km) for state in animation.states)),
            float(max(np.linalg.norm(state.state.moon_gcrf_km) for state in animation.states)),
        ],
        "ray_penetrations": penetrations,
        "maximum_tangent_residual_km": max_residual,
        "ssapy_diagnostics": animation.greatest_state.ssapy_diagnostics,
    }
