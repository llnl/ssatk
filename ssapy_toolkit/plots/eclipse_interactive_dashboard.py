"""Validated interactive eclipse dashboards with stable line geometry.

The dashboard deliberately uses one rotatable WebGL scene for the local
Earth-Moon construction and two ordinary Cartesian diagnostics beside it:

* the 3-D scene is target-centred, expressed only in kilometres, and clips
  every visible ray to the local physical window;
* the upper Cartesian strip preserves the full Sun-Earth separation in AU;
* the lower Cartesian strip shows the same finite-Sun cross-section in
  thousands of kilometres.

Keeping the one-AU source out of the local WebGL coordinate cloud avoids the
float32 precision loss that previously made otherwise straight rays jitter,
miss tangent points, or appear to cut through a body.  Footprint curves are
also kept as explicit contiguous segments so Plotly cannot join unrelated
arcs across a miss, pole, or antimeridian discontinuity.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ssapy_toolkit.io.eclipse_asset_resolver import resolve_image
import math

import numpy as np

from ssapy_toolkit.compute.eclipse_reference_events import (
    AU_KM,
    EARTH_AXES_KM,
    LUNAR_2025,
    RE_KM,
    R_MOON_MEAN_KM,
    R_SUN_KM,
    SOLAR_2024,
    SOLAR_GREATEST_SITE_LAT_DEG,
    SOLAR_GREATEST_SITE_LON_EAST_DEG,
    ReferenceEvent,
    jd_to_datetime,
    lunar_reference_state,
    solar_central_line_wgs84,
    solar_local_contacts,
    solar_local_phase_status,
    _LUNAR_MOON_SD_DEG,
    _LUNAR_P_RADIUS_DEG,
    _LUNAR_U_RADIUS_DEG,
    _unit,
)
from ssapy_toolkit.compute.eclipse_raytrace import (
    LUNAR_DANJON_EARTH_RADIUS_KM,
    ReferenceRayBundle,
    RayPath,
    solar_footprint_segments,
    tangent_cross_section_paths,
    trace_reference_rays,
)
from ssapy_toolkit.compute.eclipse_local_finite_geometry import (
    build_impact_plane_frame,
    event_view_ranges,
    peak_impact_basis,
    plotly_view_traces,
    view_ranges as impact_view_ranges,
)

from ssapy_toolkit.compute.eclipse_local_finite_geometry import (
    build_impact_plane_frame,
    peak_impact_basis,
    plotly_view_traces,
    view_ranges as impact_view_ranges,
)

from ssapy_toolkit.plots.eclipse_rendering import (
    _load_texture,
    _points_lat_lon,
    _split_longitudes,
    SOLAR_PARTIAL_OBSCURATION_BAND_COLORS,
    SOLAR_PARTIAL_OBSCURATION_BAND_EDGES,
    SOLAR_TOTALITY_PERCENT_BAND_COLORS,
    SOLAR_TOTALITY_PERCENT_BAND_EDGES,
    contact_status,
    solar_central_path_cached,
    solar_partial_visibility_grid_cached,
    solar_totality_band_polygons,
    solar_totality_duration_grid_cached,
    solar_totality_duration_raster_cached,
    solar_umbra_corridor_cached,
    solar_site_visibility_curve,
)

ASSET_DIR = Path(__file__).resolve().parent


_QUALITY_PRESETS = {
    "balanced": {
        "static": dict(body=(61, 120), atmosphere=(37, 72), shadow=81),
        "animated": dict(body=(45, 90), atmosphere=(29, 56), shadow=61),
    },
    "high": {
        "static": dict(body=(161, 320), atmosphere=(81, 160), shadow=241),
        "animated": dict(body=(61, 120), atmosphere=(37, 72), shadow=91),
    },
    "ultra": {
        # Publication/inspection meshes: about 115k surface vertices per
        # body plus a 361x361 independently resolved WGS-84 eclipse patch.
        # Animation keeps a lighter profile so contact-to-contact playback
        # remains usable in ordinary browsers.
        "static": dict(body=(241, 480), atmosphere=(121, 240), shadow=361),
        "animated": dict(body=(81, 160), atmosphere=(49, 96), shadow=141),
    },
}


def _quality_profile(name: str, *, animated: bool) -> dict:
    key = str(name).lower().strip()
    if key not in _QUALITY_PRESETS:
        raise ValueError("quality must be 'balanced', 'high', or 'ultra'")
    branch = "animated" if animated else "static"
    return dict(_QUALITY_PRESETS[key][branch])


def _optical_basis(event: ReferenceEvent) -> np.ndarray:
    """Stable peak-event impact-plane basis for the local 3-D scene.

    The previous north-referenced plane did not generally contain the target
    centre because the shadow-axis miss has both north/south and east/west
    components.  The impact-plane basis contains the source, occluder, shadow
    axis and target centre at greatest eclipse, so the selected finite-Sun
    tangents no longer look laterally detached from either body.
    """
    return peak_impact_basis(event)


# Backward-compatible internal alias used by older tests and downstream code.
_local_basis = _optical_basis


def _local_origin(event: ReferenceEvent, bundle: ReferenceRayBundle) -> np.ndarray:
    """Return the eclipsed target centre used by the local 3-D panel.

    A solar eclipse is Earth-centred and a lunar eclipse is Moon-centred.
    Keeping the target at the local origin prevents the lunar disk from
    drifting outside a peak-centred close-up as the contact-to-contact
    animation advances.
    """
    return (bundle.earth_center_km if event.mode == "solar"
            else bundle.moon_center_km)


def _transform(points, origin, basis, scale=1.0) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    return ((values - np.asarray(origin, dtype=float)) @ np.asarray(basis, dtype=float)) / float(scale)


def _transform_xyz_trace(trace, origin, basis, scale=1.0):
    x = np.asarray(trace.x, dtype=float)
    y = np.asarray(trace.y, dtype=float)
    z = np.asarray(trace.z, dtype=float)
    shape = x.shape
    points = np.column_stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)])
    result = _transform(points, origin, basis, scale=scale)
    trace.x = result[:, 0].reshape(shape)
    trace.y = result[:, 1].reshape(shape)
    trace.z = result[:, 2].reshape(shape)
    return trace


def _join_segments(segments: list[np.ndarray] | tuple[np.ndarray, ...]) -> np.ndarray:
    """Join 3-D polylines with NaN separators rather than false chords."""
    clean = [np.asarray(segment, dtype=float).reshape(-1, 3)
             for segment in segments if np.asarray(segment).size]
    if not clean:
        return np.empty((0, 3), dtype=float)
    joined: list[np.ndarray] = []
    for index, segment in enumerate(clean):
        if index:
            joined.append(np.full((1, 3), np.nan))
        joined.append(segment)
    return np.vstack(joined)




def _surface_lift(points: np.ndarray, center: np.ndarray, lift_km: float) -> np.ndarray:
    """Lift a surface overlay slightly to avoid WebGL depth fighting.

    The geographic path is unchanged.  Only the display radius is increased by
    a documented few kilometres so a line does not flicker in and out of the
    textured body mesh at identical depth.
    """
    values = np.asarray(points, dtype=float).reshape(-1, 3)
    if not len(values):
        return values
    origin = np.asarray(center, dtype=float).reshape(3)
    relative = values-origin
    radius = np.linalg.norm(relative, axis=1)
    safe = np.where(radius > 0.0, radius, 1.0)
    return origin+relative*((safe+float(lift_km))/safe)[:, None]

def _clip_segment_to_x(points: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    """Clip a straight ray segment to a local X interval."""
    values = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(values) < 2:
        return np.empty((0, 3), dtype=float)
    p0 = values[0]
    p1 = values[-1]
    delta = p1 - p0
    if abs(delta[0]) < 1.0e-15:
        if xmin <= p0[0] <= xmax:
            return np.vstack([p0, p1])
        return np.empty((0, 3), dtype=float)
    t0 = (xmin - p0[0]) / delta[0]
    t1 = (xmax - p0[0]) / delta[0]
    lower = max(0.0, min(t0, t1))
    upper = min(1.0, max(t0, t1))
    if upper < lower:
        return np.empty((0, 3), dtype=float)
    return np.vstack([p0 + lower * delta, p0 + upper * delta])


def _local_bounds(event: ReferenceEvent, basis: np.ndarray) -> dict[str, float | list[float]]:
    bundle = trace_reference_rays(event.definition, event.greatest_jd, n_azimuth=16)
    origin = _local_origin(event, bundle)
    earth_q = _transform(bundle.earth_center_km.reshape(1, 3), origin, basis)[0]
    moon_q = _transform(bundle.moon_center_km.reshape(1, 3), origin, basis)[0]
    x_margin = max(4.0 * R_MOON_MEAN_KM, 1.65 * RE_KM)
    xmin = min(float(earth_q[0]), float(moon_q[0])) - x_margin
    xmax = max(float(earth_q[0]), float(moon_q[0])) + x_margin
    transverse = max(
        2.20 * RE_KM,
        abs(float(earth_q[1])) + 2.6 * RE_KM,
        abs(float(earth_q[2])) + 2.6 * RE_KM,
        abs(float(moon_q[1])) + 2.6 * R_MOON_MEAN_KM,
        abs(float(moon_q[2])) + 2.6 * R_MOON_MEAN_KM,
    )
    return {
        "xmin": xmin,
        "xmax": xmax,
        "xrange": [xmin, xmax],
        "yrange": [-transverse, transverse],
        "zrange": [-transverse, transverse],
        "earth": earth_q,
        "moon": moon_q,
    }


def _selected_opposite_paths(bundle: ReferenceRayBundle, paths: tuple[RayPath, ...],
                             *, cross_axis_world=None) -> tuple[RayPath, RayPath]:
    """Return the upper/lower tangent rays in a fixed display cross-section."""
    occ = (bundle.moon_center_km if bundle.mode == "solar"
           else bundle.earth_center_km)
    if cross_axis_world is None:
        radial = paths[0].points_km[1] - occ
        cross_axis_world = radial - bundle.axis_hat * float(np.dot(radial, bundle.axis_hat))
    cross_axis = _unit(cross_axis_world)
    values = np.asarray([
        float(np.dot(path.points_km[1] - occ, cross_axis)) for path in paths
    ])
    return paths[int(np.argmax(values))], paths[int(np.argmin(values))]


def _select_display_pair(bundle: ReferenceRayBundle, paths: tuple[RayPath, ...],
                         tangent_points: np.ndarray, origin: np.ndarray,
                         basis: np.ndarray) -> tuple[RayPath, RayPath, np.ndarray]:
    upper, lower = _selected_opposite_paths(
        bundle, paths, cross_axis_world=np.asarray(basis, dtype=float)[:, 2],
    )
    selected = np.vstack([upper.points_km[1], lower.points_km[1]])
    return upper, lower, selected


def _line3d(points, color, width, name, *, showlegend=True, dash=None,
            legendgroup=None, legendgrouptitle_text=None, visible=True,
            scene="scene", legend="legend2", hovertemplate=None):
    import plotly.graph_objects as go

    values = np.asarray(points, dtype=float).reshape(-1, 3)
    line = dict(color=color, width=width)
    if dash is not None:
        line["dash"] = dash
    return go.Scatter3d(
        x=values[:, 0] if len(values) else [],
        y=values[:, 1] if len(values) else [],
        z=values[:, 2] if len(values) else [],
        mode="lines", line=line, name=name, showlegend=showlegend,
        hoverinfo="skip" if hovertemplate is None else None,
        hovertemplate=hovertemplate, scene=scene, legend=legend,
        legendgroup=legendgroup, visible=visible,
        legendgrouptitle=(dict(text=legendgrouptitle_text)
                          if legendgrouptitle_text else None),
        connectgaps=False,
    )


def _marker3d(points, color, size, name, text=None, *, showlegend=False,
              scene="scene", legend="legend2", legendgroup=None):
    import plotly.graph_objects as go

    values = np.asarray(points, dtype=float).reshape(-1, 3)
    labels = text if text is not None else [name] * len(values)
    return go.Scatter3d(
        x=values[:, 0] if len(values) else [],
        y=values[:, 1] if len(values) else [],
        z=values[:, 2] if len(values) else [],
        mode="markers+text" if labels else "markers",
        marker=dict(size=size, color=color, line=dict(color="white", width=0.5)),
        text=labels, textposition="top center", textfont=dict(size=10, color=color),
        name=name, showlegend=showlegend, hovertemplate=f"{name}<extra></extra>",
        scene=scene, legend=legend, legendgroup=legendgroup,
    )


def _shadow_volume_trace(bundle: ReferenceRayBundle, paths, tangent_points,
                         origin, basis, *, color: str, name: str,
                         opacity: float = 0.12):
    """Build an untwisted optical envelope clipped before the target body.

    Using each ray's independently clipped target endpoint can create a
    serrated or self-crossing ring when only part of the tangent family hits
    the target.  The interactive envelope now ends on one plane just upstream
    of the target's near limb.  The ray lines themselves still end at their
    exact first-surface intersections; this mesh is only the optional volume
    cue and therefore never enters Earth or Moon.
    """
    import plotly.graph_objects as go

    tangents = _transform(tangent_points, origin, basis)
    endpoint_candidates = _transform(
        np.asarray([path.endpoint_km for path in paths]), origin, basis,
    )
    # +X is downstream and the local origin is always the eclipsed target
    # centre.  Stop a little before the upstream limb so translucent WebGL
    # triangles cannot depth-sort through the opaque body.
    clip_x = -float(bundle.target_radius_km)-25.0
    endpoints = np.empty_like(tangents)
    for index, (tangent, candidate) in enumerate(zip(tangents, endpoint_candidates)):
        delta = candidate-tangent
        if abs(float(delta[0])) < 1.0e-12:
            endpoints[index] = candidate
            continue
        parameter = (clip_x-float(tangent[0]))/float(delta[0])
        # The clip plane is upstream of every target-surface hit.  Clamp only
        # as a guard against pathological off-event geometry.
        parameter = float(np.clip(parameter, 0.0, 1.0))
        endpoints[index] = tangent+parameter*delta
    n = len(tangents)
    vertices = np.vstack([tangents, endpoints])
    ii: list[int] = []
    jj: list[int] = []
    kk: list[int] = []
    for index in range(n):
        nxt = (index + 1) % n
        ii.extend([index, index])
        jj.extend([n + index, n + nxt])
        kk.extend([n + nxt, nxt])
    return go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=ii, j=jj, k=kk, color=color, opacity=float(opacity),
        flatshading=True, lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0),
        name=name, showlegend=True, visible="legendonly", hoverinfo="skip",
        scene="scene", legend="legend2", legendgroup="optical-volumes",
        legendgrouptitle=(dict(text="Optical volumes") if name.startswith("Umbral") else None),
    )


def _body_traces(event: ReferenceEvent, jd_value: float, bundle: ReferenceRayBundle,
                 origin: np.ndarray, basis: np.ndarray,
                 camera_hat_world: np.ndarray, *,
                 n_lat: int, n_lon: int, atmosphere_n_lat: int,
                 atmosphere_n_lon: int, shadow_resolution: int = 0):
    from ssapy_toolkit.plots.globe_orbit_daynight_plotly import (
        _earth_atmosphere_trace,
        _earth_eclipse_shadow_trace,
        _earth_mesh,
    )
    from ssapy_toolkit.plots.moon_render import moon_mesh_plotly
    from ssapy_toolkit.coordinates.eclipse_lunar_attitude import moon_attitude_for_event

    earth = bundle.earth_center_km
    moon = bundle.moon_center_km
    sun = bundle.sun_center_km
    sun_hat = _unit(sun - earth)
    solar_frame = event.mode == "solar"

    # Render the global body without embedding the eclipse into its relatively
    # coarse texture mesh.  Solar eclipses receive a separate, much denser
    # local WGS-84 shadow layer below, so a roughly 200-km totality corridor is
    # not quantised to one or two global vertices.
    earth_trace = _earth_mesh(
        sun_hat, n_lat=int(n_lat), n_lon=int(n_lon), center=tuple(earth),
        shadow_body_center_km=None,
        shadow_body_radius_km=None,
        time_jd=None if solar_frame else float(jd_value), sun_position_km=sun,
        physical_center_km=earth, night_floor=0.024, show_city_lights=False,
        texture_path=resolve_image("earth_albedo").path, exposure=1.24,
        view_hat=camera_hat_world, specular_strength=0.36,
    )
    earth_trace.name = "Earth - WGS-84 textured body"
    earth_trace.showlegend = True
    earth_trace.legend = "legend2"
    earth_trace.legendgroup = "bodies"
    earth_trace.legendgrouptitle = dict(text="Bodies")
    _transform_xyz_trace(earth_trace, origin, basis)
    earth_trace.scene = "scene"

    traces = [earth_trace]

    if solar_frame and int(shadow_resolution) >= 25:
        current_center = solar_central_line_wgs84(float(jd_value))
        patch_center = None if current_center is None else np.asarray(current_center[2], dtype=float)
        shadow = _earth_eclipse_shadow_trace(
            sun_hat,
            moon,
            bundle.umbra_optical_radius_km,
            sun_position_km=sun,
            physical_center_km=earth,
            center=earth,
            time_jd=None,
            patch_center_gcrf_km=patch_center,
            resolution=int(shadow_resolution),
            nudge=1.0010,
            night_floor=0.024,
        )
        shadow.name = "Resolved finite-disc surface shadow"
        shadow.showlegend = True
        shadow.legend = "legend2"
        shadow.legendgroup = "surface-shadow"
        shadow.legendgrouptitle = dict(text="Resolved surface eclipse")
        _transform_xyz_trace(shadow, origin, basis)
        shadow.scene = "scene"
        traces.append(shadow)

    atmosphere = _earth_atmosphere_trace(
        center=tuple(earth), sun_hat=sun_hat, view_hat=camera_hat_world,
        time_jd=None if solar_frame else float(jd_value),
        n_lat=int(atmosphere_n_lat), n_lon=int(atmosphere_n_lon),
        altitude_km=105.0, max_alpha=0.34,
    )
    _transform_xyz_trace(atmosphere, origin, basis)
    atmosphere.scene = "scene"
    atmosphere.showlegend = False
    traces.append(atmosphere)

    # A faint outer shell gives the limb a smoother falloff at close zoom
    # without laying a uniform blue haze over the disk.
    outer_atmosphere = _earth_atmosphere_trace(
        center=tuple(earth), sun_hat=sun_hat, view_hat=camera_hat_world,
        time_jd=None if solar_frame else float(jd_value),
        n_lat=max(25, int(atmosphere_n_lat)//2*2+1),
        n_lon=max(48, int(atmosphere_n_lon)//2*2),
        altitude_km=225.0, max_alpha=0.125,
    )
    _transform_xyz_trace(outer_atmosphere, origin, basis)
    outer_atmosphere.scene = "scene"
    outer_atmosphere.showlegend = False
    outer_atmosphere.name = "Outer atmosphere"
    traces.append(outer_atmosphere)

    moon_trace = moon_mesh_plotly(
        moon, R_MOON_MEAN_KM, sun_hat=sun_hat, real_center_km=moon,
        mode=event.mode, n_lat=int(n_lat), n_lon=int(n_lon),
        real_sun_position_km=sun, ambient_floor=0.010,
        texture_path=resolve_image("moon_albedo").path, view_hat=camera_hat_world,
        exposure=1.20, eclipse_occluder_radius_km=LUNAR_DANJON_EARTH_RADIUS_KM,
        body_to_gcrf=moon_attitude_for_event(event, float(jd_value)).body_to_gcrf,
    )
    moon_trace.name = "Moon - mean solid radius textured body"
    moon_trace.showlegend = True
    moon_trace.legend = "legend2"
    moon_trace.legendgroup = "bodies"
    _transform_xyz_trace(moon_trace, origin, basis)
    moon_trace.scene = "scene"
    traces.append(moon_trace)
    return traces


def _static_solar_map_traces():
    """Return the full partial-visibility footprint plus totality regions.

    Blue categories encode the maximum solar-photosphere *area* obscured at
    each WGS-84 observer over global P1-P4, provided at least the upper solar
    limb was above the ideal geometric horizon.  Warm categories are reserved
    for the physically narrower 100%-coverage corridor and encode local C2-C3
    duration relative to the 268.1-second event maximum.
    """
    import plotly.graph_objects as go
    from PIL import Image

    def categorical_colorscale(colors):
        scale = []
        count = len(colors)
        for index, color in enumerate(colors):
            scale.extend([[index/count, color], [(index+1)/count, color]])
        return scale

    # A 1440x720 background preserves map detail while keeping the complete
    # self-contained dashboard substantially smaller than the source mosaic.
    earth = _load_texture("earth")
    with Image.fromarray(earth) as image:
        image = image.resize((1440, 720), getattr(Image, "Resampling", Image).LANCZOS)
        background = np.asarray(image)
    traces = [go.Image(
        z=background, x0=-180.0, y0=90.0,
        dx=360.0/background.shape[1], dy=-180.0/background.shape[0],
        name="High-resolution Earth map", hoverinfo="skip",
    )]

    # Complete partial-eclipse visibility footprint.  Total sites are masked
    # from this blue field so the warm duration categories remain unambiguous.
    partial_grid = solar_partial_visibility_grid_cached()
    partial_lon = np.asarray(partial_grid["longitude"], dtype=float)
    partial_lat = np.asarray(partial_grid["latitude"], dtype=float)
    obscuration = np.asarray(partial_grid["max_obscuration_percent"], dtype=float)
    magnitude = np.asarray(partial_grid["max_magnitude"], dtype=float)
    altitude = np.asarray(partial_grid["sun_altitude_at_max_deg"], dtype=float)
    partial_edges = np.asarray(SOLAR_PARTIAL_OBSCURATION_BAND_EDGES, dtype=float)
    partial_colors = tuple(SOLAR_PARTIAL_OBSCURATION_BAND_COLORS)
    partial_category = np.digitize(
        obscuration, partial_edges[1:-1], right=False,
    ).astype(float)
    partial_mask = (obscuration > 0.0) & (obscuration < 99.999)
    partial_category[~partial_mask] = np.nan
    partial_custom = np.dstack([obscuration, magnitude, altitude])
    traces.append(go.Heatmap(
        x=partial_lon, y=partial_lat, z=partial_category,
        zmin=-0.5, zmax=len(partial_colors)-0.5,
        customdata=partial_custom,
        colorscale=categorical_colorscale(partial_colors),
        opacity=0.79, connectgaps=False, zsmooth=False,
        name="Maximum partial obscuration", showlegend=False, showscale=False,
        hovertemplate=(
            "Maximum photospheric area obscured: %{customdata[0]:.2f}%"
            "<br>Eclipse magnitude: %{customdata[1]:.3f}"
            "<br>Solar-center altitude at maximum: %{customdata[2]:.2f} deg"
            "<br>At least the upper solar limb is above the geometric horizon"
            "<extra></extra>"
        ),
    ))
    partial_labels = ("0-20%", "20-40%", "40-60%", "60-80%", "80-<100%")
    for index, (color, label) in enumerate(zip(partial_colors, partial_labels)):
        traces.append(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol="square", size=10, color=color,
                        line=dict(color="#d8f0ff", width=0.6)),
            name=f"Partial maximum area: {label}",
            legendgroup="partial-obscuration",
            legendgrouptitle_text=("Partial-eclipse maximum obscuration"
                                   if index == 0 else None),
            legendrank=20+index, hoverinfo="skip", showlegend=True,
            xaxis="x", yaxis="y",
        ))

    # Totality-duration field, overlaid only inside the finite-Sun corridor.
    duration_grid = solar_totality_duration_grid_cached()
    raster_x, raster_y, duration_percent = solar_totality_duration_raster_cached(
        nx=540, ny=304,
    )
    reference_s = float(duration_grid["reference_duration_s"])
    duration_seconds = duration_percent*reference_s/100.0
    total_edges = np.asarray(SOLAR_TOTALITY_PERCENT_BAND_EDGES, dtype=float)
    total_colors = tuple(SOLAR_TOTALITY_PERCENT_BAND_COLORS)
    total_labels = ["0-25%", "25-50%", "50-75%", "75-90%", "90-97.5%", "97.5-100%"]
    total_category = np.digitize(
        duration_percent, total_edges[1:-1], right=False,
    ).astype(float)
    total_category[~np.isfinite(duration_percent)] = np.nan
    total_custom = np.dstack([duration_seconds, duration_percent])
    traces.append(go.Heatmap(
        x=raster_x, y=raster_y, z=total_category,
        zmin=-0.5, zmax=len(total_colors)-0.5, customdata=total_custom,
        colorscale=categorical_colorscale(total_colors),
        opacity=0.96, connectgaps=False, zsmooth=False,
        name="Totality-duration level", showlegend=False,
        colorbar=dict(
            title=dict(text="TOTALITY<br>C2-C3", side="top",
                       font=dict(size=9, color="#f3f7fb")),
            x=0.229, xanchor="right", y=0.755, len=0.31, thickness=11,
            tickvals=list(range(len(total_colors))), ticktext=total_labels,
            tickfont=dict(size=7.7, color="#f3f7fb"),
            outlinecolor="#66758b", outlinewidth=1,
            bgcolor="rgba(7,12,20,0.88)",
        ),
        hovertemplate=(
            "Local C2-C3 totality: %{customdata[0]:.1f} s"
            "<br>%{customdata[1]:.1f}% of the 268.1 s event maximum"
            "<br>Photospheric coverage during totality: 100%<extra></extra>"
        ),
    ))

    # Exact duration-threshold polygons and corridor limits remain independent
    # of the display raster.
    for threshold in total_edges[1:-1]:
        for polygon in solar_totality_band_polygons(float(threshold)):
            traces.append(go.Scatter(
                x=polygon[:, 0], y=polygon[:, 1], mode="lines",
                line=dict(color="rgba(235,251,255,0.76)",
                          width=1.0 if threshold < 90.0 else 1.35),
                name=f">= {threshold:g}% of maximum totality duration",
                showlegend=False, hoverinfo="skip", connectgaps=False,
            ))

    corridor = solar_umbra_corridor_cached()
    if len(corridor["jd"]):
        x_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        for longitude, latitude in (
            (corridor["left_lon"], corridor["left_lat"]),
            (corridor["right_lon"], corridor["right_lat"]),
        ):
            for segment_lon, segment_lat in _split_longitudes(longitude, latitude):
                if x_parts:
                    x_parts.append(np.array([np.nan]))
                    y_parts.append(np.array([np.nan]))
                x_parts.append(np.asarray(segment_lon, dtype=float))
                y_parts.append(np.asarray(segment_lat, dtype=float))
        x = np.concatenate(x_parts) if x_parts else np.empty(0)
        y = np.concatenate(y_parts) if y_parts else np.empty(0)
        traces.append(go.Scatter(
            x=x, y=y, mode="lines", line=dict(color="#d8f5ff", width=1.7),
            name="100% coverage corridor edge", hoverinfo="skip",
            connectgaps=False, legendrank=40,
        ))

    longitude, latitude = solar_central_path_cached()
    for index, (segment_lon, segment_lat) in enumerate(_split_longitudes(longitude, latitude)):
        traces.append(go.Scatter(
            x=segment_lon, y=segment_lat, mode="lines",
            line=dict(color="#071019", width=5.2),
            name="Validated central line halo", showlegend=False, hoverinfo="skip",
        ))
        traces.append(go.Scatter(
            x=segment_lon, y=segment_lat, mode="lines",
            line=dict(color="white", width=1.7),
            name="Validated central line", showlegend=(index == 0),
            hoverinfo="skip", legendrank=41,
        ))
    traces.append(go.Scatter(
        x=[SOLAR_GREATEST_SITE_LON_EAST_DEG], y=[SOLAR_GREATEST_SITE_LAT_DEG],
        mode="markers", marker=dict(symbol="diamond", size=9, color="#05070d",
                                     line=dict(color="white", width=1.2)),
        name="NASA greatest-eclipse site", legendrank=42,
        hovertemplate="NASA greatest-eclipse site<extra></extra>",
    ))
    return traces

def _circle_xy(radius, center=(0.0, 0.0), n=181):
    theta = np.linspace(0.0, 2.0 * np.pi, int(n))
    return center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta)


def _lunar_track_geometry(jd_value: float):
    state = lunar_reference_state(float(jd_value))
    impact = state.impact_vector_deg
    impact_hat = impact / np.linalg.norm(impact)
    track_hat = np.array([impact_hat[1], -impact_hat[0]])
    center = impact + state.track_x_deg * track_hat
    return state, impact, track_hat, center


def _static_lunar_shadow_traces():
    import plotly.graph_objects as go

    traces = []
    x, y = _circle_xy(_LUNAR_P_RADIUS_DEG)
    traces.append(go.Scatter(
        x=x, y=y, mode="lines", fill="toself",
        fillcolor="rgba(155,172,199,0.13)", line=dict(color="#b8c5d8", width=2),
        name="Penumbral radius",
    ))
    x, y = _circle_xy(_LUNAR_U_RADIUS_DEG)
    traces.append(go.Scatter(
        x=x, y=y, mode="lines", fill="toself",
        fillcolor="rgba(115,20,28,0.34)", line=dict(color="#ef6659", width=2.2),
        name="Umbral radius",
    ))
    _, impact, track_hat, _ = _lunar_track_geometry(LUNAR_2025.greatest_jd)
    values = np.linspace(-1.6, 1.6, 220)
    track = impact[None, :] + values[:, None] * track_hat[None, :]
    traces.append(go.Scatter(
        x=track[:, 0], y=track[:, 1], mode="lines",
        line=dict(color="#6ed2ff", width=2, dash="dash"), name="Moon-center track",
    ))
    contact_x: list[float] = []
    contact_y: list[float] = []
    contact_text: list[str] = []
    for name in ("P1", "U1", "U2", "MAX", "U3", "U4", "P4"):
        _, _, _, center = _lunar_track_geometry(LUNAR_2025.contacts_jd[name])
        contact_x.append(float(center[0]))
        contact_y.append(float(center[1]))
        contact_text.append(name)
    traces.append(go.Scatter(
        x=contact_x, y=contact_y, mode="markers+text",
        marker=dict(size=6, color="white", line=dict(color="#27364c", width=0.7)),
        text=contact_text, textposition="top center", textfont=dict(size=9, color="white"),
        name="NASA contacts", hovertemplate="%{text}<extra></extra>",
    ))
    return traces


def _solar_map_dynamic(jd_value: float):
    import plotly.graph_objects as go

    result = []
    for family, color, name in (
        ("penumbra", "#ffd35f", "Penumbral limit"),
        ("umbra", "#ff4f61", "Umbral limit"),
    ):
        x_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        for world_segment in solar_footprint_segments(float(jd_value), family=family,
                                                       n_azimuth=144):
            longitude, latitude = _points_lat_lon(world_segment)
            for segment_lon, segment_lat in _split_longitudes(longitude, latitude):
                if x_parts:
                    x_parts.append(np.array([np.nan]))
                    y_parts.append(np.array([np.nan]))
                x_parts.append(np.asarray(segment_lon, dtype=float))
                y_parts.append(np.asarray(segment_lat, dtype=float))
        x = np.concatenate(x_parts) if x_parts else np.empty(0)
        y = np.concatenate(y_parts) if y_parts else np.empty(0)
        result.append(go.Scatter(
            x=x, y=y, mode="lines",
            line=dict(color=color, width=2.7 if family == "umbra" else 2.3),
            name=name, showlegend=True, hoverinfo="skip", xaxis="x", yaxis="y",
            connectgaps=False,
        ))
    center = solar_central_line_wgs84(float(jd_value))
    result.append(go.Scatter(
        x=[center[1]] if center is not None else [],
        y=[center[0]] if center is not None else [],
        mode="markers", marker=dict(size=9, color="white",
                                     line=dict(color="#ff4f61", width=1.5)),
        name="Shadow axis", showlegend=True,
        hovertemplate="Shadow axis<extra></extra>", xaxis="x", yaxis="y",
    ))
    return result


def _lunar_shadow_dynamic(jd_value: float):
    import plotly.graph_objects as go

    _, _, _, center = _lunar_track_geometry(float(jd_value))
    x, y = _circle_xy(_LUNAR_MOON_SD_DEG, center=center)
    return [go.Scatter(
        x=x, y=y, mode="lines", fill="toself",
        fillcolor="rgba(222,216,205,0.88)", line=dict(color="white", width=2),
        name="Moon at current UTC", xaxis="x", yaxis="y", hoverinfo="skip",
    )]


def _light_curve_data(event: ReferenceEvent):
    x = (event.jd - event.greatest_jd) * 24.0
    y = (100.0 * solar_site_visibility_curve(event.jd)
         if event.mode == "solar" else 100.0 * np.asarray(event.center_visibility))
    if event.mode == "solar" and event.metadata.get("animation_scope"):
        contacts = solar_local_contacts()
        ordered = [(name, contacts[name]) for name in ("C1", "C2")]
        ordered += [("MAX", event.greatest_jd)]
        ordered += [(name, contacts[name]) for name in ("C3", "C4")]
    else:
        ordered = sorted(event.contacts_jd.items(), key=lambda item: item[1])
    return x, y, ordered


def _light_dynamic(event: ReferenceEvent, jd_value: float, x_curve, y_curve):
    import plotly.graph_objects as go

    current = (float(jd_value) - event.greatest_jd) * 24.0
    current_y = float(np.interp(current, x_curve, y_curve))
    return [
        go.Scatter(
            x=[current, current], y=[0, 105], mode="lines",
            line=dict(color="white", width=2), name="Current UTC",
            showlegend=False, hoverinfo="skip", xaxis="x3", yaxis="y3",
        ),
        go.Scatter(
            x=[current], y=[current_y], mode="markers",
            marker=dict(size=9, color="white",
                        line=dict(color="#ffd45d" if event.mode == "solar" else "#efb9a5",
                                  width=2)),
            name="Current visibility", showlegend=False,
            hovertemplate="%{y:.3f}% visible<extra></extra>", xaxis="x3", yaxis="y3",
        ),
    ]


def _full_solar_path_world():
    start = SOLAR_2024.contacts_jd["U1"]
    stop = SOLAR_2024.contacts_jd["U4"]
    points = []
    for jd_value in np.linspace(start, stop, 180):
        value = solar_central_line_wgs84(float(jd_value))
        if value is not None:
            points.append(value[2])
    return np.asarray(points, dtype=float)



def _scene_impact_plane(bounds: dict):
    """Optional translucent sheet marking the exact peak impact plane."""
    import plotly.graph_objects as go

    xmin, xmax = [float(value) for value in bounds["xrange"]]
    zmin, zmax = [float(value) for value in bounds["zrange"]]
    return go.Mesh3d(
        x=[xmin, xmax, xmax, xmin],
        y=[0.0, 0.0, 0.0, 0.0],
        z=[zmin, zmin, zmax, zmax],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color="#66e0c2", opacity=0.055, flatshading=True,
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0),
        name="Peak impact plane - source, occluder and target cross-section",
        showlegend=True, visible="legendonly", hoverinfo="skip",
        scene="scene", legend="legend2", legendgroup="reference",
        legendgrouptitle=dict(text="Surface and path"),
    )

def _scene_static_path(event: ReferenceEvent, basis: np.ndarray):
    if event.mode == "solar":
        world = _surface_lift(_full_solar_path_world(), np.zeros(3), 12.0)
        values = _transform(world, np.zeros(3), basis) if len(world) else np.empty((0, 3))
        name = "Validated WGS-84 central path (12 km display lift)"
    else:
        # The lunar local panel is Moon-centred.  Keep this reference path
        # in the same frame by showing Earth's centre relative to the Moon.
        values = np.asarray(-event.moon_km, dtype=float) @ np.asarray(basis, dtype=float)
        name = "Earth-center event track in Moon-centered frame"
    return _line3d(
        values, "#b78cff", 2.8, name, visible="legendonly",
        legendgroup="reference", legendgrouptitle_text="Surface and path",
    )


def _local_ray_trace(path: RayPath, origin: np.ndarray, basis: np.ndarray,
                     xmin: float, xmax: float) -> np.ndarray:
    transformed = _transform(np.vstack([path.source_km, path.endpoint_km]), origin, basis)
    return _clip_segment_to_x(transformed, xmin, xmax)


def _frame_local_traces(event: ReferenceEvent, jd_value: float, basis: np.ndarray,
                        camera_hat_world: np.ndarray, bounds: dict, *,
                        n_azimuth: int, body_n_lat: int = 61,
                        body_n_lon: int = 120, atmosphere_n_lat: int = 37,
                        atmosphere_n_lon: int = 72,
                        shadow_resolution: int = 0):
    bundle = trace_reference_rays(event.definition, float(jd_value),
                                  n_azimuth=max(12, int(n_azimuth)))
    earth = bundle.earth_center_km
    moon = bundle.moon_center_km
    origin = _local_origin(event, bundle)

    traces = _body_traces(
        event, jd_value, bundle, origin, basis, camera_hat_world,
        n_lat=body_n_lat, n_lon=body_n_lon,
        atmosphere_n_lat=atmosphere_n_lat, atmosphere_n_lon=atmosphere_n_lon,
        shadow_resolution=shadow_resolution,
    )

    xmin = float(bounds["xmin"])
    xmax = float(bounds["xmax"])
    central = _local_ray_trace(bundle.central, origin, basis, xmin, xmax)
    traces.append(_line3d(
        central, "#ffe36e", 4.0,
        "Axial sunlight \u2014 local segment, first-surface stop",
        legendgroup="finite-sun-rays", legendgrouptitle_text="Local finite-Sun rays",
    ))

    # Sunlight ends at the opaque occluder.  The downstream centerline is
    # shadow geometry, not transmitted light, so it is a separate solid
    # trace.  A solid line avoids the broken, camera-dependent dash pattern
    # produced by WebGL for long 3-D segments.
    shadow_start, shadow_end = _shadow_axis_endpoint(bundle, float(jd_value))
    shadow_local = _clip_segment_to_x(
        _transform(np.vstack([shadow_start, shadow_end]), origin, basis),
        xmin, xmax,
    )
    traces.append(_line3d(
        shadow_local, "#d66cff", 2.6,
        "Blocked-light shadow axis \u2014 not transmitted sunlight",
        legendgroup="finite-sun-rays",
    ))
    central_stop = _transform(
        np.asarray(bundle.central.endpoint_km, dtype=float).reshape(1, 3),
        origin, basis,
    )
    traces.append(_marker3d(
        central_stop, "#ffe36e", 3.2,
        "Axial sunlight first opaque-surface stop",
        text=[""], showlegend=False, legendgroup="finite-sun-rays",
    ))
    shadow_stop = _transform(
        np.asarray(shadow_end, dtype=float).reshape(1, 3), origin, basis,
    )
    traces.append(_marker3d(
        shadow_stop, "#d66cff", 3.0,
        "Blocked-light axis target endpoint",
        text=[""], showlegend=False, legendgroup="finite-sun-rays",
    ))

    tangent_markers: list[np.ndarray] = []
    for family_name, family, tangent_points, color, width, label in (
        ("umbra", bundle.umbra, bundle.umbra_tangent_points_km,
         "#ff6c5c", 3.2, "Umbral / antumbral boundary rays"),
        ("penumbra", bundle.penumbra, bundle.penumbra_tangent_points_km,
         "#8bc8ff", 2.6, "Penumbral boundary rays"),
    ):
        upper, lower, selected_tangents = tangent_cross_section_paths(
            bundle, family_name, np.asarray(basis, dtype=float)[:, 2],
        )
        for ray_index, path in enumerate((upper, lower)):
            traces.append(_line3d(
                _local_ray_trace(path, origin, basis, xmin, xmax),
                color, width, label, showlegend=(ray_index == 0),
                legendgroup="finite-sun-rays",
            ))
        tangent_markers.extend(_transform(selected_tangents, origin, basis))

    traces.append(_marker3d(
        np.asarray(tangent_markers), "#ffffff", 2.3,
        "Exact optical tangent points", text=["", "", "", ""],
        showlegend=False, legendgroup="finite-sun-rays",
    ))

    traces.append(_shadow_volume_trace(
        bundle, bundle.umbra, bundle.umbra_tangent_points_km,
        origin, basis, color="#8c1f2c", name="Umbral / antumbral volume", opacity=0.17,
    ))
    traces.append(_shadow_volume_trace(
        bundle, bundle.penumbra, bundle.penumbra_tangent_points_km,
        origin, basis, color="#4d86b5", name="Penumbral volume", opacity=0.065,
    ))

    if event.mode == "solar":
        segments = [
            _transform(_surface_lift(segment, earth, 14.0), origin, basis)
            for segment in solar_footprint_segments(float(jd_value), family="umbra",
                                                     n_azimuth=180)
        ]
        footprint = _join_segments(segments)
    else:
        footprint = np.empty((0, 3))
    traces.append(_line3d(
        footprint, "#ffcf5a", 3.4, "Current WGS-84 umbral boundary (14 km display lift)",
        legendgroup="reference",
    ))

    display_north = np.asarray(basis, dtype=float)[:, 2]
    earth_label = earth+display_north*(RE_KM+420.0)
    moon_label = moon+display_north*(R_MOON_MEAN_KM+260.0)
    earth_q = _transform(earth_label.reshape(1, 3), origin, basis)[0]
    moon_q = _transform(moon_label.reshape(1, 3), origin, basis)[0]
    traces.append(_marker3d(earth_q, "#6abaff", 2.8, "Earth label", ["Earth"]))
    traces.append(_marker3d(moon_q, "#eeeeee", 2.6, "Moon label", ["Moon"]))

    # The local +Z axis is the physical impact-plane direction toward the
    # target centre, not geographic/celestial north.  Show north explicitly
    # so rotating the scene cannot be mistaken for a body-orientation error.
    target_world = earth if event.mode == "solar" else moon
    target_q = _transform(target_world.reshape(1, 3), origin, basis)[0]
    north_world = np.array([0.0, 0.0, 1.0])
    north_local = north_world @ np.asarray(basis, dtype=float)
    north_local = north_local / max(float(np.linalg.norm(north_local)), 1.0e-15)
    north_length = 1.35 * (RE_KM if event.mode == "solar" else R_MOON_MEAN_KM)
    north_end = target_q + north_local * north_length
    traces.append(_line3d(
        np.vstack([target_q, north_end]), "#65e6d2", 2.0,
        "Geographic / celestial north reference", dash="dash",
        legendgroup="reference", visible="legendonly",
    ))
    traces.append(_marker3d(
        north_end, "#65e6d2", 2.4, "North label", ["N"],
        showlegend=False, legendgroup="reference",
    ))
    return traces


def _first_sphere_hit_from(start: np.ndarray, direction: np.ndarray,
                           center: np.ndarray, radius: float) -> np.ndarray | None:
    direction = _unit(direction)
    offset = np.asarray(start, dtype=float) - np.asarray(center, dtype=float)
    b = float(np.dot(offset, direction))
    c = float(np.dot(offset, offset) - radius * radius)
    disc = b * b - c
    if disc < 0.0:
        return None
    roots = sorted((-b - math.sqrt(max(disc, 0.0)),
                    -b + math.sqrt(max(disc, 0.0))))
    for root in roots:
        if root > 1.0e-7:
            return np.asarray(start, dtype=float) + root * direction
    return None


def _shadow_axis_endpoint(bundle: ReferenceRayBundle, jd_value: float) -> tuple[np.ndarray, np.ndarray]:
    axis = bundle.axis_hat
    if bundle.mode == "solar":
        occluder = bundle.moon_center_km
        start = occluder + axis * bundle.solid_occluder_radius_km
        center = solar_central_line_wgs84(float(jd_value))
        if center is not None:
            return start, np.asarray(center[2], dtype=float)
        distance = float(np.dot(bundle.earth_center_km - start, axis))
        return start, start + max(distance, 0.0) * axis
    occluder = bundle.earth_center_km
    directional_radius = 1.0 / math.sqrt(
        float(np.sum((axis / EARTH_AXES_KM) ** 2))
    )
    start = occluder + axis * directional_radius
    hit = _first_sphere_hit_from(start, axis, bundle.moon_center_km, R_MOON_MEAN_KM)
    if hit is not None:
        return start, hit
    distance = float(np.dot(bundle.moon_center_km - start, axis))
    return start, start + max(distance, 0.0) * axis


def _frame_distance_traces(event: ReferenceEvent, jd_value: float,
                           basis: np.ndarray, bounds: dict):
    """Return readable true-distance and Earth-Moon cross-section traces.

    The old right-hand WebGL scene combined a one-AU X range with
    Earth-radius Y/Z detail.  Plotly's 3-D aspect handling made the Sun look
    enormous in some cameras and reduced Earth and Moon to sub-pixel objects
    in others.  These two Cartesian strips keep the numerical coordinates
    exact while the centre panel remains the rotatable 3-D view.
    """
    import plotly.graph_objects as go

    bundle = trace_reference_rays(event.definition, float(jd_value), n_azimuth=12)
    earth = bundle.earth_center_km
    sun_q = _transform(bundle.sun_center_km.reshape(1, 3), earth, basis, scale=AU_KM)[0]
    earth_q = np.zeros(3)
    moon_q = _transform(bundle.moon_center_km.reshape(1, 3), earth, basis, scale=AU_KM)[0]
    central = _transform(bundle.central.points_km, earth, basis, scale=AU_KM)
    shadow_start, shadow_end = _shadow_axis_endpoint(bundle, float(jd_value))
    shadow = _transform(np.vstack([shadow_start, shadow_end]), earth, basis, scale=AU_KM)
    top = [
        go.Scatter(
            x=central[:, 0], y=np.zeros(len(central)), mode="lines",
            line=dict(color="#ffe36e", width=3.0),
            name="True-distance direct sunlight", showlegend=False,
            xaxis="x2", yaxis="y2",
            hovertemplate="Optical X: %{x:.6f} AU<extra></extra>",
        ),
        go.Scatter(
            x=shadow[:, 0], y=np.zeros(len(shadow)), mode="lines",
            line=dict(color="#d66cff", width=3.0),
            name="True-distance blocked-light axis", showlegend=False,
            xaxis="x2", yaxis="y2",
            hovertemplate="Optical X: %{x:.6f} AU<extra></extra>",
        ),
        go.Scatter(
            x=[sun_q[0], moon_q[0], earth_q[0]], y=[0.0, 0.0, 0.0], mode="markers+text",
            marker=dict(size=[15, 7, 10], color=["#ffb347", "#d7d7d7", "#65a8ff"],
                        line=dict(color="white", width=1.1)),
            text=["Sun", "", "Earth + Moon"],
            textposition=["middle right", "top center", "middle left"],
            textfont=dict(size=10, color="white"), showlegend=False,
            customdata=["Actual Sun centre", "Moon centre", "Earth centre"],
            hovertemplate="%{customdata}<br>X=%{x:.9f} AU<extra></extra>",
            name="True-distance locators",
            xaxis="x2", yaxis="y2",
        ),
    ]

    # Panel E uses the exact impact plane rather than a north-only projection.
    # Draw the complete Earth-Moon cross-section once, then use axis-range
    # controls to inspect the target, the occluder, or the full baseline.
    # The trace coordinates remain unwarped kilometres in every view.
    impact_state = build_impact_plane_frame(
        event, float(jd_value), n_azimuth=48,
    )
    bottom = plotly_view_traces(
        impact_state, "system", xaxis="x4", yaxis="y4", showlegend=False,
    )
    return top, bottom


def _event_status(event: ReferenceEvent, jd_value: float) -> str:
    if event.mode == "solar" and event.metadata.get("animation_scope"):
        return solar_local_phase_status(float(jd_value))
    return contact_status(event.definition, float(jd_value))


def _single_peak_event(event: ReferenceEvent) -> ReferenceEvent:
    index = event.peak_index
    selection = np.array([index], dtype=int)
    return replace(
        event,
        jd=event.jd[selection], moon_km=event.moon_km[selection],
        sun_km=event.sun_km[selection],
        center_visibility=event.center_visibility[selection],
        separation_deg=event.separation_deg[selection],
    )


def generate_interactive_dashboard(event: ReferenceEvent, output_path: str | Path,
                                   *, animated: bool = True,
                                   n_azimuth: int = 16,
                                   include_plotlyjs: bool = True,
                                   quality: str = "balanced") -> str:
    """Write a validated dashboard with one rotatable 3-D optical scene.

    ``quality='high'`` increases the body meshes and adds a resolved local
    finite-disc shadow patch while keeping full-event HTML sizes practical.
    ``quality='ultra'`` is intended mainly for peak inspection views.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    event_to_use = event if animated else _single_peak_event(event)
    basis = _optical_basis(event)
    bounds = _local_bounds(event, basis)
    impact_ranges = event_view_ranges(event_to_use, event_to_use.jd)
    profile = _quality_profile(quality, animated=animated)
    default_eye_local = np.array([-1.30, -1.62, 0.68])
    camera_hat_world = _unit(basis @ default_eye_local)
    body_n_lat, body_n_lon = profile["body"]
    atmosphere_n_lat, atmosphere_n_lon = profile["atmosphere"]
    shadow_resolution = int(profile["shadow"])

    fig = make_subplots(
        rows=2, cols=3,
        specs=[[{"type": "xy"}, {"type": "scene", "rowspan": 2},
                {"type": "xy"}],
               [{"type": "xy"}, None, {"type": "xy"}]],
        column_widths=[0.255, 0.535, 0.210], row_heights=[0.57, 0.43],
        horizontal_spacing=0.036, vertical_spacing=0.11,
        subplot_titles=(
            ("A · Full WGS-84 eclipse visibility — partial obscuration bands, totality duration, and current shadow"
             if event.mode == "solar" else
             "A · Danjon shadow plane — Moon track and exact eclipse contacts"),
            "B · Interactive finite-Sun impact geometry — source, occluder and target share one principal plane",
            "C · True source distance - one-AU Cartesian scale — verifies the Sun remains one AU away",
            "D · Contact-labelled geometric visibility — fraction of the solar disk visible",
            "E · Exact impact-plane cross-section — target, occluder, or full Earth-Moon baseline",
        ),
    )

    for trace in (_static_solar_map_traces() if event.mode == "solar"
                  else _static_lunar_shadow_traces()):
        fig.add_trace(trace, row=1, col=1)

    x_curve, y_curve, ordered_contacts = _light_curve_data(event)
    curve_color = "#ffd45d" if event.mode == "solar" else "#efb9a5"
    fig.add_trace(go.Scatter(
        x=x_curve, y=y_curve, mode="lines", line=dict(color=curve_color, width=3),
        fill="tozeroy",
        fillcolor="rgba(255,212,93,0.10)" if event.mode == "solar"
        else "rgba(239,185,165,0.10)",
        name="Unobscured solar disk",
        hovertemplate="%{x:.3f} h<br>%{y:.3f}% visible<extra></extra>",
    ), row=2, col=1)
    contact_x = [(value - event.greatest_jd) * 24.0 for _, value in ordered_contacts]
    fig.add_trace(go.Scatter(
        x=contact_x, y=[102.0] * len(contact_x), mode="markers+text",
        marker=dict(size=5, color="#c9d6e5"),
        text=[name for name, _ in ordered_contacts], textposition="top center",
        textfont=dict(size=9, color="#d9e3ef"), name="Contacts",
        hovertemplate="%{text}<extra></extra>", showlegend=False,
    ), row=2, col=1)

    fig.add_trace(_scene_static_path(event, basis), row=1, col=2)
    fig.add_trace(_scene_impact_plane(bounds), row=1, col=2)

    first_jd = float(event_to_use.jd[0])
    dynamic: list = []
    dynamic.extend(_solar_map_dynamic(first_jd) if event.mode == "solar"
                   else _lunar_shadow_dynamic(first_jd))
    dynamic.extend(_light_dynamic(event, first_jd, x_curve, y_curve))
    local_first = _frame_local_traces(
        event, first_jd, basis, camera_hat_world, bounds,
        n_azimuth=n_azimuth, body_n_lat=body_n_lat, body_n_lon=body_n_lon,
        atmosphere_n_lat=atmosphere_n_lat, atmosphere_n_lon=atmosphere_n_lon,
        shadow_resolution=shadow_resolution,
    )
    distance_top_first, distance_bottom_first = _frame_distance_traces(
        event, first_jd, basis, bounds,
    )
    dynamic.extend(local_first)
    dynamic.extend(distance_top_first)
    dynamic.extend(distance_bottom_first)

    top_count = 3 if event.mode == "solar" else 1
    light_count = 2
    local_count = len(local_first)
    distance_top_count = len(distance_top_first)
    dynamic_indices: list[int] = []
    for index, trace in enumerate(dynamic):
        if index < top_count:
            fig.add_trace(trace, row=1, col=1)
        elif index < top_count + light_count:
            fig.add_trace(trace, row=2, col=1)
        elif index < top_count + light_count + local_count:
            fig.add_trace(trace, row=1, col=2)
        elif index < top_count + light_count + local_count + distance_top_count:
            fig.add_trace(trace, row=1, col=3)
        else:
            fig.add_trace(trace, row=2, col=3)
        dynamic_indices.append(len(fig.data) - 1)

    frames = []
    slider_steps = []
    if animated and len(event_to_use.jd) > 1:
        slider_contacts = [(str(name), float(value)) for name, value in ordered_contacts]
        if not any(name == "MAX" for name, _ in slider_contacts):
            slider_contacts.append(("MAX", float(event.greatest_jd)))
        slider_stride = max(1, int(math.ceil(len(event_to_use.jd) / 8.0)))
        for frame_index, jd_value in enumerate(event_to_use.jd):
            jd_value = float(jd_value)
            frame_data: list = []
            frame_data.extend(_solar_map_dynamic(jd_value) if event.mode == "solar"
                              else _lunar_shadow_dynamic(jd_value))
            frame_data.extend(_light_dynamic(event, jd_value, x_curve, y_curve))
            frame_data.extend(_frame_local_traces(
                event, jd_value, basis, camera_hat_world, bounds,
                n_azimuth=n_azimuth, body_n_lat=body_n_lat, body_n_lon=body_n_lon,
                atmosphere_n_lat=atmosphere_n_lat,
                atmosphere_n_lon=atmosphere_n_lon,
                shadow_resolution=shadow_resolution,
            ))
            distance_top, distance_bottom = _frame_distance_traces(
                event, jd_value, basis, bounds,
            )
            frame_data.extend(distance_top)
            frame_data.extend(distance_bottom)
            dt = jd_to_datetime(jd_value)
            status = _event_status(event, jd_value)
            frame_name = str(frame_index)
            frames.append(go.Frame(
                data=frame_data, traces=dynamic_indices, name=frame_name,
                layout=go.Layout(title=dict(
                    text=(f"{event.definition.title} - validated interactive eclipse dashboard"
                          f"<br><sub>{dt.strftime('%Y-%m-%d %H:%M:%S')} UTC - {status}; "
                          "local rays are precision-safe impact-plane clips and remain physically collinear to the first target intersection</sub>"),
                    x=0.5,
                )),
            ))
            contact_label = next(
                (name for name, contact_jd in slider_contacts
                 if abs(jd_value-contact_jd)*86400.0 < 0.75),
                None,
            )
            if contact_label is not None:
                slider_label = f"{contact_label} {dt.strftime('%H:%M')}"
            elif frame_index % slider_stride == 0 or frame_index == len(event_to_use.jd)-1:
                slider_label = dt.strftime("%H:%M")
            else:
                slider_label = ""
            slider_steps.append(dict(
                method="animate", label=slider_label,
                args=[[frame_name], dict(mode="immediate",
                                         frame=dict(duration=0, redraw=True),
                                         transition=dict(duration=0))],
            ))
        fig.frames = frames

    if event.mode == "solar":
        fig.update_xaxes(range=[-180, 30], title_text="Longitude [deg east]",
                         gridcolor="rgba(180,195,215,0.18)", row=1, col=1)
        fig.update_yaxes(range=[-10, 85], title_text="Latitude [deg]",
                         gridcolor="rgba(180,195,215,0.18)", row=1, col=1)
    else:
        fig.update_xaxes(range=[-1.58, 1.58], title_text="East or west offset [deg]",
                         gridcolor="rgba(180,195,215,0.18)", row=1, col=1)
        fig.update_yaxes(range=[-1.58, 1.58], title_text="North or south offset [deg]",
                         gridcolor="rgba(180,195,215,0.18)", scaleanchor="x", scaleratio=1,
                         row=1, col=1)
    fig.update_xaxes(title_text="Hours from greatest eclipse",
                     gridcolor="rgba(180,195,215,0.18)", row=2, col=1)
    fig.update_yaxes(range=[0, 106], title_text="Solar disk visible [%]",
                     gridcolor="rgba(180,195,215,0.18)", row=2, col=1)

    peak_bundle = trace_reference_rays(event.definition, event.greatest_jd, n_azimuth=16)
    earth = peak_bundle.earth_center_km
    peak_origin = _local_origin(event, peak_bundle)
    earth_peak = _transform(
        peak_bundle.earth_center_km.reshape(1, 3), peak_origin, basis,
    )[0]
    moon_peak = _transform(
        peak_bundle.moon_center_km.reshape(1, 3), peak_origin, basis,
    )[0]
    local_x = bounds["xrange"]
    local_y = bounds["yrange"]
    local_z = bounds["zrange"]

    camera_up = dict(x=0.0, y=0.0, z=1.0)
    beauty_camera = dict(
        eye=dict(x=-1.30, y=-1.62, z=0.68), up=camera_up,
        center=dict(x=0.0, y=0.0, z=0.0), projection=dict(type="perspective"),
    )
    optics_camera = dict(
        eye=dict(x=0.02, y=-2.25, z=0.22), up=camera_up,
        center=dict(x=0.0, y=0.0, z=0.0), projection=dict(type="orthographic"),
    )
    perspective_camera = dict(
        eye=dict(x=-1.05, y=-1.72, z=0.78), up=camera_up,
        center=dict(x=0.0, y=0.0, z=0.0), projection=dict(type="perspective"),
    )
    earth_half = 15_500.0
    earth_range = {
        "scene.xaxis.range": [earth_peak[0] - earth_half, earth_peak[0] + earth_half],
        "scene.yaxis.range": [earth_peak[1] - earth_half, earth_peak[1] + earth_half],
        "scene.zaxis.range": [earth_peak[2] - earth_half, earth_peak[2] + earth_half],
    }
    # P1-P4 moves the lunar disk across the full Danjon penumbra.  A narrow
    # peak-only cube clipped the outer boundary rays near first and last
    # contact, making apparently broken lines.  Eighteen thousand kilometres
    # contains the complete contact-to-contact envelope without scaling or
    # warping the geometry.
    moon_half = 18_000.0
    moon_range = {
        "scene.xaxis.range": [moon_peak[0] - moon_half, moon_peak[0] + moon_half],
        "scene.yaxis.range": [moon_peak[1] - moon_half, moon_peak[1] + moon_half],
        "scene.zaxis.range": [moon_peak[2] - moon_half, moon_peak[2] + moon_half],
    }
    system_range = {
        "scene.xaxis.range": local_x,
        "scene.yaxis.range": local_y,
        "scene.zaxis.range": local_z,
    }
    target_range = earth_range if event.mode == "solar" else moon_range
    occluder_range = moon_range if event.mode == "solar" else earth_range

    def with_camera(ranges, camera, *, aspectmode):
        result = dict(ranges)
        result["scene.camera"] = camera
        result["scene.aspectmode"] = aspectmode
        return result

    fig.update_scenes(
        xaxis=dict(range=target_range["scene.xaxis.range"],
                   title="Optical X [km] (downstream)",
                   gridcolor="#2c394d", zerolinecolor="#60718a",
                   backgroundcolor="#010207", visible=False, showspikes=False),
        yaxis=dict(range=target_range["scene.yaxis.range"], title="Out-of-impact-plane Y [km]",
                   gridcolor="#2c394d", zerolinecolor="#60718a",
                   backgroundcolor="#010207", visible=False, showspikes=False),
        zaxis=dict(range=target_range["scene.zaxis.range"], title="Impact-plane Z [km] (toward target centre)",
                   gridcolor="#2c394d", zerolinecolor="#60718a",
                   backgroundcolor="#010207", visible=False, showspikes=False),
        aspectmode="cube", bgcolor="#010207", camera=beauty_camera,
        dragmode="orbit", uirevision="validated-eclipse-camera-v4-impact-plane",
        row=1, col=2,
    )

    sun_peak = _transform(peak_bundle.sun_center_km.reshape(1, 3), earth, basis,
                          scale=AU_KM)[0]
    moon_peak_earth = _transform(
        peak_bundle.moon_center_km.reshape(1, 3), earth, basis,
    )[0]
    moon_peak_au = moon_peak_earth / AU_KM
    overview_xmin = float(sun_peak[0] - 3.0 * R_SUN_KM / AU_KM)
    overview_xmax = float(max(0.025, moon_peak_au[0] + 0.018))
    fig.update_xaxes(
        range=[overview_xmin, overview_xmax], title_text="Optical X [AU]",
        gridcolor="rgba(180,195,215,0.18)", zerolinecolor="#60718a",
        row=1, col=3,
    )
    fig.update_yaxes(
        range=[-1.0, 1.0], visible=False, fixedrange=True, row=1, col=3,
    )
    # Add the actual solar diameter directly against the upper-right
    # true-distance strip.  ``add_vrect(row=..., col=...)`` inspects all
    # subplot traces and can incorrectly query ``xaxis`` on the neighboring
    # Scatter3d traces in mixed 2-D/3-D figures.  Explicit axis references
    # avoid that Plotly bug.  In this subplot layout the upper-right strip is
    # x2/y2 (the lower-left visibility curve is x3/y3).
    solar_left = float(sun_peak[0] - R_SUN_KM / AU_KM)
    solar_right = float(sun_peak[0] + R_SUN_KM / AU_KM)
    fig.add_shape(
        type="rect", x0=solar_left, x1=solar_right, y0=0.0, y1=1.0,
        xref="x2", yref="y2 domain",
        fillcolor="#ffb347", opacity=0.22,
        line=dict(color="#ffd785", width=1), layer="below",
    )
    fig.add_annotation(
        x=0.5 * (solar_left + solar_right), y=0.86,
        xref="x2", yref="y2 domain", showarrow=False,
        text="actual solar diameter", font=dict(color="#ffd785", size=9),
    )
    # Panel E defaults to a readable target-interception magnification.
    # The plotted traces retain the complete unwarped Earth-Moon baseline,
    # so the dropdown below can switch to the occluder or full-system view
    # without rebuilding or rescaling the physical geometry.
    cross_target_x, cross_target_z = impact_ranges["target"]
    cross_occ_x, cross_occ_z = impact_ranges["occluder"]
    cross_system_x, cross_system_z = impact_ranges["system"]
    fig.update_xaxes(
        range=list(cross_target_x), title_text="Optical X [km] - downstream",
        gridcolor="rgba(180,195,215,0.18)", zerolinecolor="#60718a",
        row=2, col=3,
    )
    fig.update_yaxes(
        range=list(cross_target_z),
        title_text="Impact-plane Z [km] - toward target centre",
        gridcolor="rgba(180,195,215,0.18)", zerolinecolor="#60718a",
        scaleanchor="x4", scaleratio=1.0, constrain="domain",
        row=2, col=3,
    )


    view_buttons = [
        dict(label="Eclipse target close-up (default)", method="relayout",
             args=[with_camera(target_range, beauty_camera, aspectmode="cube")]),
        dict(label="Optics side - orthographic", method="relayout",
             args=[with_camera(target_range, optics_camera, aspectmode="cube")]),
        dict(label="Full optical system - equal physical scale", method="relayout",
             args=[with_camera(system_range, optics_camera, aspectmode="data")]),
        dict(label="Perspective Earth-Moon system", method="relayout",
             args=[with_camera(system_range, perspective_camera, aspectmode="data")]),
        dict(label="Occluder close-up", method="relayout",
             args=[with_camera(occluder_range, optics_camera, aspectmode="cube")]),
        dict(label="Earth close-up", method="relayout",
             args=[with_camera(earth_range, perspective_camera, aspectmode="cube")]),
        dict(label="Moon close-up", method="relayout",
             args=[with_camera(moon_range, perspective_camera, aspectmode="cube")]),
    ]
    style_buttons = [
        dict(label="Scientific axes", method="relayout", args=[{
            "scene.xaxis.visible": True, "scene.yaxis.visible": True,
            "scene.zaxis.visible": True,
        }]),
        dict(label="Clean 3-D views", method="relayout", args=[{
            "scene.xaxis.visible": False, "scene.yaxis.visible": False,
            "scene.zaxis.visible": False,
        }]),
    ]
    cross_section_buttons = [
        dict(label="Panel E - target interception (default)", method="relayout", args=[{
            "xaxis4.range": list(cross_target_x),
            "yaxis4.range": list(cross_target_z),
        }]),
        dict(label="Panel E - occluder tangency", method="relayout", args=[{
            "xaxis4.range": list(cross_occ_x),
            "yaxis4.range": list(cross_occ_z),
        }]),
        dict(label="Panel E - full Earth-Moon baseline", method="relayout", args=[{
            "xaxis4.range": list(cross_system_x),
            "yaxis4.range": list(cross_system_z),
        }]),
    ]
    menus = [
        dict(type="dropdown", direction="down", x=1.008, xanchor="left",
             y=1.075, yanchor="top", buttons=view_buttons,
             bgcolor="#1d2838", bordercolor="#718096", font=dict(color="white")),
        dict(type="dropdown", direction="down", x=1.008, xanchor="left",
             y=1.025, yanchor="top", buttons=style_buttons,
             bgcolor="#1d2838", bordercolor="#718096", font=dict(color="white")),
        dict(type="dropdown", direction="down", x=1.008, xanchor="left",
             y=0.975, yanchor="top", buttons=cross_section_buttons,
             bgcolor="#1d2838", bordercolor="#718096", font=dict(color="white")),
    ]
    if animated and len(event_to_use.jd) > 1:
        menus.append(dict(
            type="buttons", direction="left", x=0.015, y=-0.105,
            showactive=False,
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, dict(fromcurrent=True,
                                      frame=dict(duration=165, redraw=True),
                                      transition=dict(duration=0))]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(mode="immediate",
                                       frame=dict(duration=0, redraw=False),
                                       transition=dict(duration=0))]),
            ],
            bgcolor="#1d2838", bordercolor="#718096", font=dict(color="white"),
        ))

    sliders = []
    if slider_steps:
        sliders = [dict(
            active=0, x=0.18, y=-0.095, len=0.75,
            currentvalue=dict(prefix="UTC ", font=dict(color="white", size=13)),
            steps=slider_steps, tickcolor="#7e90a8",
            font=dict(color="#cbd6e5", size=10),
        )]

    initial_dt = jd_to_datetime(first_jd)
    initial_status = _event_status(event, first_jd)
    distance_million = np.linalg.norm(peak_bundle.sun_center_km - earth) / 1.0e6
    moon_distance = np.linalg.norm(peak_bundle.moon_center_km - earth)
    figure_height = 1120 if str(quality).lower() in {"high", "ultra"} else 1040
    fig.update_layout(
        title=dict(
            text=(f"{event.definition.title} - validated high-fidelity interactive eclipse dashboard"
                  f"<br><sub>{initial_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC - {initial_status}; "
                  "yellow = direct sunlight; magenta centerline = blocked-light shadow axis; "
                  "blue/red pairs = exact tangent limits</sub>"),
            x=0.5, y=0.985, yanchor="top", font=dict(size=20),
        ),
        paper_bgcolor="#010207", plot_bgcolor="#02040a",
        font=dict(color="white", family="Arial, sans-serif", size=12),
        autosize=True, height=figure_height,
        margin=dict(l=42, r=385, t=155, b=190),
        legend=dict(
            title=dict(text="Map and visibility key", font=dict(size=11)),
            orientation="v", x=1.008, y=0.405, xanchor="left", yanchor="top",
            bgcolor="rgba(7,12,20,0.96)", bordercolor="#3d4b60", borderwidth=1,
            font=dict(size=9.2), itemwidth=30,
            groupclick="toggleitem",
        ),
        legend2=dict(
            title=dict(text="Impact-plane local 3-D key", font=dict(size=12)),
            orientation="v", x=1.008, y=0.935, xanchor="left", yanchor="top",
            bgcolor="rgba(7,12,20,0.96)", bordercolor="#3d4b60", borderwidth=1,
            font=dict(size=10.0), itemwidth=31, tracegroupgap=4,
            groupclick="toggleitem",
        ),
        updatemenus=menus, sliders=sliders,
        annotations=list(fig.layout.annotations) + [
            dict(
                x=1.008, y=0.225, xref="paper", yref="paper",
                xanchor="left", yanchor="top", showarrow=False,
                text=("<b>How to read the panels</b><br>"
                      "A: full solar visibility footprint plus current shadow, or the Moon's Danjon-plane track.<br>"
                      "B: rotatable target-centred 3-D impact view; precision-safe clips stop at the first opaque surface and the peak source, occluder and target share one principal plane.<br>"
                      "C: one-AU distance check; the Sun is never moved into the close-up.<br>"
                      "D: contact-to-contact geometric visibility curve.<br>"
                      "E: exact impact-plane cross-section; use its dropdown for target, occluder, or full baseline.<br><br>"
                      "<b>Geometry notes</b><br>"
                      "Target-centred kilometres; no spatial warp.<br>"
                      f"3-D quality: {quality}; body mesh {body_n_lat}x{body_n_lon}; "
                      f"shadow patch {shadow_resolution}x{shadow_resolution}.<br>"
                      "Yellow = direct sunlight and can be off-frame in a target close-up; magenta = blocked-light axis.<br>"
                      "Exact distance strips: upper = one AU; lower = finite-Sun impact plane.<br>"
                      f"Sun-Earth: {distance_million:.3f} million km; Earth-Moon: {moon_distance:,.0f} km."),
                font=dict(color="#a9bbcf", size=9.3), align="left",
                bgcolor="rgba(7,12,20,0.96)", bordercolor="#3d4b60",
                borderwidth=1, borderpad=6, width=318,
            ),
        ],
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs=include_plotlyjs, full_html=True,
                   config={"displaylogo": False, "responsive": True,
                           "scrollZoom": True, "plotGlPixelRatio": 2.0,
                           "toImageButtonOptions": {
                               "format": "png", "scale": 2,
                               "filename": path.stem,
                           }})
    return str(path)


def generate_high_fidelity_3d_view(
    event: ReferenceEvent,
    output_path: str | Path,
    *,
    n_azimuth: int = 72,
    quality: str = "ultra",
    include_plotlyjs: bool = True,
) -> str:
    """Write a full-screen peak 3-D inspection view.

    This companion to :func:`generate_interactive_dashboard` devotes the full
    browser canvas to the precision-safe Earth-Moon scene.  It uses the same
    validated rays and first-surface clipping, but raises the texture mesh and
    local solar-shadow resolution for close inspection.
    """
    import plotly.graph_objects as go

    basis = _optical_basis(event)
    bounds = _local_bounds(event, basis)
    profile = _quality_profile(quality, animated=False)
    body_n_lat, body_n_lon = profile["body"]
    atmosphere_n_lat, atmosphere_n_lon = profile["atmosphere"]
    shadow_resolution = int(profile["shadow"])
    jd_value = float(event.greatest_jd)

    default_eye_local = np.array([-1.30, -1.62, 0.68])
    camera_hat_world = _unit(basis @ default_eye_local)
    local_traces = _frame_local_traces(
        event, jd_value, basis, camera_hat_world, bounds,
        n_azimuth=max(24, int(n_azimuth)),
        body_n_lat=body_n_lat, body_n_lon=body_n_lon,
        atmosphere_n_lat=atmosphere_n_lat,
        atmosphere_n_lon=atmosphere_n_lon,
        shadow_resolution=shadow_resolution,
    )
    fig = go.Figure(data=[
        _scene_static_path(event, basis),
        _scene_impact_plane(bounds),
    ] + local_traces)

    peak_bundle = trace_reference_rays(
        event.definition, jd_value, n_azimuth=max(24, int(n_azimuth)),
    )
    peak_origin = _local_origin(event, peak_bundle)
    earth_peak = _transform(
        peak_bundle.earth_center_km.reshape(1, 3), peak_origin, basis,
    )[0]
    moon_peak = _transform(
        peak_bundle.moon_center_km.reshape(1, 3), peak_origin, basis,
    )[0]

    earth_half = 15_500.0
    moon_half = 18_000.0
    earth_range = {
        "scene.xaxis.range": [earth_peak[0]-earth_half, earth_peak[0]+earth_half],
        "scene.yaxis.range": [earth_peak[1]-earth_half, earth_peak[1]+earth_half],
        "scene.zaxis.range": [earth_peak[2]-earth_half, earth_peak[2]+earth_half],
    }
    moon_range = {
        "scene.xaxis.range": [moon_peak[0]-moon_half, moon_peak[0]+moon_half],
        "scene.yaxis.range": [moon_peak[1]-moon_half, moon_peak[1]+moon_half],
        "scene.zaxis.range": [moon_peak[2]-moon_half, moon_peak[2]+moon_half],
    }
    system_range = {
        "scene.xaxis.range": bounds["xrange"],
        "scene.yaxis.range": bounds["yrange"],
        "scene.zaxis.range": bounds["zrange"],
    }
    target_range = earth_range if event.mode == "solar" else moon_range
    occluder_range = moon_range if event.mode == "solar" else earth_range

    up = dict(x=0.0, y=0.0, z=1.0)
    beauty_camera = dict(
        eye=dict(x=-1.30, y=-1.62, z=0.68), up=up,
        center=dict(x=0.0, y=0.0, z=0.0), projection=dict(type="perspective"),
    )
    optics_camera = dict(
        eye=dict(x=0.02, y=-2.25, z=0.22), up=up,
        center=dict(x=0.0, y=0.0, z=0.0), projection=dict(type="orthographic"),
    )
    system_camera = dict(
        eye=dict(x=-1.05, y=-1.72, z=0.78), up=up,
        center=dict(x=0.0, y=0.0, z=0.0), projection=dict(type="perspective"),
    )

    def view_args(ranges, camera, aspectmode="cube"):
        result = dict(ranges)
        result["scene.camera"] = camera
        result["scene.aspectmode"] = aspectmode
        return result

    view_buttons = [
        dict(label="High-fidelity target view", method="relayout",
             args=[view_args(target_range, beauty_camera)]),
        dict(label="Optics side - orthographic", method="relayout",
             args=[view_args(target_range, optics_camera)]),
        dict(label="Full Earth-Moon system", method="relayout",
             args=[view_args(system_range, system_camera, aspectmode="data")]),
        dict(label="Occluder close-up", method="relayout",
             args=[view_args(occluder_range, beauty_camera)]),
        dict(label="Earth close-up", method="relayout",
             args=[view_args(earth_range, beauty_camera)]),
        dict(label="Moon close-up", method="relayout",
             args=[view_args(moon_range, beauty_camera)]),
    ]
    axes_buttons = [
        dict(label="Clean view", method="relayout", args=[{
            "scene.xaxis.visible": False,
            "scene.yaxis.visible": False,
            "scene.zaxis.visible": False,
        }]),
        dict(label="Scientific axes", method="relayout", args=[{
            "scene.xaxis.visible": True,
            "scene.yaxis.visible": True,
            "scene.zaxis.visible": True,
        }]),
    ]

    dt = jd_to_datetime(jd_value)
    sun_distance = np.linalg.norm(
        peak_bundle.sun_center_km-peak_bundle.earth_center_km,
    )
    moon_distance = np.linalg.norm(
        peak_bundle.moon_center_km-peak_bundle.earth_center_km,
    )
    fig.update_layout(
        title=dict(
            text=(f"{event.definition.title} - high-fidelity validated 3-D view"
                  f"<br><sub>{dt.strftime('%Y-%m-%d %H:%M:%S')} UTC at greatest eclipse; "
                  "unwarped target-centred kilometres; direct sunlight stops at the first opaque surface</sub>"),
            x=0.5, y=0.985, yanchor="top", font=dict(size=22),
        ),
        scene=dict(
            xaxis=dict(
                range=target_range["scene.xaxis.range"],
                title="Optical X [km] (downstream)",
                gridcolor="#2c394d", zerolinecolor="#60718a",
                backgroundcolor="#010207", visible=False, showspikes=False,
            ),
            yaxis=dict(
                range=target_range["scene.yaxis.range"],
                title="Out-of-impact-plane Y [km]", gridcolor="#2c394d",
                zerolinecolor="#60718a", backgroundcolor="#010207",
                visible=False, showspikes=False,
            ),
            zaxis=dict(
                range=target_range["scene.zaxis.range"],
                title="Impact-plane Z [km] (toward target centre)", gridcolor="#2c394d",
                zerolinecolor="#60718a", backgroundcolor="#010207",
                visible=False, showspikes=False,
            ),
            aspectmode="cube", bgcolor="#010207", camera=beauty_camera,
            dragmode="orbit", uirevision="validated-eclipse-high-fidelity-v6",
        ),
        paper_bgcolor="#010207",
        font=dict(color="white", family="Arial, sans-serif", size=12),
        height=1120,
        margin=dict(l=24, r=345, t=120, b=55),
        legend2=dict(
            title=dict(text="High-fidelity 3-D scene key", font=dict(size=13)),
            orientation="v", x=1.005, y=0.94, xanchor="left", yanchor="top",
            bgcolor="rgba(7,12,20,0.97)", bordercolor="#3d4b60",
            borderwidth=1, font=dict(size=10.5), itemwidth=32,
            tracegroupgap=5, groupclick="toggleitem",
        ),
        updatemenus=[
            dict(
                type="dropdown", direction="down", x=1.005, xanchor="left",
                y=1.03, yanchor="top", buttons=view_buttons,
                bgcolor="#1d2838", bordercolor="#718096", font=dict(color="white"),
            ),
            dict(
                type="dropdown", direction="down", x=1.005, xanchor="left",
                y=0.985, yanchor="top", buttons=axes_buttons,
                bgcolor="#1d2838", bordercolor="#718096", font=dict(color="white"),
            ),
        ],
        annotations=[
            dict(
                x=1.005, y=0.16, xref="paper", yref="paper",
                xanchor="left", yanchor="top", showarrow=False,
                text=(f"<b>Physical scale</b><br>"
                      f"Sun-Earth: {sun_distance/1.0e6:.3f} million km<br>"
                      f"Earth-Moon: {moon_distance:,.0f} km<br>"
                      f"Body mesh: {body_n_lat}x{body_n_lon}<br>"
                      f"Solar shadow patch: {shadow_resolution}x{shadow_resolution}<br>"
                      "No distance compression or body enlargement.<br>"
                      "Optical volumes are hidden initially and can be enabled from the key."),
                font=dict(color="#a9bbcf", size=10), align="left",
                bgcolor="rgba(7,12,20,0.97)", bordercolor="#3d4b60",
                borderwidth=1, borderpad=7, width=285,
            ),
        ],
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        path, include_plotlyjs=include_plotlyjs, full_html=True,
        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
            "plotGlPixelRatio": 2.0,
            "toImageButtonOptions": {
                "format": "png", "scale": 2, "filename": path.stem,
            },
        },
    )
    return str(path)
