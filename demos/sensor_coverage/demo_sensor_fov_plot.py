#!/usr/bin/env python3
"""
Demo: sensor FOVs for a GEO-radius spherical constellation.

This example builds a simple full-sphere coverage geometry from three mutually
orthogonal circular orbit planes at geostationary radius.  Each plane has six
nadir-pointing spacecraft.  The Plotly legend is intentionally compact: one
entry per orbit plane for the color, with a single symbol guide separated from
the title and a ground coverage map below the 3D viewer.
"""

GALLERY_CATEGORY = "sensor_coverage"

import os
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from ssatk.constants import EARTH_MU, RGEO
from ssatk.plots.figpath import figpath
from ssatk.plots.layers import _footprint_on_earth
from ssatk.plots.sensor_fov_plot import (
    DEFAULT_CFG,
    add_sensor_fov_to_figure,
    build_figure,
)

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None

GEO_RADIUS_KM = RGEO / 1000.0
GEO_SPEED_KMS = np.sqrt(EARTH_MU / RGEO) / 1000.0

ORBIT_PLANES = [
    {
        "name": "Equatorial GEO-radius orbit",
        "short": "EQ",
        "inclination_deg": 0.0,
        "raan_deg": 0.0,
        "phase_deg": 0.0,
        "color": "#FF6B6B",
    },
    {
        "name": "Polar XZ GEO-radius orbit",
        "short": "PXZ",
        "inclination_deg": 90.0,
        "raan_deg": 0.0,
        "phase_deg": 30.0,
        "color": "#4DABF7",
    },
    {
        "name": "Polar YZ GEO-radius orbit",
        "short": "PYZ",
        "inclination_deg": 90.0,
        "raan_deg": 90.0,
        "phase_deg": 15.0,
        "color": "#69DB7C",
    },
]

_SYMBOL_GUIDE = (
    "● spacecraft · cone = sensor FOV · dotted = boresight<br>"
    "Map shading = instantaneous ground coverage"
)


def _rotation_matrix(inclination_deg, raan_deg):
    """Return the circular-orbit plane rotation Rz(RAAN) Rx(inclination)."""
    inc = np.radians(inclination_deg)
    raan = np.radians(raan_deg)
    cos_i, sin_i = np.cos(inc), np.sin(inc)
    cos_o, sin_o = np.cos(raan), np.sin(raan)
    raan_rot = np.array([
        [cos_o, -sin_o, 0.0],
        [sin_o, cos_o, 0.0],
        [0.0, 0.0, 1.0],
    ])
    inc_rot = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_i, -sin_i],
        [0.0, sin_i, cos_i],
    ])
    return raan_rot @ inc_rot


def geo_state(inclination_deg, raan_deg, anomaly_deg):
    """Return one circular GEO-radius state in the requested orbit plane."""
    theta = np.radians(anomaly_deg)
    perifocal_r = GEO_RADIUS_KM * np.array([np.cos(theta), np.sin(theta), 0.0])
    perifocal_v = GEO_SPEED_KMS * np.array([-np.sin(theta), np.cos(theta), 0.0])
    rotation = _rotation_matrix(inclination_deg, raan_deg)
    return rotation @ perifocal_r, rotation @ perifocal_v


def geo_orbit(plane, n_points):
    """Return a full circular reference orbit for one GEO-radius plane."""
    anomalies = np.linspace(0.0, 360.0, n_points, endpoint=True)
    states = [
        geo_state(plane["inclination_deg"], plane["raan_deg"], anomaly)
        for anomaly in anomalies
    ]
    r_km = np.vstack([state[0] for state in states])
    v_kms = np.vstack([state[1] for state in states])
    return r_km, v_kms


def satellite_anomalies(plane, n_satellites=6):
    """Return staggered true anomalies for one orbit plane."""
    return plane["phase_deg"] + np.arange(n_satellites) * 360.0 / n_satellites


def _set_orbit_trace_style(trace, plane):
    trace.name = plane["name"]
    trace.line.color = plane["color"]
    trace.line.width = 5
    trace.legendgroup = plane["short"]
    trace.showlegend = True


def _hide_symbol_legend(fig, trace_count, plane):
    """Keep FOV symbols colored by plane, but out of the repeated legend."""
    for trace in fig.data[-trace_count:]:
        trace.legendgroup = plane["short"]
        trace.showlegend = False


def _add_orbit_trace(fig, r_km, plane):
    fig.add_trace(go.Scatter3d(
        x=r_km[:, 0],
        y=r_km[:, 1],
        z=r_km[:, 2],
        mode="lines",
        line=dict(color=plane["color"], width=5),
        name=plane["name"],
        legendgroup=plane["short"],
        showlegend=True,
    ))


def _hex_to_rgba(hex_color, opacity):
    value = hex_color.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{opacity})"


def _unit_vectors_from_lonlat(lon_deg, lat_deg):
    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)
    cos_lat = np.cos(lat)
    return np.stack((
        cos_lat * np.cos(lon),
        cos_lat * np.sin(lon),
        np.sin(lat),
    ), axis=-1)


def _coverage_mask_from_footprint(footprint_eci_km, sat_r_eci_km, theta_rad, lon_grid, lat_grid):
    footprint_ecef = _eci_to_ecef(np.column_stack(footprint_eci_km), theta_rad)
    footprint_norms = np.linalg.norm(footprint_ecef, axis=1)
    footprint_dirs = footprint_ecef[footprint_norms > 0.0] / footprint_norms[footprint_norms > 0.0, None]
    if len(footprint_dirs) == 0:
        return np.zeros_like(lon_grid, dtype=bool)

    center_ecef = _eci_to_ecef(np.asarray(sat_r_eci_km, dtype=float)[None, :], theta_rad)[0]
    center_norm = np.linalg.norm(center_ecef)
    if center_norm == 0.0:
        return np.zeros_like(lon_grid, dtype=bool)
    center_dir = center_ecef / center_norm

    footprint_angles = np.arccos(np.clip(footprint_dirs @ center_dir, -1.0, 1.0))
    angular_radius = float(np.nanmax(footprint_angles))
    grid_dirs = _unit_vectors_from_lonlat(lon_grid, lat_grid)
    return (grid_dirs @ center_dir) >= np.cos(angular_radius)


def _gmst_angle_rad(epoch):
    """Return a lightweight Greenwich sidereal angle for Earth-fixed mapping."""
    try:
        from astropy.time import Time
        time = Time(epoch, format="iso", scale="utc") if isinstance(epoch, str) else Time(epoch)
        jd = float(time.jd)
    except Exception:
        return 0.0
    centuries = (jd - 2451545.0) / 36525.0
    theta_deg = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * centuries**2
        - centuries**3 / 38710000.0
    )
    return np.radians(theta_deg % 360.0)


def _eci_to_ecef(vectors_km, theta_rad):
    vectors = np.asarray(vectors_km, dtype=float).reshape(-1, 3)
    theta = np.asarray(theta_rad, dtype=float)
    if theta.ndim == 0:
        theta = np.full(len(vectors), float(theta))
    theta = np.broadcast_to(theta.reshape(-1), (len(vectors),))
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x = vectors[:, 0] * cos_t + vectors[:, 1] * sin_t
    y = -vectors[:, 0] * sin_t + vectors[:, 1] * cos_t
    z = vectors[:, 2]
    return np.column_stack((x, y, z))


def _latlon_from_vectors(vectors_km, theta_rad=0.0):
    vectors = _eci_to_ecef(vectors_km, theta_rad)
    norms = np.linalg.norm(vectors, axis=1)
    valid = norms > 0.0
    lon = np.full(len(vectors), np.nan)
    lat = np.full(len(vectors), np.nan)
    lon[valid] = np.degrees(np.arctan2(vectors[valid, 1], vectors[valid, 0]))
    lat[valid] = np.degrees(np.arcsin(np.clip(vectors[valid, 2] / norms[valid], -1.0, 1.0)))
    return lon, lat


def _break_dateline(lon, lat):
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    out_lon = [lon[0]]
    out_lat = [lat[0]]
    for previous_lon, current_lon, current_lat in zip(lon[:-1], lon[1:], lat[1:]):
        if abs(current_lon - previous_lon) > 180.0:
            out_lon.append(None)
            out_lat.append(None)
        out_lon.append(current_lon)
        out_lat.append(current_lat)
    return out_lon, out_lat


def _map_grid():
    lon = np.linspace(-180.0, 180.0, 181)
    lat = np.linspace(-90.0, 90.0, 91)
    return lon, lat, *np.meshgrid(lon, lat)


def _add_map_background(fig):
    lon, lat, lon_grid, lat_grid = _map_grid()
    try:
        from global_land_mask import globe
        land = globe.is_land(lat_grid, lon_grid).astype(float)
        fig.add_trace(go.Heatmap(
            x=lon,
            y=lat,
            z=land,
            colorscale=[[0.0, "#0b1f3a"], [0.49, "#0b1f3a"], [0.5, "#253b2b"], [1.0, "#4b5f38"]],
            showscale=False,
            hoverinfo="skip",
            opacity=0.72,
            xaxis="x",
            yaxis="y",
            name="Land/water background",
        ))
    except Exception:
        pass

    for lat_line in [-60, -30, 0, 30, 60]:
        fig.add_trace(go.Scatter(
            x=[-180, 180],
            y=[lat_line, lat_line],
            mode="lines",
            line=dict(color="rgba(180,180,180,0.22)", width=1),
            hoverinfo="skip",
            showlegend=False,
            xaxis="x",
            yaxis="y",
            name="Latitude grid",
        ))


def _add_ground_coverage_map(fig, satellites, cfg):
    _add_map_background(fig)
    gmst = _gmst_angle_rad(cfg.get("epoch"))
    lon_axis, lat_axis, lon_grid, lat_grid = _map_grid()

    coverage_by_plane = {}
    outlines = []
    half_angle_deg = float(cfg["fov_half_angle_deg"])
    for sat in satellites:
        sat_r_km = np.asarray(sat["r_km"], dtype=float)
        direction = -sat_r_km / np.linalg.norm(sat_r_km)
        footprint = _footprint_on_earth(sat_r_km, direction, half_angle_deg, n_pts=180)
        if footprint is None:
            continue

        mask = _coverage_mask_from_footprint(footprint, sat_r_km, gmst, lon_grid, lat_grid)
        plane_mask = coverage_by_plane.setdefault(
            sat["short"],
            {"color": sat["color"], "mask": np.zeros_like(mask, dtype=bool)},
        )
        plane_mask["mask"] |= mask

        lon, lat = _latlon_from_vectors(np.column_stack(footprint), theta_rad=gmst)
        lon_line, lat_line = _break_dateline(lon, lat)
        outlines.append((sat, lon_line, lat_line))

    for short, coverage in coverage_by_plane.items():
        z = np.where(coverage["mask"], 1.0, np.nan)
        fig.add_trace(go.Heatmap(
            x=lon_axis,
            y=lat_axis,
            z=z,
            colorscale=[[0.0, coverage["color"]], [1.0, coverage["color"]]],
            zmin=0.0,
            zmax=1.0,
            opacity=0.20,
            showscale=False,
            hoverinfo="skip",
            xaxis="x",
            yaxis="y",
            name=f"{short} coverage shading",
        ))

    for sat, lon_line, lat_line in outlines:
        fig.add_trace(go.Scatter(
            x=lon_line,
            y=lat_line,
            mode="lines",
            line=dict(color=_hex_to_rgba(sat["color"], 0.88), width=1),
            name=f"{sat['name']} coverage boundary",
            legendgroup=sat["short"],
            showlegend=False,
            hovertemplate=f"{sat['name']} coverage boundary<br>lon=%{{x:.1f}}°<br>lat=%{{y:.1f}}°<extra></extra>",
            xaxis="x",
            yaxis="y",
        ))

    fig.add_annotation(
        text="Ground coverage map: instantaneous sensor coverage (Earth-fixed lon/lat projection)",
        x=0.5,
        y=0.306,
        xref="paper",
        yref="paper",
        xanchor="center",
        yanchor="bottom",
        showarrow=False,
        font=dict(color="#DDDDDD", size=12),
    )


def _finalize_layout(fig):
    fig.update_layout(
        title=dict(
            text="GEO-Radius Spherical Constellation Sensor FOV",
            x=0.5,
            y=0.952,
            xanchor="center",
            yanchor="top",
        ),
        annotations=list(fig.layout.annotations or []) + [dict(
            text=_SYMBOL_GUIDE,
            x=0.66,
            y=0.925,
            xref="paper",
            yref="paper",
            xanchor="center",
            yanchor="top",
            showarrow=False,
            align="center",
            font=dict(color="#DDDDDD", size=12),
            bgcolor="rgba(0,0,0,0.55)",
            bordercolor="#444444",
            borderwidth=1,
        )],
        legend=dict(
            orientation="v",
            x=0.01,
            y=0.925,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(0,0,0,0.55)",
            bordercolor="#333333",
            borderwidth=1,
            font=dict(size=11),
            itemsizing="constant",
        ),
        scene=dict(domain=dict(x=[0.0, 1.0], y=[0.34, 0.875])),
        xaxis=dict(
            domain=[0.06, 0.96],
            range=[-180, 180],
            title="Longitude (deg)",
            showgrid=False,
            zeroline=False,
            tickmode="array",
            tickvals=[-180, -120, -60, 0, 60, 120, 180],
        ),
        yaxis=dict(
            domain=[0.04, 0.28],
            range=[-90, 90],
            title="Latitude (deg)",
            showgrid=False,
            zeroline=False,
            tickmode="array",
            tickvals=[-90, -60, -30, 0, 30, 60, 90],
        ),
        height=760,
        margin=dict(l=0, r=0, t=40, b=18),
    )


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    # 1. Build the first GEO-radius orbit plane through the standard scene
    #    constructor so Earth, stars, Moon, Sun, and the subsatellite track are
    #    added once.  Additional planes are added explicitly below.
    orbit_points = 121 if fast else 241
    reference_r_km, reference_v_kms = geo_orbit(ORBIT_PLANES[0], n_points=orbit_points)

    # 2. Configure the shared scene and FOV properties.  show_sensor=False keeps
    #    build_figure from adding the old single-satellite example; each
    #    constellation spacecraft is added explicitly below.
    cfg = DEFAULT_CFG.copy()
    cfg["title"] = "GEO-Radius Spherical Constellation Sensor FOV"
    cfg["show_sensor"] = False
    cfg["fov_animate"] = False
    cfg["fov_pointing_mode"] = "nadir"
    cfg["fov_half_angle_deg"] = 6.0
    cfg["fov_cone_length_km"] = 36_000.0
    cfg["fov_opacity"] = 0.22
    cfg["fov_show_boresight"] = True
    cfg["fov_show_ground_intercept"] = False
    cfg["axis_range_km"] = 50_000.0
    cfg["earth_n_lat"] = 50 if fast else 90
    cfg["earth_n_lon"] = 100 if fast else 180
    cfg["show_stars"] = True
    cfg["star_mag_limit"] = 4.8 if fast else 6.0
    cfg["star_sphere_factor"] = 60.0
    cfg["show_sun"] = True
    cfg["show_moon"] = True
    cfg["moon_n_lat"] = 24 if fast else 45
    cfg["moon_n_lon"] = 48 if fast else 90
    cfg["show_axis_ticks"] = False

    # 3. Build the scene context and convert the base orbit trace into the first
    #    compact legend entry.
    fig = build_figure(cfg, reference_r_km, reference_v_kms)
    for trace in fig.data:
        if getattr(trace, "name", None) == "Orbit":
            _set_orbit_trace_style(trace, ORBIT_PLANES[0])
        elif getattr(trace, "name", None) == "Subsatellite ground track":
            trace.showlegend = False

    # 4. Add the other orthogonal GEO-radius orbit planes.  Together these three
    #    rings make a simple full-sphere coverage shell at GEO radius.
    orbit_tracks = [{
        "name": ORBIT_PLANES[0]["name"],
        "short": ORBIT_PLANES[0]["short"],
        "color": ORBIT_PLANES[0]["color"],
        "r_km": reference_r_km,
    }]
    for plane in ORBIT_PLANES[1:]:
        orbit_r_km, _ = geo_orbit(plane, n_points=orbit_points)
        _add_orbit_trace(fig, orbit_r_km, plane)
        orbit_tracks.append({
            "name": plane["name"],
            "short": plane["short"],
            "color": plane["color"],
            "r_km": orbit_r_km,
        })

    # 5. Add six nadir-pointing spacecraft on each orbit plane.  Their markers,
    #    FOV cones, footprints, and intercepts are colored by plane but hidden
    #    from the legend so the legend remains one row per orbit.
    satellites = []
    for plane_index, plane in enumerate(ORBIT_PLANES, start=1):
        for sat_index, anomaly_deg in enumerate(satellite_anomalies(plane), start=1):
            sat_r_km, sat_v_kms = geo_state(
                plane["inclination_deg"],
                plane["raan_deg"],
                anomaly_deg,
            )
            sat_name = f"{plane['short']}-{sat_index} ({anomaly_deg % 360:.0f}°)"
            traces = add_sensor_fov_to_figure(
                fig,
                sat_r_km[None, :],
                sat_v_kms[None, :],
                cfg,
                time_index=0,
                sat_name=sat_name,
                satellite_color=plane["color"],
                fov_color=plane["color"],
            )
            _hide_symbol_legend(fig, len(traces), plane)
            satellites.append({
                "name": sat_name,
                "orbit": plane["name"],
                "short": plane["short"],
                "color": plane["color"],
                "plane_index": plane_index,
                "anomaly_deg": float(anomaly_deg % 360.0),
                "r_km": sat_r_km,
                "v_kms": sat_v_kms,
            })

    # 6. Add a lower lon/lat map showing only instantaneous FOV coverage
    #    footprints.  We intentionally omit orbit ground-track curves here so
    #    users do not read a polar orbit path as continuous ground coverage.
    _add_ground_coverage_map(fig, satellites, cfg)
    _finalize_layout(fig)

    html_path = None
    if make_figures:
        html_path = Path(figpath("demo_sensor_fov_plot.html"))
        fig.write_html(str(html_path))
        print(f"Saved: {html_path}")

    return {
        "figure": fig,
        "r_km": reference_r_km,
        "v_kms": reference_v_kms,
        "n_steps": len(reference_r_km),
        "orbits": orbit_tracks,
        "satellites": satellites,
        "html": str(html_path) if html_path else None,
    }


if __name__ == "__main__":
    main(make_figures=True, fast=False)
