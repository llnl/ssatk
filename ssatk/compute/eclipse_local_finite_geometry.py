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

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np

from ssatk.compute.eclipse_reference_events import (
    AU_KM,
    EARTH_AXES_KM,
    RE_KM,
    R_MOON_MEAN_KM,
    R_SUN_KM,
    ReferenceEvent,
    build_reference_event,
    build_solar_local_event,
    jd_to_datetime,
    solar_central_line_wgs84,
    _unit,
)
from ssatk.compute.eclipse_raytrace import (
    LUNAR_DANJON_EARTH_RADIUS_KM,
    LUNAR_OPTICAL_MOON_RADIUS_KM,
    ReferenceRayBundle,
    RayPath,
    tangent_cross_section_paths,
    trace_reference_rays,
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


@dataclass(frozen=True)
class ImpactPlaneFrame:
    """One exact two-dimensional finite-Sun cross-section."""

    bundle: ReferenceRayBundle
    origin_world_km: np.ndarray
    basis_world_from_local: np.ndarray
    source_local_km: np.ndarray
    occluder_local_km: np.ndarray
    target_local_km: np.ndarray
    impact_parameter_km: float
    direct_local_km: np.ndarray
    shadow_axis_local_km: np.ndarray
    umbra_paths_local_km: tuple[np.ndarray, np.ndarray]
    penumbra_paths_local_km: tuple[np.ndarray, np.ndarray]
    umbra_tangent_local_km: np.ndarray
    penumbra_tangent_local_km: np.ndarray
    target_hit_labels: dict[str, tuple[bool, bool]]

    @property
    def mode(self) -> str:
        return self.bundle.mode

    @property
    def occluder_name(self) -> str:
        return self.bundle.occluder_name

    @property
    def target_name(self) -> str:
        return self.bundle.target_name


def transform_points(points, origin_world_km, basis_world_from_local) -> np.ndarray:
    """World vectors to local impact-plane coordinates."""
    values = np.asarray(points, dtype=float)
    return (values - np.asarray(origin_world_km, dtype=float)) @ np.asarray(
        basis_world_from_local, dtype=float
    )


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


def peak_impact_basis(event: ReferenceEvent) -> np.ndarray:
    """Stable event basis used by the rotatable local 3-D scene."""
    bundle = trace_reference_rays(event.definition, event.greatest_jd, n_azimuth=24)
    _, basis, _ = impact_plane_basis(bundle)
    return basis


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


def build_impact_plane_frame(
    event: ReferenceEvent, jd_value: float, *, n_azimuth: int = 48
) -> ImpactPlaneFrame:
    bundle = trace_reference_rays(
        event.definition, float(jd_value), n_azimuth=max(12, int(n_azimuth))
    )
    origin, basis, impact = impact_plane_basis(bundle)
    source = transform_points(bundle.sun_center_km.reshape(1, 3), origin, basis)[0]
    occluder_world = (
        bundle.moon_center_km if bundle.mode == "solar" else bundle.earth_center_km
    )
    target_world = (
        bundle.earth_center_km if bundle.mode == "solar" else bundle.moon_center_km
    )
    occluder = transform_points(np.asarray(occluder_world).reshape(1, 3), origin, basis)[0]
    target = transform_points(np.asarray(target_world).reshape(1, 3), origin, basis)[0]
    direct = transform_points(bundle.central.points_km, origin, basis)
    shadow_axis = transform_points(shadow_axis_world(bundle, float(jd_value)), origin, basis)

    local_paths: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    local_tangents: dict[str, np.ndarray] = {}
    hit_labels: dict[str, tuple[bool, bool]] = {}
    for family in ("umbra", "penumbra"):
        upper, lower, tangents = tangent_cross_section_paths(
            bundle, family, basis[:, 2]
        )
        local_paths[family] = (
            transform_points(upper.points_km, origin, basis),
            transform_points(lower.points_km, origin, basis),
        )
        local_tangents[family] = transform_points(tangents, origin, basis)
        hit_labels[family] = (bool(upper.target_hit), bool(lower.target_hit))

    return ImpactPlaneFrame(
        bundle=bundle,
        origin_world_km=origin,
        basis_world_from_local=basis,
        source_local_km=source,
        occluder_local_km=occluder,
        target_local_km=target,
        impact_parameter_km=impact,
        direct_local_km=direct,
        shadow_axis_local_km=shadow_axis,
        umbra_paths_local_km=local_paths["umbra"],
        penumbra_paths_local_km=local_paths["penumbra"],
        umbra_tangent_local_km=local_tangents["umbra"],
        penumbra_tangent_local_km=local_tangents["penumbra"],
        target_hit_labels=hit_labels,
    )


def _radial_earth_outline_local(
    frame: ImpactPlaneFrame, earth_center_world: np.ndarray, *, n: int = 721
) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * math.pi, int(n))
    x_axis = frame.basis_world_from_local[:, 0]
    z_axis = frame.basis_world_from_local[:, 2]
    directions = (
        np.cos(theta)[:, None] * x_axis[None, :]
        + np.sin(theta)[:, None] * z_axis[None, :]
    )
    radii = 1.0 / np.sqrt(np.sum((directions / EARTH_AXES_KM[None, :]) ** 2, axis=1))
    world = np.asarray(earth_center_world, dtype=float)[None, :] + directions * radii[:, None]
    return transform_points(world, frame.origin_world_km, frame.basis_world_from_local)


def _circle_outline(center_local: np.ndarray, radius: float, *, n: int = 721) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * math.pi, int(n))
    center = np.asarray(center_local, dtype=float)
    result = np.repeat(center.reshape(1, 3), len(theta), axis=0)
    result[:, 0] += float(radius) * np.cos(theta)
    result[:, 2] += float(radius) * np.sin(theta)
    return result


def body_outlines(frame: ImpactPlaneFrame) -> dict[str, list[dict]]:
    """Solid and optical limbs in the exact impact plane."""
    bundle = frame.bundle
    result: dict[str, list[dict]] = {"earth": [], "moon": []}
    earth_solid = _radial_earth_outline_local(frame, bundle.earth_center_km)
    result["earth"].append(
        dict(points=earth_solid, label="WGS-84 solid Earth", kind="solid")
    )
    # Keep the construction explicit so the solid and event-specific optical
    # lunar limbs remain independently auditable.
    moon_center = transform_points(
        bundle.moon_center_km.reshape(1, 3),
        frame.origin_world_km,
        frame.basis_world_from_local,
    )[0]
    result["moon"] = [
        dict(
            points=_circle_outline(moon_center, R_MOON_MEAN_KM),
            label="Mean solid Moon",
            kind="solid",
        )
    ]

    if bundle.mode == "solar":
        result["moon"].extend([
            dict(
                points=_circle_outline(moon_center, bundle.umbra_optical_radius_km),
                label="NASA k2 umbral optical limb",
                kind="optical_umbra",
            ),
            dict(
                points=_circle_outline(moon_center, bundle.penumbra_optical_radius_km),
                label="NASA k1 penumbral optical limb",
                kind="optical_penumbra",
            ),
        ])
    else:
        earth_center = transform_points(
            bundle.earth_center_km.reshape(1, 3),
            frame.origin_world_km,
            frame.basis_world_from_local,
        )[0]
        result["earth"].append(
            dict(
                points=_circle_outline(earth_center, LUNAR_DANJON_EARTH_RADIUS_KM),
                label="Danjon effective optical Earth limb",
                kind="optical_umbra",
            )
        )
        result["moon"].append(
            dict(
                points=_circle_outline(moon_center, LUNAR_OPTICAL_MOON_RADIUS_KM),
                label="NASA optical lunar limb",
                kind="optical_target",
            )
        )
    return result


def clip_line_to_x(points: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    """Clip a straight path to an X interval while retaining exact endpoints."""
    values = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(values) < 2:
        return np.empty((0, 3), dtype=float)
    p0, p1 = values[0], values[-1]
    delta = p1 - p0
    if abs(float(delta[0])) < 1.0e-14:
        return np.vstack([p0, p1]) if xmin <= p0[0] <= xmax else np.empty((0, 3))
    t0 = (float(xmin) - p0[0]) / delta[0]
    t1 = (float(xmax) - p0[0]) / delta[0]
    lo = max(0.0, min(t0, t1))
    hi = min(1.0, max(t0, t1))
    if hi < lo:
        return np.empty((0, 3), dtype=float)
    return np.vstack([p0 + lo * delta, p0 + hi * delta])


def _z_on_line(points: np.ndarray, x: np.ndarray | float) -> np.ndarray:
    values = np.asarray(points, dtype=float).reshape(-1, 3)
    p0, p1 = values[0], values[-1]
    x_values = np.asarray(x, dtype=float)
    if abs(float(p1[0] - p0[0])) < 1.0e-14:
        return np.full_like(x_values, p0[2], dtype=float)
    t = (x_values - p0[0]) / (p1[0] - p0[0])
    return p0[2] + t * (p1[2] - p0[2])


def target_plane_radii(frame: ImpactPlaneFrame) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, pair in (
        ("umbra", frame.umbra_paths_local_km),
        ("penumbra", frame.penumbra_paths_local_km),
    ):
        values = [float(_z_on_line(path, 0.0)) for path in pair]
        result[name] = 0.5 * abs(values[0] - values[1])
        result[f"{name}_center_z_km"] = 0.5 * (values[0] + values[1])
    return result


def view_ranges(frame: ImpactPlaneFrame) -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
    """Readable exact windows for tangency, target and full baseline views."""
    occ = frame.occluder_local_km
    target = frame.target_local_km
    occ_radius = max(
        frame.bundle.solid_occluder_radius_km,
        frame.bundle.umbra_optical_radius_km,
        frame.bundle.penumbra_optical_radius_km,
    )
    if frame.mode == "solar":
        target_radius = RE_KM
    else:
        target_radius = max(R_MOON_MEAN_KM, LUNAR_OPTICAL_MOON_RADIUS_KM)
    radii = target_plane_radii(frame)
    target_vertical = max(
        abs(float(target[2])) + 1.35 * target_radius,
        1.15 * radii["penumbra"],
    )
    tangent = (
        (float(occ[0] - 3.0 * occ_radius), float(occ[0] + 3.0 * occ_radius)),
        (float(-2.8 * occ_radius), float(2.8 * occ_radius)),
    )
    target_view = (
        (float(-3.0 * target_radius), float(1.45 * target_radius)),
        (float(-target_vertical), float(target_vertical)),
    )
    full_margin = 3.0 * max(occ_radius, target_radius)
    full_vertical = max(
        abs(float(target[2])) + 1.25 * target_radius,
        1.15 * radii["penumbra"],
        1.35 * occ_radius,
    )
    full = (
        (float(occ[0] - full_margin), float(full_margin)),
        (float(-full_vertical), float(full_vertical)),
    )
    return {"occluder": tangent, "target": target_view, "system": full}



def event_view_ranges(event: ReferenceEvent, jd_values=None) -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
    """Union of readable impact-plane windows across an event sequence."""
    values = np.asarray(event.jd if jd_values is None else jd_values, dtype=float)
    if values.size == 0:
        values = np.asarray([event.greatest_jd], dtype=float)
    accum = {
        key: [float("inf"), float("-inf"), float("inf"), float("-inf")]
        for key in ("occluder", "target", "system")
    }
    for jd_value in values:
        state = build_impact_plane_frame(event, float(jd_value), n_azimuth=24)
        for key, (xlim, ylim) in view_ranges(state).items():
            box = accum[key]
            box[0] = min(box[0], float(xlim[0]))
            box[1] = max(box[1], float(xlim[1]))
            box[2] = min(box[2], float(ylim[0]))
            box[3] = max(box[3], float(ylim[1]))
    result = {}
    for key, box in accum.items():
        xpad = 0.025 * max(box[1] - box[0], 1.0)
        ypad = 0.04 * max(box[3] - box[2], 1.0)
        result[key] = (
            (box[0] - xpad, box[1] + xpad),
            (box[2] - ypad, box[3] + ypad),
        )
    return result

def geometry_metrics(frame: ImpactPlaneFrame) -> dict[str, float | str | bool]:
    radii = target_plane_radii(frame)
    out: dict[str, float | str | bool] = {
        "mode": frame.mode,
        "impact_parameter_km": frame.impact_parameter_km,
        "earth_moon_distance_km": float(
            np.linalg.norm(frame.bundle.moon_center_km - frame.bundle.earth_center_km)
        ),
        "sun_earth_distance_km": float(
            np.linalg.norm(frame.bundle.sun_center_km - frame.bundle.earth_center_km)
        ),
        "umbra_radius_at_target_plane_km": radii["umbra"],
        "penumbra_radius_at_target_plane_km": radii["penumbra"],
        "umbra_upper_hits_target": frame.target_hit_labels["umbra"][0],
        "umbra_lower_hits_target": frame.target_hit_labels["umbra"][1],
        "penumbra_upper_hits_target": frame.target_hit_labels["penumbra"][0],
        "penumbra_lower_hits_target": frame.target_hit_labels["penumbra"][1],
    }
    return out


def _matplotlib_body(ax, frame: ImpactPlaneFrame, body: str, *, zorder: int = 8) -> None:
    outlines = body_outlines(frame)[body]
    center = frame.target_local_km if (
        (body == "earth" and frame.mode == "solar")
        or (body == "moon" and frame.mode == "lunar")
    ) else frame.occluder_local_km
    fill_color = COLORS["earth_fill"] if body == "earth" else COLORS["moon_fill"]
    edge_color = COLORS["earth"] if body == "earth" else COLORS["moon"]
    solid = next(item for item in outlines if item["kind"] == "solid")
    points = solid["points"]
    ax.fill(points[:, 0], points[:, 2], color=fill_color, alpha=0.97, zorder=zorder)
    ax.plot(points[:, 0], points[:, 2], color=edge_color, lw=2.2, zorder=zorder + 1)
    for item in outlines:
        if item["kind"] == "solid":
            continue
        line_color = (
            COLORS["umbra"] if item["kind"] == "optical_umbra"
            else COLORS["penumbra"] if item["kind"] == "optical_penumbra"
            else COLORS["optical"]
        )
        ax.plot(
            item["points"][:, 0], item["points"][:, 2],
            color=line_color, lw=1.0, ls=(0, (3, 3)), alpha=0.85,
            zorder=zorder + 2,
        )
    ax.plot(center[0], center[2], marker="+", color=edge_color, ms=9, mew=1.7, zorder=zorder + 3)


def _matplotlib_envelopes(ax, frame: ImpactPlaneFrame, xlim: tuple[float, float]) -> None:
    for family, pair, color, alpha in (
        ("penumbra", frame.penumbra_paths_local_km, COLORS["penumbra"], 0.085),
        ("umbra", frame.umbra_paths_local_km, COLORS["umbra"], 0.16),
    ):
        start = max(float(pair[0][1, 0]), float(pair[1][1, 0]), float(xlim[0]))
        stop = min(float(pair[0][-1, 0]), float(pair[1][-1, 0]), float(xlim[1]))
        if stop <= start:
            continue
        x = np.linspace(start, stop, 400)
        z0 = _z_on_line(pair[0], x)
        z1 = _z_on_line(pair[1], x)
        ax.fill_between(x, np.minimum(z0, z1), np.maximum(z0, z1), color=color, alpha=alpha, zorder=1)


def draw_matplotlib_view(ax, frame: ImpactPlaneFrame, view: str) -> None:
    """Draw one exact view in kilometres with equal physical X/Z scale."""
    ranges = view_ranges(frame)[view]
    xlim, ylim = ranges
    ax.set_facecolor(COLORS["panel"])
    ax.grid(color=COLORS["grid"], alpha=0.55, lw=0.65)
    ax.axhline(0.0, color=COLORS["target_plane"], lw=0.8, alpha=0.5)
    if view != "occluder":
        ax.axvline(0.0, color=COLORS["target_plane"], lw=1.0, ls=(0, (2, 4)), alpha=0.75)
    _matplotlib_envelopes(ax, frame, xlim)

    # Incoming axial sunlight stops on the Sun-facing solid surface.
    direct = clip_line_to_x(frame.direct_local_km, *xlim)
    if len(direct):
        ax.plot(direct[:, 0], direct[:, 2], color=COLORS["sunlight"], lw=2.6, zorder=6)
    shadow = clip_line_to_x(frame.shadow_axis_local_km, *xlim)
    if len(shadow):
        ax.plot(shadow[:, 0], shadow[:, 2], color=COLORS["axis"], lw=2.1, zorder=5)

    for family, pair, color, lw in (
        ("umbra", frame.umbra_paths_local_km, COLORS["umbra"], 2.0),
        ("penumbra", frame.penumbra_paths_local_km, COLORS["penumbra"], 1.65),
    ):
        for path, hit in zip(pair, frame.target_hit_labels[family]):
            clipped = clip_line_to_x(path, *xlim)
            if len(clipped):
                ax.plot(clipped[:, 0], clipped[:, 2], color=color, lw=lw, zorder=5)
            endpoint = path[-1]
            if xlim[0] <= endpoint[0] <= xlim[1] and ylim[0] <= endpoint[2] <= ylim[1]:
                marker = "o" if hit else "x"
                ax.plot(endpoint[0], endpoint[2], marker=marker, color=color, ms=5.0, mew=1.3, zorder=9)

    for tangents, color in (
        (frame.umbra_tangent_local_km, COLORS["umbra"]),
        (frame.penumbra_tangent_local_km, COLORS["penumbra"]),
    ):
        visible = (
            (tangents[:, 0] >= xlim[0]) & (tangents[:, 0] <= xlim[1])
            & (tangents[:, 2] >= ylim[0]) & (tangents[:, 2] <= ylim[1])
        )
        if np.any(visible):
            ax.scatter(tangents[visible, 0], tangents[visible, 2], s=20, facecolors="none", edgecolors=color, lw=1.2, zorder=10)

    _matplotlib_body(ax, frame, "earth")
    _matplotlib_body(ax, frame, "moon")

    # Full two-dimensional axis miss at the target-centre plane.
    if view in {"target", "system"}:
        z_target = float(frame.target_local_km[2])
        ax.annotate(
            "", xy=(0.0, z_target), xytext=(0.0, 0.0),
            arrowprops=dict(arrowstyle="<->", color=COLORS["impact"], lw=1.6),
            zorder=12,
        )
        label_x = 0.018 * (xlim[1] - xlim[0])
        ax.text(
            label_x, 0.5 * z_target,
            f"axis miss b = {frame.impact_parameter_km:,.1f} km",
            color=COLORS["impact"], fontsize=8.5, va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.22", facecolor=COLORS["panel"], edgecolor=COLORS["impact"], alpha=0.9),
            zorder=13,
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(colors=COLORS["muted"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#516078")
    ax.set_xlabel("Optical X [km] — downstream →", color=COLORS["text"], fontsize=9)
    ax.set_ylabel("Impact-plane Z [km] — toward target centre", color=COLORS["text"], fontsize=9)

    if view == "occluder":
        ax.set_title(
            f"{frame.occluder_name} tangency — solid and optical limbs",
            color=COLORS["text"], fontsize=10.5, pad=8,
        )
    elif view == "target":
        radii = target_plane_radii(frame)
        ax.set_title(
            f"{frame.target_name} interception — first-surface clipping\n"
            f"target-plane umbra {radii['umbra']:,.1f} km; penumbra {radii['penumbra']:,.1f} km",
            color=COLORS["text"], fontsize=10.5, pad=8,
        )
    else:
        ax.set_title(
            "Complete Earth–Moon baseline — true unwarped physical scale",
            color=COLORS["text"], fontsize=10.5, pad=8,
        )
        ax.text(
            0.012, 0.96,
            f"Sun is {np.linalg.norm(frame.bundle.sun_center_km-frame.bundle.earth_center_km)/1e6:.3f} million km upstream and remains off-frame",
            transform=ax.transAxes, color=COLORS["sunlight"], fontsize=8.5,
            ha="left", va="top",
        )


def generate_validation_figure(output_path: str | Path) -> str:
    """Generate a large, readable solar/lunar geometry validation sheet."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    solar_event = build_solar_local_event(n_frames=17)
    lunar_event = build_reference_event("lunar", n_frames=17)
    solar = build_impact_plane_frame(solar_event, solar_event.greatest_jd, n_azimuth=72)
    lunar = build_impact_plane_frame(lunar_event, lunar_event.greatest_jd, n_azimuth=72)

    fig = plt.figure(figsize=(22, 18), dpi=150, facecolor=COLORS["background"])
    outer = fig.add_gridspec(4, 2, height_ratios=[1.0, 0.48, 1.0, 0.48], hspace=0.33, wspace=0.18)
    axes = [
        fig.add_subplot(outer[0, 0]), fig.add_subplot(outer[0, 1]),
        fig.add_subplot(outer[1, :]),
        fig.add_subplot(outer[2, 0]), fig.add_subplot(outer[2, 1]),
        fig.add_subplot(outer[3, :]),
    ]
    for ax, frame, view in (
        (axes[0], solar, "occluder"), (axes[1], solar, "target"),
        (axes[2], solar, "system"),
        (axes[3], lunar, "occluder"), (axes[4], lunar, "target"),
        (axes[5], lunar, "system"),
    ):
        draw_matplotlib_view(ax, frame, view)

    axes[0].text(
        -0.02, 1.18, "SOLAR ECLIPSE · 8 APRIL 2024 · GREATEST ECLIPSE",
        transform=axes[0].transAxes, color="#ffd785", fontsize=13, fontweight="bold",
        ha="left", va="bottom",
    )
    axes[3].text(
        -0.02, 1.18, "LUNAR ECLIPSE · 14 MARCH 2025 · GREATEST ECLIPSE",
        transform=axes[3].transAxes, color="#efb9a5", fontsize=13, fontweight="bold",
        ha="left", va="bottom",
    )

    legend = [
        Line2D([0], [0], color=COLORS["sunlight"], lw=3, label="direct sunlight — stops at first opaque surface"),
        Line2D([0], [0], color=COLORS["axis"], lw=2.4, label="blocked-light shadow axis"),
        Line2D([0], [0], color=COLORS["umbra"], lw=2.2, label="umbral / antumbral tangent boundaries"),
        Line2D([0], [0], color=COLORS["penumbra"], lw=2.0, label="penumbral tangent boundaries"),
        Patch(facecolor=COLORS["umbra"], alpha=0.16, label="umbral / antumbral envelope"),
        Patch(facecolor=COLORS["penumbra"], alpha=0.085, label="penumbral envelope"),
        Line2D([0], [0], color=COLORS["impact"], marker="|", lw=1.8, label="full impact parameter at target plane"),
        Line2D([0], [0], color=COLORS["optical"], lw=1.1, ls=(0, (3, 3)), label="model optical limb distinct from solid surface"),
    ]
    fig.legend(
        handles=legend, loc="lower center", ncol=4, frameon=True,
        bbox_to_anchor=(0.5, 0.012), fontsize=9.2, labelcolor=COLORS["text"],
        facecolor="#0b111c", edgecolor="#53627a", framealpha=0.98,
    )
    fig.suptitle(
        "Exact finite-Sun impact-plane geometry — source, occluder, target and selected tangents share one physical plane",
        color=COLORS["text"], fontsize=17, y=0.992,
    )
    fig.text(
        0.5, 0.965,
        "Local windows are magnifications of the same unwarped kilometre coordinates. The full-system strips retain equal X/Z scale; the real Sun remains one AU upstream.",
        ha="center", color=COLORS["muted"], fontsize=10.5,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return str(path)


def _plotly_body_traces(frame: ImpactPlaneFrame, xaxis: str, yaxis: str, *, showlegend: bool):
    import plotly.graph_objects as go
    result = []
    outlines = body_outlines(frame)
    for body, edge, fill in (
        ("earth", COLORS["earth"], COLORS["earth_fill"]),
        ("moon", COLORS["moon"], COLORS["moon_fill"]),
    ):
        items = outlines[body]
        solid = next(item for item in items if item["kind"] == "solid")
        pts = solid["points"]
        result.append(go.Scatter(
            x=pts[:, 0], y=pts[:, 2], mode="lines", fill="toself",
            fillcolor=fill, line=dict(color=edge, width=2.0),
            name=solid["label"], showlegend=showlegend,
            legendgroup="bodies", legendgrouptitle_text="Solid bodies",
            hovertemplate=f"{solid['label']}<extra></extra>", xaxis=xaxis, yaxis=yaxis,
        ))
        for item in items:
            if item["kind"] == "solid":
                continue
            color = (
                COLORS["umbra"] if item["kind"] == "optical_umbra"
                else COLORS["penumbra"] if item["kind"] == "optical_penumbra"
                else COLORS["optical"]
            )
            pts = item["points"]
            result.append(go.Scatter(
                x=pts[:, 0], y=pts[:, 2], mode="lines",
                line=dict(color=color, width=1.2, dash="dot"),
                name=item["label"], showlegend=showlegend,
                legendgroup="optical-limbs", legendgrouptitle_text="Model optical limbs",
                hovertemplate=f"{item['label']}<extra></extra>", xaxis=xaxis, yaxis=yaxis,
            ))
    return result


def _plotly_envelope_trace(frame: ImpactPlaneFrame, family: str, xlim, xaxis, yaxis, *, showlegend: bool):
    import plotly.graph_objects as go
    pair = frame.umbra_paths_local_km if family == "umbra" else frame.penumbra_paths_local_km
    color = COLORS["umbra"] if family == "umbra" else COLORS["penumbra"]
    rgba = "rgba(255,108,92,0.14)" if family == "umbra" else "rgba(139,200,255,0.075)"
    start = max(float(pair[0][1, 0]), float(pair[1][1, 0]), float(xlim[0]))
    stop = min(float(pair[0][-1, 0]), float(pair[1][-1, 0]), float(xlim[1]))
    if stop <= start:
        polygon_x = np.asarray([], dtype=float)
        polygon_z = np.asarray([], dtype=float)
    else:
        x = np.linspace(start, stop, 320)
        z0 = _z_on_line(pair[0], x)
        z1 = _z_on_line(pair[1], x)
        polygon_x = np.concatenate([x, x[::-1]])
        polygon_z = np.concatenate([z0, z1[::-1]])
    label = "Umbral / antumbral optical envelope" if family == "umbra" else "Penumbral optical envelope"
    return go.Scatter(
        x=polygon_x, y=polygon_z, mode="lines", fill="toself",
        line=dict(color=color, width=0.6), fillcolor=rgba,
        name=label, showlegend=showlegend, visible="legendonly",
        legendgroup="envelopes", legendgrouptitle_text="Optional optical envelopes",
        hoverinfo="skip", xaxis=xaxis, yaxis=yaxis,
    )


def plotly_view_traces(
    frame: ImpactPlaneFrame,
    view: str,
    *,
    xaxis: str,
    yaxis: str,
    showlegend: bool,
) -> list:
    """Plotly traces for one exact cross-section view."""
    import plotly.graph_objects as go
    xlim, ylim = view_ranges(frame)[view]
    traces: list = []
    for family in ("penumbra", "umbra"):
        envelope = _plotly_envelope_trace(
            frame, family, xlim, xaxis, yaxis, showlegend=showlegend
        )
        traces.append(envelope)

    direct = clip_line_to_x(frame.direct_local_km, *xlim)
    traces.append(go.Scatter(
        x=direct[:, 0] if len(direct) else [],
        y=direct[:, 2] if len(direct) else [], mode="lines",
        line=dict(color=COLORS["sunlight"], width=3.4),
        name="Direct sunlight - first-surface stop", showlegend=showlegend,
        legendgroup="rays", legendgrouptitle_text="Finite-Sun paths",
        hovertemplate="Direct sunlight<br>X=%{x:,.1f} km<br>Z=%{y:,.1f} km<extra></extra>",
        xaxis=xaxis, yaxis=yaxis,
    ))
    shadow = clip_line_to_x(frame.shadow_axis_local_km, *xlim)
    traces.append(go.Scatter(
        x=shadow[:, 0] if len(shadow) else [],
        y=shadow[:, 2] if len(shadow) else [], mode="lines",
        line=dict(color=COLORS["axis"], width=2.8),
        name="Blocked-light axis - not transmitted light", showlegend=showlegend,
        legendgroup="rays",
        hovertemplate="Blocked-light axis<br>X=%{x:,.1f} km<br>Z=%{y:,.1f} km<extra></extra>",
        xaxis=xaxis, yaxis=yaxis,
    ))

    for family, pair, color, width, label in (
        ("umbra", frame.umbra_paths_local_km, COLORS["umbra"], 2.6, "Umbral / antumbral tangent"),
        ("penumbra", frame.penumbra_paths_local_km, COLORS["penumbra"], 2.1, "Penumbral tangent"),
    ):
        for index, (path, hit) in enumerate(zip(pair, frame.target_hit_labels[family])):
            clipped = clip_line_to_x(path, *xlim)
            custom = np.repeat(
                [["hits target" if hit else "misses target; ends on target-centre plane"]],
                len(clipped), axis=0,
            )
            traces.append(go.Scatter(
                x=clipped[:, 0] if len(clipped) else [],
                y=clipped[:, 2] if len(clipped) else [], mode="lines",
                line=dict(color=color, width=width),
                name=label, showlegend=showlegend and index == 0,
                legendgroup="rays", customdata=custom,
                hovertemplate=(f"{label}<br>%{{customdata[0]}}<br>"
                               "X=%{x:,.1f} km<br>Z=%{y:,.1f} km<extra></extra>"),
                xaxis=xaxis, yaxis=yaxis,
            ))
            endpoint = path[-1]
            traces.append(go.Scatter(
                x=[endpoint[0]], y=[endpoint[2]], mode="markers",
                marker=dict(
                    size=7, color=color, symbol="circle" if hit else "x",
                    line=dict(color="white", width=0.7),
                ),
                name=f"{label} endpoint", showlegend=False,
                hovertemplate=(f"{label}<br>"
                               f"{'first target-surface hit' if hit else 'target-centre-plane endpoint'}"
                               "<br>X=%{x:,.1f} km<br>Z=%{y:,.1f} km<extra></extra>"),
                xaxis=xaxis, yaxis=yaxis,
            ))

    traces.extend(_plotly_body_traces(frame, xaxis, yaxis, showlegend=showlegend))

    if view in {"target", "system"}:
        target_z = float(frame.target_local_km[2])
        traces.append(go.Scatter(
            x=[0.0, 0.0], y=[0.0, target_z], mode="lines+markers",
            line=dict(color=COLORS["impact"], width=2.4, dash="dot"),
            marker=dict(size=[6, 6], color=COLORS["impact"], symbol=["line-ew", "line-ew"]),
            name="Full target-axis impact parameter", showlegend=showlegend,
            legendgroup="reference", legendgrouptitle_text="Reference geometry",
            customdata=[[frame.impact_parameter_km], [frame.impact_parameter_km]],
            hovertemplate="Axis miss = %{customdata[0]:,.1f} km<extra></extra>",
            xaxis=xaxis, yaxis=yaxis,
        ))
    return traces


def generate_interactive_geometry(
    event: ReferenceEvent,
    output_path: str | Path,
    *,
    animated: bool = False,
    include_plotlyjs: bool = True,
) -> str:
    """Write a dedicated, readable interactive finite-Sun geometry view."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    jd_values = np.asarray(event.jd if animated else [event.greatest_jd], dtype=float)
    first = build_impact_plane_frame(event, float(jd_values[0]), n_azimuth=72)
    ranges = event_view_ranges(event, jd_values)

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy", "colspan": 2}, None]],
        row_heights=[0.66, 0.34], column_widths=[0.5, 0.5],
        vertical_spacing=0.14, horizontal_spacing=0.08,
        subplot_titles=(
            f"A · {first.occluder_name} tangency — solid and optical limbs",
            f"B · {first.target_name} interception — exact first-surface clipping",
            "C · Complete Earth–Moon baseline — true unwarped kilometre scale",
        ),
    )

    trace_map: list[tuple[int, int]] = []
    for view, row, col, xaxis, yaxis, legend in (
        ("occluder", 1, 1, "x", "y", True),
        ("target", 1, 2, "x2", "y2", False),
        ("system", 2, 1, "x3", "y3", False),
    ):
        for trace in plotly_view_traces(
            first, view, xaxis=xaxis, yaxis=yaxis, showlegend=legend
        ):
            fig.add_trace(trace, row=row, col=col)
            trace_map.append((row, col))

    frames = []
    slider_steps = []
    if animated and len(jd_values) > 1:
        for index, jd in enumerate(jd_values):
            state = build_impact_plane_frame(event, float(jd), n_azimuth=72)
            data = []
            for view, xaxis, yaxis, legend in (
                ("occluder", "x", "y", True),
                ("target", "x2", "y2", False),
                ("system", "x3", "y3", False),
            ):
                data.extend(plotly_view_traces(
                    state, view, xaxis=xaxis, yaxis=yaxis, showlegend=legend
                ))
            dt = jd_to_datetime(float(jd))
            frames.append(go.Frame(
                name=str(index), data=data,
                traces=list(range(len(data))),
                layout=go.Layout(title=dict(
                    text=(f"{event.definition.title} — exact interactive finite-Sun impact geometry"
                          f"<br><sub>{dt.strftime('%Y-%m-%d %H:%M:%S')} UTC · "
                          f"impact parameter {state.impact_parameter_km:,.1f} km</sub>"),
                    x=0.5,
                )),
            ))
            slider_steps.append(dict(
                method="animate", label=dt.strftime("%H:%M"),
                args=[[str(index)], dict(
                    mode="immediate", frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0),
                )],
            ))
        fig.frames = frames

    # Equal physical scale in all panels.  The bottom strip is intentionally
    # shallow because a 400,000-km baseline is being shown without distortion.
    for axis_name, view in (("xaxis", "occluder"), ("xaxis2", "target"), ("xaxis3", "system")):
        xlim, _ = ranges[view]
        getattr(fig.layout, axis_name).update(
            range=list(xlim), title="Optical X [km] — downstream →",
            gridcolor="rgba(155,176,204,0.20)", zerolinecolor="#71839a",
            showspikes=True, spikemode="across", spikesnap="cursor",
        )
    for axis_name, xref, view in (
        ("yaxis", "x", "occluder"),
        ("yaxis2", "x2", "target"),
        ("yaxis3", "x3", "system"),
    ):
        _, ylim = ranges[view]
        getattr(fig.layout, axis_name).update(
            range=list(ylim), title="Impact-plane Z [km] — toward target centre",
            gridcolor="rgba(155,176,204,0.20)", zerolinecolor="#71839a",
            scaleanchor=xref, scaleratio=1.0, constrain="domain",
            showspikes=True, spikemode="across", spikesnap="cursor",
        )

    peak_state = build_impact_plane_frame(event, event.greatest_jd, n_azimuth=72)
    metrics = geometry_metrics(peak_state)
    title_dt = jd_to_datetime(float(jd_values[0]))
    menus = []
    if animated and len(jd_values) > 1:
        menus.append(dict(
            type="buttons", direction="left", x=0.01, y=-0.075,
            showactive=False,
            buttons=[
                dict(label="▶ Play", method="animate", args=[
                    None, dict(fromcurrent=True, frame=dict(duration=170, redraw=True), transition=dict(duration=0))
                ]),
                dict(label="⏸ Pause", method="animate", args=[
                    [None], dict(mode="immediate", frame=dict(duration=0, redraw=False), transition=dict(duration=0))
                ]),
            ],
            bgcolor="#1d2838", bordercolor="#718096", font=dict(color="white"),
        ))

    fig.update_layout(
        title=dict(
            text=(f"{event.definition.title} — exact interactive finite-Sun impact geometry"
                  f"<br><sub>{title_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC · "
                  "source, occluder and target are coplanar; no ray enters an opaque body</sub>"),
            x=0.5, y=0.985, yanchor="top", font=dict(size=21),
        ),
        template="plotly_dark", paper_bgcolor=COLORS["background"], plot_bgcolor=COLORS["panel"],
        font=dict(color=COLORS["text"], family="Arial, sans-serif", size=11),
        height=1180, margin=dict(l=70, r=330, t=125, b=90),
        legend=dict(
            title=dict(text="Finite-Sun geometry key"), x=1.01, y=0.97,
            xanchor="left", yanchor="top", bgcolor="rgba(8,13,21,0.97)",
            bordercolor="#53627a", borderwidth=1, font=dict(size=10),
            tracegroupgap=5, groupclick="toggleitem",
        ),
        updatemenus=menus,
        sliders=([dict(
            active=0, x=0.14, len=0.72, y=-0.075,
            currentvalue=dict(prefix="UTC ", font=dict(size=12)),
            steps=slider_steps,
        )] if slider_steps else []),
        annotations=list(fig.layout.annotations) + [
            dict(
                x=1.01, y=0.28, xref="paper", yref="paper", xanchor="left", yanchor="top",
                showarrow=False, align="left", width=275,
                bgcolor="rgba(8,13,21,0.97)", bordercolor="#53627a", borderwidth=1, borderpad=8,
                text=(
                    f"<b>Peak physical geometry</b><br>"
                    f"Full target-axis miss: {metrics['impact_parameter_km']:,.1f} km<br>"
                    f"Target-plane umbra: {metrics['umbra_radius_at_target_plane_km']:,.1f} km<br>"
                    f"Target-plane penumbra: {metrics['penumbra_radius_at_target_plane_km']:,.1f} km<br>"
                    f"Earth–Moon centres: {metrics['earth_moon_distance_km']:,.1f} km<br>"
                    f"Sun–Earth: {metrics['sun_earth_distance_km']/1e6:.3f} million km<br><br>"
                    "<b>Reading the panels</b><br>"
                    "A resolves exact tangency at the occluder.<br>"
                    "B resolves target hits or target-plane misses.<br>"
                    "C retains the complete Earth–Moon baseline at equal physical scale.<br>"
                    "The real Sun remains one AU upstream and is not moved into the local scene."
                ),
                font=dict(color=COLORS["muted"], size=10.5),
            )
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
            "toImageButtonOptions": {"format": "png", "scale": 2, "filename": path.stem},
        },
    )
    return str(path)


__all__ = [
    "ImpactPlaneFrame",
    "build_impact_plane_frame",
    "body_outlines",
    "clip_line_to_x",
    "draw_matplotlib_view",
    "event_view_ranges",
    "generate_interactive_geometry",
    "generate_validation_figure",
    "geometry_metrics",
    "impact_plane_basis",
    "peak_impact_basis",
    "plotly_view_traces",
    "shadow_axis_world",
    "target_plane_radii",
    "transform_points",
    "view_ranges",
]
