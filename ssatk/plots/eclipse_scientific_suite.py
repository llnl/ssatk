"""Event-aware scientific eclipse products.

The approved publication renderer is specialized for the two NASA/GSFC
reference events.  This module keeps that layout for those exact reference
states while providing a backend-neutral scientific renderer for discovered,
serialized, or strict-provider events.  Every product consumes the caller's
immutable event; no renderer silently substitutes another eclipse.
"""
from __future__ import annotations

from pathlib import Path
import csv
import json
import math
import re

import numpy as np

from ssatk.compute.eclipse_reference_events import (
    EARTH_AXES_KM,
    LUNAR_2025,
    RE_KM,
    R_MOON_MEAN_KM,
    R_SUN_KM,
    SOLAR_2024,
    ReferenceEvent,
    jd_to_datetime,
)
from ssatk.compute.eclipse_raytrace import (
    bundle_penetrations,
    tangent_cross_section_paths,
    tangent_residuals,
    trace_event_rays,
)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_") or "eclipse"


def _greatest_index(event: ReferenceEvent) -> int:
    return int(np.argmin(np.abs(np.asarray(event.jd, dtype=float) - float(event.greatest_jd))))


def _contact_items(event: ReferenceEvent) -> list[tuple[str, float]]:
    return sorted(
        ((str(name), float(value)) for name, value in event.definition.contacts_jd.items()),
        key=lambda item: item[1],
    )


def _is_exact_reference_publication_event(event: ReferenceEvent) -> bool:
    return (
        str(event.metadata.get("state_source", "reference")) == "reference"
        and event.definition.key in {SOLAR_2024.key, LUNAR_2025.key}
        and not bool(event.metadata.get("dynamic_event", False))
    )


def _angular_geometry(event: ReferenceEvent, index: int) -> tuple[float, float, float, str]:
    sun = np.asarray(event.sun_km[index], dtype=float)
    moon = np.asarray(event.moon_km[index], dtype=float)
    if event.mode == "solar":
        sun_vec, occ_vec = sun, moon
        sun_radius = math.asin(np.clip(R_SUN_KM / np.linalg.norm(sun_vec), -1.0, 1.0))
        occ_radius = math.asin(np.clip(R_MOON_MEAN_KM / np.linalg.norm(occ_vec), -1.0, 1.0))
        separation = math.acos(np.clip(
            np.dot(sun_vec, occ_vec) / (np.linalg.norm(sun_vec) * np.linalg.norm(occ_vec)), -1.0, 1.0
        ))
        return sun_radius, occ_radius, separation, "Moon"
    sun_vec, occ_vec = sun - moon, -moon
    sun_radius = math.asin(np.clip(R_SUN_KM / np.linalg.norm(sun_vec), -1.0, 1.0))
    occ_radius = math.asin(np.clip(RE_KM / np.linalg.norm(occ_vec), -1.0, 1.0))
    separation = math.acos(np.clip(
        np.dot(sun_vec, occ_vec) / (np.linalg.norm(sun_vec) * np.linalg.norm(occ_vec)), -1.0, 1.0
    ))
    return sun_radius, occ_radius, separation, "Earth"


def _impact_basis(bundle):
    axis = np.asarray(bundle.axis_hat, dtype=float)
    axis /= np.linalg.norm(axis)
    occluder = np.asarray(
        bundle.moon_center_km if bundle.mode == "solar" else bundle.earth_center_km,
        dtype=float,
    )
    target = np.asarray(
        bundle.earth_center_km if bundle.mode == "solar" else bundle.moon_center_km,
        dtype=float,
    )
    transverse = target - occluder - axis * float(np.dot(target - occluder, axis))
    if np.linalg.norm(transverse) < 1.0e-9:
        reference = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        transverse = np.cross(reference, axis)
    transverse /= np.linalg.norm(transverse)
    return occluder, axis, transverse


def _project(points, origin, axis, transverse) -> np.ndarray:
    values = np.asarray(points, dtype=float) - np.asarray(origin, dtype=float)
    return np.stack([values @ axis, values @ transverse], axis=-1)


def _clip_polyline(points_2d: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    points = np.asarray(points_2d, dtype=float)
    dense_parts: list[np.ndarray] = []
    for start, stop in zip(points[:-1], points[1:]):
        t = np.linspace(0.0, 1.0, 220, endpoint=False)
        dense_parts.append(start[None, :] + (stop - start)[None, :] * t[:, None])
    dense_parts.append(points[-1:])
    dense = np.concatenate(dense_parts, axis=0)
    return dense[(dense[:, 0] >= xmin) & (dense[:, 0] <= xmax)]


def _draw_timeline(ax, event: ReferenceEvent, index: int) -> None:
    jd = np.asarray(event.jd, dtype=float)
    hours = (jd - float(event.greatest_jd)) * 24.0
    visibility = 100.0 * np.asarray(event.center_visibility, dtype=float)
    ax.plot(hours, visibility, color="#8fd9f0", lw=2.2)
    ax.fill_between(hours, visibility, 100.0, color="#29405e", alpha=0.35)
    for label, value in _contact_items(event):
        location = (value - float(event.greatest_jd)) * 24.0
        ax.axvline(location, color="#d8a85d", lw=0.8, alpha=0.65)
        ax.text(location, 102.0, label, ha="center", va="bottom", fontsize=7.4, color="#f2d49b")
    ax.scatter([hours[index]], [visibility[index]], s=48, color="#ffe27a", zorder=4)
    ax.set_title("A. Eclipse timeline", loc="left", fontweight="bold")
    ax.set_xlabel("Hours from greatest eclipse")
    ax.set_ylabel("Direct solar photosphere visible [%]")
    ax.set_ylim(-2.0, 108.0)
    ax.grid(alpha=0.25)


def _draw_apparent_discs(ax, event: ReferenceEvent, index: int) -> None:
    from matplotlib.patches import Circle

    sun_radius, occ_radius, separation, occ_name = _angular_geometry(event, index)
    scale = max(sun_radius, occ_radius, separation + occ_radius, 1.0e-12)
    sr, rr, dd = sun_radius / scale, occ_radius / scale, separation / scale
    ax.add_patch(Circle((0.0, 0.0), sr, facecolor="#ffd96a", edgecolor="#fff3c4", lw=1.6))
    ax.add_patch(Circle((dd, 0.0), rr, facecolor="#080a0f", edgecolor="#b7c1d2", lw=1.2))
    limit = max(sr, dd + rr) * 1.22
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("B. Sun and occluder at greatest eclipse", loc="left", fontweight="bold")
    ax.text(
        0.02, 0.03,
        f"Sun radius: {math.degrees(sun_radius):.5f}°\n"
        f"{occ_name} radius: {math.degrees(occ_radius):.5f}°\n"
        f"center separation: {math.degrees(separation):.5f}°\n"
        f"direct photosphere visible: {100.0 * float(event.center_visibility[index]):.5f}%",
        transform=ax.transAxes, va="bottom", ha="left", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="#101a28", ec="#52647a", alpha=0.92),
    )


def _draw_ray_geometry(ax, bundle) -> None:
    from matplotlib.patches import Circle, Ellipse

    origin, axis, transverse = _impact_basis(bundle)
    occluder = np.asarray(
        bundle.moon_center_km if bundle.mode == "solar" else bundle.earth_center_km,
        dtype=float,
    )
    target = np.asarray(
        bundle.earth_center_km if bundle.mode == "solar" else bundle.moon_center_km,
        dtype=float,
    )
    occ2 = _project(occluder, origin, axis, transverse)
    target2 = _project(target, origin, axis, transverse)
    baseline = abs(float(target2[0] - occ2[0]))
    xmin = -max(8.0 * bundle.solid_occluder_radius_km, 0.08 * baseline)
    xmax = float(target2[0]) + 1.25 * bundle.target_radius_km

    umbra = tangent_cross_section_paths(bundle, "umbra", transverse)[:2]
    penumbra = tangent_cross_section_paths(bundle, "penumbra", transverse)[:2]
    direct = _clip_polyline(_project(bundle.central.points_km, origin, axis, transverse), xmin, xmax)
    if len(direct):
        ax.plot(direct[:, 0], direct[:, 1], color="#ffe27a", lw=2.1, label="direct light / first-surface stop")
    for paths, color, label in (
        (umbra, "#ef5d68", "umbra / antumbra tangents"),
        (penumbra, "#63b3ff", "penumbra tangents"),
    ):
        for number, path in enumerate(paths):
            line = _clip_polyline(_project(path.points_km, origin, axis, transverse), xmin, xmax)
            if not len(line):
                continue
            ax.plot(line[:, 0], line[:, 1], color=color, lw=1.25, label=label if number == 0 else None)
            ax.scatter([line[-1, 0]], [line[-1, 1]], color=color,
                       marker="o" if path.target_hit else "x", s=22, zorder=4)

    if bundle.mode == "solar":
        ax.add_patch(Circle(tuple(occ2), bundle.solid_occluder_radius_km,
                            facecolor="#777d89", edgecolor="white", lw=0.8, zorder=3))
        ax.add_patch(Ellipse(tuple(target2), 2 * EARTH_AXES_KM[0], 2 * EARTH_AXES_KM[2],
                             facecolor="#194f76", edgecolor="#92d7ff", lw=1.0, zorder=2))
    else:
        ax.add_patch(Ellipse(tuple(occ2), 2 * EARTH_AXES_KM[0], 2 * EARTH_AXES_KM[2],
                             facecolor="#194f76", edgecolor="#92d7ff", lw=1.0, zorder=3))
        ax.add_patch(Circle(tuple(target2), R_MOON_MEAN_KM,
                            facecolor="#777d89", edgecolor="white", lw=0.8, zorder=2))

    ax.axhline(0.0, color="#bd72d8", lw=0.8, ls="--", alpha=0.75, label="blocked-light axis")
    span = max(
        bundle.target_radius_km * 1.5,
        abs(float(target2[1])) + bundle.target_radius_km * 1.3,
        bundle.solid_occluder_radius_km * 2.2,
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-span, span)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Downstream distance in exact impact plane [km]")
    ax.set_ylabel("Complete target-axis offset [km]")
    ax.set_title("C. How the eclipse shadow forms", loc="left", fontweight="bold")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.19), ncol=2, fontsize=7.2, frameon=True)


def _draw_distance_locator(ax, bundle) -> None:
    sun = np.asarray(bundle.sun_center_km, dtype=float)
    earth = np.asarray(bundle.earth_center_km, dtype=float)
    moon = np.asarray(bundle.moon_center_km, dtype=float)
    sun_earth = float(np.linalg.norm(sun - earth))
    earth_moon = float(np.linalg.norm(moon - earth))
    ax.set_axis_off()
    ax.set_title("D. True Sun–Earth–Moon distances", loc="left", fontweight="bold")
    ax.text(0.02, 0.77, "Sun — Earth", transform=ax.transAxes, fontweight="bold", fontsize=10)
    ax.text(0.02, 0.61, f"{sun_earth:,.3f} km  ({sun_earth / 149_597_870.7:.9f} AU)", transform=ax.transAxes)
    ax.text(0.02, 0.40, "Earth — Moon", transform=ax.transAxes, fontweight="bold", fontsize=10)
    ax.text(0.02, 0.24, f"{earth_moon:,.3f} km", transform=ax.transAxes)
    ax.text(
        0.02, 0.03,
        "The 3-D counterpart uses separate floating origins for AU and kilometre views. No physical body is moved closer, enlarged, or nonlinearly warped.",
        transform=ax.transAxes, fontsize=8.1, color="#aebcd0", wrap=True,
    )


def _draw_contacts(ax, event: ReferenceEvent) -> None:
    ax.set_axis_off()
    ax.set_title("E. Contacts and data sources", loc="left", fontweight="bold")
    contacts = [f"{name:>4}  {jd_to_datetime(value).strftime('%Y-%m-%d %H:%M:%S')} UTC"
                for name, value in _contact_items(event)]
    metadata = event.metadata
    provenance = [
        f"state: {metadata.get('state_source', event.backend)}",
        f"ephemeris: {metadata.get('ephemeris_backend', event.backend)}",
        f"frame: {metadata.get('frame_backend', event.frame)}",
        f"classification: {metadata.get('event_classification', event.definition.title)}",
    ]
    ax.text(0.02, 0.94, "\n".join(contacts), transform=ax.transAxes, va="top", family="monospace", fontsize=8.3)
    ax.text(0.55, 0.94, "\n".join(provenance), transform=ax.transAxes, va="top", fontsize=8.3)


def _draw_validation(ax, event: ReferenceEvent, bundle, validation) -> None:
    ax.set_axis_off()
    ax.set_title("F. Accuracy checks", loc="left", fontweight="bold")
    penetration = bundle_penetrations(bundle)
    residual = tangent_residuals(bundle)
    radius_errors = [abs(float(value)) for key, value in residual.items() if "radius_error" in key]
    orthogonality = [abs(float(value)) for key, value in residual.items() if "orthogonality" in key]
    rows = [
        ("validation", "PASS" if validation.passed else "FAIL"),
        ("requested minimum states", str(event.metadata.get("requested_minimum_state_count", "not recorded"))),
        ("resolved solver states", str(len(event.jd))),
        ("Earth-interior ray penetrations", str(penetration["earth"])),
        ("Moon-interior ray penetrations", str(penetration["moon"])),
        ("max tangent radius error [km]", f"{max(radius_errors or [0.0]):.3e}"),
        ("max tangent orthogonality [km]", f"{max(orthogonality or [0.0]):.3e}"),
    ]
    y = 0.92
    for label, value in rows:
        ax.text(0.02, y, label, transform=ax.transAxes, color="#aebcd0", fontsize=8.5)
        ax.text(0.98, y, value, transform=ax.transAxes, ha="right", fontsize=8.5, fontweight="bold")
        y -= 0.12


def _render_generic_frame(event: ReferenceEvent, index: int, *, width: int, height: int, dpi: int) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    from ssatk.eclipse_api_impl import validate_event

    index = int(index)
    jd = float(event.jd[index])
    bundle = trace_event_rays(event, jd, n_azimuth=32)
    validation = validate_event(event, ray_azimuth=24)
    with plt.rc_context({
        "figure.facecolor": "#02050b", "axes.facecolor": "#07101c",
        "axes.edgecolor": "#52647a", "axes.labelcolor": "#e7edf5",
        "xtick.color": "#c7d2df", "ytick.color": "#c7d2df",
        "text.color": "#f3f7fb", "font.size": 9.2,
    }):
        fig = plt.Figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="#02050b")
        canvas = FigureCanvasAgg(fig)
        grid = fig.add_gridspec(
            3, 2, height_ratios=[1.0, 1.45, 0.95], hspace=0.42, wspace=0.22,
            left=0.055, right=0.97, top=0.89, bottom=0.075,
        )
        axes = [fig.add_subplot(grid[row, col]) for row in range(3) for col in range(2)]
        _draw_timeline(axes[0], event, index)
        _draw_apparent_discs(axes[1], event, index)
        _draw_ray_geometry(axes[2], bundle)
        _draw_distance_locator(axes[3], bundle)
        _draw_contacts(axes[4], event)
        _draw_validation(axes[5], event, bundle, validation)
        fig.suptitle(f"{event.definition.title} — Scientific Summary",
                     fontsize=18.5, fontweight="bold", y=0.975)
        fig.text(
            0.5, 0.935,
            f"State {jd_to_datetime(jd).strftime('%Y-%m-%d %H:%M:%S')} UTC  •  "
            f"backend {event.metadata.get('state_source', event.backend)}  •  "
            "contact geometry uses uniform finite-disc overlap; surface irradiance may use limb darkening",
            ha="center", color="#aebcd0", fontsize=9.3,
        )
        fig.text(
            0.5, 0.018,
            "Yellow light ends at the first opaque surface. Magenta is blocked-light geometry, not transmitted sunlight. The Sun remains at its real distance in the interactive counterpart.",
            ha="center", color="#aebcd0", fontsize=8.6,
        )
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        return np.asarray(rgba[..., :3], dtype=np.uint8).copy()


class GenericScientificFrameRenderer:
    """Backend-neutral frame renderer with the same contract as FrameRenderer."""

    def __init__(self, event: ReferenceEvent, *, width: int = 1600, height: int = 960,
                 dpi: int = 100, photometry: str = "quadratic-visible") -> None:
        self.event = event
        self.width = int(width)
        self.height = int(height)
        self.dpi = int(dpi)
        self.photometry = str(photometry)

    def render(self, index: int) -> np.ndarray:
        return _render_generic_frame(
            self.event, int(index), width=self.width, height=self.height, dpi=self.dpi
        )


def renderer_for_event(event: ReferenceEvent, *, width: int = 1600, height: int = 960,
                       dpi: int = 100, photometry: str = "quadratic-visible"):
    """Return the approved reference renderer or the generic event renderer."""
    if _is_exact_reference_publication_event(event):
        from ssatk.plots.eclipse_animation import FrameRenderer
        return FrameRenderer(event, width=int(width), height=int(height), dpi=int(dpi))
    return GenericScientificFrameRenderer(
        event, width=width, height=height, dpi=dpi, photometry=photometry
    )


def generate_event_scientific_summary(
    event: ReferenceEvent,
    output_path: str | Path,
    *,
    photometry: str = "quadratic-visible",
    width: int = 2400,
    height: int = 1440,
    dpi: int = 120,
) -> str:
    """Write a selected-event PNG without substituting another eclipse."""
    from matplotlib.figure import Figure

    from ssatk.plots.plotutils import figsave

    renderer = renderer_for_event(event, width=width, height=height, dpi=dpi, photometry=photometry)
    frame = renderer.render(_greatest_index(event))
    figure = Figure(figsize=(width / dpi, height / dpi), dpi=dpi, frameon=False)
    figure.figimage(np.asarray(frame, dtype=np.uint8))
    output = figsave(figure, output_path, dpi=dpi)
    if output is None:
        raise OSError(f"Could not save eclipse summary to {output_path}")
    return output


def write_event_timeline_csv(
    event: ReferenceEvent,
    output_path: str | Path,
    *,
    photometry: str = "quadratic-visible",
) -> str:
    """Write a GUI-friendly timeline table for the selected event."""
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "index", "jd_utc", "utc", "center_visibility", "center_obscuration",
            "separation_deg", "photometry_mode",
        ])
        for index, (jd, visible, separation) in enumerate(zip(
            event.jd, event.center_visibility, event.separation_deg
        )):
            writer.writerow([
                index, f"{float(jd):.12f}", jd_to_datetime(float(jd)).isoformat(),
                f"{float(visible):.12g}", f"{1.0 - float(visible):.12g}",
                f"{float(separation):.12g}", photometry,
            ])
    return str(output)


def generate_event_scientific_suite(
    event: ReferenceEvent,
    output_dir: str | Path,
    *,
    quality: str = "ultra",
    animate: bool = False,
    playback_seconds: float | None = None,
    photometry: str = "quadratic-visible",
) -> dict[str, object]:
    """Generate a selected-event summary, table, state, validation, and 3-D view."""
    from ssatk.eclipse_api_impl import event_to_dict, validate_event
    from ssatk.plots.eclipse_cinematic_light_3d import generate_cinematic_view

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    slug = _slug(event.definition.key)
    summary = generate_event_scientific_summary(
        event, output / f"{slug}_scientific_summary.png", photometry=photometry
    )
    timeline = write_event_timeline_csv(
        event, output / f"{slug}_timeline.csv", photometry=photometry
    )
    event_path = output / f"{slug}_event_state.json"
    event_path.write_text(json.dumps(event_to_dict(event), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    validation = validate_event(event)
    validation_path = output / f"{slug}_validation.json"
    validation_path.write_text(json.dumps(validation.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    interactive_path = output / f"{slug}_{'full_animation' if animate else 'greatest'}_interactive_3d.html"
    interactive = generate_cinematic_view(
        event, interactive_path, animate=bool(animate), quality=quality,
        n_frames=len(event.jd), backend=str(event.metadata.get("state_source", "reference")),
        solar_scope=str(event.metadata.get("solar_scope", "global")),
        playback_seconds=playback_seconds, photometry=photometry,
    )
    manifest = {
        "event_key": event.definition.key,
        "event_title": event.definition.title,
        "requested_minimum_state_count": event.metadata.get("requested_minimum_state_count"),
        "resolved_state_count": int(len(event.jd)),
        "sample_count_semantics": event.metadata.get(
            "sample_count_semantics", "minimum; exact contact states are inserted"
        ),
        "summary_renderer": (
            "approved reference publication panel"
            if _is_exact_reference_publication_event(event)
            else "backend-neutral selected-event scientific panel"
        ),
        "validation_passed": bool(validation.passed),
        "products": {
            "scientific_summary_png": Path(summary).name,
            "interactive_3d_html": Path(interactive).name,
            "timeline_csv": Path(timeline).name,
            "event_state_json": event_path.name,
            "validation_json": validation_path.name,
        },
    }
    manifest_path = output / f"{slug}_scientific_suite_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {
        "scientific_summary_png": summary,
        "interactive_3d_html": interactive,
        "timeline_csv": timeline,
        "event_state_json": str(event_path),
        "validation_json": str(validation_path),
        "manifest_json": str(manifest_path),
        "requested_minimum_state_count": event.metadata.get("requested_minimum_state_count"),
        "resolved_state_count": int(len(event.jd)),
    }


# Descriptive alias retained for applications that adopted the pre-release name.
generate_selected_scientific_suite = generate_event_scientific_suite


__all__ = [
    "GenericScientificFrameRenderer", "renderer_for_event",
    "generate_event_scientific_summary", "generate_event_scientific_suite",
    "generate_selected_scientific_suite", "write_event_timeline_csv",
]
