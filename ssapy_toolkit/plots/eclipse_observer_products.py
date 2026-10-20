"""Observer-specific solar-eclipse figures and machine-readable products.

This module turns the custom April 8, 2024 figure into a reusable Toolkit API.
The supplied event object is never replaced.  Event-wide visibility maps are
available when the event carries a matching precomputed/validated map product;
otherwise the local observer products still render and the full-map request
fails explicitly instead of substituting a different eclipse.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo
import csv
import json
import math
import re

import numpy as np

from ssapy_toolkit.coordinates.eclipse_observer_geometry import ObserverConfig, observer_state, refine_solar_contacts
from ssapy_toolkit.compute.eclipse_reference_events import SOLAR_2024, R_SUN_KM, R_MOON_MEAN_KM
# ``event_positions_gcrf`` belongs to eclipse_state; keep the import separate
# to avoid obscuring the state-provider ownership boundary.
from ssapy_toolkit.compute.eclipse_state import event_positions_gcrf
from ssapy_toolkit.compute.eclipse_reference_events import itrf_surface_point, jd_to_datetime
from ssapy_toolkit.compute.eclipse_raytrace import trace_event_rays
from ssapy_toolkit.plots.eclipse_rendering import (
    SOLAR_TOTALITY_PERCENT_BAND_COLORS,
    SOLAR_TOTALITY_PERCENT_BAND_EDGES,
    _solar_granulation_field,
    _split_longitudes,
    draw_north_indicator,
    draw_ray_diagram,
    draw_solar_combined_visibility_key,
    draw_solar_map,
    draw_solar_partial_visibility_map,
    render_earth_disk,
    solar_central_path_cached,
    solar_totality_band_polygons,
    solar_umbra_corridor_cached,
)


def _unit(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= np.finfo(float).tiny:
        raise ValueError("vector must be nonzero")
    return vector / norm


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_") or "observer"


def _observer_slug(observer: ObserverConfig) -> str:
    lat = f"{abs(observer.latitude_deg):.6f}{'N' if observer.latitude_deg >= 0 else 'S'}"
    lon = f"{abs(observer.longitude_east_deg):.6f}{'E' if observer.longitude_east_deg >= 0 else 'W'}"
    return f"{lat}_{lon}".replace(".", "p")


def _format_clock(dt) -> str:
    """Portable 12-hour clock without platform-specific %-I directives."""
    return dt.strftime("%I:%M:%S %p %Z").lstrip("0")


@dataclass(frozen=True)
class ObserverCircumstances:
    observer: ObserverConfig
    contacts_jd_utc: dict[str, float]
    maximum_jd_utc: float
    totality_duration_s: float
    classification: str
    maximum_state: object

    def to_dict(self, *, timezone_name: str = "UTC") -> dict[str, object]:
        tz = ZoneInfo(timezone_name)
        contacts = {
            key: {
                "jd_utc": float(value),
                "utc": jd_to_datetime(float(value)).isoformat(),
                "local": jd_to_datetime(float(value)).astimezone(tz).isoformat(),
            }
            for key, value in self.contacts_jd_utc.items()
        }
        return {
            "$schema": "ssapy-toolkit.eclipse.observer-circumstances/2.2",
            "observer": self.observer.to_dict(),
            "classification": self.classification,
            "contacts": contacts,
            "maximum_jd_utc": float(self.maximum_jd_utc),
            "totality_duration_s": float(self.totality_duration_s),
            "maximum_state": self.maximum_state.to_dict(),
            "timezone": timezone_name,
        }


def solar_observer_circumstances(
    event,
    observer: ObserverConfig,
    *,
    reference_contacts: dict[str, float] | None = None,
    limb_profile=None,
) -> ObserverCircumstances:
    """Compute local solar contacts and maximum for one immutable event."""

    if event.mode != "solar":
        raise ValueError("solar_observer_circumstances requires a solar event")
    contacts = refine_solar_contacts(
        event,
        observer,
        limb_profile=limb_profile,
        reference_contacts=reference_contacts or event.metadata.get("observer_contacts_jd"),
    )
    maximum = float(contacts["MAX"])
    state = observer_state(event, maximum, observer, limb_profile=limb_profile)
    totality_s = 0.0
    if "C2" in contacts and "C3" in contacts:
        totality_s = max(0.0, (float(contacts["C3"]) - float(contacts["C2"])) * 86400.0)
    classification = "total" if totality_s > 0.0 else "partial"
    return ObserverCircumstances(observer, contacts, maximum, totality_s, classification, state)


def render_observer_apparent_sky(
    event,
    jd_utc: float,
    observer: ObserverConfig,
    *,
    size: int = 760,
    extent_solar_radii: float = 2.1,
) -> tuple[np.ndarray, dict[str, float]]:
    """Render a north-up topocentric Sun/Moon view for a solar eclipse."""

    if event.mode != "solar":
        raise ValueError("observer apparent-sky rendering requires a solar event")
    sun, moon = event_positions_gcrf(event, float(jd_utc))
    state = observer_state(event, float(jd_utc), observer, sun_gcrf_km=sun, moon_gcrf_km=moon)
    to_sun = np.asarray(sun, dtype=float) - state.observer_gcrf_km
    to_moon = np.asarray(moon, dtype=float) - state.observer_gcrf_km
    view = _unit(to_sun)
    up = state.north_gcrf - view * float(np.dot(state.north_gcrf, view))
    if np.linalg.norm(up) < 1.0e-10:
        up = state.up_gcrf - view * float(np.dot(state.up_gcrf, view))
    up = _unit(up)
    right = _unit(np.cross(up, view))
    up = _unit(np.cross(view, right))

    d_sun = float(np.linalg.norm(to_sun))
    d_moon = float(np.linalg.norm(to_moon))
    a_sun = math.asin(np.clip(R_SUN_KM / d_sun, 0.0, 1.0))
    a_moon = math.asin(np.clip(state.lunar_limb_radius_km / d_moon, 0.0, 1.0))
    moon_hat = to_moon / d_moon
    denom = max(float(np.dot(moon_hat, view)), 1.0e-12)
    off_x = math.atan2(float(np.dot(moon_hat, right)), denom) / a_sun
    off_y = math.atan2(float(np.dot(moon_hat, up)), denom) / a_sun
    moon_r = a_moon / a_sun

    extent = float(extent_solar_radii)
    q = np.linspace(-extent, extent, int(size))
    X, Y = np.meshgrid(q, q)
    R = np.hypot(X, Y)
    sun_mask = R <= 1.0
    moon_mask = np.hypot(X - off_x, Y - off_y) <= moon_r
    z = np.sqrt(np.clip(1.0 - R * R, 0.0, 1.0))
    limb = 0.47 + 0.53 * z**0.72
    granulation = _solar_granulation_field(int(size))
    brightness = np.clip(limb * (1.0 + granulation), 0.24, 1.08)
    rgb = np.zeros((*X.shape, 3), dtype=float)
    rgb[..., 0] = brightness
    rgb[..., 1] = 0.79 * brightness**1.03
    rgb[..., 2] = 0.39 * brightness**1.12
    rgb[~sun_mask] = 0.0
    rgb[moon_mask] = np.array([0.0015, 0.0018, 0.0025])

    visible = float(state.photosphere_visible)
    if visible < 0.035:
        rr = np.maximum(R, 1.0e-6)
        corona = np.exp(-3.4 * np.clip(rr - 0.96, 0.0, None)) / rr**0.75
        corona *= rr >= 0.94
        angle = np.arctan2(Y, X)
        corona *= 0.58 + 0.42 * (0.5 + 0.5 * np.cos(4 * angle + 0.7 * np.sin(3 * angle)))
        corona = np.clip(corona, 0.0, 1.35)
        outside = ~sun_mask
        rgb[outside] += corona[outside, None] * np.array([0.72, 0.80, 1.0])
        rgb[moon_mask] = 0.0
    alpha = np.clip(np.max(rgb, axis=-1) * 1.6, 0.0, 1.0)
    alpha[sun_mask | moon_mask] = 1.0
    rgba = np.zeros((*X.shape, 4), dtype=np.uint8)
    rgba[..., :3] = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgba[..., 3] = np.rint(alpha * 255.0).astype(np.uint8)
    return rgba, {
        "photosphere_visible": visible,
        "moon_to_sun_radius_ratio": float(moon_r),
        "moon_offset_x_solar_radii": float(off_x),
        "moon_offset_y_solar_radii": float(off_y),
        "sun_angular_radius_deg": math.degrees(a_sun),
        "moon_angular_radius_deg": math.degrees(a_moon),
    }


def _require_solar_2024_map(event) -> None:
    if event.definition.key != SOLAR_2024.key:
        raise NotImplementedError(
            "the event-wide obscuration and totality-duration raster is currently "
            "validated only for the 8 April 2024 reference eclipse; local observer "
            "products remain available for other solar events"
        )


def _add_totality_regions(ax) -> None:
    from matplotlib.patches import Polygon

    corridor = solar_umbra_corridor_cached()
    colors = tuple(SOLAR_TOTALITY_PERCENT_BAND_COLORS)
    edges = np.asarray(SOLAR_TOTALITY_PERCENT_BAND_EDGES, dtype=float)
    if len(corridor["jd"]):
        outer = np.column_stack([
            np.concatenate([corridor["left_lon"], corridor["right_lon"][::-1]]),
            np.concatenate([corridor["left_lat"], corridor["right_lat"][::-1]]),
        ])
        ax.add_patch(Polygon(outer, closed=True, facecolor=colors[0], edgecolor="none", alpha=0.97, zorder=4.0))
        for index, threshold in enumerate(edges[1:-1], start=1):
            for polygon in solar_totality_band_polygons(float(threshold)):
                ax.add_patch(Polygon(
                    polygon, closed=True, facecolor=colors[index], edgecolor="none",
                    alpha=0.98, zorder=4.0 + index * 0.02,
                ))
                ax.plot(polygon[:, 0], polygon[:, 1], color="#08121d", lw=0.44, alpha=0.85, zorder=4.7)
        ax.plot(corridor["left_lon"], corridor["left_lat"], color="#d7f6ff", lw=1.25, zorder=5.3)
        ax.plot(corridor["right_lon"], corridor["right_lat"], color="#d7f6ff", lw=1.25, zorder=5.3)
    central_lon, central_lat = solar_central_path_cached()
    for lon, lat in _split_longitudes(central_lon, central_lat):
        ax.plot(lon, lat, color="#02070d", lw=3.0, zorder=5.5, solid_capstyle="round")
        ax.plot(lon, lat, color="white", lw=1.05, zorder=5.7, solid_capstyle="round")


def _mark_observer(ax, observer: ObserverConfig, totality_s: float, *, compact: bool = False) -> None:
    ax.scatter(
        [observer.longitude_east_deg], [observer.latitude_deg], marker="*",
        s=92 if compact else 150, facecolor="#fff3a2", edgecolor="#132337",
        linewidth=1.15, zorder=20,
    )
    duration = f"Totality {totality_s / 60.0:.2f} min" if totality_s > 0 else "Partial eclipse"
    label = (
        f"{observer.name}\n{abs(observer.latitude_deg):.5f}° "
        f"{'N' if observer.latitude_deg >= 0 else 'S'}, "
        f"{abs(observer.longitude_east_deg):.5f}° "
        f"{'E' if observer.longitude_east_deg >= 0 else 'W'}\n{duration}"
    )
    ax.annotate(
        label, xy=(observer.longitude_east_deg, observer.latitude_deg),
        xytext=(10 if compact else 13, 8 if compact else 10), textcoords="offset points",
        ha="left", va="bottom", color="white", fontsize=6.3 if compact else 8.0,
        fontweight="semibold", zorder=22,
        bbox=dict(boxstyle="round,pad=0.28", fc="#101b29", ec="#f2de88", alpha=0.95),
        arrowprops=dict(arrowstyle="-", color="#f2de88", lw=0.8),
    )


def draw_full_solar_eclipse_map(ax, key_ax, event, observer: ObserverConfig, *, totality_s: float, compact: bool = False) -> None:
    """Draw event-wide partial visibility plus the totality corridor and site."""

    _require_solar_2024_map(event)
    draw_solar_partial_visibility_map(ax, map_extent=(-178.0, 20.0, -8.0, 84.0), show_magnitude_contours=True)
    _add_totality_regions(ax)
    _mark_observer(ax, observer, totality_s, compact=compact)
    ax.set_title(
        "Full eclipse map\nMaximum Sun coverage, path of totality, and observer location",
        color="white", fontsize=9.4 if compact else 13.0, pad=5, fontweight="semibold",
    )
    draw_solar_combined_visibility_key(key_ax)
    key_ax.scatter([10.25], [1.74], marker="*", s=55 if compact else 85,
                   facecolor="#fff3a2", edgecolor="#132337", linewidth=0.8, zorder=10)
    key_ax.text(10.58, 1.74, "Requested observer", color="#f2f6fb",
                fontsize=5.7 if compact else 7.0, ha="left", va="center")


def _draw_observer_timeline(ax, event, observer: ObserverConfig, contacts: dict[str, float]) -> None:
    jd = np.asarray(event.jd, dtype=float)
    states = [observer_state(event, float(value), observer) for value in jd]
    visible = 100.0 * np.asarray([state.photosphere_visible for state in states])
    altitude = np.asarray([state.sun_altitude_apparent_deg for state in states])
    hours = (jd - float(contacts["MAX"])) * 24.0
    ax.plot(hours, visible, color="#8fd9f0", lw=2.0, label="Sun visible")
    ax.set_xlabel("Hours from local maximum")
    ax.set_ylabel("Solar photosphere visible [%]")
    ax.set_ylim(-2, 104)
    ax.grid(alpha=0.25)
    twin = ax.twinx()
    twin.plot(hours, altitude, color="#f2b75b", lw=1.35, alpha=0.85, label="Sun altitude")
    twin.set_ylabel("Apparent Sun altitude [deg]", color="#f2b75b")
    twin.tick_params(colors="#f2b75b")
    for label, value in sorted(contacts.items(), key=lambda item: item[1]):
        x = (float(value) - float(contacts["MAX"])) * 24.0
        ax.axvline(x, color="#d6b16e", lw=0.75, alpha=0.7)
        ax.text(x, 105.0, label, ha="center", va="bottom", fontsize=7.0, color="#f2d49b")


def generate_solar_observer_products(
    event,
    observer: ObserverConfig,
    output_dir: str | Path,
    *,
    timezone_name: str = "UTC",
    include_interactive: bool = True,
    quality: str = "ultra",
) -> dict[str, object]:
    """Generate a complete observer-specific scientific product set."""

    if event.mode != "solar":
        raise ValueError("observer product suite requires a solar event")
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from PIL import Image

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    circumstances = solar_observer_circumstances(event, observer)
    max_jd = circumstances.maximum_jd_utc
    totality_s = circumstances.totality_duration_s
    state = circumstances.maximum_state
    sky, apparent = render_observer_apparent_sky(event, max_jd, observer, size=760)
    bundle = trace_event_rays(event, max_jd, n_azimuth=48)
    tz = ZoneInfo(timezone_name)
    local_max = jd_to_datetime(max_jd).astimezone(tz)
    dt_utc = jd_to_datetime(max_jd)
    date_label = f"{dt_utc.strftime('%B')} {dt_utc.day}, {dt_utc.year}"
    stem = f"{_slug(event.definition.key)}_{_observer_slug(observer)}"

    style = {
        "figure.facecolor": "#010207", "axes.facecolor": "#02040a",
        "axes.edgecolor": "#46556a", "axes.labelcolor": "#e7edf5",
        "xtick.color": "#c7d2df", "ytick.color": "#c7d2df",
        "text.color": "#f3f7fb", "font.size": 9.0,
    }
    with plt.rc_context(style):
        fig = plt.figure(figsize=(20, 15.4), dpi=120, facecolor="#010207")
        grid = fig.add_gridspec(
            3, 24, height_ratios=[4.15, 2.15, 5.15],
            left=0.045, right=0.985, bottom=0.125, top=0.915,
            hspace=0.40, wspace=0.55,
        )
        map_spec = grid[0, :].subgridspec(2, 1, height_ratios=[0.79, 0.21], hspace=0.045)
        ax_map = fig.add_subplot(map_spec[0, 0]); ax_map_key = fig.add_subplot(map_spec[1, 0])
        try:
            draw_full_solar_eclipse_map(ax_map, ax_map_key, event, observer, totality_s=totality_s, compact=True)
        except NotImplementedError:
            ax_map.set_axis_off(); ax_map_key.set_axis_off()
            ax_map.text(0.5, 0.5, "Event-wide raster unavailable for this event\nLocal observer geometry remains valid.",
                        ha="center", va="center", fontsize=14)

        ax_ray = fig.add_subplot(grid[1, :])
        draw_ray_diagram(ax_ray, bundle)
        ax_ray.set_title("How the Moon's shadow reaches Earth\nFinite-Sun light paths at local maximum",
                         color="white", fontsize=9.4, pad=5, fontweight="semibold")

        ax_earth = fig.add_subplot(grid[2, 0:4]); ax_sky = fig.add_subplot(grid[2, 4:8])
        local_spec = grid[2, 8:19].subgridspec(2, 1, height_ratios=[0.77, 0.23], hspace=0.045)
        ax_local = fig.add_subplot(local_spec[0, 0]); ax_local_key = fig.add_subplot(local_spec[1, 0])
        ax_curve = fig.add_subplot(grid[2, 20:24])

        view = _unit(itrf_surface_point(observer.latitude_deg, observer.longitude_east_deg, observer.elevation_m / 1000.0))
        earth = render_earth_disk(max_jd, size=760, view_hat=view, supersample=2)
        ax_earth.imshow(earth, interpolation="lanczos"); ax_earth.axis("off"); draw_north_indicator(ax_earth)
        center = (earth.shape[1] - 1) / 2.0
        ax_earth.scatter([center], [center], marker="*", s=72, facecolor="#fff1a8", edgecolor="#172638", linewidth=0.8)
        ax_earth.set_title("Earth at local maximum\nObserver-centered eclipse-shadow view",
                           color="white", fontsize=8.8, pad=5, fontweight="semibold")

        ax_sky.imshow(sky, interpolation="lanczos"); ax_sky.axis("off"); draw_north_indicator(ax_sky, label="N")
        ax_sky.set_title(
            f"Sky from the requested location\n{circumstances.classification.title()} eclipse • {_format_clock(local_max)} • Sun {state.sun_altitude_apparent_deg:.2f}° high",
            color="white", fontsize=8.8, pad=5, fontweight="semibold",
        )
        ax_sky.text(0.5, 0.035,
                    f"Moon/Sun radius ratio {apparent['moon_to_sun_radius_ratio']:.4f} • photosphere visible {100*apparent['photosphere_visible']:.5f}%",
                    transform=ax_sky.transAxes, ha="center", va="bottom", color="#dce7f2", fontsize=6.9,
                    bbox=dict(boxstyle="round,pad=0.26", fc="#07101a", ec="#52637a", alpha=0.88))

        if event.definition.key == SOLAR_2024.key:
            draw_solar_map(ax_local, max_jd, footprint_azimuth=300, key_ax=ax_local_key,
                           key_layout="horizontal", map_extent=(-104.5, -92.0, 26.0, 37.0),
                           show_penumbra=False, show_umbra=True, show_key=True)
            _mark_observer(ax_local, observer, totality_s, compact=True)
            ax_local.set_title("Observer inside the totality corridor\nLocal duration bands and instantaneous umbra",
                               color="white", fontsize=9.1, pad=5, fontweight="semibold")
        else:
            ax_local.set_axis_off(); ax_local_key.set_axis_off()
            ax_local.text(0.5, 0.5, "Local map not yet available for this discovered event",
                          ha="center", va="center")
        _draw_observer_timeline(ax_curve, event, observer, circumstances.contacts_jd_utc)
        ax_curve.set_title("Eclipse at the requested location\nSun visibility and altitude through local contacts",
                           color="white", fontsize=8.55, pad=5, fontweight="semibold")

        title = f"{circumstances.classification.title()} Solar Eclipse — {date_label}"
        fig.suptitle(title, color="white", fontsize=19.0, fontweight="bold", y=0.984)
        fig.text(0.5, 0.953,
                 f"Observer scientific summary • {observer.latitude_deg:.8f}°, {observer.longitude_east_deg:.8f}° • local maximum {local_max.isoformat()} • WGS-84",
                 ha="center", va="top", color="#aebcd0", fontsize=9.25)
        contacts_text = " • ".join(
            f"{key} {_format_clock(jd_to_datetime(value).astimezone(tz)).rsplit(' ', 1)[0]}"
            for key, value in sorted(circumstances.contacts_jd_utc.items(), key=lambda item: item[1])
        )
        if totality_s > 0:
            contacts_text += f" • totality {totality_s / 60.0:.2f} min ({totality_s:.1f} s)"
        fig.text(0.5, 0.038, contacts_text, color="#d7e1ec", fontsize=8.25, ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.36", facecolor="#0b111d", edgecolor="#34435a", alpha=0.95))
        fig.text(0.5, 0.011,
                 "Apparent altitude includes the selected refraction model. Terrain obstruction is included only when the observer has a horizon profile.",
                 color="#8292a8", fontsize=7.25, ha="center", va="bottom")
        summary_path = output / f"{stem}_scientific_summary.png"
        fig.savefig(summary_path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)

    # Standalone full map.
    full_map_path = None
    if event.definition.key == SOLAR_2024.key:
        with plt.rc_context(style):
            fig = plt.figure(figsize=(17.5, 10.5), dpi=140, facecolor="#010207")
            gs = fig.add_gridspec(2, 12, height_ratios=[5.8, 1.35], left=0.045, right=0.98,
                                  bottom=0.08, top=0.90, hspace=0.08, wspace=0.25)
            ax = fig.add_subplot(gs[0, :]); key_ax = fig.add_subplot(gs[1, :])
            draw_full_solar_eclipse_map(ax, key_ax, event, observer, totality_s=totality_s)
            ax.set_title(f"{date_label} Solar Eclipse — Full Visibility Map\nObserver: {observer.latitude_deg:.8f}°, {observer.longitude_east_deg:.8f}°",
                         color="white", fontsize=15.5, pad=8, fontweight="bold")
            full_map_path = output / f"{stem}_full_eclipse_map.png"
            fig.savefig(full_map_path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
            plt.close(fig)

    local_map_path = None
    if event.definition.key == SOLAR_2024.key:
        with plt.rc_context(style):
            fig = plt.figure(figsize=(15.5, 9.4), dpi=140, facecolor="#010207")
            gs = fig.add_gridspec(2, 1, height_ratios=[5.2, 1.25], left=0.055, right=0.98,
                                  bottom=0.08, top=0.90, hspace=0.07)
            ax = fig.add_subplot(gs[0, 0]); key_ax = fig.add_subplot(gs[1, 0])
            draw_solar_map(ax, max_jd, footprint_azimuth=360, key_ax=key_ax,
                           key_layout="horizontal", map_extent=(-104.5, -92.0, 26.0, 37.0),
                           show_penumbra=False, show_umbra=True, show_key=True)
            _mark_observer(ax, observer, totality_s, compact=False)
            ax.set_title(
                f"{date_label} — Observer Location Inside the Totality Corridor\n"
                f"{observer.latitude_deg:.8f}°, {observer.longitude_east_deg:.8f}°",
                color="white", fontsize=14.5, pad=8, fontweight="bold",
            )
            local_map_path = output / f"{stem}_local_totality_map.png"
            fig.savefig(local_map_path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
            plt.close(fig)

    contacts_path = output / f"{stem}_contacts.txt"
    lines = [
        f"Observer: {observer.name}",
        f"Latitude: {observer.latitude_deg:.12f} deg",
        f"Longitude east: {observer.longitude_east_deg:.12f} deg",
        f"Date: {date_label}",
        f"Classification: {circumstances.classification}",
    ]
    for key, value in sorted(circumstances.contacts_jd_utc.items(), key=lambda item: item[1]):
        local_dt = jd_to_datetime(value).astimezone(tz)
        lines.append(f"{key}: {_format_clock(local_dt)} | {jd_to_datetime(value).isoformat()}")
    lines.append(f"Totality duration: {totality_s:.3f} s")
    contacts_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    sky_path = output / f"{stem}_maximum_sky.png"
    Image.fromarray(sky, mode="RGBA").save(sky_path)
    circumstances_path = output / f"{stem}_circumstances.json"
    circumstances_path.write_text(json.dumps(circumstances.to_dict(timezone_name=timezone_name), indent=2), encoding="utf-8")
    timeline_path = output / f"{stem}_timeline.csv"
    with timeline_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["jd_utc", "utc", "local", "photosphere_visible", "sun_altitude_apparent_deg"])
        for jd in event.jd:
            sample = observer_state(event, float(jd), observer)
            writer.writerow([f"{float(jd):.12f}", jd_to_datetime(float(jd)).isoformat(),
                             jd_to_datetime(float(jd)).astimezone(tz).isoformat(),
                             f"{sample.photosphere_visible:.12g}", f"{sample.sun_altitude_apparent_deg:.8f}"])

    interactive_path = None
    if include_interactive:
        from ssapy_toolkit.plots.eclipse_cinematic_light_3d import generate_cinematic_view
        interactive_path = output / f"{stem}_interactive.html"
        generate_cinematic_view(event, output_path=interactive_path, quality=quality, animate=True)

    output_paths = {
        "scientific_summary": summary_path,
        "full_eclipse_map": full_map_path,
        "local_totality_map": local_map_path,
        "maximum_sky": sky_path,
        "contacts": contacts_path,
        "circumstances": circumstances_path,
        "timeline": timeline_path,
        "interactive": interactive_path,
    }
    # Keep the sealed manifest portable: products are siblings of the manifest,
    # so filenames are sufficient and do not leak build-machine paths.  The
    # returned API result still contains resolved paths for GUI convenience.
    manifest_record = {
        "$schema": "ssapy-toolkit.eclipse.observer-products/2.2",
        "event_key": event.definition.key,
        "observer": observer.to_dict(),
        "classification": circumstances.classification,
        "path_base": "manifest-directory",
        "outputs": {
            key: None if path is None else Path(path).name
            for key, path in output_paths.items()
        },
    }
    manifest_path = output / f"{stem}_manifest.json"
    manifest_path.write_text(json.dumps(manifest_record, indent=2) + "\n", encoding="utf-8")
    return {
        **manifest_record,
        "outputs": {
            key: None if path is None else str(Path(path).resolve())
            for key, path in output_paths.items()
        },
        "manifest": str(manifest_path.resolve()),
    }


__all__ = [
    "ObserverCircumstances",
    "solar_observer_circumstances",
    "render_observer_apparent_sky",
    "draw_full_solar_eclipse_map",
    "generate_solar_observer_products",
]
