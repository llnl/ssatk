"""Full-contact animations and interactive peak views for validated eclipses."""
from __future__ import annotations

from dataclasses import dataclass


import numpy as np

from ssatk.compute.eclipse_reference_events import (
    LUNAR_2025,
    SOLAR_2024,
    ReferenceEvent,
    jd_to_datetime,
    itrf_surface_point,
    lunar_reference_state,
    solar_besselian_state,
    solar_central_line_wgs84,
    solar_local_contacts,
    solar_local_phase_status,
    _unit,
)
from ssatk.compute.eclipse_raytrace import (
    solar_cross_track_width_km,
    trace_reference_rays,
)
from ssatk.plots.eclipse_rendering import (
    _camera_basis,
    contact_status,
    draw_light_curve,
    draw_lunar_shadow_plane,
    draw_ray_diagram,
    draw_solar_map,
    draw_solar_partial_visibility_map,
    draw_solar_combined_visibility_key,
    draw_lunar_geographic_visibility_map,
    draw_true_distance_locator,
    draw_north_indicator,
    render_earth_disk,
    render_moon_disk,
    render_solar_apparent_disk,
    solar_site_geometry,
)


@dataclass
class FrameRenderer:
    """Publication-style scientific frame with readable, validated panels."""

    event: ReferenceEvent
    width: int = 1600
    height: int = 960
    dpi: int = 100

    def __post_init__(self):
        if self.event.mode == "solar":
            peak_state = solar_besselian_state(self.event.greatest_jd)
            self.fixed_body_view = _unit(peak_state.sun_itrf_km)
            _, peak_sun, _, *_ = solar_site_geometry(self.event.greatest_jd)
            self.fixed_solar_basis = _camera_basis(peak_sun)
            peak_line = solar_central_line_wgs84(self.event.greatest_jd)
            contacts = solar_local_contacts()
            totality_s = (contacts["C3"]-contacts["C2"])*86400.0
            width_km = solar_cross_track_width_km(self.event.greatest_jd, n_azimuth=720)
            self.metrics_line = (
                f"NASA greatest eclipse: {peak_line[0]:.4f} deg N, {abs(peak_line[1]):.4f} deg W"
                f"   |   central-path width {width_km:.1f} km"
                f"   |   totality at the reference site {totality_s:.1f} s"
                "   |   eclipse magnitude 1.0566"
            )
        else:
            peak_state = lunar_reference_state(self.event.greatest_jd)
            self.fixed_body_view = _unit(-peak_state.moon_gcrf_km)
            self.fixed_solar_basis = None
            contacts = self.event.contacts_jd
            totality_s = (contacts["U3"]-contacts["U2"])*86400.0
            event_s = (contacts["P4"]-contacts["P1"])*86400.0
            self.metrics_line = (
                f"NASA umbral magnitude 1.1784   |   shadow-axis offset 0.3171 deg"
                f"   |   totality {totality_s/60.0:.1f} min"
                f"   |   P1-P4 duration {event_s/3600.0:.3f} h"
                "   |   red totality brightness/color is atmospheric and illustrative"
            )

    def render(self, index: int) -> np.ndarray:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        idx = int(index)
        jd = float(self.event.jd[idx])
        bundle = trace_reference_rays(self.event.definition, jd, n_azimuth=12)

        fig = plt.Figure(
            figsize=(self.width/self.dpi, self.height/self.dpi),
            dpi=self.dpi,
            facecolor="#010207",
        )
        canvas = FigureCanvasAgg(fig)
        gs = fig.add_gridspec(
            3, 24,
            height_ratios=[0.58, 2.45, 5.35],
            left=0.048, right=0.985, bottom=0.155, top=0.895,
            hspace=0.56, wspace=0.52,
        )

        ax_locator = fig.add_subplot(gs[0, :])
        ax_ray = fig.add_subplot(gs[1, :])
        draw_true_distance_locator(ax_locator, bundle)
        draw_ray_diagram(ax_ray, bundle)
        ax_locator.set_title("True Sun–Earth–Moon scale", color="white", fontsize=10.0, pad=3, fontweight="semibold")

        def _panel_badge(ax, letter: str, *, x: float = 0.008, y: float = 0.965):
            ax.text(
                x, y, letter, transform=ax.transAxes, ha="left", va="top",
                color="white", fontsize=8.4, fontweight="bold", zorder=30,
                bbox=dict(boxstyle="round,pad=0.20", facecolor="#17324f",
                          edgecolor="#8fd9f0", linewidth=0.75, alpha=0.96),
            )

        _panel_badge(ax_locator, "A", y=0.90)
        _panel_badge(ax_ray, "B", y=0.94)

        dt = jd_to_datetime(jd)
        if self.event.mode == "solar" and self.event.metadata.get("animation_scope"):
            title_status = solar_local_phase_status(jd)
        else:
            title_status = contact_status(self.event.definition, jd)
        event_title = (
            "Total Solar Eclipse — 8 April 2024" if self.event.definition.key == SOLAR_2024.key
            else "Total Lunar Eclipse — 14 March 2025" if self.event.definition.key == LUNAR_2025.key
            else self.event.definition.title
        )
        fig.suptitle(
            f"{event_title}   |   {dt.strftime('%H:%M:%S')} UTC   |   {title_status}",
            color="white", fontsize=17.2, fontweight="bold", y=0.983,
        )
        fig.text(
            0.5, 0.943,
            f"Reference geometry: {self.event.backend}   •   "
            f"{self.event.metadata.get('animation_scope', 'global P1-P4 event')}   •   north is up in body renderings",
            color="#aebcd0", fontsize=9.0, ha="center", va="top",
        )

        image_size = max(600, min(740, int(self.height*0.62)))
        if self.event.mode == "solar":
            ax_body = fig.add_subplot(gs[2, 0:4])
            ax_appearance = fig.add_subplot(gs[2, 4:8])
            # Two complementary geographic views answer two different physical
            # questions without forcing one color scale to do both jobs:
            # (1) the full partial-eclipse visibility footprint and maximum
            # photospheric obscuration, and (2) a true-scale zoom of the much
            # narrower totality corridor shaded by local C2-C3 duration.
            map_spec = gs[2, 8:19].subgridspec(
                2, 2, height_ratios=[0.76, 0.24], width_ratios=[1.38, 1.0],
                hspace=0.050, wspace=0.105,
            )
            ax_partial_map = fig.add_subplot(map_spec[0, 0])
            ax_corridor_map = fig.add_subplot(map_spec[0, 1])
            ax_map_key = fig.add_subplot(map_spec[1, :])
            ax_curve = fig.add_subplot(gs[2, 20:24])

            earth_state = solar_besselian_state(jd)
            earth_view = _unit(earth_state.sun_itrf_km)
            shadow_center = solar_central_line_wgs84(jd)
            if shadow_center is not None:
                shadow_view = _unit(itrf_surface_point(
                    float(shadow_center[0]), float(shadow_center[1]), 0.0,
                ))
                earth_view = _unit(0.58*earth_view + 0.42*shadow_view)
            earth_rgba = render_earth_disk(
                jd, size=image_size, view_hat=earth_view, supersample=2,
            )
            ax_body.imshow(earth_rgba, interpolation="lanczos")
            ax_body.set_facecolor("#010207")
            ax_body.axis("off")
            draw_north_indicator(ax_body)
            ax_body.set_title(
                "Earth at greatest eclipse\nThe Moon's shadow on the WGS-84 surface",
                color="white", fontsize=9.25, pad=5, fontweight="semibold",
            )

            appearance, visible = render_solar_apparent_disk(
                jd, size=image_size, fixed_basis=self.fixed_solar_basis,
            )
            ax_appearance.imshow(appearance, interpolation="lanczos")
            ax_appearance.set_facecolor("#010207")
            ax_appearance.axis("off")
            draw_north_indicator(ax_appearance, label="N")
            ax_appearance.set_title(
                "What an observer saw\n"
                f"Unobscured solar disk = {100.0*visible:.3f}%",
                color="white", fontsize=9.25, pad=5, fontweight="semibold",
            )

            draw_solar_partial_visibility_map(
                ax_partial_map,
                map_extent=(-180.0, 30.0, -10.0, 85.0),
                show_magnitude_contours=True,
            )
            draw_solar_map(
                ax_corridor_map, jd, footprint_azimuth=240,
                map_extent=(-126.0, -64.0, 15.0, 55.0),
                show_penumbra=False, show_key=False,
            )
            ax_corridor_map.set_title(
                "Where totality occurred\nLocal C2–C3 duration",
                color="white", fontsize=8.4, pad=4, fontweight="semibold",
            )
            draw_solar_combined_visibility_key(ax_map_key)
            draw_light_curve(ax_curve, self.event, jd)
            _panel_badge(ax_body, "C")
            _panel_badge(ax_appearance, "D")
            _panel_badge(ax_partial_map, "E1")
            _panel_badge(ax_corridor_map, "E2")
            _panel_badge(ax_curve, "F")
        else:
            ax_body = fig.add_subplot(gs[2, 0:5])
            ax_shadow = fig.add_subplot(gs[2, 5:12])
            lunar_map_spec = gs[2, 12:19].subgridspec(
                2, 1, height_ratios=[0.79, 0.21], hspace=0.045,
            )
            ax_lunar_map = fig.add_subplot(lunar_map_spec[0, 0])
            ax_lunar_key = fig.add_subplot(lunar_map_spec[1, 0])
            ax_curve = fig.add_subplot(gs[2, 20:24])

            moon_rgba = render_moon_disk(jd, size=image_size, view_hat=self.fixed_body_view)
            ax_body.imshow(moon_rgba, interpolation="lanczos")
            ax_body.set_facecolor("#010207")
            ax_body.axis("off")
            draw_north_indicator(ax_body)
            state = lunar_reference_state(jd)
            ax_body.set_title(
                f"Moon at greatest eclipse\n"
                f"Shadow-axis separation = {state.shadow_separation_deg:.4f}°",
                color="white", fontsize=10.2, pad=6, fontweight="semibold",
            )
            ax_body.text(
                0.5, 0.035,
                "Geometric boundary is validated; totality hue/brightness depends on Earth's atmosphere",
                transform=ax_body.transAxes, color="#c7a594", fontsize=7.2,
                ha="center", va="bottom",
            )

            draw_lunar_shadow_plane(ax_shadow, jd)
            draw_lunar_geographic_visibility_map(
                ax_lunar_map, key_ax=ax_lunar_key,
                map_extent=(-180.0, 180.0, -75.0, 75.0),
            )
            draw_light_curve(ax_curve, self.event, jd)
            _panel_badge(ax_body, "C")
            _panel_badge(ax_shadow, "D")
            _panel_badge(ax_lunar_map, "E")
            _panel_badge(ax_curve, "F")

        if self.event.mode == "solar":
            panel_guide = (
                "A  True-distance locator: verifies the Sun remains at its physical distance.   "
                "B  Finite-Sun rays: shows axial light, tangent limits, and first-surface clipping.   "
                "C  Earth: resolves the WGS-84 surface and lunar shadow.\n"
                "D  Observer sky view: shows the Sun/Moon overlap at the NASA reference site.   "
                "E1  Partial region: maximum photospheric obscuration wherever the Sun was above the horizon.   "
                "E2  Totality zoom: local C2-C3 duration inside the physical corridor.   "
                "F  Visibility curve: follows the unobscured solar disk through C1-MAX-C4."
            )
        else:
            panel_guide = (
                "A  True-distance locator: verifies the physical Sun-Earth-Moon order and scale.   "
                "B  Finite-Sun rays: shows Earth's umbral and penumbral tangent families.   "
                "C  Moon: renders the textured lunar disk with the geometric eclipse shadow.\n"
                "D  Danjon shadow plane: plots the Moon's track through penumbra and umbra at every contact.   "
                "E  Geographic visibility: separates complete-event, totality-only, partial-phase, and no-view regions.   "
                "F  Visibility curve: shows the fraction of the solar disk visible from the Moon through P1-MAX-P4."
            )
        fig.text(
            0.5, 0.096, panel_guide, color="#c2d0df", fontsize=7.05,
            ha="center", va="center", linespacing=1.30,
            bbox=dict(boxstyle="round,pad=0.42", facecolor="#07101a",
                      edgecolor="#344a63", alpha=0.97),
        )

        fig.text(
            0.5, 0.045, self.metrics_line,
            color="#d2dce8", fontsize=8.5, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.36", facecolor="#0b111d",
                      edgecolor="#34435a", alpha=0.95),
        )
        fig.text(
            0.5, 0.010,
            "Distances, contacts, path coordinates, and ray intersections are physical. "
            "The Sun is never relocated into the Earth-Moon close-up.",
            color="#8292a8", fontsize=7.5, ha="center", va="bottom",
        )

        canvas.draw()
        rgb = np.asarray(canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return rgb


