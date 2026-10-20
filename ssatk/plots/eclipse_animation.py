"""Full-contact animations and interactive peak views for validated eclipses."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from ssatk.io.eclipse_asset_resolver import resolve_image
import base64
import json
import math

import numpy as np

from ssatk.compute.eclipse_reference_events import (
    AU_KM,
    EARTH_AXES_KM,
    LUNAR_2025,
    RE_KM,
    R_MOON_MEAN_KM,
    RP_KM,
    R_SUN_KM,
    SOLAR_2024,
    ReferenceEvent,
    build_reference_event,
    build_solar_local_event,
    contact_label_at_jd,
    jd_to_datetime,
    itrf_surface_point,
    lunar_reference_state,
    local_contact_label_at_jd,
    reference_summary,
    solar_besselian_state,
    solar_central_line_wgs84,
    solar_local_contacts,
    solar_local_phase_status,
    _unit,
)
from ssatk.compute.eclipse_raytrace import (
    LUNAR_DANJON_EARTH_RADIUS_KM,
    LUNAR_OPTICAL_MOON_RADIUS_KM,
    ReferenceRayBundle,
    bundle_penetrations,
    solar_cross_track_width_km,
    solar_footprint_points,
    tangent_residuals,
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


def _render_frame_worker(payload):
    """Process-pool worker; short worker lifetimes bound Matplotlib memory."""
    event, index, width, height, dpi = payload
    from ssatk.plots.eclipse_scientific_suite import renderer_for_event
    renderer = renderer_for_event(event, width=int(width), height=int(height), dpi=int(dpi))
    return int(index), renderer.render(int(index))


def _create_animation_pool(*, workers: int = 2, tasks_per_worker: int = 8):
    """Create one short-lived frame-render pool.

    Workers are not recycled inside a live ``multiprocessing.Pool``.  CPython's
    pool worker-replacement path can lose completion notifications under heavy
    Matplotlib allocations.  Instead, the caller closes the whole pool after a
    bounded batch; the process exit is the memory boundary.
    """
    import multiprocessing as mp

    del tasks_per_worker  # batching is managed by the caller, not Pool internals
    count = max(1, int(workers))
    # Always use a fresh interpreter.  Linux ``fork`` inherits every open
    # descriptor in the parent process, including pytest capture pipes and any
    # encoder handles created by a previous export.  The rendering payload is
    # fully pickleable, so ``spawn`` is slower but deterministic and prevents
    # otherwise successful test/CLI processes from hanging at exit.
    return mp.get_context("spawn").Pool(processes=count)


def _write_html_player(frames_jpeg: list[bytes], labels: list[str], title: str,
                       path: Path, *, fps: int = 10):
    encoded = [base64.b64encode(frame).decode("ascii") for frame in frames_jpeg]
    payload = json.dumps(encoded, separators=(",", ":"))
    label_payload = json.dumps(labels, separators=(",", ":"))
    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>{title}</title>
<style>
html,body{{margin:0;background:#010207;color:#fff;font-family:system-ui,Segoe UI,sans-serif;height:100%;}}
main{{height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;}}
img{{max-width:98vw;max-height:86vh;box-shadow:0 0 24px #000;}}
.controls{{display:flex;align-items:center;gap:10px;width:min(98vw,1280px);}}
button{{background:#273248;color:white;border:1px solid #64738c;border-radius:5px;padding:6px 13px;font-size:15px;}}
input{{flex:1;}} #label{{min-width:220px;text-align:right;color:#c8d3e2;font-variant-numeric:tabular-nums;}}
</style></head><body><main>
<img id='frame' alt='Validated eclipse animation frame'>
<div class='controls'><button id='play'>Play</button><input id='slider' type='range' min='0' max='{len(encoded)-1}' value='0' step='1'><span id='label'></span></div>
</main><script>
const frames={payload}; const labels={label_payload}; const fps={int(fps)};
const img=document.getElementById('frame'), slider=document.getElementById('slider'), label=document.getElementById('label'), play=document.getElementById('play');
let timer=null; function show(i){{i=Number(i);img.src='data:image/jpeg;base64,'+frames[i];slider.value=i;label.textContent=labels[i];}}
slider.addEventListener('input',e=>{{show(e.target.value);}});
play.addEventListener('click',()=>{{if(timer){{clearInterval(timer);timer=null;play.textContent='Play';return;}} play.textContent='Pause';timer=setInterval(()=>{{let i=(Number(slider.value)+1)%frames.length;show(i);}},1000/fps);}});
show(0);
</script></body></html>"""
    path.write_text(html, encoding="utf-8")


def generate_event_animation(event: ReferenceEvent, output_dir: str | Path,
                             *, width: int = 1280, height: int = 720,
                             fps: int = 10, jpeg_quality: int = 87,
                             stem: str | None = None,
                             title: str | None = None,
                             workers: int = 2,
                             tasks_per_worker: int = 8,
                             _render_pool=None) -> dict:
    """Generate MP4, self-contained HTML player, and greatest-eclipse PNG.

    Rendering and video encoding are deliberately separated.  Starting FFmpeg
    with a stdin stream while a Linux fork pool is recycling workers lets new
    workers inherit that pipe and can make encoder shutdown wait forever.  Frames
    are rendered first to numbered lossless PNG files, then FFmpeg reads those
    files directly with stdin disabled.  The export is therefore safe even when
    the long-lived render pool replaces a worker between events.
    """
    from PIL import Image
    from tempfile import TemporaryDirectory
    import shutil
    import subprocess

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = stem or event.definition.key
    title = title or event.definition.title
    scope_suffix = "local_C1_to_C4" if event.metadata.get("animation_scope") else "global_P1_to_P4"
    mp4_path = out / f"{stem}_{scope_suffix}.mp4"
    mp4_partial = out / f".{stem}_{scope_suffix}.partial.mp4"
    html_path = out / f"{stem}_{scope_suffix}.html"
    peak_path = out / f"{stem}_{scope_suffix}_greatest_frame.png"
    for stale in (mp4_path, mp4_partial, html_path, peak_path):
        stale.unlink(missing_ok=True)

    frames_jpeg: list[bytes] = []
    labels: list[str] = []
    peak_idx = event.peak_index
    external_pool = _render_pool

    with TemporaryDirectory(prefix=f"{stem}_{scope_suffix}_frames_", dir=out) as frame_dir_name:
        frame_dir = Path(frame_dir_name)

        if external_pool is not None:
            batch_size = max(1, int(tasks_per_worker))*max(1, int(workers) or 1)

            def _batched_rendered():
                for batch_start in range(0, len(event.jd), batch_size):
                    batch_stop = min(batch_start + batch_size, len(event.jd))
                    payloads = [(event, idx, width, height, 100)
                                for idx in range(batch_start, batch_stop)]
                    for item in external_pool.map(_render_frame_worker, payloads, chunksize=1):
                        yield item

            rendered = _batched_rendered()
        elif int(workers) > 0 and len(event.jd) > max(4, 2*int(workers)):
            worker_count = max(1, int(workers))
            batch_size = worker_count*max(1, int(tasks_per_worker))

            def _short_lived_pool_rendered():
                for batch_start in range(0, len(event.jd), batch_size):
                    batch_stop = min(batch_start + batch_size, len(event.jd))
                    payloads = [(event, idx, width, height, 100)
                                for idx in range(batch_start, batch_stop)]
                    batch_pool = _create_animation_pool(workers=worker_count)
                    try:
                        results = batch_pool.map(_render_frame_worker, payloads, chunksize=1)
                    finally:
                        batch_pool.close()
                        batch_pool.join()
                    for item in results:
                        yield item

            rendered = _short_lived_pool_rendered()
        else:
            # A process pool is counterproductive for very small contact-aware
            # regressions: importing Matplotlib and the complete eclipse package
            # in multiple spawned interpreters can exceed the render time by two
            # orders of magnitude and, if an outer test timeout kills the parent,
            # leave capture descriptors open in descendants.  Render tiny jobs
            # serially while retaining spawned workers for actual animations.
            from ssatk.plots.eclipse_scientific_suite import renderer_for_event
            renderer = renderer_for_event(event, width=width, height=height)
            rendered = ((idx, renderer.render(idx)) for idx in range(len(event.jd)))

        for idx, frame in rendered:
            idx = int(idx)
            jd = float(event.jd[idx])
            Image.fromarray(frame).save(
                frame_dir / f"frame_{idx:05d}.png",
                format="PNG", compress_level=1,
            )
            if idx == peak_idx:
                Image.fromarray(frame).save(peak_path, optimize=True)
            buffer = BytesIO()
            Image.fromarray(frame).save(
                buffer, format="JPEG", quality=int(jpeg_quality),
                optimize=True, subsampling=1,
            )
            frames_jpeg.append(buffer.getvalue())
            dt = jd_to_datetime(jd)
            if event.mode == "solar" and event.metadata.get("animation_scope"):
                label_status = solar_local_phase_status(jd)
            else:
                label_status = contact_status(event.definition, jd)
            labels.append(f"{dt.strftime('%Y-%m-%d %H:%M:%S')} UTC — {label_status}")

        # Encode from numbered lossless image files rather than an FFmpeg
        # stdin pipe.  Even if the long-lived render pool replaces a worker at
        # this instant, there is no pipe descriptor for that worker to inherit,
        # so FFmpeg termination cannot be held open by an unrelated child.
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_exe = shutil.which("ffmpeg")
        if not ffmpeg_exe:
            raise RuntimeError(
                "FFmpeg is required for MP4 export; install imageio-ffmpeg or provide ffmpeg on PATH"
            )
        command = [
            str(ffmpeg_exe), "-y",
            "-framerate", str(int(fps)),
            "-start_number", "0",
            "-i", str(frame_dir / "frame_%05d.png"),
            "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(mp4_partial),
        ]
        completed = subprocess.run(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, text=True, close_fds=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"FFmpeg MP4 export failed: {completed.stderr.strip()}")
        mp4_partial.replace(mp4_path)

    _write_html_player(frames_jpeg, labels, title, html_path, fps=fps)
    return {
        "mp4": str(mp4_path),
        "html": str(html_path),
        "peak_png": str(peak_path),
        "frames": len(event.jd),
        "fps": int(fps),
    }


def _world_clip_path(points, origin, axis, xmin, xmax):
    p = np.asarray(points, dtype=float)
    origin = np.asarray(origin, dtype=float)
    axis = _unit(axis)
    out = []
    for a, b in zip(p[:-1], p[1:]):
        xa = float(np.dot(a-origin, axis)); xb = float(np.dot(b-origin, axis))
        if max(xa, xb) < xmin or min(xa, xb) > xmax:
            continue
        lo, hi = 0.0, 1.0
        dx = xb-xa
        if abs(dx) > 1.0e-14:
            t1, t2 = (xmin-xa)/dx, (xmax-xa)/dx
            lo=max(lo,min(t1,t2)); hi=min(hi,max(t1,t2))
        if hi < lo: continue
        pa=a+lo*(b-a); pb=a+hi*(b-a)
        if not out or np.linalg.norm(out[-1]-pa)>1e-7: out.append(pa)
        out.append(pb)
    return np.asarray(out)


def _sphere_trace(center, radius, color, name, *, n=28, opacity=1.0):
    import plotly.graph_objects as go
    u=np.linspace(0,2*np.pi,n); v=np.linspace(-np.pi/2,np.pi/2,n//2)
    U,V=np.meshgrid(u,v)
    X=center[0]+radius*np.cos(V)*np.cos(U)
    Y=center[1]+radius*np.cos(V)*np.sin(U)
    Z=center[2]+radius*np.sin(V)
    return go.Surface(x=X,y=Y,z=Z,surfacecolor=np.zeros_like(X),
                      colorscale=[[0,color],[1,color]],showscale=False,opacity=opacity,
                      lighting=dict(ambient=.8,diffuse=.5,specular=.1,roughness=.9),
                      name=name,hovertemplate=f"{name}<extra></extra>")


def generate_peak_3d(event: ReferenceEvent, output_path: str | Path) -> str:
    """Create a two-scene interactive view with no local fake Sun or ray warp."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from ssatk.coordinates.eclipse_lunar_attitude import moon_attitude_for_event
    from ssatk.plots.globe_orbit_daynight_plotly import _earth_atmosphere_trace, _earth_mesh, _sun_sphere_traces
    from ssatk.plots.moon_render import moon_mesh_plotly

    idx=event.peak_index; jd=float(event.jd[idx]); bundle=trace_reference_rays(event.definition,jd,n_azimuth=8)
    fig=make_subplots(rows=1,cols=2,specs=[[{"type":"scene"},{"type":"scene"}]],
                      subplot_titles=("True Earth–Moon local scale; Sun remains off-frame",
                                      "True Sun–Earth–Moon distances and radii"),
                      horizontal_spacing=.03)
    earth=np.zeros(3); moon=bundle.moon_center_km; sun=bundle.sun_center_km
    sun_hat=_unit(sun-earth)
    target=earth if event.mode=="solar" else moon
    occ=moon if event.mode=="solar" else earth
    axis=bundle.axis_hat; target_x=float(np.dot(target-occ,axis)); xmin=-.055*target_x; xmax=1.045*target_x
    # Fixed side-on camera direction.
    rel=bundle.umbra_tangent_points_km[0]-occ; transverse=_unit(rel-axis*np.dot(rel,axis)); camera_hat=_unit(np.cross(axis,transverse))
    earth_trace=_earth_mesh(sun_hat,n_lat=91,n_lon=180,center=(0,0,0),
                            sun_position_km=sun,physical_center_km=(0,0,0),
                            time_jd=None,view_hat=camera_hat,specular_strength=.24,
                            texture_path=resolve_image("earth_albedo").path)
    fig.add_trace(earth_trace,row=1,col=1)
    fig.add_trace(_earth_atmosphere_trace(center=(0,0,0),sun_hat=sun_hat,view_hat=camera_hat,
                                          time_jd=None,n_lat=42,n_lon=84),row=1,col=1)
    moon_mode="solar" if event.mode=="solar" else "lunar"
    moon_radius=bundle.solid_occluder_radius_km if event.mode=="solar" else LUNAR_OPTICAL_MOON_RADIUS_KM
    fig.add_trace(moon_mesh_plotly(moon,moon_radius,sun_hat=sun_hat,
                                   real_center_km=moon,mode=moon_mode,n_lat=91,n_lon=180,
                                   real_sun_position_km=sun,view_hat=camera_hat,
                                   eclipse_occluder_radius_km=LUNAR_DANJON_EARTH_RADIUS_KM,
                                   texture_path=resolve_image("moon_albedo").path,
                                   body_to_gcrf=moon_attitude_for_event(event,jd).body_to_gcrf),
                  row=1,col=1)
    # Only the central ray and one opposite pair from each tangent family.
    def selected(paths):
        values=[np.dot(path.points_km[1]-occ,transverse) for path in paths]
        return [paths[int(np.argmax(values))],paths[int(np.argmin(values))]]
    for path,color,width,name in [(bundle.central,"#ffe36e",7,"Direct sunlight")]:
        q=_world_clip_path(path.points_km,occ,axis,xmin,xmax)
        fig.add_trace(go.Scatter3d(x=q[:,0],y=q[:,1],z=q[:,2],mode="lines",
                                   line=dict(color=color,width=width),name=name),row=1,col=1)
    for paths,color,width,name in ((bundle.umbra,"#ff6c5c",5,"Umbra tangents"),
                                   (bundle.penumbra,"#8bc8ff",4,"Penumbra tangents")):
        for k,path in enumerate(selected(paths)):
            q=_world_clip_path(path.points_km,occ,axis,xmin,xmax)
            fig.add_trace(go.Scatter3d(x=q[:,0],y=q[:,1],z=q[:,2],mode="lines",
                                       line=dict(color=color,width=width),name=name,
                                       showlegend=(k==0)),row=1,col=1)
    # True-scale scene in a rigid optical basis, expressed in AU.
    xhat=_unit(target-sun); cross=moon-np.dot(moon-sun,xhat)*xhat
    if np.linalg.norm(cross)<1e-6: yhat=transverse
    else: yhat=_unit(cross)
    zhat=_unit(np.cross(xhat,yhat)); yhat=_unit(np.cross(zhat,xhat))
    def optical(points):
        q=np.asarray(points)-sun
        return np.column_stack([q@xhat,q@yhat,q@zhat])/AU_KM
    sun_q=np.zeros(3); earth_q=optical(earth.reshape(1,3))[0]; moon_q=optical(moon.reshape(1,3))[0]
    fig.add_trace(_sphere_trace(sun_q,R_SUN_KM/AU_KM,"#ffad32","Sun",n=36),row=1,col=2)
    fig.add_trace(_sphere_trace(earth_q,RE_KM/AU_KM,"#3e8fd6","Earth",n=20),row=1,col=2)
    fig.add_trace(_sphere_trace(moon_q,moon_radius/AU_KM,"#bcbcbc","Moon",n=18),row=1,col=2)
    fig.add_trace(go.Scatter3d(x=[earth_q[0]],y=[earth_q[1]],z=[earth_q[2]],mode="markers+text",
                               marker=dict(size=5,color="#68baff"),text=["Earth"],textposition="top center",
                               name="Earth locator"),row=1,col=2)
    fig.add_trace(go.Scatter3d(x=[moon_q[0]],y=[moon_q[1]],z=[moon_q[2]],mode="markers+text",
                               marker=dict(size=4,color="#eeeeee"),text=["Moon"],textposition="bottom center",
                               name="Moon locator"),row=1,col=2)
    for path,color,width,name in ((bundle.central,"#ffe36e",4,"Direct sunlight"),):
        q=optical(path.points_km)
        fig.add_trace(go.Scatter3d(x=q[:,0],y=q[:,1],z=q[:,2],mode="lines",
                                   line=dict(color=color,width=width),name=name,showlegend=False),row=1,col=2)
    for paths,color,width in ((selected(bundle.umbra),"#ff6c5c",2),(selected(bundle.penumbra),"#8bc8ff",2)):
        for path in paths:
            q=optical(path.points_km)
            fig.add_trace(go.Scatter3d(x=q[:,0],y=q[:,1],z=q[:,2],mode="lines",
                                       line=dict(color=color,width=width),showlegend=False),row=1,col=2)
    # Local ranges stay in the original frame; data aspect prevents visual ray/body penetration.
    centers=np.vstack([earth,moon]); center_mean=centers.mean(axis=0); span=np.linalg.norm(moon-earth)*.62
    local_range=[[center_mean[k]-span,center_mean[k]+span] for k in range(3)]
    fig.update_scenes(xaxis=dict(range=local_range[0],title="X [km]",gridcolor="#263246"),
                      yaxis=dict(range=local_range[1],title="Y [km]",gridcolor="#263246"),
                      zaxis=dict(range=local_range[2],title="Z [km]",gridcolor="#263246"),
                      aspectmode="data",bgcolor="#010207",
                      camera=dict(eye=dict(x=float(camera_hat[0]*1.55),y=float(camera_hat[1]*1.55),z=float(camera_hat[2]*1.55))),
                      row=1,col=1)
    xmax_true=max(earth_q[0],moon_q[0])+0.01
    fig.update_scenes(xaxis=dict(range=[-0.012,xmax_true],title="Optical axis [AU]",gridcolor="#263246"),
                      yaxis=dict(range=[-.009,.009],title="Transverse [AU]",gridcolor="#263246"),
                      zaxis=dict(range=[-.009,.009],title="Normal [AU]",gridcolor="#263246"),
                      aspectmode="data",bgcolor="#010207",row=1,col=2)
    fig.update_layout(title=dict(text=f"{event.definition.title} — greatest eclipse<br><sub>Sun is at its real ephemeris distance; the local scene contains no relocated Sun and no coordinate warp.</sub>",x=.5),
                      paper_bgcolor="#010207",plot_bgcolor="#010207",font=dict(color="white"),
                      width=1500,height=760,margin=dict(l=0,r=0,t=90,b=0),legend=dict(orientation="h",y=.02,x=.02))
    path=Path(output_path);path.parent.mkdir(parents=True,exist_ok=True);fig.write_html(path,include_plotlyjs=True)
    return str(path)



def _hex_rgb(value: str) -> tuple[int, int, int]:
    text = value.lstrip("#")
    if len(text) != 6:
        raise ValueError(f"expected #RRGGBB, got {value!r}")
    return tuple(int(text[i:i+2], 16) for i in (0, 2, 4))


def _shade_scale(base_color: str, *, floor: float = 0.055):
    r, g, b = _hex_rgb(base_color)
    dark = tuple(max(0, int(round(channel*floor))) for channel in (r, g, b))
    return [[0.0, f"rgb{dark}"], [1.0, f"rgb({r},{g},{b})"]]


def _basis_transform(points, origin, basis) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    return (pts-np.asarray(origin, dtype=float))@np.asarray(basis, dtype=float)


def _clip_axis_x(points, xmin: float, xmax: float) -> np.ndarray:
    """Clip a polyline already expressed in a fixed Cartesian display basis."""
    p = np.asarray(points, dtype=float)
    out: list[np.ndarray] = []
    for a, b in zip(p[:-1], p[1:]):
        x0, x1 = float(a[0]), float(b[0])
        if max(x0, x1) < xmin or min(x0, x1) > xmax:
            continue
        lo, hi = 0.0, 1.0
        dx = x1-x0
        if abs(dx) > 1.0e-15:
            t0, t1 = (xmin-x0)/dx, (xmax-x0)/dx
            lo = max(lo, min(t0, t1))
            hi = min(hi, max(t0, t1))
        if hi < lo:
            continue
        pa = a+lo*(b-a)
        pb = a+hi*(b-a)
        if not out or np.linalg.norm(out[-1]-pa) > 1.0e-9:
            out.append(pa)
        out.append(pb)
    return np.asarray(out, dtype=float).reshape(-1, 3)


def _lit_body_trace(center_world, radii_km, transform_origin, transform_basis,
                    sun_world, base_color: str, name: str, *, scene: str,
                    n_lat: int = 25, n_lon: int = 49, opacity: float = 1.0):
    """A physically placed, smoothly lit ellipsoid in an unwarped basis."""
    import plotly.graph_objects as go
    radii = np.broadcast_to(np.asarray(radii_km, dtype=float), (3,))
    lat = np.linspace(-np.pi/2.0, np.pi/2.0, int(n_lat))
    lon = np.linspace(-np.pi, np.pi, int(n_lon))
    Lon, Lat = np.meshgrid(lon, lat)
    unit = np.stack([
        np.cos(Lat)*np.cos(Lon),
        np.cos(Lat)*np.sin(Lon),
        np.sin(Lat),
    ], axis=-1)
    world = np.asarray(center_world, dtype=float)+unit*radii
    # Ellipsoid outward normal in the source frame.
    normal = unit/np.maximum(radii, 1.0e-15)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    to_sun = np.asarray(sun_world, dtype=float)-world
    to_sun /= np.linalg.norm(to_sun, axis=-1, keepdims=True)
    mu = np.clip(np.sum(normal*to_sun, axis=-1), 0.0, 1.0)
    intensity = 0.055+0.945*mu**0.76
    q = _basis_transform(world.reshape(-1, 3), transform_origin, transform_basis).reshape(world.shape)
    return go.Surface(
        x=q[..., 0], y=q[..., 1], z=q[..., 2],
        surfacecolor=intensity, cmin=0.0, cmax=1.0,
        colorscale=_shade_scale(base_color), showscale=False,
        opacity=float(opacity), name=name, scene=scene,
        hovertemplate=f"{name}<extra></extra>",
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0),
    )


def _constant_body_trace(center_world, radii_km, transform_origin, transform_basis,
                         color: str, name: str, *, scene: str,
                         n_lat: int = 25, n_lon: int = 49, opacity: float = 1.0):
    import plotly.graph_objects as go
    radii = np.broadcast_to(np.asarray(radii_km, dtype=float), (3,))
    lat = np.linspace(-np.pi/2.0, np.pi/2.0, int(n_lat))
    lon = np.linspace(-np.pi, np.pi, int(n_lon))
    Lon, Lat = np.meshgrid(lon, lat)
    world = np.asarray(center_world, dtype=float)+np.stack([
        radii[0]*np.cos(Lat)*np.cos(Lon),
        radii[1]*np.cos(Lat)*np.sin(Lon),
        radii[2]*np.sin(Lat),
    ], axis=-1)
    q = _basis_transform(world.reshape(-1, 3), transform_origin, transform_basis).reshape(world.shape)
    return go.Surface(
        x=q[..., 0], y=q[..., 1], z=q[..., 2],
        surfacecolor=np.zeros(q.shape[:2]), colorscale=[[0, color], [1, color]],
        showscale=False, opacity=float(opacity), name=name, scene=scene,
        hovertemplate=f"{name}<extra></extra>",
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0),
    )


def _selected_opposite_paths(bundle: ReferenceRayBundle, paths):
    occ = bundle.moon_center_km if bundle.mode == "solar" else bundle.earth_center_km
    rel = bundle.umbra_tangent_points_km[0]-occ
    transverse = rel-bundle.axis_hat*np.dot(rel, bundle.axis_hat)
    transverse = _unit(transverse)
    values = [float(np.dot(path.points_km[1]-occ, transverse)) for path in paths]
    return paths[int(np.argmax(values))], paths[int(np.argmin(values))]


def _event_status_label(event: ReferenceEvent, jd_value: float) -> str:
    if event.mode == "solar" and event.metadata.get("animation_scope"):
        return solar_local_phase_status(jd_value)
    return contact_status(event.definition, jd_value)


def generate_event_3d_animation(event: ReferenceEvent, output_path: str | Path,
                                *, n_azimuth: int = 8,
                                include_plotlyjs: bool = True) -> str:
    """Write a full-contact, two-panel 3-D animation with no distance warp.

    The local panel uses true Earth-Moon separation and radii; the real Sun is
    off-frame.  The second panel uses true Sun-Earth-Moon coordinates in AU.
    Every displayed light segment comes directly from ``trace_reference_rays``
    and is clipped before entering the WGS-84 Earth or the modeled lunar limb.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    peak_bundle = trace_reference_rays(event.definition, event.greatest_jd,
                                       n_azimuth=max(8, int(n_azimuth)))
    earth = np.zeros(3)
    moon_peak = peak_bundle.moon_center_km
    sun_peak = peak_bundle.sun_center_km

    # Rigid local basis, centered on Earth.  No nonlinear compression.
    x_local = _unit(moon_peak-earth)
    axis_peak = peak_bundle.axis_hat
    rel = peak_bundle.umbra_tangent_points_km[0]-(moon_peak if event.mode == "solar" else earth)
    y_seed = rel-axis_peak*np.dot(rel, axis_peak)
    y_local = _unit(y_seed-x_local*np.dot(y_seed, x_local))
    z_local = _unit(np.cross(x_local, y_local))
    y_local = _unit(np.cross(z_local, x_local))
    local_basis = np.stack([x_local, y_local, z_local], axis=1)

    # Rigid true-scale basis centered on the Sun.
    x_true = _unit(earth-sun_peak)
    moon_cross = moon_peak-sun_peak-x_true*np.dot(moon_peak-sun_peak, x_true)
    y_true = _unit(moon_cross) if np.linalg.norm(moon_cross) > 1.0e-8 else y_local
    z_true = _unit(np.cross(x_true, y_true))
    y_true = _unit(np.cross(z_true, x_true))
    true_basis = np.stack([x_true, y_true, z_true], axis=1)

    moon_distances = np.linalg.norm(event.moon_km, axis=1)
    local_distance = float(np.max(moon_distances))
    local_xmin, local_xmax = -0.105*local_distance, 1.105*local_distance
    local_transverse = max(0.13*local_distance, 3.2*RE_KM)

    # Complete event track in the same unwarped local basis.
    track_local = _basis_transform(event.moon_km, earth, local_basis)
    solar_path_world = []
    if event.mode == "solar":
        for jd_value in event.jd:
            point = solar_central_line_wgs84(float(jd_value))
            if point is not None:
                solar_path_world.append(point[2])
    solar_path_local = (_basis_transform(np.asarray(solar_path_world), earth, local_basis)
                        if solar_path_world else np.empty((0, 3)))

    def line_trace(points, color, width, name, scene, *, dash=None, showlegend=True):
        p = np.asarray(points, dtype=float).reshape(-1, 3)
        line = dict(color=color, width=width)
        if dash is not None:
            line["dash"] = dash
        return go.Scatter3d(
            x=p[:, 0] if len(p) else [], y=p[:, 1] if len(p) else [],
            z=p[:, 2] if len(p) else [], mode="lines", line=line,
            name=name, scene=scene, showlegend=showlegend, hoverinfo="skip",
        )

    def marker_trace(point, color, size, name, scene, text=None):
        p = np.asarray(point, dtype=float).reshape(3)
        mode = "markers+text" if text else "markers"
        return go.Scatter3d(
            x=[p[0]], y=[p[1]], z=[p[2]], mode=mode,
            marker=dict(color=color, size=size), text=[text] if text else None,
            textposition="top center", name=name, scene=scene,
            showlegend=False, hovertemplate=f"{name}<extra></extra>",
        )

    def frame_traces(jd_value: float):
        bundle = trace_reference_rays(event.definition, float(jd_value),
                                      n_azimuth=max(8, int(n_azimuth)))
        sun, moon = bundle.sun_center_km, bundle.moon_center_km
        sun_hat_earth = _unit(sun-earth)
        local_moon_radius = R_MOON_MEAN_KM
        traces = [
            _lit_body_trace(earth, EARTH_AXES_KM, earth, local_basis, sun,
                            "#2d79b9", "Earth — WGS-84", scene="scene",
                            n_lat=31, n_lon=61),
            _constant_body_trace(earth, EARTH_AXES_KM*1.012, earth, local_basis,
                                 "#6eb8ff", "Atmosphere", scene="scene",
                                 n_lat=25, n_lon=49, opacity=0.07),
            _lit_body_trace(moon, local_moon_radius, earth, local_basis, sun,
                            "#b8b5ad", "Moon", scene="scene",
                            n_lat=27, n_lon=53),
        ]
        # Local ray segments: exact physical points transformed by a rigid basis.
        central_local = _clip_axis_x(
            _basis_transform(bundle.central.points_km, earth, local_basis),
            local_xmin, local_xmax,
        )
        traces.append(line_trace(central_local, "#ffe36e", 7,
                                 "Direct sunlight — blocked at first body",
                                 "scene"))
        for family, color, width, label in (
            (bundle.umbra, "#ff6c5c", 5, "Umbra tangents"),
            (bundle.penumbra, "#8bc8ff", 4, "Penumbra tangents"),
        ):
            for index, path in enumerate(_selected_opposite_paths(bundle, family)):
                q = _clip_axis_x(_basis_transform(path.points_km, earth, local_basis),
                                 local_xmin, local_xmax)
                traces.append(line_trace(q, color, width, label, "scene",
                                         showlegend=(index == 0)))
        # Current physical solar footprint; empty outside central eclipse.
        if event.mode == "solar":
            footprint_world = solar_footprint_points(float(jd_value), family="umbra",
                                                     n_azimuth=96)
            footprint_local = (_basis_transform(footprint_world, earth, local_basis)
                               if len(footprint_world) else np.empty((0, 3)))
        else:
            footprint_local = np.empty((0, 3))
        traces.append(line_trace(footprint_local, "#ffcf5a", 5,
                                 "Current WGS-84 umbral boundary", "scene"))
        event_path = solar_path_local if event.mode == "solar" else track_local
        traces.append(line_trace(event_path, "#b47cff", 3,
                                 "Eclipse path" if event.mode == "solar" else "Moon track",
                                 "scene", dash="dot"))
        current_local = _basis_transform(moon.reshape(1, 3), earth, local_basis)[0]
        traces.append(marker_trace(current_local, "#ffffff", 4,
                                   "Current Moon center", "scene"))

        # True-scale panel in AU, also using a rigid basis.
        def true(points):
            return _basis_transform(points, sun_peak, true_basis)/AU_KM
        sun_q = true(sun.reshape(1, 3))[0]
        earth_q = true(earth.reshape(1, 3))[0]
        moon_q = true(moon.reshape(1, 3))[0]
        traces.extend([
            _constant_body_trace(sun, R_SUN_KM, sun_peak, true_basis,
                                 "#ffae32", "Sun — actual radius", scene="scene2",
                                 n_lat=29, n_lon=57),
            _constant_body_trace(earth, EARTH_AXES_KM, sun_peak, true_basis,
                                 "#3c8fd3", "Earth — actual radius", scene="scene2",
                                 n_lat=17, n_lon=33),
            _constant_body_trace(moon, local_moon_radius, sun_peak, true_basis,
                                 "#bdbdbd", "Moon — actual radius", scene="scene2",
                                 n_lat=15, n_lon=29),
        ])
        central_true = true(bundle.central.points_km)
        traces.append(line_trace(central_true, "#ffe36e", 4,
                                 "Direct sunlight (true scale)", "scene2",
                                 showlegend=False))
        for family, color, width in ((bundle.umbra, "#ff6c5c", 2),
                                     (bundle.penumbra, "#8bc8ff", 2)):
            for path in _selected_opposite_paths(bundle, family):
                traces.append(line_trace(true(path.points_km), color, width,
                                         "Boundary ray", "scene2", showlegend=False))
        traces.append(marker_trace(earth_q, "#68baff", 5, "Earth locator",
                                   "scene2", text="Earth"))
        traces.append(marker_trace(moon_q, "#eeeeee", 4, "Moon locator",
                                   "scene2", text="Moon"))
        # Sun center locator makes the true-distance panel readable at any zoom.
        traces.append(marker_trace(sun_q, "#ffc14f", 5, "Sun center",
                                   "scene2", text="Sun"))
        return traces

    first = frame_traces(float(event.jd[0]))
    fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(
            "True Earth-Moon scale — Sun remains at its real off-frame distance",
            "True Sun-Earth-Moon distances and radii",
        ), horizontal_spacing=0.025,
    )
    for trace in first:
        fig.add_trace(trace)

    frames = []
    slider_steps = []
    for index, jd_value in enumerate(event.jd):
        dt = jd_to_datetime(float(jd_value))
        status = _event_status_label(event, float(jd_value))
        frame_name = str(index)
        frames.append(go.Frame(
            data=frame_traces(float(jd_value)), name=frame_name,
            traces=list(range(len(first))),
            layout=go.Layout(title=dict(
                text=(f"{event.definition.title} — full-contact 3-D ray trace"
                      f"<br><sub>{dt.strftime('%Y-%m-%d %H:%M:%S')} UTC — {status}; "
                      "all coordinates are unwarped</sub>"), x=0.5,
            )),
        ))
        slider_steps.append(dict(
            method="animate", label=dt.strftime("%H:%M"),
            args=[[frame_name], dict(mode="immediate",
                                     frame=dict(duration=0, redraw=True),
                                     transition=dict(duration=0))],
        ))
    fig.frames = frames

    true_earth_peak = _basis_transform(earth.reshape(1, 3), sun_peak, true_basis)[0]/AU_KM
    true_moon_all = _basis_transform(event.moon_km, sun_peak, true_basis)/AU_KM
    true_sun_all = _basis_transform(event.sun_km, sun_peak, true_basis)/AU_KM
    true_xmin = min(-R_SUN_KM/AU_KM*1.35, float(np.min(true_sun_all[:, 0]))-0.006)
    true_xmax = max(float(true_earth_peak[0]), float(np.max(true_moon_all[:, 0])))+0.012
    true_transverse = max(0.010, float(np.max(np.abs(true_moon_all[:, 1:])))+0.006)

    fig.update_scenes(
        xaxis=dict(range=[local_xmin, local_xmax], title="Earth-centered X [km]",
                   gridcolor="#273349", zerolinecolor="#53627a"),
        yaxis=dict(range=[-local_transverse, local_transverse], title="Y [km]",
                   gridcolor="#273349", zerolinecolor="#53627a"),
        zaxis=dict(range=[-local_transverse, local_transverse], title="Z [km]",
                   gridcolor="#273349", zerolinecolor="#53627a"),
        aspectmode="data", bgcolor="#010207",
        camera=dict(eye=dict(x=1.45, y=1.15, z=0.72)),
        row=1, col=1,
    )
    fig.update_scenes(
        xaxis=dict(range=[true_xmin, true_xmax], title="Optical-axis distance [AU]",
                   gridcolor="#273349", zerolinecolor="#53627a"),
        yaxis=dict(range=[-true_transverse, true_transverse], title="Transverse [AU]",
                   gridcolor="#273349", zerolinecolor="#53627a"),
        zaxis=dict(range=[-true_transverse, true_transverse], title="Normal [AU]",
                   gridcolor="#273349", zerolinecolor="#53627a"),
        aspectmode="data", bgcolor="#010207",
        camera=dict(eye=dict(x=0.12, y=1.5, z=0.65)),
        row=1, col=2,
    )
    initial_dt = jd_to_datetime(float(event.jd[0]))
    fig.update_layout(
        title=dict(
            text=(f"{event.definition.title} — full-contact 3-D ray trace"
                  f"<br><sub>{initial_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC — "
                  f"{_event_status_label(event, float(event.jd[0]))}; all coordinates are unwarped</sub>"),
            x=0.5,
        ),
        paper_bgcolor="#010207", plot_bgcolor="#010207",
        font=dict(color="white"), width=1600, height=820,
        margin=dict(l=0, r=0, t=105, b=70),
        legend=dict(orientation="h", x=0.02, y=0.01, bgcolor="rgba(1,2,7,0.72)"),
        updatemenus=[dict(
            type="buttons", direction="left", x=0.02, y=-0.04,
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, dict(fromcurrent=True,
                                      frame=dict(duration=150, redraw=True),
                                      transition=dict(duration=0))]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(mode="immediate",
                                       frame=dict(duration=0, redraw=False),
                                       transition=dict(duration=0))]),
            ],
        )],
        sliders=[dict(
            active=0, x=0.17, y=-0.03, len=0.80,
            currentvalue=dict(prefix="UTC ", font=dict(color="white")),
            steps=slider_steps,
        )],
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs=include_plotlyjs, full_html=True)
    return str(path)

def generate_validation_report(output_dir: str | Path, solar: ReferenceEvent,
                               lunar: ReferenceEvent) -> dict:
    """Write numerical metrics and a visual validation sheet."""
    import matplotlib
    matplotlib.use("Agg",force=True)
    import matplotlib.pyplot as plt
    from PIL import Image

    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)

    def all_frame_ray_metrics(event):
        penetration_totals = {"earth": 0, "moon": 0}
        maxima = {}
        sun_distances = []
        moon_distances = []
        for jd_value in event.jd:
            frame_bundle = trace_reference_rays(event.definition, float(jd_value), n_azimuth=12)
            counts = bundle_penetrations(frame_bundle)
            for key in penetration_totals:
                penetration_totals[key] += int(counts[key])
            for key, value in tangent_residuals(frame_bundle).items():
                maxima[key] = max(float(value), maxima.get(key, 0.0))
            sun_distances.append(float(np.linalg.norm(frame_bundle.sun_center_km-frame_bundle.earth_center_km)))
            moon_distances.append(float(np.linalg.norm(frame_bundle.moon_center_km-frame_bundle.earth_center_km)))
        return {
            "frame_count": int(len(event.jd)),
            "segment_penetration_totals": penetration_totals,
            "max_tangent_residuals": maxima,
            "sun_earth_distance_range_km": [min(sun_distances), max(sun_distances)],
            "earth_moon_distance_range_km": [min(moon_distances), max(moon_distances)],
        }

    solar_bundle=trace_reference_rays("solar",solar.greatest_jd,n_azimuth=24)
    lunar_bundle=trace_reference_rays("lunar",lunar.greatest_jd,n_azimuth=24)
    width=solar_cross_track_width_km(solar.greatest_jd,n_azimuth=720)
    computed=solar_central_line_wgs84(solar.greatest_jd)
    reference_lat=float(solar.metadata["greatest_lat_deg"]); reference_lon=float(solar.metadata["greatest_lon_east_deg"])
    metrics={
        "solar":{
            **reference_summary(solar),
            "computed_greatest_lat_deg":computed[0],
            "computed_greatest_lon_east_deg":computed[1],
            "greatest_lat_error_deg":computed[0]-reference_lat,
            "greatest_lon_error_deg":computed[1]-reference_lon,
            "computed_cross_track_width_km":width,
            "reference_path_width_km":solar.metadata["path_width_at_greatest_km"],
            "path_width_error_km":width-float(solar.metadata["path_width_at_greatest_km"]),
            "computed_local_totality_duration_s":(solar_local_contacts()["C3"]-solar_local_contacts()["C2"])*86400.0,
            "reference_local_totality_duration_s":solar.metadata["central_duration_at_greatest_s"],
            "penetrations_at_greatest":bundle_penetrations(solar_bundle),
            "tangent_residuals_at_greatest":tangent_residuals(solar_bundle),
            "animation_ray_audit":all_frame_ray_metrics(solar),
        },
        "lunar":{
            **reference_summary(lunar),
            "penetrations_at_greatest":bundle_penetrations(lunar_bundle),
            "tangent_residuals_at_greatest":tangent_residuals(lunar_bundle),
            "animation_ray_audit":all_frame_ray_metrics(lunar),
            "danjon_effective_earth_radius_km":LUNAR_DANJON_EARTH_RADIUS_KM,
        },
    }
    metrics_path=out/"validated_metrics.json";metrics_path.write_text(json.dumps(metrics,indent=2),encoding="utf-8")

    fig,axes=plt.subplots(2,2,figsize=(14,8),dpi=150,facecolor="#010207")
    for ax in axes.ravel(): ax.set_facecolor("#010207")
    # NASA path checkpoints versus Besselian result.
    checkpoints=[
        (17,0,1+42.7/60,-(129+41.2/60)),(18,0,20+19.2/60,-(108+45.8/60)),
        (18,16,24+54.8/60,-(104+29.7/60)),(18,18,25+29.1/60,-(103+56.8/60)),
        (19,0,37+19.7/60,-(89+46.6/60)),(19,40,47+40.1/60,-(60+44.7/60)),
        (19,54,48+28.2/60,-(27+19.0/60)),
    ]
    times=[];lat_err=[];lon_err=[]
    from ssatk.compute.eclipse_reference_events import datetime_to_jd,utc
    for h,m,lat_ref,lon_ref in checkpoints:
        jd=datetime_to_jd(utc(2024,4,8,h,m,0));point=solar_central_line_wgs84(jd)
        times.append(f"{h:02d}:{m:02d}");lat_err.append((point[0]-lat_ref)*111.2);lon_err.append((point[1]-lon_ref)*111.2*math.cos(math.radians(lat_ref)))
    x=np.arange(len(times));axes[0,0].plot(x,lat_err,marker="o",label="latitude error");axes[0,0].plot(x,lon_err,marker="s",label="longitude error")
    axes[0,0].axhline(0,color="white",lw=.6);axes[0,0].set_xticks(x,times,rotation=30);axes[0,0].set_ylabel("Approximate path error [km]");axes[0,0].legend(frameon=False,labelcolor="white",fontsize=8);axes[0,0].set_title("2024 solar central line vs NASA WGS-84 path table",color="white")
    axes[0,0].tick_params(colors="#aab7c9");axes[0,0].grid(alpha=.2)
    # Lunar contact geometry.
    draw_lunar_shadow_plane(axes[0,1],lunar.greatest_jd)
    # Peak body renders.
    axes[1,0].imshow(render_earth_disk(solar.greatest_jd,size=520));axes[1,0].axis("off");axes[1,0].set_title("Solar maximum — exact finite-disc shadow",color="white")
    axes[1,1].imshow(render_moon_disk(lunar.greatest_jd,size=520));axes[1,1].axis("off");axes[1,1].set_title("Lunar totality — Danjon geometry; color illustrative",color="white")
    fig.suptitle("Validated eclipse geometry and renderer diagnostics",color="white",fontsize=16)
    fig.tight_layout(rect=[0,0,1,.95])
    sheet=out/"validated_eclipse_diagnostics.png";fig.savefig(sheet,facecolor=fig.get_facecolor(),bbox_inches="tight");plt.close(fig)
    return {"metrics":str(metrics_path),"diagnostic_png":str(sheet)}


def generate_all_validated_outputs(output_dir: str | Path,
                                   *, n_frames: int = 65,
                                   width: int = 1280,
                                   height: int = 720,
                                   fps: int = 10,
                                   workers: int = 2,
                                   tasks_per_worker: int = 8) -> dict:
    """Generate all validated animations, true-scale 3-D views, and reports.

    Each raster animation uses short-lived worker batches.  No live worker is
    recycled, and FFmpeg reads numbered PNG files rather than an inherited stdin
    pipe.  This keeps memory bounded and makes consecutive exports deterministic.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    solar = build_reference_event("solar", n_frames=n_frames)
    solar_local = build_solar_local_event(n_frames=n_frames)
    lunar = build_reference_event("lunar", n_frames=n_frames)

    result = {
        "solar_global": generate_event_animation(
            solar, out, width=width, height=height, fps=fps,
            stem="solar_2024_04_08",
            title="2024 Total Solar Eclipse — global P1 to P4",
            workers=workers, tasks_per_worker=tasks_per_worker,
        ),
        "solar_local": generate_event_animation(
            solar_local, out, width=width, height=height, fps=fps,
            stem="solar_2024_04_08",
            title="2024 Total Solar Eclipse — NASA greatest-site C1 to C4",
            workers=workers, tasks_per_worker=tasks_per_worker,
        ),
        "lunar_global": generate_event_animation(
            lunar, out, width=width, height=height, fps=fps,
            stem="lunar_2025_03_14",
            title="2025 Total Lunar Eclipse — P1 to P4",
            workers=workers, tasks_per_worker=tasks_per_worker,
        ),
    }
    # Restore the interactive 3-D view as a combined scientific dashboard:
    # validated 2-D path/shadow and visibility panels on the left, true-scale
    # rotatable Earth-Moon-Sun geometry on the right.  Import lazily to avoid
    # a module cycle during package initialization.
    from ssatk.plots.eclipse_interactive_dashboard import generate_interactive_dashboard
    result["solar_local"]["peak_3d_html"] = generate_interactive_dashboard(
        solar_local, out/"solar_2024_greatest_interactive_3d.html", animated=False
    )
    result["lunar_global"]["peak_3d_html"] = generate_interactive_dashboard(
        lunar, out/"lunar_2025_greatest_interactive_3d.html", animated=False
    )
    result["solar_local"]["animation_3d_html"] = generate_interactive_dashboard(
        solar_local, out/"solar_2024_greatest_site_C1_C4_interactive_3d.html",
        animated=True
    )
    result["lunar_global"]["animation_3d_html"] = generate_interactive_dashboard(
        lunar, out/"lunar_2025_P1_P4_interactive_3d.html", animated=True
    )
    result["validation"] = generate_validation_report(out, solar, lunar)
    manifest = out/"validated_output_manifest.json"
    manifest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["manifest"] = str(manifest)
    return result
