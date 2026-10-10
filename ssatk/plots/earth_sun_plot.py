"""
ssatk/plots/earth_sun_plot.py
======================================
Focused Earth–Sun(–Moon) visualization — a zoomed-in companion to
solar_view_plot.py's full 8-planet scene.

Unlike solar_view_plot.py (a single frozen snapshot of the whole solar
system), this renders Earth's orbit as an actual animation: Earth moves
around the Sun over one full year, the Moon moves around Earth on its own
real date-based ephemeris, and the title/hover text update every frame with
the current date, Sun–Earth distance, Earth–Moon distance, and Moon phase.

Run via GUI "Export Plots" tab or directly:

    python -m ssatk.plots.earth_sun_plot

Reads GUI_CONFIG from environment or falls back to defaults.

Output
------
  <output_dir>/earth_sun_plot.html   — interactive, animated Plotly HTML
  <output_dir>/earth_sun_plot.jpg    — static snapshot (requires the ``static`` extra)
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from ssatk._paths import output_root
from ssatk.plots.solar_bodies import (
    _AU_KM,
    _R_AU,
    make_moon_traces,
    make_planet_traces,
    make_sun_traces,
    moon_geocentric_ecliptic,
)

# ── Project imports ───────────────────────────────────────────────────────────
# Earth's Keplerian elements and orbit math are already correct and validated
# in solar_view_plot.py — reused here rather than re-derived, so both scenes
# stay in sync if the elements are ever updated.
from ssatk.plots.solar_view_plot import (
    _PLANETS,
    _orbit_trail_au,
    _planet_pos_au,
)

# Real catalog + IAU 1976 precession (see _static_backdrop for why this calls
# starfield.py's private loader directly rather than its public add_starfield,
# which is matplotlib-only).
from ssatk.plots.starfield import _load_stars as _sf_load_stars

_EARTH = _PLANETS["Earth"]
_YEAR_DAYS = 365.25 * math.sqrt(_EARTH["a"] ** 3)   # ≈ 365.256 days
_SIDEREAL_MONTH_DAYS = 27.321661


def _shrink_floats(fig: go.Figure) -> go.Figure:
    """
    Downcast large float64 array attributes (x/y/z/surfacecolor/etc.) to
    float32 before writing HTML.

    Plotly embeds numeric arrays as raw binary (base64), sized by dtype, not
    by how many decimals the values print with — so rounding values does
    nothing, but the dtype itself is a real, free win. float32 keeps ~7
    significant digits, far finer than anything visible at this AU-to-sphere
    scale (the Moon's own display radius alone is ~1e-4 AU), and cuts the
    animated file size by roughly 40% since x/y/z resend every frame.
    Called only at save time — build_figure()/build_static_figure() still
    return full-precision figures for anyone doing further math on them.
    """
    def _shrink_trace(trace):
        for attr in list(trace):
            try:
                val = trace[attr]
            except Exception:
                continue
            if val is None:
                continue
            arr = np.asarray(val)
            if arr.dtype == np.float64 and arr.size > 8:
                trace[attr] = arr.astype(np.float32)

    for trace in fig.data:
        _shrink_trace(trace)
    for frame in fig.frames:
        for trace in frame.data:
            _shrink_trace(trace)
    return fig


# ── Moon phase (illuminated fraction, as seen from Earth) ─────────────────────
def _moon_phase_fraction(earth_pos_au: np.ndarray, moon_offset_km: np.ndarray) -> float:
    """
    Fraction of the Moon's disk illuminated as seen from Earth, using the
    Sun–Earth and Earth–Moon direction vectors (Sun >> Moon in distance, so
    phase angle ≈ π − elongation is an excellent approximation here).
    New Moon → 0.0, Full Moon → 1.0.
    """
    E = np.asarray(earth_pos_au, dtype=float)
    M = np.asarray(moon_offset_km, dtype=float)
    denom = np.linalg.norm(E) * np.linalg.norm(M)
    if denom < 1e-30:
        return 0.5
    cos_elongation = float(np.dot(E, M) / denom)
    cos_elongation = max(-1.0, min(1.0, cos_elongation))
    return 0.5 * (1.0 + cos_elongation)


def _jd_to_date(t_jd: float) -> _dt.date:
    return _dt.date(2000, 1, 1) + _dt.timedelta(days=round(t_jd - 2_451_545.0))


def _title_text(t_jd: float, earth_pos_au: np.ndarray, moon_offset_km: np.ndarray) -> str:
    r_se_km = float(np.linalg.norm(earth_pos_au)) * _AU_KM
    r_em_km = float(np.linalg.norm(moon_offset_km))
    phase = _moon_phase_fraction(earth_pos_au, moon_offset_km)
    return (
        f"Earth–Sun System — {_jd_to_date(t_jd).isoformat()}"
        f"  |  Sun–Earth {r_se_km/1e6:.2f} M km"
        f"  |  Earth–Moon {r_em_km:,.0f} km"
        f"  |  Moon illum. {phase*100:.0f}%"
    )


# ── Static backdrop (built once, shared by every frame) ───────────────────────
def _static_backdrop(cfg: dict, t_jd0: float) -> list:
    traces = []

    # Earth's full-year orbit — computed once; over one year the ellipse
    # itself doesn't visibly change, so it's a static reference rather than
    # something recomputed per frame.
    ex, ey, ez = _orbit_trail_au("Earth", t_jd0, n_pts=300)
    traces.append(go.Scatter3d(
        x=ex, y=ey, z=ez, mode="lines",
        line=dict(color="rgba(90,160,230,0.45)", width=2),
        name="Earth's orbit", hoverinfo="skip",
    ))

    for t in make_sun_traces(r_display_au=_R_AU["Sun"] * float(cfg.get("planet_scale", 1.0))):
        traces.append(t)

    if bool(cfg.get("show_stars", True)):
        # Real catalog + IAU 1976 precession to this scene's epoch, via the
        # shared starfield.py module (same one the rest of the toolkit's
        # matplotlib plots use). starfield.py's own add_starfield() draws
        # directly onto a matplotlib Axes3D, which doesn't apply here — so
        # this calls its underlying _load_stars() directly (the same
        # pattern starfield_verification_plotly.py uses) and builds a
        # Plotly trace from the returned precessed positions.
        mag_limit = float(cfg.get("star_mag_limit", 6.5))
        stars = _sf_load_stars(mag_limit=mag_limit, epoch_jd=t_jd0)

        R_star = 1.8   # AU — just outside the orbit; must stay within the
                       # scene's axis range (_scene_layout) or every star
                       # gets clipped out of the visible scene entirely.
                       # (Previously 6.0 AU against a ±1.3 AU range — every
                       # star was silently outside the plot bounds.)
        if stars is not None:
            cx, cy, cz = stars["cx"], stars["cy"], stars["cz"]
            colors = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
                      for r, g, b in stars["colors"]]
            # starfield.py's sizes are tuned for matplotlib's ax.scatter(s=...),
            # which is an AREA in points² (needs values in the tens-to-hundreds
            # to read clearly). Plotly's marker size is a pixel DIAMETER, where
            # even "4.0" — starfield.py's own max — is a small dot, and its
            # dimmer-star values (many stars sit well under 1.0) round down to
            # sub-pixel and simply don't render. Rescaling by the actual loaded
            # brightness range, rather than reusing the raw values, keeps the
            # brightest-to-dimmest ordering but makes every star land in a
            # size Plotly actually draws.
            #
            # IMPORTANT: keep this range fairly narrow (below ~3x smallest-to-
            # largest). A wider range (originally 1.2–5.2) was confirmed, via
            # extensive bisection against real headless-Chromium WebGL
            # rendering, to trigger a Plotly/WebGL depth-sorting bug: larger
            # star markers would render *in front of* the opaque Earth sphere
            # instead of being occluded behind it, showing up as a dark blob
            # on Earth's lit hemisphere. This wasn't a data bug (no single
            # "bad" star was responsible — removing any one star didn't fix
            # it) and wasn't fixed by angular exclusion, opacity changes, or
            # reducing star count down to 168 — only reducing the marker
            # *size range* eliminated it in every test.
            raw = np.asarray(stars["sizes"], dtype=float)
            lo, hi = raw.min(), raw.max()
            sizes = 0.8 + 2.2 * (raw - lo) / (hi - lo + 1e-9)
            print(f"[earth_sun_plot] Loaded {stars['n']} stars from the real "
                  f"catalog, precessed to JD {t_jd0:.1f}.")
        else:
            print("[earth_sun_plot] No star catalog found via starfield.py's "
                  "search paths — using a synthetic starfield.")
            rng = np.random.default_rng(7)
            n_syn = 1200
            th = rng.uniform(0, 2 * np.pi, n_syn)
            ph = np.arccos(rng.uniform(-1, 1, n_syn))
            cx, cy, cz = np.sin(ph) * np.cos(th), np.sin(ph) * np.sin(th), np.cos(ph)
            sizes = rng.uniform(1.2, 3.5, n_syn)   # same Plotly-pixel scale as above
            colors = "white"

        traces.append(go.Scatter3d(
            x=cx * R_star, y=cy * R_star, z=cz * R_star,
            mode="markers",
            marker=dict(size=sizes, color=colors, opacity=0.85),
            hoverinfo="skip", name="Stars", showlegend=True,
        ))

    return traces


# ── Dynamic traces (Earth, Moon, trails — recomputed every frame) ────────────
def _dynamic_traces(cfg: dict, t_jd: float, n: int):
    scale_au    = float(cfg.get("planet_scale", 1.0))
    moon_scale  = float(cfg.get("moon_orbit_scale", 20.0))
    show_labels = bool(cfg.get("show_labels", True))

    earth_pos = np.array(_planet_pos_au(_EARTH, t_jd))
    traces = list(make_planet_traces("Earth", tuple(earth_pos), scale_au=scale_au,
                                      show_label=show_labels, n=n, time=t_jd))

    moon_offset_km = np.array(moon_geocentric_ecliptic(t_jd))
    traces.extend(make_moon_traces(tuple(earth_pos), t_jd, orbit_scale=moon_scale,
                                    show_label=show_labels, n=n))

    if bool(cfg.get("show_moon_trail", True)):
        # The Moon's recent path around Earth, drawn relative to Earth's
        # *current* position (not its historical position a month ago) —
        # this is a local Earth–Moon diagram, so the trail should show the
        # Earth-relative geometry, not get smeared by Earth's own motion
        # around the Sun over that same month.
        pts = []
        for k in range(41):
            tj = t_jd - _SIDEREAL_MONTH_DAYS + k * _SIDEREAL_MONTH_DAYS / 40.0
            off_km = np.array(moon_geocentric_ecliptic(tj))
            pts.append(earth_pos + off_km / _AU_KM * moon_scale)
        pts = np.array(pts)
        traces.append(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="lines",
            line=dict(color="rgba(205,205,215,0.35)", width=1.5),
            hoverinfo="skip", showlegend=False, name="Moon's path",
        ))

    if bool(cfg.get("show_radius_line", True)):
        traces.append(go.Scatter3d(
            x=[0, earth_pos[0]], y=[0, earth_pos[1]], z=[0, earth_pos[2]],
            mode="lines",
            line=dict(color="rgba(255,210,120,0.35)", width=1, dash="dot"),
            hoverinfo="skip", showlegend=False, name="Sun–Earth line",
        ))

    return traces, earth_pos, moon_offset_km


def _scene_layout(cfg: dict, bg: str) -> dict:
    rng = 2.0   # must exceed the star radius (R_star, in _static_backdrop) or
                # stars render outside the visible scene and get clipped
    axis_kw = dict(showbackground=False, showgrid=False, zeroline=False,
                   showticklabels=False, title="")
    # A shallow camera elevation (previously ~27°) means that whenever the
    # *current date* happens to put Earth near the same azimuth as the
    # camera, Earth visually overlaps the Sun in the projection — they're
    # still 1 AU apart in the data, but the near-edge-on view hides that
    # separation. A steep, near-top-down elevation (~55°+) keeps Sun and
    # Earth visibly separated across the whole orbit regardless of date
    # (checked across several dates spanning the year before settling here).
    return dict(
        xaxis=dict(range=[-rng, rng], **axis_kw),
        yaxis=dict(range=[-rng, rng], **axis_kw),
        # Was rng*0.25 (±0.5) — fine for the orbit itself (genuinely flat,
        # z≈0), but the star sphere is isotropic and needs z-range out to
        # its own radius (R_star=1.8) or every star outside a thin
        # equatorial band gets clipped, which is exactly why stars were
        # rendering in a ring instead of covering the sky. aspectratio z
        # widened to match so spheres still render round, not squashed.
        zaxis=dict(range=[-rng, rng], **axis_kw),
        bgcolor=bg, aspectmode="manual", aspectratio=dict(x=1, y=1, z=1),
        camera=dict(eye=dict(x=0.31, y=-0.54, z=0.89), up=dict(x=0, y=0, z=1)),
        # (Steepening the elevation angle last round also accidentally
        # increased the camera *distance* — same angle, farther away — which
        # shrank the whole scene down to a small fraction of the frame. This
        # eye vector keeps the corrected elevation but at a distance that
        # actually fills the frame.)
    )


def _resolve_start_jd(cfg: dict) -> float:
    yr = cfg.get("start_year")
    mo = cfg.get("start_month")
    dy = cfg.get("start_day")
    if yr is None or mo is None or dy is None:
        d0 = _dt.date.today()
    else:
        try:
            d0 = _dt.date(int(yr), int(mo), int(dy))
        except ValueError:
            d0 = _dt.date.today()
    return 2_451_545.0 + (d0 - _dt.date(2000, 1, 1)).days


# ── Figure builders ────────────────────────────────────────────────────────────
def build_figure(cfg: dict) -> go.Figure:
    """Animated Earth-orbits-the-Sun figure — one frame per step across a year."""
    t_jd0      = _resolve_start_jd(cfg)
    n_frames   = int(cfg.get("n_frames", 48))
    frame_ms   = int(cfg.get("frame_duration_ms", 110))
    sphere_res = int(cfg.get("sphere_resolution", 28))
    bg         = cfg.get("bg_color", "#060810")

    static_traces = _static_backdrop(cfg, t_jd0)
    n_static = len(static_traces)

    frame0_traces, earth_pos0, moon_off0 = _dynamic_traces(cfg, t_jd0, sphere_res)
    n_dynamic = len(frame0_traces)
    dyn_idx = list(range(n_static, n_static + n_dynamic))

    day_steps = np.linspace(0, _YEAR_DAYS, n_frames, endpoint=False)
    frames = []
    for i, dstep in enumerate(day_steps):
        t_jd = t_jd0 + dstep
        dyn_traces, epos, moff = _dynamic_traces(cfg, t_jd, sphere_res)
        frames.append(go.Frame(
            data=dyn_traces, traces=dyn_idx, name=str(i),
            layout=go.Layout(title=dict(text=_title_text(t_jd, epos, moff))),
        ))
    fig = go.Figure(data=static_traces + frame0_traces, frames=frames)

    slider_steps = [
        dict(
            label=_jd_to_date(t_jd0 + d).strftime("%b %d"),
            method="animate",
            args=[[str(i)], dict(mode="immediate",
                                  frame=dict(duration=0, redraw=True),
                                  transition=dict(duration=0))],
        )
        for i, d in enumerate(day_steps)
    ]

    fig.update_layout(
        scene=_scene_layout(cfg, bg),
        paper_bgcolor=bg,
        font=dict(color="#C8D8E8"),
        title=dict(text=_title_text(t_jd0, earth_pos0, moon_off0),
                   x=0.5, font=dict(color="#00FF9C", size=13)),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="#333",
                    borderwidth=1, font=dict(size=10)),
        margin=dict(l=0, r=0, t=60, b=0),
        updatemenus=[dict(
            type="buttons", showactive=False,
            x=0.05, y=0.02, xanchor="left", yanchor="bottom",
            pad=dict(t=0, r=6),
            buttons=[
                dict(label="▶  Play", method="animate",
                     args=[None, dict(frame=dict(duration=frame_ms, redraw=True),
                                       transition=dict(duration=0),
                                       fromcurrent=True, mode="immediate")]),
                dict(label="⏸  Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                         mode="immediate")]),
            ],
        )],
        sliders=[dict(
            active=0, x=0.15, y=0.02, len=0.8,
            currentvalue=dict(prefix="", visible=False),
            pad=dict(t=0), steps=slider_steps,
        )],
    )
    return fig


def build_static_figure(cfg: dict) -> go.Figure:
    """A single high-resolution, non-animated frame — for the JPG snapshot."""
    t_jd0   = _resolve_start_jd(cfg)
    hero_n  = int(cfg.get("hero_resolution", 64))
    bg      = cfg.get("bg_color", "#060810")

    static_traces = _static_backdrop(cfg, t_jd0)
    dyn_traces, earth_pos0, moon_off0 = _dynamic_traces(cfg, t_jd0, hero_n)

    fig = go.Figure(data=static_traces + dyn_traces)
    fig.update_layout(
        scene=_scene_layout(cfg, bg),
        paper_bgcolor=bg,
        font=dict(color="#C8D8E8"),
        title=dict(text=_title_text(t_jd0, earth_pos0, moon_off0),
                   x=0.5, font=dict(color="#00FF9C", size=13)),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="#333",
                    borderwidth=1, font=dict(size=10)),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────
DEFAULT_CFG = dict(
    start_year=None, start_month=None, start_day=None,   # None → today
    n_frames=40,
    frame_duration_ms=110,
    planet_scale=3.5,               # Earth's true display radius (0.021 AU) is
                                     # ~1.6% of the scene width — a few pixels,
                                     # too small to show shading or texture at
                                     # all. Exaggerated further here (on top of
                                     # solar_bodies.py's already-exaggerated
                                     # base sizes) purely for this close-in view.
    moon_orbit_scale=65.0,          # Moon's real distance is stretched, but
                                    # NOT by the same ~500x/3.5x exaggeration
                                    # as planet sizes — just enough to clear
                                    # Earth's own display radius. This must
                                    # be recalibrated any time planet_scale
                                    # changes: Earth's display radius is
                                    # 0.021*planet_scale AU, and the Moon's
                                    # exaggerated distance is (real distance,
                                    # ~0.0024-0.0027 AU) * moon_orbit_scale —
                                    # the latter must clearly exceed the
                                    # former even at perigee, or the Moon
                                    # renders *inside* the Earth sphere (as
                                    # it did here: at planet_scale=3.5, the
                                    # old moon_orbit_scale=20 put the Moon's
                                    # orbit at 0.048-0.054 AU against an Earth
                                    # radius of 0.0735 AU — always inside).
    sphere_resolution=90,           # per-frame animated sphere mesh — real Earth
                                     # texture now backs this (see solar_bodies.py).
                                     # x/y/z resend every frame, but _shrink_floats()
                                     # downcasts to float32 at save time (~40% smaller,
                                     # no visible loss), which is what makes n=90
                                     # affordable here (~23MB) instead of n=65.
    hero_resolution=320,            # single high-res frame for the JPG
    show_labels=True,
    show_stars=False,               # Off by default. Extensive debugging (bisection
                                     # against real headless-Chromium WebGL rendering,
                                     # not just the static JPG) traced a real dark-blob
                                     # artifact on Earth to the Stars trace, and found
                                     # it's sensitive to marker size range — but ALSO,
                                     # unexpectedly, to the browser's viewport size
                                     # (clean at 1000x700, still present at 1400x900
                                     # with identical data). That points to a Plotly.js
                                     # WebGL depth-precision issue I can't reliably
                                     # eliminate from this side. The size-range fix
                                     # below is a genuine improvement and stays in,
                                     # but rather than risk shipping a residual visual
                                     # bug by default, this is opt-in: set True if you
                                     # want to try it (works cleanly in many window
                                     # sizes, just not guaranteed in all of them).
    star_mag_limit=6.5,             # deeper = more (dimmer) stars; starfield.py
                                     # finds its own catalog file (same search
                                     # paths it already uses for the matplotlib
                                     # plots) — no path to configure here.
    show_radius_line=True,
    show_moon_trail=True,
    bg_color="#060810",
    output_dir=str(output_root() / "figures"),
)

if __name__ == "__main__":
    cfg = DEFAULT_CFG.copy()
    env_cfg = os.environ.get("GUI_CONFIG", "")
    if env_cfg:
        try:
            cfg.update(json.loads(env_cfg))
            print("[earth_sun_plot] Loaded GUI_CONFIG from environment.")
        except json.JSONDecodeError as e:
            print(f"[earth_sun_plot] Warning: bad GUI_CONFIG ({e}); using defaults.")

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[earth_sun_plot] Building animated Earth–Sun scene...")
    fig = build_figure(cfg)
    fig = _shrink_floats(fig)
    html_path = output_dir / "earth_sun_plot.html"
    fig.write_html(str(html_path))
    print(f"[earth_sun_plot] Saved → {html_path}")

    print("[earth_sun_plot] Building high-res static snapshot...")
    hero_fig = build_static_figure(cfg)
    jpg_path = output_dir / "earth_sun_plot.jpg"
    try:
        hero_fig.write_image(str(jpg_path), width=1920, height=1080, scale=2)
        print(f"[earth_sun_plot] Saved → {jpg_path}")
    except Exception as e:
        print(f"[earth_sun_plot] write_image failed: {e} — install ssatk[static]")

    print("[earth_sun_plot] Done.")
