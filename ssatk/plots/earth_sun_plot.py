"""
ssatk/plots/earth_sun_plot.py
======================================
Solar System — Sun, Earth and Moon: a zoomed-in companion to
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
  <output_dir>/solar_system_earth_sun_moon.html   — interactive, animated Plotly HTML
  <output_dir>/solar_system_earth_sun_moon.jpg    — static snapshot (requires the ``static`` extra)
"""

from __future__ import annotations

from .rendering import _with_low_spec

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
    _moon_offset_au,
)

# ── Project imports ───────────────────────────────────────────────────────────
# Share the same ephemeris and J2000 ecliptic convention as solar_view_plot.
from ssatk.plots.solar_view_plot import (
    _PLANETS,
    _orbit_trail_au,
    _body_positions_au,
)

from ssatk.plots._solar_scene import SolarFigure, add_sky, scale_caption
from astropy.time import Time

# earth_sun_plot.py, sensor_fov_plot.py and solar_view_plot.py each define
# their own unrelated build_figure()/DEFAULT_CFG -- ssatk.plots flattens
# every public module-level name into one namespace (see ssatk._namespace),
# so without this, whichever of the three sorted last alphabetically would
# silently shadow the other two at ssatk.plots.build_figure. Every test and
# demo already imports these from the specific submodule, so omitting them
# here does not change any existing import.
__all__ = ["build_static_figure"]

_EARTH = _PLANETS["Earth"]
_YEAR_DAYS = 365.25 * math.sqrt(_EARTH["a"] ** 3)   # ≈ 365.256 days
_SIDEREAL_MONTH_DAYS = 27.321661


def _shrink_floats(fig: go.Figure) -> go.Figure:
    """Compact surface colours while preserving physical position precision.

    Float32 AU coordinates lose kilometres near Earth: retain x/y/z as
    float64 now that bodies use their physical radii.
    """
    def _shrink_trace(trace):
        for attr in ("surfacecolor",):
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
    return Time(t_jd, format="jd", scale="utc").to_datetime().date()


def _title_text(t_jd: float, earth_pos_au: np.ndarray, moon_offset_km: np.ndarray, cfg: dict) -> str:
    r_se_km = float(np.linalg.norm(earth_pos_au)) * _AU_KM
    r_em_km = float(np.linalg.norm(moon_offset_km))
    phase = _moon_phase_fraction(earth_pos_au, moon_offset_km)
    return (
        f"Solar System — Sun, Earth and Moon — {_jd_to_date(t_jd).isoformat()}"
        f"  |  Sun–Earth {r_se_km/1e6:.2f} M km"
        f"  |  Earth–Moon {r_em_km:,.0f} km"
        f"  |  Moon illum. {phase*100:.0f}%"
        f"<br><sup>J2000 ecliptic · {scale_caption(cfg)}</sup>"
    )


# ── Static backdrop (built once, shared by every frame) ───────────────────────
def _static_backdrop(cfg: dict, t_jd0: float) -> list:
    traces = []

    # Earth's full-year orbit — computed once; over one year the ellipse
    # itself doesn't visibly change, so it's a static reference rather than
    # something recomputed per frame.
    ex, ey, ez = _orbit_trail_au("Earth", t_jd0, ephemeris=cfg.get("sol_ephemeris", "builtin"), n_pts=720)
    traces.append(go.Scatter3d(
        x=ex, y=ey, z=ez, mode="lines",
        line=dict(color="rgba(90,160,230,0.45)", width=2),
        name="Earth's orbit", hoverinfo="skip",
    ))

    for t in make_sun_traces(r_display_au=_R_AU["Sun"] * float(cfg.get("sun_scale", 1.0))):
        traces.append(t)

    return traces


# ── Dynamic traces (Earth, Moon, trails — recomputed every frame) ────────────
def _dynamic_traces(cfg: dict, t_jd: float, n: int):
    scale_au    = float(cfg.get("planet_scale", 1.0))
    moon_scale  = float(cfg.get("moon_orbit_scale", 1.0))
    ephemeris = cfg.get("sol_ephemeris", "builtin")
    show_labels = bool(cfg.get("show_labels", True))

    earth_pos = _body_positions_au("Earth", t_jd, ephemeris=ephemeris)
    traces = list(make_planet_traces("Earth", tuple(earth_pos), scale_au=scale_au,
                                      show_label=show_labels, n=n, time=t_jd))

    moon_offset_km = _moon_offset_au(t_jd, ephemeris=ephemeris) * _AU_KM
    traces.extend(make_moon_traces(tuple(earth_pos), t_jd, orbit_scale=moon_scale,
                                    show_label=show_labels, n=n, scale_au=scale_au, ephemeris=ephemeris))

    if bool(cfg.get("show_moon_trail", True)):
        # The Moon's recent path around Earth, drawn relative to Earth's
        # *current* position (not its historical position a month ago) —
        # this is a local Earth–Moon diagram, so the trail should show the
        # Earth-relative geometry, not get smeared by Earth's own motion
        # around the Sun over that same month.
        samples = np.linspace(t_jd - _SIDEREAL_MONTH_DAYS, t_jd, 361)
        pts = earth_pos + _moon_offset_au(samples, ephemeris=ephemeris) * moon_scale
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
    rng = 1.2
    axis_kw = dict(showbackground=False, showgrid=False, zeroline=False,
                   showticklabels=False, title="")
    return dict(
        xaxis=dict(range=[-rng, rng], **axis_kw),
        yaxis=dict(range=[-rng, rng], **axis_kw),
        zaxis=dict(range=[-rng, rng], **axis_kw),
        bgcolor=bg, aspectmode="cube",
        camera=dict(eye=dict(x=0.31, y=-0.54, z=0.89), up=dict(x=0, y=0, z=1)),
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
    return float(Time(d0.isoformat(), scale="utc").jd)


# ── Figure builders ────────────────────────────────────────────────────────────
@_with_low_spec
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
    camera_positions = []
    for i, dstep in enumerate(day_steps):
        t_jd = t_jd0 + dstep
        dyn_traces, epos, moff = _dynamic_traces(cfg, t_jd, sphere_res)
        camera_positions.append(epos)
        frames.append(go.Frame(
            data=dyn_traces, traces=dyn_idx, name=str(i),
            layout=go.Layout(title=dict(text=_title_text(t_jd, epos, moff, cfg))),
        ))
    fig = SolarFigure(data=static_traces + frame0_traces, frames=frames)

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
        title=dict(text=_title_text(t_jd0, earth_pos0, moon_off0, cfg),
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
    from .camera import _plotly_camera
    _plotly_camera(fig, cfg.get("camera", "overview"), camera_positions, times=t_jd0+day_steps)
    if cfg.get("show_stars", True):
        add_sky(fig, cfg, Time(t_jd0, format="jd"))
    return fig


@_with_low_spec
def build_static_figure(cfg: dict) -> go.Figure:
    """A single high-resolution, non-animated frame — for the JPG snapshot."""
    t_jd0   = _resolve_start_jd(cfg)
    hero_n  = int(cfg.get("hero_resolution", 64))
    bg      = cfg.get("bg_color", "#060810")

    static_traces = _static_backdrop(cfg, t_jd0)
    dyn_traces, earth_pos0, moon_off0 = _dynamic_traces(cfg, t_jd0, hero_n)

    fig = SolarFigure(data=static_traces + dyn_traces)
    fig.update_layout(
        scene=_scene_layout(cfg, bg),
        paper_bgcolor=bg,
        font=dict(color="#C8D8E8"),
        title=dict(text=_title_text(t_jd0, earth_pos0, moon_off0, cfg),
                   x=0.5, font=dict(color="#00FF9C", size=13)),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="#333",
                    borderwidth=1, font=dict(size=10)),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    if cfg.get("show_stars", True):
        add_sky(fig, cfg, Time(t_jd0, format="jd"))
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────
DEFAULT_CFG = dict(
    start_year=None, start_month=None, start_day=None,
    n_frames=365, frame_duration_ms=80,
    planet_scale=1.0, sun_scale=1.0, moon_orbit_scale=1.0,
    sol_ephemeris="builtin",
    sphere_resolution=28, hero_resolution=64,
    show_labels=True, show_stars=True, star_mag_limit=6.5,
    show_radius_line=True, show_moon_trail=True,
    bg_color="#060810", output_dir=str(output_root() / "figures"),
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

    print("[earth_sun_plot] Building animated Solar System — Sun, Earth and Moon scene...")
    fig = build_figure(cfg)
    fig = _shrink_floats(fig)
    html_path = output_dir / "solar_system_earth_sun_moon.html"
    fig.write_html(str(html_path))
    print(f"[earth_sun_plot] Saved → {html_path}")

    print("[earth_sun_plot] Building high-res static snapshot...")
    hero_fig = build_static_figure(cfg)
    jpg_path = output_dir / "solar_system_earth_sun_moon.jpg"
    try:
        hero_fig.write_image(str(jpg_path), width=1920, height=1080, scale=2)
        print(f"[earth_sun_plot] Saved → {jpg_path}")
    except Exception as e:
        print(f"[earth_sun_plot] write_image failed: {e} — install ssatk[static]")

    print("[earth_sun_plot] Done.")
