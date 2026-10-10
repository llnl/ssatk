"""
ssatk/plots/solar_view_plot.py
========================================
Standalone solar system visualization script.
Renders properly 3D-shaded planets using solar_bodies.py.

Run via GUI "Export Plots" tab or directly:

    python -m ssatk.plots.solar_view_plot

Reads GUI_CONFIG from environment or falls back to defaults.

Output
------
  <output_dir>/solar_view_plot.html   — interactive Plotly HTML
  <output_dir>/solar_view_plot.jpg    — static snapshot (requires the ``static`` extra)
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from ssatk._paths import output_root

# ── Project imports ───────────────────────────────────────────────────────────
from ssatk.plots.solar_bodies import (
    _R_AU,
    make_moon_traces,
    make_planet_traces,
    make_saturn_ring_traces,
    make_sun_traces,
)
from ssatk.plots.starfield import starfield_traces

# Always define these locally — importing toolkit_gui.py here is unsafe:
# it's a Streamlit *app* script, not a library, and merely importing it
# executes its entire top-level code (page config, CSS, sidebar, all 8
# tabs) outside of a real Streamlit session. That doesn't just fail
# quietly — it floods the terminal with one "missing ScriptRunContext"
# warning per Streamlit call in that file (thousands of lines).
_PLANETS = {
    "Mercury": dict(a=0.38710, e=0.20563, i=7.005,  Om=48.331,  w=29.125,  M0=174.795),
    "Venus":   dict(a=0.72333, e=0.00677, i=3.395,  Om=76.680,  w=54.884,  M0=50.416),
    "Earth":   dict(a=1.00000, e=0.01671, i=0.000,  Om=0.000,   w=102.937, M0=357.527),
    "Mars":    dict(a=1.52366, e=0.09341, i=1.850,  Om=49.558,  w=286.502, M0=19.393),
    "Jupiter": dict(a=5.20336, e=0.04839, i=1.303,  Om=100.464, w=273.867, M0=20.065),
    "Saturn":  dict(a=9.53707, e=0.05415, i=2.485,  Om=113.666, w=339.391, M0=317.020),
    "Uranus":  dict(a=19.1913, e=0.04717, i=0.773,  Om=74.230,  w=96.998,  M0=142.827),
    "Neptune": dict(a=30.0690, e=0.00859, i=1.770,  Om=131.722, w=272.847, M0=259.780),
}

def _solve_kepler(M: float, e: float) -> float:
    E = M
    for _ in range(60):
        dE = (M - E + e * math.sin(E)) / (1.0 - e * math.cos(E))
        E += dE
        if abs(dE) < 1e-12:
            break
    return E

def _planet_pos_au(p: dict, t_jd: float):
    J2000 = 2_451_545.0
    T_days = t_jd - J2000
    n_deg_day = 360.0 / (365.25 * math.sqrt(p["a"] ** 3))
    M = math.radians((p["M0"] + n_deg_day * T_days) % 360.0)
    E = _solve_kepler(M, p["e"])
    nu = 2.0 * math.atan2(math.sqrt(1 + p["e"]) * math.sin(E / 2),
                           math.sqrt(1 - p["e"]) * math.cos(E / 2))
    r = p["a"] * (1.0 - p["e"] * math.cos(E))
    xo, yo = r * math.cos(nu), r * math.sin(nu)
    w  = math.radians(p["w"])
    i  = math.radians(p["i"])
    Om = math.radians(p["Om"])
    cw, sw = math.cos(w), math.sin(w)
    ci, si = math.cos(i), math.sin(i)
    cO, sO = math.cos(Om), math.sin(Om)
    x = cO*(cw*xo - sw*yo) - sO*(sw*xo + cw*yo)*ci
    y = sO*(cw*xo - sw*yo) + cO*(sw*xo + cw*yo)*ci
    z = (sw*xo + cw*yo)*si
    return x, y, z

def _fallback_planet_positions_au(name: str, t_jd):
    p = _PLANETS[name]
    jd = np.atleast_1d(np.asarray(t_jd, dtype=float))
    out = np.array([_planet_pos_au(p, float(t)) for t in jd], dtype=float)
    return out[0] if np.ndim(t_jd) == 0 else out


def _body_positions_au(name: str, time, ephemeris: str = "builtin"):
    """Return heliocentric body positions in AU using Astropy ephemerides."""
    try:
        from astropy import units as u
        from astropy.coordinates import get_body_barycentric, solar_system_ephemeris

        with solar_system_ephemeris.set(ephemeris):
            body = get_body_barycentric(name.lower(), time)
            sun = get_body_barycentric("sun", time)
        xyz = (body.xyz - sun.xyz).to_value(u.au)
        return np.moveaxis(np.asarray(xyz, dtype=float), 0, -1)
    except Exception as exc:
        jd = getattr(time, "jd", time)
        print(
            f"[solar_view_plot] Astropy ephemeris '{ephemeris}' failed for {name} "
            f"({exc}); using low-precision Kepler fallback."
        )
        return _fallback_planet_positions_au(name, jd)


def _orbit_trail_au(
    name: str,
    t_jd: float,
    time=None,
    ephemeris: str = "builtin",
    n_pts: int = 360,
):
    p = _PLANETS[name]
    T_orbit_days = 365.25 * math.sqrt(p["a"] ** 3)
    offsets = np.linspace(-T_orbit_days, 0.0, n_pts + 1)
    if time is not None:
        sample_years = float(time.jyear) + offsets / 365.25
        if (
            np.nanmin(sample_years) >= 1972.0
            and np.nanmax(sample_years) <= 2100.0
        ):
            try:
                from astropy import units as u
                samples = time + offsets * u.day
                pos = np.asarray(
                    _body_positions_au(name, samples, ephemeris=ephemeris),
                    dtype=float,
                )
                return pos[:, 0], pos[:, 1], pos[:, 2]
            except Exception:
                pass

    pos = _fallback_planet_positions_au(name, t_jd + offsets)
    return pos[:, 0], pos[:, 1], pos[:, 2]


def _mars_view_radius_au(margin: float = 1.10) -> float:
    mars = _PLANETS["Mars"]
    return float(mars["a"] * (1.0 + mars["e"]) * margin)


def _view_radius_au(cfg: dict, full_content_radius_au: float) -> float:
    value = cfg.get("view_radius_au", cfg.get("sol_view_radius_au", "mars"))
    if isinstance(value, str):
        cleaned = value.strip().lower().replace("-", "_").replace(" ", "_")
        if cleaned in {"mars", "inner", "inner_solar_system"}:
            return _mars_view_radius_au(float(cfg.get("mars_view_margin", 1.10)))
        if cleaned in {"full", "auto", "all", "all_planets"}:
            return float(full_content_radius_au)
    try:
        return max(float(value), 0.1)
    except Exception:
        return _mars_view_radius_au(float(cfg.get("mars_view_margin", 1.10)))


def _camera_eye_from_cfg(cfg: dict):
    value = cfg.get("camera_eye", cfg.get("sol_camera_eye"))
    default = dict(x=0.75, y=-0.95, z=0.09)
    if value is None:
        return default
    if isinstance(value, dict):
        return {axis: float(value.get(axis, default[axis])) for axis in ("x", "y", "z")}
    try:
        vals = list(value)
        return {axis: float(vals[i]) for i, axis in enumerate(("x", "y", "z"))}
    except Exception:
        return default


def _scene_time(cfg: dict):
    import datetime as _dt

    yr = int(cfg.get("sol_year", 2025))
    mo = int(cfg.get("sol_month", 1))
    try:
        d = _dt.date(yr, mo, 1)
    except ValueError:
        d = _dt.date(2025, 1, 1)

    try:
        from astropy.time import Time
        time = Time(f"{d.isoformat()}T00:00:00", scale="utc")
        return d, float(time.jd), time
    except Exception:
        t_jd = 2_451_545.0 + (d - _dt.date(2000, 1, 1)).days - 0.5
        return d, t_jd, None



# ── Figure builder ────────────────────────────────────────────────────────────

def build_figure(cfg: dict) -> go.Figure:
    # Resolve epoch. Planet positions are heliocentric: Astropy barycentric
    # body positions minus the Astropy Sun position, with a Kepler fallback.
    d, t_jd, time = _scene_time(cfg)
    ephemeris = str(cfg.get("sol_ephemeris", "builtin"))

    show_planets = {
        name: bool(cfg.get(f"sol_show_{name.lower()}", True))
        for name in _PLANETS
    }
    show_trails   = bool(cfg.get("sol_show_trails",   True))
    show_stars    = bool(cfg.get("sol_show_stars",    True))
    show_ecliptic = bool(cfg.get("sol_show_ecliptic", True))
    show_labels   = bool(cfg.get("sol_show_labels",   True))
    show_moon     = bool(cfg.get("sol_show_moon",     True))
    scale_au      = float(cfg.get("planet_scale",     1.0))
    sun_scale     = float(cfg.get("sun_scale",        1.0))
    star_mag_limit = float(cfg.get("star_mag_limit", 6.5))
    catalog_path = cfg.get("star_catalog")
    sphere_res    = int(cfg.get("sphere_resolution",  50))
    bg            = cfg.get("bg_color", "#060810")
    outer_a = max((p["a"] for n, p in _PLANETS.items() if show_planets.get(n)), default=1.5)
    full_content_radius_au = max(outer_a * 1.25, 3.0)
    view_radius_au = _view_radius_au(cfg, full_content_radius_au)
    star_radius_au = view_radius_au * float(cfg.get("star_backdrop_radius_factor", 0.98))
    camera_eye = _camera_eye_from_cfg(cfg)

    fig = go.Figure()

    # ── Starfield ─────────────────────────────────────────────────────────────
    # Stars use real catalogue directions, projected onto the back half of the
    # current view shell so the default Mars view reads as a distant backdrop.
    if show_stars:
        for trace in starfield_traces(
            star_radius_au,
            when=time if time is not None else t_jd,
            frame="gcrf",
            mag_limit=star_mag_limit,
            opacity=0.72,
            fallback_random=True,
            catalog_path=catalog_path,
            hemisphere_away_from=(camera_eye["x"], camera_eye["y"], camera_eye["z"]),
        ):
            trace.showlegend = True
            fig.add_trace(trace)

    # ── Ecliptic grid — faint teal reference circles (radii 1/5/10/20/30 AU)
    # + 12 radial spokes every 30°, all centred on the Sun. Purely a visual
    # scale reference, not orbits — labelled here (showlegend on the first
    # circle) so it doesn't look like unexplained stray lines.
    if show_ecliptic:
        th = np.linspace(0, 2*np.pi, 200)
        _first_ring = True
        for _r in [1, 5, 10, 20, 30]:
            if _r > outer_a * 1.1:
                continue
            fig.add_trace(go.Scatter3d(
                x=np.cos(th)*_r, y=np.sin(th)*_r, z=np.zeros(200),
                mode="lines", line=dict(color="rgba(0,255,156,0.07)", width=1),
                hoverinfo="skip", showlegend=_first_ring,
                name="Ecliptic reference grid" if _first_ring else None,
                legendgroup="ecliptic_grid",
            ))
            _first_ring = False
        for ang in range(0, 360, 30):
            a = math.radians(ang)
            fig.add_trace(go.Scatter3d(
                x=[0, math.cos(a)*outer_a*1.1],
                y=[0, math.sin(a)*outer_a*1.1],
                z=[0, 0],
                mode="lines", line=dict(color="rgba(0,255,156,0.04)", width=1),
                hoverinfo="skip", showlegend=False,
                legendgroup="ecliptic_grid",
            ))

    # ── Sun ───────────────────────────────────────────────────────────────────
    for t in make_sun_traces(r_display_au=_R_AU["Sun"] * sun_scale):
        fig.add_trace(t)

    # ── Planets ───────────────────────────────────────────────────────────────
    earth_pos = None
    trail_colors = {
        "Mercury": "rgba(170,170,170,0.35)", "Venus":   "rgba(255,204,68,0.35)",
        "Earth":   "rgba(26,143,209,0.40)",  "Mars":    "rgba(212,90,42,0.38)",
        "Jupiter": "rgba(200,164,110,0.30)", "Saturn":  "rgba(232,217,160,0.28)",
        "Uranus":  "rgba(125,232,232,0.28)", "Neptune": "rgba(63,84,186,0.28)",
    }

    for name in _PLANETS:
        if not show_planets.get(name, False):
            continue
        pos = tuple(np.asarray(
            _body_positions_au(
                name,
                time if time is not None else t_jd,
                ephemeris=ephemeris,
            ),
            dtype=float,
        ))
        if name == "Earth":
            earth_pos = pos

        # Orbit trail
        if show_trails:
            tx, ty, tz = _orbit_trail_au(name, t_jd, time=time, ephemeris=ephemeris)
            fig.add_trace(go.Scatter3d(
                x=tx, y=ty, z=tz, mode="lines",
                line=dict(
                    color=trail_colors.get(name, "rgba(200,200,200,0.3)"),
                    width=1,
                ),
                hoverinfo="skip", showlegend=False,
            ))

        # Planet sphere(s)
        for t in make_planet_traces(
            name,
            pos,
            scale_au=scale_au,
            show_label=show_labels,
            n=sphere_res,
            time=time if time is not None else t_jd,
        ):
            fig.add_trace(t)

        # Saturn rings
        if name == "Saturn":
            for t in make_saturn_ring_traces(pos, scale_au=scale_au):
                fig.add_trace(t)

    # ── Moon ──────────────────────────────────────────────────────────────────
    # Was a fixed offset at a constant angle regardless of date (and smaller
    # than Earth's own display radius, so it rendered inside the Earth
    # sphere). Now a real position for t_jd, shaded like the other bodies.
    if show_moon and earth_pos is not None:
        for t in make_moon_traces(earth_pos, t_jd, show_label=show_labels):
            fig.add_trace(t)

    # ── Layout ────────────────────────────────────────────────────────────────
    T_yr = (t_jd - 2_451_545.0) / 365.25
    rng = view_radius_au

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-rng, rng], showbackground=False,
                       showgrid=False, zeroline=False, title="X (AU)"),
            yaxis=dict(range=[-rng, rng], showbackground=False,
                       showgrid=False, zeroline=False, title="Y (AU)"),
            zaxis=dict(range=[-rng, rng], showbackground=False,
                       showgrid=False, zeroline=False, title="Z (AU)"),
            bgcolor=bg,
            aspectmode="cube",
            camera=dict(
                eye=camera_eye,
                up=dict(x=0, y=0, z=1),
                projection=dict(type=str(cfg.get("camera_projection", "perspective"))),
            ),
        ),
        paper_bgcolor=bg,
        font=dict(color="#C8D8E8"),
        title=dict(
            text=(
                f"Heliocentric Solar System — Mars view — "
                f"{2000+T_yr:.3f} ({ephemeris} ephemeris)"
            ),
            x=0.5, font=dict(color="#00FF9C", size=14),
        ),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="#333",
                    borderwidth=1, font=dict(size=10)),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────
DEFAULT_CFG = dict(
    sol_year=2025, sol_month=6,
    sol_show_mercury=True, sol_show_venus=True, sol_show_earth=True,
    sol_show_mars=True,    sol_show_jupiter=True, sol_show_saturn=True,
    sol_show_uranus=True,  sol_show_neptune=True,
    sol_show_moon=True,    sol_show_trails=True,
    sol_show_stars=True,   sol_show_ecliptic=True, sol_show_labels=True,
    star_mag_limit=6.5,
    view_radius_au="mars",
    mars_view_margin=1.10,
    star_backdrop_radius_factor=0.98,
    camera_projection="perspective",
    star_catalog=None,
    sol_ephemeris="builtin",
    planet_scale=1.0,
    sun_scale=1.0,
    sphere_resolution=50,
    bg_color="#060810",
    output_dir=str(output_root() / "figures"),
)

if __name__ == "__main__":
    cfg = DEFAULT_CFG.copy()
    env_cfg = os.environ.get("GUI_CONFIG", "")
    if env_cfg:
        try:
            cfg.update(json.loads(env_cfg))
            print("[solar_view_plot] Loaded GUI_CONFIG from environment.")
        except json.JSONDecodeError as e:
            print(f"[solar_view_plot] Warning: bad GUI_CONFIG ({e}); using defaults.")

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[solar_view_plot] Building scene for "
          f"{cfg['sol_year']}-{cfg['sol_month']:02d} ...")
    fig = build_figure(cfg)

    html_path = output_dir / "solar_view_plot.html"
    fig.write_html(str(html_path))
    print(f"[solar_view_plot] Saved → {html_path}")

    jpg_path = output_dir / "solar_view_plot.jpg"
    try:
        fig.write_image(str(jpg_path), width=1920, height=1080, scale=2)
        print(f"[solar_view_plot] Saved → {jpg_path}")
    except Exception as e:
        print(f"[solar_view_plot] write_image failed: {e} — install ssatk[static]")

    print("[solar_view_plot] Done.")
