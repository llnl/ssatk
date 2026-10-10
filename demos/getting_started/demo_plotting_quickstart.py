#!/usr/bin/env python3
"""
Beginner plotting quickstart for SSATK.

This file is intentionally written top-to-bottom.  Each section makes one
important kind of plot from the same simple SSAPy orbit so a new user can copy a
single call, change the inputs, and get a useful figure.

Run from a cloned repo with:

    python -m demos.getting_started.demo_plotting_quickstart

Outputs are saved under ``~/ssatk_output/figures`` by default.
"""

GALLERY_CATEGORY = "getting_started"

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

import ssatk
from ssatk.plots.divergence_gif import divergence_gif
from ssatk.plots.divergence_plot import divergence_plot
from ssatk.plots.earth_sun_plot import DEFAULT_CFG as EARTH_SUN_CFG
from ssatk.plots.earth_sun_plot import build_static_figure as build_earth_sun_static
from ssatk.plots.figpath import figpath, ssatk_path
from ssatk.plots.globe_plot import globe_plot
from ssatk.plots.groundtrack_plot import groundtrack_plot
from ssatk.plots.magfield_plot_3d import plot_magfield_3d
from ssatk.plots.moon_plot_3d import moon_plot_3d
from ssatk.plots.orbit_plot import orbit_plot
from ssatk.plots.sensor_fov_plot import DEFAULT_CFG as SENSOR_FOV_CFG
from ssatk.plots.sensor_fov_plot import plot_sensor_fov
from ssatk.plots.solar_view_plot import DEFAULT_CFG as SOLAR_VIEW_CFG
from ssatk.plots.solar_view_plot import build_figure as build_solar_view
from ssatk.plots.van_allen_plot_3d import plot_van_allen_3d

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None
FIGDIR = "figures"


def _make_demo_orbit(fast=False):
    """Return one propagated SSAPy orbit as position/velocity/time arrays."""
    t0 = Time("2025-01-01T00:00:00", scale="utc")

    orbit = ssatk.Orbit.fromKeplerianElements(
        a=ssatk.EARTH_RADIUS + 700e3,       # semi-major axis [m]
        e=0.01,                             # eccentricity [-]
        i=np.radians(51.6),                 # inclination [rad]
        pa=np.radians(40.0),                # argument of perigee [rad]
        raan=np.radians(20.0),              # right ascension of ascending node [rad]
        trueAnomaly=0.0,                    # starting true anomaly [rad]
        t=t0,
    )

    n_points = 48 if fast else 120
    duration_seconds = 2.0 * 3600.0 if fast else 6.0 * 3600.0
    times = Time(t0.gps + np.linspace(0.0, duration_seconds, n_points), format="gps")

    r_m, v_mps = ssatk.rv(orbit, times)
    r_m = np.asarray(r_m, dtype=float).reshape((-1, 3))
    v_mps = np.asarray(v_mps, dtype=float).reshape((-1, 3))
    return orbit, times, r_m, v_mps


def _make_divergence_cloud(r_m, v_mps, times, fast=False):
    """Return a tiny deterministic ensemble for divergence_plot/divergence_gif."""
    rng = np.random.default_rng(7)
    n_samples = 8 if fast else 20
    n_snapshots = min(8 if fast else 12, len(times))
    sample_idx = np.linspace(0, len(times) - 1, n_snapshots).astype(int)

    offsets_m = rng.normal(loc=0.0, scale=350.0, size=(n_samples, 3))
    growth = np.linspace(0.15, 1.0, n_snapshots)
    r_histories = r_m[sample_idx][None, :, :] + offsets_m[:, None, :] * growth[None, :, None]

    return {
        "r_histories": r_histories,
        "r_nominal_hist": r_m[sample_idx],
        "v_nominal_hist": v_mps[sample_idx],
        "times_gps": times.gps[sample_idx],
    }


def main(make_figures=None, fast=None, include_heavy=False):
    """
    Build a small orbit and show direct calls to the main SSATK plot routines.

    Parameters
    ----------
    make_figures : bool or None
        If True, save the example figures.  Defaults to False under pytest and
        True when the file is run by a user.
    fast : bool or None
        Use fewer orbit samples and lower mesh resolutions.
    include_heavy : bool
        If True, also run the slower magnetic-field and Van Allen examples.
        The calls are included here so users can see them, but they are off by
        default because they take longer than the quickstart plots.
    """
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    orbit, times, r_m, v_mps = _make_demo_orbit(fast=fast)
    divergence = _make_divergence_cloud(r_m, v_mps, times, fast=fast)

    outputs = {
        "orbit": orbit,
        "times": times,
        "r_m": r_m,
        "v_mps": v_mps,
        "files": [],
    }

    if not make_figures:
        return outputs

    # ------------------------------------------------------------------
    # 1) Main entry point: orbit_plot
    # ------------------------------------------------------------------
    # Use orbit_plot first.  It accepts SSAPy position arrays directly and can
    # switch between common views with the `view=` keyword.
    orbit_plot(
        r_m,
        t=times,
        view=("xy", "xz", "yz", "3d"),
        title="orbit_plot: Cartesian views",
        save=f"{FIGDIR}/plotting_quickstart_orbit_views.jpg",
        show=False,
    )
    outputs["files"].append(ssatk_path(f"{FIGDIR}/plotting_quickstart_orbit_views.jpg"))

    # Dashboard combines the orbit, ground track, and globe context.
    orbit_plot(
        r_m,
        t=times,
        view="dashboard",
        title="orbit_plot: dashboard view",
        save=f"{FIGDIR}/plotting_quickstart_dashboard.jpg",
        show=False,
    )
    outputs["files"].append(ssatk_path(f"{FIGDIR}/plotting_quickstart_dashboard.jpg"))

    # A GIF is made by saving orbit_plot to a .gif path.  Keep the frame count
    # small in a beginner demo; increase max_frames for smoother animations.
    orbit_plot(
        r_m,
        t=times,
        view=("xy", "groundtrack", "globe"),
        title="orbit_plot: animated orbit summary",
        save=f"{FIGDIR}/plotting_quickstart_orbit_animation.gif",
        max_frames=8 if fast else 14,
        fps=2,
        tail=8,
        show=False,
    )
    outputs["files"].append(ssatk_path(f"{FIGDIR}/plotting_quickstart_orbit_animation.gif"))

    # ------------------------------------------------------------------
    # 2) Standalone Earth orbit views: groundtrack_plot and globe_plot
    # ------------------------------------------------------------------
    groundtrack_fig = groundtrack_plot(
        r_m,
        times,
        title="groundtrack_plot: sub-satellite track",
        ground_stations=[(35.0, -106.0)],
    )
    groundtrack_path = figpath(f"{FIGDIR}/plotting_quickstart_groundtrack.jpg")
    groundtrack_fig.savefig(groundtrack_path, dpi=180, bbox_inches="tight")
    plt.close(groundtrack_fig)
    outputs["files"].append(groundtrack_path)

    globe_fig, _ = globe_plot(
        r_m,
        times,
        title="globe_plot: orbit over rotating Earth",
        scale=16,
        globe_time=times[-1],
        save_path=figpath(f"{FIGDIR}/plotting_quickstart_globe.jpg"),
    )
    plt.close(globe_fig)
    outputs["files"].append(figpath(f"{FIGDIR}/plotting_quickstart_globe.jpg"))

    # ------------------------------------------------------------------
    # 3) Sensor field of view: plot_sensor_fov
    # ------------------------------------------------------------------
    # plot_sensor_fov works with Orbit objects or raw r/v/t arrays.  Here the
    # inputs are converted to km and km/s to make the units explicit.
    fov_cfg = SENSOR_FOV_CFG.copy()
    fov_cfg["fov_time_index"] = len(r_m) // 2
    fov_cfg["fov_animate"] = False
    fov_cfg["earth_n_lat"] = 18 if fast else 36
    fov_cfg["earth_n_lon"] = 36 if fast else 72
    fov_cfg["show_stars"] = False
    fov_cfg["show_moon"] = False
    fov_cfg["show_sun"] = True

    fov_path = figpath(f"{FIGDIR}/plotting_quickstart_sensor_fov.html")
    plot_sensor_fov(
        r=r_m / 1000.0,
        v=v_mps / 1000.0,
        t=times,
        cfg=fov_cfg,
        r_units="km",
        v_units="km/s",
        save_path=fov_path,
        show=False,
    )
    outputs["files"].append(fov_path)

    # ------------------------------------------------------------------
    # 4) Divergence plots: divergence_plot and divergence_gif
    # ------------------------------------------------------------------
    final_cloud = divergence["r_histories"][:, -1, :]
    div_fig = divergence_plot(
        final_cloud,
        r_center=divergence["r_nominal_hist"][-1],
        v_center=divergence["v_nominal_hist"][-1],
        title="divergence_plot: final position-error cloud",
        show=False,
    )
    div_path = figpath(f"{FIGDIR}/plotting_quickstart_divergence.jpg")
    div_fig.savefig(div_path, dpi=180, bbox_inches="tight")
    plt.close(div_fig)
    outputs["files"].append(div_path)

    gif_path = ssatk_path(f"{FIGDIR}/plotting_quickstart_divergence.gif")
    divergence_gif(
        r_histories=divergence["r_histories"],
        times_gps=divergence["times_gps"],
        output_path=gif_path,
        r_nominal_hist=divergence["r_nominal_hist"],
        v_nominal_hist=divergence["v_nominal_hist"],
        duration=1.0,
    )
    outputs["files"].append(gif_path)

    # ------------------------------------------------------------------
    # 5) Moon plot: moon_plot_3d
    # ------------------------------------------------------------------
    moon_fig, _ = moon_plot_3d(
        title="moon_plot_3d: Moon surface context",
        save_path=figpath(f"{FIGDIR}/plotting_quickstart_moon.jpg"),
        show=False,
    )
    plt.close(moon_fig)
    outputs["files"].append(figpath(f"{FIGDIR}/plotting_quickstart_moon.jpg"))

    # ------------------------------------------------------------------
    # 6) Space-environment Plotly figures: Earth-Sun and solar-system views
    # ------------------------------------------------------------------
    earth_sun_cfg = EARTH_SUN_CFG.copy()
    earth_sun_cfg["n_frames"] = 3
    earth_sun_cfg["sphere_resolution"] = 12 if fast else 20
    earth_sun_cfg["hero_resolution"] = 24 if fast else 48
    earth_sun_cfg["show_stars"] = False
    earth_sun_fig = build_earth_sun_static(earth_sun_cfg)
    earth_sun_path = figpath(f"{FIGDIR}/plotting_quickstart_earth_sun.html")
    earth_sun_fig.write_html(str(earth_sun_path))
    outputs["files"].append(earth_sun_path)

    solar_cfg = SOLAR_VIEW_CFG.copy()
    solar_cfg["sol_show_moon"] = True
    solar_cfg["sol_show_trails"] = True
    solar_cfg["sol_show_stars"] = False
    solar_cfg["sphere_resolution"] = 10 if fast else 18
    solar_fig = build_solar_view(solar_cfg)
    solar_path = figpath(f"{FIGDIR}/plotting_quickstart_solar_view.html")
    solar_fig.write_html(str(solar_path))
    outputs["files"].append(solar_path)

    # ------------------------------------------------------------------
    # 7) Slower optional science-context plots
    # ------------------------------------------------------------------
    # These are important SSATK plotting routines, but they are slower than a
    # quickstart.  Enable them explicitly with main(include_heavy=True).
    if include_heavy:
        try:
            mag_path = figpath(f"{FIGDIR}/plotting_quickstart_magfield")
            plot_magfield_3d(
                title="plot_magfield_3d: magnetic field and belts",
                fidelity="low",
                max_r_re=6.0,
                show_stars=False,
                save_path=mag_path,
                show=False,
            )
            outputs["files"].append(mag_path)
        except ImportError as exc:
            outputs["magfield_skipped"] = str(exc)

        try:
            belts_path = figpath(f"{FIGDIR}/plotting_quickstart_van_allen")
            plot_van_allen_3d(
                title="plot_van_allen_3d: radiation belts",
                fidelity="low",
                show_stars=False,
                save_path=belts_path,
                show=False,
            )
            outputs["files"].append(belts_path)
        except ImportError as exc:
            outputs["van_allen_skipped"] = str(exc)

    return outputs


if __name__ == "__main__":
    main(make_figures=True, fast=False)
