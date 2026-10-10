import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from ssapy import rv, Orbit
from ssapy.propagator import default_numerical
from astropy.time import Time

from ssatk.coordinates.lunar import get_lunar_rv
from ssatk.plots.figpath import figpath
from ssatk.plots.orbit_animation import _animation_writer

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None
GALLERY_CATEGORY = "video_animation"


def _set_axes_equal_3d(ax, lim):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))


def solve_collinear_l1_x(mu, tol=1e-12, max_iter=50):
    x = 1.0 - mu - (mu / 3.0) ** (1.0 / 3.0)
    for _ in range(max_iter):
        r1 = x + mu
        r2 = x - (1.0 - mu)

        a1 = (1.0 - mu) / np.abs(r1) ** 3
        a2 = mu / np.abs(r2) ** 3

        f = x - a1 * r1 - a2 * r2

        da1 = -3.0 * (1.0 - mu) / np.abs(r1) ** 5
        da2 = -3.0 * mu / np.abs(r2) ** 5
        fp = 1.0 - (da1 * r1 * r1 + a1) - (da2 * r2 * r2 + a2)

        step = f / fp
        x_new = x - step
        if np.abs(step) < tol:
            return float(x_new)
        x = x_new
    return float(x)


def _propagate_with_fallback(orbit, t_frames):
    prop = default_numerical()
    n = len(t_frames)
    min_n = max(2, int(0.05 * n))
    k = n

    while k >= min_n:
        try:
            times = Time(t_frames[:k], format="gps")
            r_sc_f, v_sc_f = rv(orbit=orbit, time=times, propagator=prop)
            r_sc_f = np.array(r_sc_f, dtype=float).reshape((-1, 3))
            return r_sc_f, t_frames[:k]
        except Exception:
            k = int(0.9 * k)

    times = Time(t_frames[:2], format="gps")
    r_sc_f, v_sc_f = rv(orbit=orbit, time=times, propagator=prop)
    r_sc_f = np.array(r_sc_f, dtype=float).reshape((-1, 3))
    return r_sc_f, t_frames[:2]


def _draw_sphere(ax, center, radius, color, alpha=1.0, resolution=30):
    """Draw a filled sphere at `center` with given radius and color."""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, antialiased=True)


def orbit_moon_video_demo(
    t0="2024-01-01",
    duration_days=60.0,
    fps=12,
    seconds_per_frame=6 * 3600,
    trail_len=120,
    out_name="figures/demo_orbit_moon_video.mp4",
    save_gif=False,
    make_figures=True,
    fast=True,
):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    if fast:
        duration_days = min(duration_days, 2.0)
        fps = min(fps, 8)
        seconds_per_frame = max(seconds_per_frame, 12 * 3600)
        save_gif = False

    t0_gps = float(np.array(Time(t0).gps, dtype=float).ravel()[0])
    dt_frame = float(seconds_per_frame)
    n_frames = int(np.ceil((duration_days * 86400.0) / dt_frame)) + 1
    t_frames = t0_gps + dt_frame * np.arange(n_frames, dtype=float)

    r_moon, v_moon = get_lunar_rv(t_frames)
    r_moon = np.array(r_moon, dtype=float).reshape((-1, 3))
    v_moon = np.array(v_moon, dtype=float).reshape((-1, 3))

    rm0 = r_moon[0]
    vm0 = v_moon[0]

    r_em = np.linalg.norm(rm0)
    x_hat = rm0 / r_em

    h_vec = np.cross(rm0, vm0)
    h_norm = np.linalg.norm(h_vec)
    k_hat = np.array([0.0, 0.0, 1.0]) if h_norm == 0.0 else h_vec / h_norm

    y_hat = np.cross(k_hat, x_hat)
    y_norm = np.linalg.norm(y_hat)
    y_hat = np.array([0.0, 1.0, 0.0]) if y_norm == 0.0 else y_hat / y_norm

    mu = 0.0121505856
    x_l1 = solve_collinear_l1_x(mu)
    d_l1_from_moon = (1.0 - mu) - x_l1

    r_l1 = rm0 - (d_l1_from_moon * r_em) * x_hat

    y_offset = 70_000e3
    z_offset = 2_000e3
    r0 = r_l1 + y_offset * y_hat + z_offset * k_hat

    omega = h_norm / (r_em * r_em)
    omega_vec = omega * k_hat

    r_rel_m = r0 - rm0
    v_rel = -np.cross(omega_vec, r_rel_m)
    v_rel = 0.95 * v_rel
    radial_nudge = 5.0
    v0 = vm0 + v_rel + radial_nudge * x_hat

    orbit = Orbit(r=r0, v=v0, t=t0_gps)
    r_sc_f, t_frames_ok = _propagate_with_fallback(orbit, t_frames)

    n_ok = len(t_frames_ok)
    r_moon = r_moon[:n_ok, :]
    max_norm = np.max(np.linalg.norm(np.vstack([r_moon, r_sc_f]), axis=1))
    lim = 1.10 * max_norm

    if not make_figures:
        return {"r_sc_f": r_sc_f, "r_moon": r_moon, "t_frames_ok": t_frames_ok}

    # --- Colors ---
    COLOR_MOON      = "#A0A0A8"   # moon-grey dot
    COLOR_MOON_TRAIL = "#7A7A82"  # slightly darker grey trail
    COLOR_SC        = "#FF8C00"   # orange satellite dot
    COLOR_SC_TRAIL  = "#FF6A00"   # deeper orange trail
    COLOR_EARTH     = "#1E6FD9"   # blue Earth sphere

    # Earth radius in metres (visual scale — scaled up for visibility)
    EARTH_RADIUS_VIS = 0.04 * lim

    fig = plt.figure(figsize=(10, 10), dpi=160)
    fig.patch.set_facecolor("black")
    fig.suptitle("Initialize L1 Lunar Orbit", color="white")

    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")

    # Kill all panes, grid lines, and tick labels
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("black")
    ax.yaxis.pane.set_edgecolor("black")
    ax.zaxis.pane.set_edgecolor("black")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Earth sphere at origin
    _draw_sphere(ax, center=(0.0, 0.0, 0.0), radius=EARTH_RADIUS_VIS, color=COLOR_EARTH, alpha=1.0)

    # Animated artists
    moon_pt,   = ax.plot([], [], [], marker="o", markersize=7,  color=COLOR_MOON,       linewidth=0)
    sc_pt,     = ax.plot([], [], [], marker="o", markersize=4,  color=COLOR_SC,         linewidth=0)
    moon_trail,= ax.plot([], [], [], color=COLOR_MOON_TRAIL,    linewidth=1.2, alpha=0.45)
    sc_trail,  = ax.plot([], [], [], color=COLOR_SC_TRAIL,      linewidth=1.6, alpha=0.65)
    title      = ax.text2D(0.03, 0.92, "", transform=ax.transAxes, color="white")

    _set_axes_equal_3d(ax, lim)

    def init():
        moon_pt.set_data([], [])
        moon_pt.set_3d_properties([])
        sc_pt.set_data([], [])
        sc_pt.set_3d_properties([])
        moon_trail.set_data([], [])
        moon_trail.set_3d_properties([])
        sc_trail.set_data([], [])
        sc_trail.set_3d_properties([])
        title.set_text("")
        return moon_pt, sc_pt, moon_trail, sc_trail, title

    def update(i):
        rm = r_moon[i]
        rs = r_sc_f[i]
        moon_pt.set_data([rm[0]], [rm[1]])
        moon_pt.set_3d_properties([rm[2]])
        sc_pt.set_data([rs[0]], [rs[1]])
        sc_pt.set_3d_properties([rs[2]])

        j0 = max(0, i - trail_len)
        rm_tr = r_moon[j0:i + 1]
        rs_tr = r_sc_f[j0:i + 1]
        moon_trail.set_data(rm_tr[:, 0], rm_tr[:, 1])
        moon_trail.set_3d_properties(rm_tr[:, 2])
        sc_trail.set_data(rs_tr[:, 0], rs_tr[:, 1])
        sc_trail.set_3d_properties(rs_tr[:, 2])

        days = (t_frames_ok[i] - t_frames_ok[0]) / 86400.0
        title.set_text(f"t0 = {t0}   +{days:6.2f} days")
        return moon_pt, sc_pt, moon_trail, sc_trail, title

    ani = FuncAnimation(
        fig,
        update,
        frames=len(t_frames_ok),
        init_func=init,
        blit=False,
        interval=1000.0 / fps,
    )

    out_mp4 = figpath(out_name)
    writer = _animation_writer(".mp4", fps=fps, bitrate=8000)
    ani.save(out_mp4, writer=writer)
    print(f"Saved MP4: {out_mp4}")

    if save_gif:
        out_gif = figpath(out_name.replace(".mp4", ".gif"))
        ani.save(out_gif, writer=PillowWriter(fps=fps))
        print(f"Saved GIF: {out_gif}")

    plt.close(fig)
    return {"r_sc_f": r_sc_f, "r_moon": r_moon, "t_frames_ok": t_frames_ok, "out_mp4": out_mp4}


def main(make_figures=None, fast=None, save_gif=False):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    return orbit_moon_video_demo(
        t0="2024-01-01",
        duration_days=2.0 if fast else 7.0,
        fps=6 if fast else 8,
        seconds_per_frame=12 * 3600 if fast else 6 * 3600,
        trail_len=24 if fast else 48,
        out_name="figures/demo_orbit_moon_video.mp4",
        save_gif=save_gif,
        make_figures=make_figures,
        fast=fast,
    )


if __name__ == "__main__":
    main(make_figures=True, fast=False)
    print("VIDEO DEMO DONE.")
