import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from ssapy import rv, Orbit, get_body
from ssapy.propagator import default_numerical
from astropy.time import Time

from ssatk.coordinates.lunar import get_lunar_rv
from ssatk.constants import EARTH_RADIUS, MOON_RADIUS
from ssatk.plots._body_surfaces import BodySurface, body_rotation
from ssatk.plots.figpath import figpath
from ssatk.plots.orbit_animation import _animation_writer

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None
GALLERY_CATEGORY = "video_animation"
# NASA Moon Fact Sheet: mean sidereal revolution period, not the phase cycle.
# https://nssdc.gsfc.nasa.gov/planetary/factsheet/moonfact.html
LUNAR_ORBIT_DAYS = 27.3217
TITLE = "Fancy video — retrograde lunar encounter"
DESCRIPTION = (
    "The original DRO-like retrograde initial state near the Moon, propagated "
    "for one mean lunar revolution (27.3217 days) in a 30-second movie. "
    "This is a perturbed encounter trajectory, not a validated periodic DRO. "
    "Full trails retain at most one-hour spacing independently of playback. "
    "A dated, Earth-facing Moon close-up shows the surface on a separate scale."
)


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


def _propagate_complete(orbit, t_frames):
    """Propagate the requested span or fail; never silently shorten the movie."""
    r_sc, _ = rv(orbit=orbit, time=Time(t_frames, format="gps"),
                 propagator=default_numerical())
    r_sc = np.asarray(r_sc, dtype=float).reshape((-1, 3))
    if len(r_sc) != len(t_frames) or not np.isfinite(r_sc).all():
        raise RuntimeError("Incomplete cislunar propagation")
    return r_sc


def _movie_times(t0_gps, duration_days, playback_seconds, fps):
    """Exactly round(playback_seconds*fps) frames spanning the physical interval."""
    if min(duration_days, playback_seconds, fps) <= 0:
        raise ValueError("Durations and frame rate must be positive")
    count = max(2, int(round(playback_seconds * fps)))
    return t0_gps + np.linspace(0.0, duration_days * 86400.0, count)


def orbit_moon_video_demo(
    t0="2024-01-01",
    duration_days=LUNAR_ORBIT_DAYS,
    fps=12,
    seconds_per_frame=3600,
    playback_seconds=30.0,
    trail_len=None,
    out_name="demo_orbit_moon_video.mp4",
    save_gif=False,
    make_figures=True,
    fast=False,
    camera="overview",
):
    """Animate a mean lunar month by default; trail_len=None retains all history.

    duration_days is physical time, playback_seconds is encoded video length,
    and seconds_per_frame bounds the path sampling interval independently.
    The ephemeris orbit is perturbed, so a mean month need not close exactly.
    """
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    # Fast mode reduces spatial resolution, never the movie duration.
    t0_gps = float(Time(t0).gps)
    t_frames = _movie_times(t0_gps, duration_days, playback_seconds, fps)
    # Optional finer propagation cadence affects the line geometry only.
    if not np.isfinite(seconds_per_frame) or seconds_per_frame <= 0:
        raise ValueError("seconds_per_frame must be positive and finite")
    n_geometry = max(len(t_frames), int(np.ceil(duration_days * 86400 / seconds_per_frame)) + 1)
    # Form both grids as elapsed seconds before adding the GPS epoch. When
    # their counts match, this also avoids duplicate epochs one float ULP apart.
    t_geometry = np.unique(np.concatenate((
        t_frames, t0_gps + np.linspace(0.0, duration_days * 86400.0, n_geometry)
    )))
    frame_indices = np.searchsorted(t_geometry, t_frames)

    r_moon, v_moon = get_lunar_rv(t_geometry)
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
    # Restore the original retrograde seed from commit 8103d54. The negative
    # sign is intentional: Moon-relative angular momentum opposes its orbit
    # about Earth. This seed is DRO-like, not a corrected periodic solution.
    v_rel = -np.cross(omega_vec, r_rel_m)
    v_rel = 0.95 * v_rel
    radial_nudge = 5.0
    v0 = vm0 + v_rel + radial_nudge * x_hat

    orbit = Orbit(r=r0, v=v0, t=t0_gps)
    r_sc_f = _propagate_complete(orbit, t_geometry)
    t_frames_ok = t_geometry
    all_positions = np.vstack([np.zeros((1, 3)), r_moon, r_sc_f])
    lower = all_positions.min(axis=0)-EARTH_RADIUS
    upper = all_positions.max(axis=0)+EARTH_RADIUS

    if not make_figures:
        return {"r_sc_f": r_sc_f, "r_moon": r_moon, "t_frames_ok": t_frames_ok, "frame_times": t_frames}

    # --- Colors ---
    COLOR_MOON      = "#A0A0A8"   # moon-grey dot
    COLOR_MOON_TRAIL = "#7A7A82"  # slightly darker grey trail
    COLOR_SC        = "#FF8C00"   # orange satellite dot
    COLOR_SC_TRAIL  = "#FF6A00"   # deeper orange trail


    fig = plt.figure(figsize=(10, 8), dpi=100 if fast else 150)
    fig.patch.set_facecolor("black")
    fig.suptitle(f"Retrograde lunar encounter · {duration_days:.3f} days", color="white")
    fig.text(0.5, 0.03, "Main view: Earth/Moon at physical scale · inset: Moon close-up",
             ha="center", color="#aebacb", fontsize=9)

    ax = fig.add_subplot(111, projection="3d")
    ax.set_position([.02, .13, .75, .75])
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
    earth_surface = BodySurface(ax, "earth", EARTH_RADIUS, n=30)
    moon_surface = BodySurface(ax, "moon", MOON_RADIUS, r_moon[0], n=30)

    # A physical lunar radius is only a few pixels across in a cislunar scene.
    # Show the same dated mesh in an explicitly separate, Earth-facing view.
    moon_ax = fig.add_axes([.76, .61, .23, .26], projection="3d", facecolor="black")
    moon_ax.set_axis_off()
    moon_ax.set_proj_type("ortho")
    _set_axes_equal_3d(moon_ax, 1.05*MOON_RADIUS)
    closeup = BodySurface(moon_ax, "moon", MOON_RADIUS, n=120)
    moon_ax.set_title("Moon close-up\nEarth-facing · separate scale", color="white", fontsize=10, pad=8)
    sun_positions = np.asarray(get_body("sun").position(t_geometry)).reshape(3, -1).T

    # Animated artists
    sc_pt,     = ax.plot([], [], [], marker="o", markersize=4,  color=COLOR_SC,         linewidth=0)
    moon_trail,= ax.plot([], [], [], color=COLOR_MOON_TRAIL,    linewidth=1.2, alpha=0.45)
    sc_trail,  = ax.plot([], [], [], color=COLOR_SC_TRAIL,      linewidth=1.6, alpha=0.65)
    title      = ax.text2D(0.03, 0.92, "", transform=ax.transAxes, color="white")

    for j, limits in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        limits(lower[j], upper[j])
    ax.set_box_aspect((1, 1, 1))
    ax.text2D(0.03, 0.06, "Moon: textured surface", transform=ax.transAxes, color=COLOR_MOON)
    ax.text2D(0.03, 0.03, "● Spacecraft", transform=ax.transAxes, color=COLOR_SC)

    from ssatk.plots.camera import _mpl_camera, _camera_spec
    if camera == "overview":
        # View along the initial lunar orbit normal instead of almost edge-on.
        camera = dict(mode="overview", elevation=float(np.degrees(np.arcsin(k_hat[2]))),
                      azimuth=float(np.degrees(np.arctan2(k_hat[1], k_hat[0]))))
    if _camera_spec(camera).target != 0:
        raise ValueError("the cislunar video has one spacecraft camera target (index 0)")
    shot = _mpl_camera(camera, ax, r_sc_f, times=t_frames_ok)

    def init():
        sc_pt.set_data([], [])
        sc_pt.set_3d_properties([])
        moon_trail.set_data([], [])
        moon_trail.set_3d_properties([])
        sc_trail.set_data([], [])
        sc_trail.set_3d_properties([])
        title.set_text("")
        return sc_pt, moon_trail, sc_trail, title

    def update(frame):
        i = frame_indices[frame]
        shot.mpl(ax, i)
        rm = r_moon[i]
        rs = r_sc_f[i]
        earth_surface.update(rotation=body_rotation("earth", t_geometry[i]))
        rotation = body_rotation("moon", t_geometry[i])
        sun_hat = sun_positions[i]-rm
        sun_hat /= np.linalg.norm(sun_hat)
        moon_surface.update(rm, rotation, sun_hat)
        closeup.update(rotation=rotation, sun_hat=sun_hat)
        earth_hat = -rm/np.linalg.norm(rm)
        moon_ax.view_init(elev=np.degrees(np.arcsin(earth_hat[2])),
                          azim=np.degrees(np.arctan2(earth_hat[1], earth_hat[0])))
        sc_pt.set_data([rs[0]], [rs[1]])
        sc_pt.set_3d_properties([rs[2]])

        j0 = 0 if trail_len is None else frame_indices[max(0, frame - trail_len)]
        rm_tr = r_moon[j0:i + 1]
        rs_tr = r_sc_f[j0:i + 1]
        moon_trail.set_data(rm_tr[:, 0], rm_tr[:, 1])
        moon_trail.set_3d_properties(rm_tr[:, 2])
        sc_trail.set_data(rs_tr[:, 0], rs_tr[:, 1])
        sc_trail.set_3d_properties(rs_tr[:, 2])

        days = (t_frames_ok[i] - t_frames_ok[0]) / 86400.0
        title.set_text(f"t0 = {t0}   +{days:6.2f} days")
        return sc_pt, moon_trail, sc_trail, title, closeup.artist

    ani = FuncAnimation(
        fig,
        update,
        frames=len(t_frames),
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
    return {"r_sc_f": r_sc_f, "r_moon": r_moon, "t_frames_ok": t_frames_ok,
            "frame_times": t_frames, "out_mp4": out_mp4}


def main(make_figures=None, fast=None, save_gif=False):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    # gallery-call: *.gif | *.mp4
    return orbit_moon_video_demo(
        t0="2024-01-01",
        duration_days=LUNAR_ORBIT_DAYS,
        fps=12,
        playback_seconds=30.0,
        seconds_per_frame=3600,
        trail_len=None,
        out_name="demo_orbit_moon_video.mp4",
        save_gif=save_gif,
        make_figures=make_figures,
        fast=fast,
    )
    # gallery-end


if __name__ == "__main__":
    main(make_figures=True, fast=False)
    print("VIDEO DEMO DONE.")
