"""Compare four presentation shots using one SSAPy circular orbit."""

from pathlib import Path

import numpy as np
from astropy.time import Time
from ssapy import Orbit, rv

from ssatk.constants import EARTH_MU, EARTH_RADIUS
from ssatk.plots.camera import AnimationCamera, _CameraPath
from ssatk.plots._body_surfaces import BodySurface, body_rotation
from ssatk.plots.orbit_animation import _animation_writer

TITLE = "Animation Camera Shots"
DESCRIPTION = "The same circular orbit and physical Earth scale in overview, fixed follow, chase and orbit-sweep cameras."
GALLERY_CATEGORY = "orbit_visualization"


def main(make_figures=True, fast=False, output_dir=None):
    if not make_figures:
        return {"skipped": True, "reason": "figures_disabled"}
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from ssatk.plots.figpath import figpath

    destination = (
        Path(output_dir) if output_dir else Path(figpath("camera_modes.mp4")).parent
    )
    destination.mkdir(parents=True, exist_ok=True)
    fps, seconds = (8 if fast else 24), 15
    radius = EARTH_RADIUS + 1_200_000.0
    period = 2 * np.pi * np.sqrt(radius**3 / EARTH_MU)
    epoch = Time("2025-01-01T00:00:00")
    elapsed = np.linspace(0, period, fps * seconds)
    times = Time(epoch.gps + elapsed, format="gps")
    inclination = np.deg2rad(35)
    state = Orbit(
        [radius, 0, 0],
        np.sqrt(EARTH_MU / radius)
        * np.array([0, np.cos(inclination), np.sin(inclination)]),
        epoch,
        mu=EARTH_MU,
    )
    positions, _ = rv(state, times)
    positions /= 1000.0
    earth_radius = EARTH_RADIUS / 1000.0
    limit = radius / 1000.0 * 1.06
    bounds = np.array([[-limit] * 3, [limit] * 3])
    # gallery-call: camera_modes.mp4 | camera_modes.png
    fig = plt.figure(figsize=(12, 9), dpi=90 if fast else 120, facecolor="#07111e")
    fig.suptitle("One orbit · four camera shots", color="white", fontsize=19, y=0.98)
    fig.text(
        0.5,
        0.025,
        "Same SSATK trajectory, epochs and physical scale · + marks spacecraft centre",
        ha="center",
        color="#bacadb",
        fontsize=10,
    )
    panels = []
    names = [
        ("overview", "Overview · fixed full scene"),
        ("follow", "Fixed follow · constant scene offset"),
        ("chase", "Chase · behind direction of travel"),
        ("orbit", "Orbit sweep · around the full scene"),
    ]
    for index, (mode, label) in enumerate(names):
        ax = fig.add_subplot(2, 2, index + 1, projection="3d", computed_zorder=False)
        ax.set_facecolor("#07111e")
        ax.set_axis_off()
        ax.set_title(label, color="white", fontsize=12, pad=0)
        spec = AnimationCamera(
            mode=mode,
            distance=18000.0 if mode in {"follow", "chase"} else None,
            azimuth=-55,
        )
        shot = _CameraPath(spec, positions, bounds, times=elapsed)
        body = BodySurface(ax, "earth", earth_radius, n=30 if fast else 75)
        body.artist.set_zorder(1)
        (line,) = ax.plot([], [], [], color="#dda667", lw=1.3, zorder=2)
        (dot,) = ax.plot(
            [], [], [], marker="+", markersize=10, color="#ffe4a8", zorder=3
        )
        stamp = ax.text2D(
            0.04, 0.02, "", transform=ax.transAxes, color="#b8c9dd", fontsize=9
        )
        panels.append((ax, shot, body, line, dot, stamp))
    fig.subplots_adjust(
        left=0.01, right=0.99, top=0.92, bottom=0.07, wspace=0.01, hspace=0.09
    )

    def update(i):
        rotation = body_rotation("earth", times[i])
        for ax, shot, body, line, dot, stamp in panels:
            shot.mpl(ax, i)
            body.update(rotation=rotation)
            line.set_data_3d(positions.T)
            dot.set_data_3d(positions[i : i + 1].T)
            stamp.set_text(f"t + {elapsed[i] / 60:.1f} min / {period / 60:.1f} min")
        return []

    update(len(times) // 8)
    still = destination / "camera_modes.png"
    fig.savefig(still, facecolor=fig.get_facecolor())
    video = destination / "camera_modes.mp4"
    animation = FuncAnimation(
        fig, update, frames=len(times), interval=1000 / fps, blit=False
    )
    animation.save(video, writer=_animation_writer(".mp4", fps=fps, bitrate=3500))
    # gallery-end
    plt.close(fig)
    return {"title": TITLE, "files": [str(video), str(still)]}


if __name__ == "__main__":
    main()
