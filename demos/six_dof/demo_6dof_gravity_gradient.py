"""Show gravity-gradient torque changing attitude along a low-Earth orbit."""

from __future__ import annotations

import numpy as np

from demos.six_dof._common import (
    body_axis_history,
    circular_leo_state,
    demo_flags,
    hours,
    plot_earth,
    plot_orbit_km,
    save_demo_figure,
    set_equal_3d,
    vector_angle_deg,
)
from ssatk.constants import EARTH_MU
from ssatk.propagators_6dof import Spacecraft, normalize_quaternion
from ssatk.satellites import SpacecraftBody

TITLE = "6-DoF Gravity-Gradient Torque"
DESCRIPTION = "Compares torque-free and gravity-gradient attitude propagation for an elongated spacecraft."
GALLERY_CATEGORY = "six_dof"


def main(make_figures=None, fast=None):
    make_figures, fast = demo_flags(make_figures, fast)

    r0, v0 = circular_leo_state(altitude_m=400_000.0)
    orbit_period_s = 2.0 * np.pi * np.sqrt(np.linalg.norm(r0) ** 3 / EARTH_MU)
    duration_s = (2.0 if fast else 5.0) * orbit_period_s
    times = np.linspace(0.0, duration_s, 90 if fast else 180)

    elongated_body = SpacecraftBody.box(
        name="elongated_bus",
        mass=80.0,
        size=(0.3, 1.0, 5.0),
    )

    initial_yaw_deg = 45.0
    q0 = normalize_quaternion(
        [
            np.cos(np.radians(initial_yaw_deg) / 2.0),
            0.0,
            0.0,
            np.sin(np.radians(initial_yaw_deg) / 2.0),
        ]
    )

    spacecraft = Spacecraft(
        r=r0,
        v=v0,
        q=q0,
        omega=[0.0, 0.0, 0.0],
        body=elongated_body,
    )

    torque_free = spacecraft.propagate(times=times, max_step=60.0)
    gravity_gradient = spacecraft.propagate(times=times, gravity_gradient=True, max_step=60.0)

    radial_free = torque_free.r / np.linalg.norm(torque_free.r, axis=1)[:, None]
    radial_gg = gravity_gradient.r / np.linalg.norm(gravity_gradient.r, axis=1)[:, None]
    free_body_x = body_axis_history(torque_free.q)
    gg_body_x = body_axis_history(gravity_gradient.q)
    free_radial_angle_deg = vector_angle_deg(free_body_x, radial_free)
    gg_radial_angle_deg = vector_angle_deg(gg_body_x, radial_gg)

    figure_path = None
    if make_figures:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(14, 9), constrained_layout=True)
        ax_orbit = fig.add_subplot(2, 2, 1, projection="3d")
        ax_angle = fig.add_subplot(2, 2, 2)
        ax_rate = fig.add_subplot(2, 2, 3)
        ax_quat = fig.add_subplot(2, 2, 4)
        fig.suptitle("6-DoF gravity-gradient torque on an elongated spacecraft", fontsize=18)

        plot_earth(ax_orbit)
        r_km = plot_orbit_km(ax_orbit, gravity_gradient, label="LEO trajectory", color="tab:blue")
        set_equal_3d(ax_orbit, r_km)
        ax_orbit.legend()

        time_hr = hours(times)
        ax_angle.plot(time_hr, free_radial_angle_deg, label="torque-free", lw=2)
        ax_angle.plot(time_hr, gg_radial_angle_deg, label="gravity-gradient", lw=2)
        ax_angle.set_ylabel("body +X angle from radial [deg]")
        ax_angle.grid(True, alpha=0.3)
        ax_angle.legend()

        ax_rate.plot(time_hr, np.degrees(np.linalg.norm(torque_free.omega, axis=1)), label="torque-free", lw=2)
        ax_rate.plot(time_hr, np.degrees(np.linalg.norm(gravity_gradient.omega, axis=1)), label="gravity-gradient", lw=2)
        ax_rate.set_xlabel("time [hr]")
        ax_rate.set_ylabel("angular rate norm [deg/s]")
        ax_rate.grid(True, alpha=0.3)
        ax_rate.legend()

        for index, label in enumerate(("q0", "q1", "q2", "q3")):
            ax_quat.plot(time_hr, gravity_gradient.q[:, index], label=label, lw=2)
        ax_quat.set_xlabel("time [hr]")
        ax_quat.set_ylabel("gravity-gradient quaternion")
        ax_quat.grid(True, alpha=0.3)
        ax_quat.legend()

        figure_path = save_demo_figure(fig, "demo_6dof_gravity_gradient.png", make_figures)

    return {
        "title": TITLE,
        "description": DESCRIPTION,
        "times": times,
        "torque_free": torque_free,
        "gravity_gradient": gravity_gradient,
        "free_radial_angle_deg": free_radial_angle_deg,
        "gravity_gradient_radial_angle_deg": gg_radial_angle_deg,
        "figure": figure_path,
    }


if __name__ == "__main__":
    main(make_figures=True, fast=False)

