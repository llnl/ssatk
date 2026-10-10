"""Compare uncontrolled tumbling with a simple 6-DoF attitude controller."""

from __future__ import annotations

import numpy as np

from demos.six_dof._common import (
    body_axis_history,
    demo_flags,
    hours,
    quaternion_angle_deg,
    save_demo_figure,
)
from ssatk.accelerations_6dof import SpacecraftAttitudePD
from ssatk.propagators_6dof import Spacecraft, normalize_quaternion
from ssatk.satellites import SpacecraftBody

TITLE = "6-DoF Attitude Control"
DESCRIPTION = "Propagates one spacecraft with and without a body-frame quaternion PD attitude controller."
GALLERY_CATEGORY = "six_dof"


def main(make_figures=None, fast=None):
    make_figures, fast = demo_flags(make_figures, fast)

    body = SpacecraftBody.box(
        name="rectangular_bus",
        mass=100.0,
        size=(0.8, 1.0, 1.2),
    )

    initial_attitude_error_deg = 25.0
    q0 = normalize_quaternion(
        [
            np.cos(np.radians(initial_attitude_error_deg) / 2.0),
            0.0,
            0.0,
            np.sin(np.radians(initial_attitude_error_deg) / 2.0),
        ]
    )
    omega0 = np.radians([0.05, -0.03, 0.20])

    spacecraft = Spacecraft(
        r=[0.0, 0.0, 0.0],
        v=[0.0, 0.0, 0.0],
        q=q0,
        omega=omega0,
        body=body,
    )

    times = np.linspace(0.0, 300.0 if fast else 600.0, 80 if fast else 160)

    uncontrolled = spacecraft.propagate(times=times, mu=0.0, max_step=5.0)
    controlled = spacecraft.propagate(
        times=times,
        mu=0.0,
        torque=SpacecraftAttitudePD(kp=0.002, kd=0.08, max_torque=0.003),
        max_step=5.0,
    )

    time_hr = hours(times)
    uncontrolled_error_deg = quaternion_angle_deg(uncontrolled.q)
    controlled_error_deg = quaternion_angle_deg(controlled.q)
    uncontrolled_x_axis = body_axis_history(uncontrolled.q)
    controlled_x_axis = body_axis_history(controlled.q)

    figure_path = None
    if make_figures:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
        fig.suptitle("6-DoF attitude control: free tumble vs PD torque", fontsize=18)

        axes[0, 0].plot(time_hr, uncontrolled_error_deg, label="uncontrolled", lw=2)
        axes[0, 0].plot(time_hr, controlled_error_deg, label="controlled", lw=2)
        axes[0, 0].set_ylabel("attitude error [deg]")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(time_hr, np.degrees(np.linalg.norm(uncontrolled.omega, axis=1)), label="uncontrolled", lw=2)
        axes[0, 1].plot(time_hr, np.degrees(np.linalg.norm(controlled.omega, axis=1)), label="controlled", lw=2)
        axes[0, 1].set_ylabel("angular rate norm [deg/s]")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        for index, label in enumerate(("x", "y", "z")):
            axes[1, 0].plot(time_hr, uncontrolled.omega[:, index], label=f"ω{label}", lw=2)
            axes[1, 1].plot(time_hr, controlled_x_axis[:, index], label=f"body +X {label}", lw=2)
        axes[1, 0].set_ylabel("uncontrolled ω [rad/s]")
        axes[1, 1].set_ylabel("controlled body +X in GCRF")
        for ax in axes[1, :]:
            ax.set_xlabel("time [hr]")
            ax.legend()
            ax.grid(True, alpha=0.3)

        figure_path = save_demo_figure(fig, "demo_6dof_attitude_control.png", make_figures)

    return {
        "title": TITLE,
        "description": DESCRIPTION,
        "times": times,
        "uncontrolled": uncontrolled,
        "controlled": controlled,
        "uncontrolled_error_deg": uncontrolled_error_deg,
        "controlled_error_deg": controlled_error_deg,
        "figure": figure_path,
    }


if __name__ == "__main__":
    main(make_figures=True, fast=False)

