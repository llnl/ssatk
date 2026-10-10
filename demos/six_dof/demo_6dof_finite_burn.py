"""Compare finite-burn trajectory fidelity over several orbit periods."""

from __future__ import annotations

import numpy as np

from demos.six_dof._common import (
    circular_leo_state,
    demo_flags,
    hours,
    plot_earth,
    save_demo_figure,
    set_equal_3d,
)
from ssatk.constants import EARTH_MU
from ssatk.coordinates.satellite_frames import frame_to_gcrf_matrix
from ssatk.accelerations_6dof import (
    SpacecraftAttitudePD,
    SpacecraftManeuverAccel,
    SpacecraftThrusterAccel,
    thrust_profile_trapezoid,
)
from ssatk.propagators_6dof import Spacecraft, attitude_quaternion_from_frame
from ssatk.plots.plotutils import (
    apply_auto_log_scale,
    log_safe_values,
    should_use_log_scale,
)
from ssatk.engines import thruster_spec
from ssatk.satellites import SpacecraftBody

TITLE = "6-DoF Finite-Burn Fidelity"
DESCRIPTION = (
    "Shows how fixed-frame, orbit-frame, and body-mounted thruster fidelity "
    "changes the resulting trajectory."
)
GALLERY_CATEGORY = "six_dof"


def main(make_figures=None, fast=None):
    make_figures, fast = demo_flags(make_figures, fast)

    r0, v0 = circular_leo_state(altitude_m=700_000.0)
    orbit_period_s = 2.0 * np.pi * np.sqrt(np.linalg.norm(r0) ** 3 / EARTH_MU)
    duration_s = 900.0 if fast else 4.0 * orbit_period_s
    times = np.linspace(0.0, duration_s, 60 if fast else 360)
    burn_max_step = 60.0 if fast else 30.0

    cold_gas = thruster_spec("cold_gas_micro")
    throttle = thrust_profile_trapezoid(1.0, start=60.0, burn_time=300.0, rise_time=30.0, fall_time=30.0)
    thrust_profile = lambda t, *args: cold_gas.nominal_thrust_n * throttle(t)

    centered_body = SpacecraftBody.box(
        name="3u_centered_burn_demo_bus",
        mass=30.0,
        size=(0.5, 0.5, 0.8),
    ).with_thrusters(
        cold_gas.to_thruster(
            name="centered_y_thruster",
            direction_body=[0.0, 1.0, 0.0],
            position_body=[0.0, 0.0, 0.0],
        )
    )
    off_axis_body = SpacecraftBody.box(
        name="3u_burn_demo_bus",
        mass=30.0,
        size=(0.5, 0.5, 0.8),
    ).with_thrusters(
        cold_gas.to_thruster(
            name="off_axis_y_thruster",
            direction_body=[0.0, 1.0, 0.0],
            position_body=[0.03, 0.0, 0.0],
        )
    )

    q_rtn0 = attitude_quaternion_from_frame("rtn", r=r0, v=v0)
    initial_rtn = frame_to_gcrf_matrix("rtn", r=r0, v=v0)
    initial_prograde = initial_rtn[:, 1]

    centered_spacecraft = Spacecraft(
        r=r0,
        v=v0,
        q=q_rtn0,
        omega=[0.0, 0.0, 0.0],
        body=centered_body,
    )
    off_axis_spacecraft = centered_spacecraft.with_body(off_axis_body)

    def rtn_target(t, r, v, q, omega, spacecraft):
        return attitude_quaternion_from_frame("rtn", r=r, v=v)

    attitude_hold = SpacecraftAttitudePD(q_target=rtn_target, kp=0.01, kd=0.5, max_torque=0.004)

    coast = centered_spacecraft.propagate(times=times, max_step=120.0)

    fixed_inertial_burn = centered_spacecraft.propagate(
        times=times,
        acceleration=SpacecraftManeuverAccel(
            thrust_profile,
            direction=initial_prograde,
            frame="gcrf",
            mass=centered_body.current_mass,
        ),
        max_step=burn_max_step,
    )
    rtn_guided_burn = centered_spacecraft.propagate(
        times=times,
        acceleration=SpacecraftManeuverAccel(
            thrust_profile,
            direction=[0.0, 1.0, 0.0],
            frame="rtn",
            mass=centered_body.current_mass,
        ),
        max_step=burn_max_step,
    )
    body_centered_free = centered_spacecraft.propagate(
        times=times,
        models=[SpacecraftThrusterAccel(throttle=throttle)],
        max_step=burn_max_step,
    )
    body_centered_rtn_hold = centered_spacecraft.propagate(
        times=times,
        models=[SpacecraftThrusterAccel(throttle=throttle), attitude_hold],
        max_step=burn_max_step,
    )
    body_off_axis_free = off_axis_spacecraft.propagate(
        times=times,
        models=[SpacecraftThrusterAccel(throttle=throttle)],
        max_step=burn_max_step,
    )
    body_off_axis_rtn_hold = off_axis_spacecraft.propagate(
        times=times,
        models=[SpacecraftThrusterAccel(throttle=throttle), attitude_hold],
        max_step=burn_max_step,
    )

    trajectories = {
        "coast": coast,
        "fixed inertial burn": fixed_inertial_burn,
        "RTN guided burn": rtn_guided_burn,
        "centered body thruster": body_centered_free,
        "centered body + RTN hold": body_centered_rtn_hold,
        "off-axis body thruster": body_off_axis_free,
        "off-axis body + RTN hold": body_off_axis_rtn_hold,
    }
    colors = {
        "coast": "0.35",
        "fixed inertial burn": "tab:blue",
        "RTN guided burn": "tab:green",
        "centered body thruster": "tab:cyan",
        "centered body + RTN hold": "tab:olive",
        "off-axis body thruster": "tab:red",
        "off-axis body + RTN hold": "tab:purple",
    }

    offset_vs_coast_m = {
        name: np.linalg.norm(trajectory.r - coast.r, axis=1)
        for name, trajectory in trajectories.items()
        if name != "coast"
    }
    offset_vs_rtn_m = {
        name: np.linalg.norm(trajectory.r - rtn_guided_burn.r, axis=1)
        for name, trajectory in trajectories.items()
        if name != "RTN guided burn"
    }
    angular_rate_deg_s = {
        name: np.degrees(np.linalg.norm(trajectory.omega, axis=1))
        for name, trajectory in trajectories.items()
    }
    summary = {
        name: {
            "final_offset_vs_coast_m": float(offset_vs_coast_m.get(name, np.zeros_like(times))[-1]),
            "final_offset_vs_rtn_m": float(offset_vs_rtn_m.get(name, np.zeros_like(times))[-1]),
            "final_speed_offset_vs_coast_mps": float(np.linalg.norm(trajectory.v[-1] - coast.v[-1])),
            "max_angular_rate_deg_s": float(angular_rate_deg_s[name].max()),
        }
        for name, trajectory in trajectories.items()
    }
    throttle_values = np.array([throttle(t) for t in times])

    figure_path = None
    if make_figures:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(16, 11), constrained_layout=True)
        ax_orbit = fig.add_subplot(2, 2, 1, projection="3d")
        ax_offset = fig.add_subplot(2, 2, 2)
        ax_relative = fig.add_subplot(2, 2, 3)
        ax_rtn = fig.add_subplot(2, 2, 4)
        fig.suptitle("6-DoF finite-burn fidelity over multiple orbits", fontsize=18)

        plot_earth(ax_orbit)
        orbit_lines = []
        for name, trajectory in trajectories.items():
            r_km = trajectory.r / 1000.0
            (line,) = ax_orbit.plot(
                r_km[:, 0],
                r_km[:, 1],
                r_km[:, 2],
                color=colors[name],
                lw=1.5 if name == "coast" else 2.0,
                alpha=0.75,
                label=name,
            )
            orbit_lines.append(line)
        set_equal_3d(ax_orbit, *(trajectory.r / 1000.0 for trajectory in trajectories.values()))
        ax_orbit.set_title("Absolute GCRF trajectories")
        ax_orbit.legend(handles=orbit_lines, loc="upper left", fontsize=8)

        time_hr = hours(times)
        use_coast_log = should_use_log_scale(offset_vs_coast_m.values())
        for name, offset in offset_vs_coast_m.items():
            plot_offset = log_safe_values(offset) if use_coast_log else offset
            ax_offset.plot(time_hr, plot_offset, color=colors[name], label=name, lw=2)
        if use_coast_log:
            apply_auto_log_scale(ax_offset, offset_vs_coast_m.values())
        ax_offset.set_ylabel("relative position [m]")
        ax_offset.set_xlabel("time [hr]")
        ax_offset.set_title("Distance from no-burn coast")
        ax_offset.grid(True, which="both" if use_coast_log else "major", alpha=0.3)
        ax_offset.legend(fontsize=8)

        relative_offsets = {
            name: offset
            for name, offset in offset_vs_rtn_m.items()
            if name != "coast"
        }
        use_relative_log = should_use_log_scale(relative_offsets.values())
        for name, offset in offset_vs_rtn_m.items():
            if name == "coast":
                continue
            plot_offset = log_safe_values(offset) if use_relative_log else offset
            ax_relative.plot(time_hr, plot_offset, color=colors[name], label=name, lw=2)
        if use_relative_log:
            apply_auto_log_scale(ax_relative, relative_offsets.values())
        ax_relative.set_xlabel("time [hr]")
        ax_relative.set_ylabel("position difference [m]")
        ax_relative.set_title("Difference from RTN-guided finite burn")
        ax_relative.grid(True, which="both" if use_relative_log else "major", alpha=0.3)
        ax_relative.legend(fontsize=8)

        for name, trajectory in trajectories.items():
            if name == "coast":
                continue
            rel_gcrf = trajectory.r - coast.r
            rel_rtn = rel_gcrf @ initial_rtn
            ax_rtn.plot(
                rel_rtn[:, 1],
                rel_rtn[:, 0],
                color=colors[name],
                label=name,
                lw=2,
            )
            ax_rtn.scatter(rel_rtn[-1, 1], rel_rtn[-1, 0], color=colors[name], s=25)
        ax_rtn.set_xlabel("initial along-track difference [m]")
        ax_rtn.set_ylabel("initial radial difference [m]")
        ax_rtn.set_title("Relative trajectory in initial RTN plane")
        ax_rtn.grid(True, alpha=0.3)
        ax_rtn.legend(fontsize=8)

        figure_path = save_demo_figure(fig, "demo_6dof_finite_burn.png", make_figures)

    return {
        "title": TITLE,
        "description": DESCRIPTION,
        "times": times,
        "orbit_period_s": orbit_period_s,
        "fidelity_order": tuple(trajectories),
        "trajectories": trajectories,
        "coast": coast,
        "fixed_inertial_burn": fixed_inertial_burn,
        "rtn_guided_burn": rtn_guided_burn,
        "body_centered_free": body_centered_free,
        "body_centered_rtn_hold": body_centered_rtn_hold,
        "body_off_axis_free": body_off_axis_free,
        "body_off_axis_rtn_hold": body_off_axis_rtn_hold,
        "ideal_burn": fixed_inertial_burn,
        "body_thruster": body_off_axis_free,
        "ideal_offset_m": offset_vs_coast_m["fixed inertial burn"],
        "body_offset_m": offset_vs_coast_m["off-axis body thruster"],
        "body_vs_ideal_m": np.linalg.norm(body_off_axis_free.r - fixed_inertial_burn.r, axis=1),
        "offset_vs_coast_m": offset_vs_coast_m,
        "offset_vs_rtn_m": offset_vs_rtn_m,
        "angular_rate_deg_s": angular_rate_deg_s,
        "summary": summary,
        "throttle": throttle_values,
        "figure": figure_path,
    }


if __name__ == "__main__":
    main(make_figures=True, fast=False)
