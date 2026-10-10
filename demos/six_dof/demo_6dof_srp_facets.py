"""Show attitude-dependent facet solar-radiation pressure and self-shadowing."""

from __future__ import annotations

import numpy as np

from demos.six_dof._common import demo_flags, hours, save_demo_figure
from ssatk.accelerations_6dof import SpacecraftFacetSolRad
from ssatk.constants import AU
from ssatk.propagators_6dof import Spacecraft
from ssatk.satellites import Facet, SpacecraftBody

TITLE = "6-DoF Facet Solar-Radiation Pressure"
DESCRIPTION = "Compares SRP force and torque with and without facet self-shadowing."
GALLERY_CATEGORY = "six_dof"


def _square_yz(x, y, z, side):
    half = side / 2.0
    return (
        (x, y - half, z - half),
        (x, y + half, z - half),
        (x, y + half, z + half),
        (x, y - half, z + half),
    )


def main(make_figures=None, fast=None):
    make_figures, fast = demo_flags(make_figures, fast)

    panel_area_m2 = 6.0
    panel_side_m = np.sqrt(panel_area_m2)
    front_panel = Facet(
        name="front_panel",
        area=panel_area_m2,
        normal_body=[1.0, 0.0, 0.0],
        center_of_pressure=[0.20, 0.35, 0.0],
        specular_reflectivity=0.25,
        diffuse_reflectivity=0.45,
        vertices_body=_square_yz(0.20, 0.35, 0.0, panel_side_m),
    )
    rear_panel = Facet(
        name="shadowed_rear_panel",
        area=panel_area_m2,
        normal_body=[1.0, 0.0, 0.0],
        center_of_pressure=[-0.20, 0.35, 0.0],
        specular_reflectivity=0.25,
        diffuse_reflectivity=0.45,
        vertices_body=_square_yz(-0.20, 0.35, 0.0, panel_side_m),
    )
    body = SpacecraftBody(
        name="two_parallel_solar_panels",
        mass=40.0,
        inertia=np.diag([6.0, 8.0, 10.0]),
        facets=(front_panel, rear_panel),
        reference_area=2.0 * panel_area_m2,
    )

    spacecraft = Spacecraft(
        r=[0.0, 0.0, 0.0],
        v=[0.0, 0.0, 0.0],
        q=[1.0, 0.0, 0.0, 0.0],
        omega=[0.0, 0.0, 0.0],
        body=body,
    )
    sun_position = [AU, 0.0, 0.0]
    times = np.linspace(0.0, 1_800.0 if fast else 7_200.0, 80 if fast else 180)

    no_shadow_model = SpacecraftFacetSolRad(sun_position, self_shadowing=False)
    shadow_model = SpacecraftFacetSolRad(sun_position, self_shadowing=True)

    no_shadow = spacecraft.propagate(times=times, mu=0.0, models=[no_shadow_model], max_step=60.0)
    with_shadow = spacecraft.propagate(times=times, mu=0.0, models=[shadow_model], max_step=60.0)

    no_shadow_torque = np.array([no_shadow_model.torque(spacecraft, t=t, r=r, v=v, q=q, omega=w) for t, r, v, q, w in zip(no_shadow.t, no_shadow.r, no_shadow.v, no_shadow.q, no_shadow.omega)])
    with_shadow_torque = np.array([shadow_model.torque(spacecraft, t=t, r=r, v=v, q=q, omega=w) for t, r, v, q, w in zip(with_shadow.t, with_shadow.r, with_shadow.v, with_shadow.q, with_shadow.omega)])

    figure_path = None
    if make_figures:
        import matplotlib.pyplot as plt

        time_hr = hours(times)
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
        fig.suptitle("6-DoF facet SRP: optical properties and self-shadowing", fontsize=18)

        axes[0, 0].plot(time_hr, np.linalg.norm(no_shadow.v, axis=1) * 1000.0, label="no self-shadowing", lw=2)
        axes[0, 0].plot(time_hr, np.linalg.norm(with_shadow.v, axis=1) * 1000.0, label="self-shadowing", lw=2)
        axes[0, 0].set_ylabel("SRP-induced speed [mm/s]")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        axes[0, 1].plot(time_hr, np.degrees(np.linalg.norm(no_shadow.omega, axis=1)), label="no self-shadowing", lw=2)
        axes[0, 1].plot(time_hr, np.degrees(np.linalg.norm(with_shadow.omega, axis=1)), label="self-shadowing", lw=2)
        axes[0, 1].set_ylabel("angular rate norm [deg/s]")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

        axes[1, 0].plot(time_hr, no_shadow_torque[:, 2], label="no self-shadowing", lw=2)
        axes[1, 0].plot(time_hr, with_shadow_torque[:, 2], label="self-shadowing", lw=2)
        axes[1, 0].set_xlabel("time [hr]")
        axes[1, 0].set_ylabel("body torque z [N m]")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

        axes[1, 1].bar(["no shadow", "self-shadow"], [len(body.facets), 1])
        axes[1, 1].set_ylabel("sunlit panel count at t=0")
        axes[1, 1].set_ylim(0, len(body.facets) + 0.5)
        axes[1, 1].grid(True, axis="y", alpha=0.3)

        figure_path = save_demo_figure(fig, "demo_6dof_srp_facets.png", make_figures)

    return {
        "title": TITLE,
        "description": DESCRIPTION,
        "times": times,
        "no_shadow": no_shadow,
        "with_shadow": with_shadow,
        "no_shadow_torque": no_shadow_torque,
        "with_shadow_torque": with_shadow_torque,
        "figure": figure_path,
    }


if __name__ == "__main__":
    main(make_figures=True, fast=False)

