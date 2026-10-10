import os
import sys
import numpy as np

from astropy.time import Time

from ssatk.ssapy_wrappers.ssapy_orbits import ssapy_orbit
from ssatk.plots.orbit_plot import orbit_plot
from ssatk.constants import RGEO
from ssatk.coordinates.lunar import get_lunar_rv

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None
GALLERY_CATEGORY = "orbit_visualization"
FIGDIR = "figures"


def main(make_figures=None, fast=None):
    """
    Demo for the unified orbit_plot entry point, including standard orbit
    slices, cislunar views, ground-track/globe/dashboard views, and animated
    GIF/MP4 saves.

    Parameters
    ----------
    make_figures : bool or None
        If None, defaults to False under pytest and True otherwise.
    fast : bool or None
        If None, defaults to True under pytest and False otherwise.

    Returns
    -------
    dict
        Computed orbit arrays and times.
    """
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    times = Time("2024-1-1").gps
    print(times)

    r_moon, v_moon = get_lunar_rv(times)
    print(r_moon, v_moon)

    r0 = r_moon[0] + (1000e3 * r_moon[0] / np.linalg.norm(r_moon[0]))
    v0 = v_moon[0] + 100
    print(r0, v0)

    # single orbit
    print("\nCalculating orbit.")
    single_duration = (3, "day") if fast else (4, "day")
    r, v, t = ssapy_orbit(r=r0, v=v0, duration=single_duration)
    print(f"Plotting orbit. {np.shape(r)} {np.shape(t)}")

    if make_figures:
        orbit_plot(r=r, t=t, save=f"{FIGDIR}/demo_orbit_plot")
        orbit_plot(r=r, t=t, view="cislunar", save=f"{FIGDIR}/demo_cislunar_plot")
        orbit_plot(r=r, t=t, view="cislunar_3d", save=f"{FIGDIR}/demo_cislunar_plot_3d")
        orbit_plot(r=r, t=t, view="globe", save=f"{FIGDIR}/demo_globe_plot_black", scale=16)
        orbit_plot(r=r, t=t, view="globe", save=f"{FIGDIR}/demo_globe_plot_white", scale=16, c="white")

    # two same length orbits
    print("\nCalculating 2 orbit.")
    r2, v2, t2 = ssapy_orbit(
        a=9 * RGEO,
        e=0.5,
        i=0.25,
        pa=np.pi / 2,
        duration=single_duration,
    )
    print(f"Plotting two orbits same length. {np.shape(r)} {np.shape(r2)} {np.shape(t)}")

    if make_figures:
        orbit_plot(r=[r, r2], t=t, save=f"{FIGDIR}/demo_orbit_plot_two_orbits")
        orbit_plot(r=[r, r2], t=t, view="cislunar", save=f"{FIGDIR}/demo_cislunar_plot_two_orbits")
        orbit_plot(r=[r, r2], t=t, view="globe", save=f"{FIGDIR}/demo_globe_two_orbits", scale=16, c="black")

    # two orbits different lengths
    print("\nCalculating 2 different orbit.")
    diff_duration = (2, "day") if fast else (3, "day")
    r3, v3, t3 = ssapy_orbit(a=5 * RGEO, e=0.5, i=0.75, duration=diff_duration, t0="2024-1-1")
    print("Plotting two orbits different lengths.")

    if make_figures:
        orbit_plot(
            r=[r, r3],
            t=[t, t3],
            save=f"{FIGDIR}/demo_orbit_plot_two_different_length_orbits",
        )
        orbit_plot(
            r=[r, r3],
            t=[t, t3],
            save=f"{FIGDIR}/demo_orbit_plot_two_different_length_orbits_itrf",
            frame="itrf",
        )
        orbit_plot(
            r=[r, r3],
            t=[t, t3],
            view="cislunar",
            save=f"{FIGDIR}/demo_cislunar_plot_two_different_length_orbits",
        )
        orbit_plot(
            r=[r, r3],
            t=[t, t3],
            view="globe",
            save=f"{FIGDIR}/demo_globe_two_different_length_orbits",
            scale=16,
            c="black",
        )

    # groundtrack dashboard
    r_dash, v_dash, t_dash = ssapy_orbit(a=RGEO, e=0.2, duration=((6, "hour") if fast else (12, "hour")))

    if make_figures:
        orbit_plot(
            r_dash,
            t_dash,
            view="dashboard",
            save=f"{FIGDIR}/demo_ground_dashboard_test",
        )
        orbit_plot(
            r=[r_dash, r3],
            t=[t_dash, t3],
            view="dashboard",
            save=f"{FIGDIR}/demo_ground_dashboard_two_different_length_orbits",
        )
        orbit_plot(
            r_dash,
            t_dash,
            view=("groundtrack", "globe", "xy"),
            save=f"{FIGDIR}/demo_orbit_plot_dashboard_animation.gif",
            fps=8,
            max_frames=18,
            tail=8,
        )
        orbit_plot(
            r_dash,
            t_dash,
            view="xy",
            save=f"{FIGDIR}/demo_orbit_plot_xy_animation.mp4",
            fps=12,
            max_frames=24,
            tail=10,
        )

    print("PLOT DEMO DONE.")

    return {
        "single": (r, v, t),
        "same_length": (r2, v2, t2),
        "different_length": (r3, v3, t3),
        "dashboard": (r_dash, v_dash, t_dash),
        "r_moon": r_moon,
        "v_moon": v_moon,
        "r0": r0,
        "v0": v0,
    }


if __name__ == "__main__":
    main(make_figures=True, fast=False)
