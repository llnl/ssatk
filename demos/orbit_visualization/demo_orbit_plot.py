import os
import sys
import numpy as np

from astropy.time import Time

from ssatk.ssapy_wrappers.ssapy_orbits import ssapy_orbit
from ssatk.plots.orbit_plot import orbit_plot
from ssatk.constants import RGEO, MOON_RADIUS, MOON_MU
from ssatk.coordinates.lunar import get_lunar_rv

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None
GALLERY_CATEGORY = "orbit_visualization"


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

    epoch = Time("2024-01-01", scale="utc")
    times = epoch.gps
    print(times)

    r_moon, v_moon = get_lunar_rv(times)
    print(r_moon, v_moon)

    # 1,000 km altitude above the lunar surface, not 1,000 km from its center.
    radial = r_moon[0] / np.linalg.norm(r_moon[0])
    normal = np.cross(r_moon[0], v_moon[0])
    normal /= np.linalg.norm(normal)
    tangent = np.cross(normal, radial)
    radius = MOON_RADIUS + 1000e3
    r0 = r_moon[0] + radius * radial
    v0 = v_moon[0] + np.sqrt(MOON_MU / radius) * tangent
    print(r0, v0)

    # single orbit
    print("\nCalculating orbit.")
    single_duration = (3, "day") if fast else (4, "day")
    r, v, t = ssapy_orbit(r=r0, v=v0, duration=single_duration, t0=epoch)
    print(f"Plotting orbit. {np.shape(r)} {np.shape(t)}")

    if make_figures:
        # gallery-call: demo_orbit_plot.jpg
        orbit_plot(r=r, t=t, save=f"demo_orbit_plot")
        # gallery-end
        # gallery-call: demo_cislunar_plot.jpg
        orbit_plot(r=r, t=t, view="cislunar", save=f"demo_cislunar_plot")
        # gallery-end
        # gallery-call: demo_cislunar_plot_3d.jpg
        orbit_plot(r=r, t=t, view="cislunar_3d", save=f"demo_cislunar_plot_3d")
        # gallery-end
        # gallery-call: demo_globe_plot_black.jpg
        orbit_plot(r=r, t=t, view="globe", save=f"demo_globe_plot_black", scale=16)
        # gallery-end
        # gallery-call: demo_globe_plot_white.jpg
        orbit_plot(r=r, t=t, view="globe", save=f"demo_globe_plot_white", scale=16, c="white")
        # gallery-end

    # two same length orbits
    print("\nCalculating 2 orbit.")
    r2, v2, t2 = ssapy_orbit(
        a=9 * RGEO,
        e=0.5,
        i=0.25,
        pa=np.pi / 2,
        duration=single_duration,
        t0=epoch,
    )
    print(f"Plotting two orbits same length. {np.shape(r)} {np.shape(r2)} {np.shape(t)}")

    if make_figures:
        # gallery-call: demo_orbit_plot_two_orbits.jpg
        orbit_plot(r=[r, r2], t=t, save=f"demo_orbit_plot_two_orbits")
        # gallery-end
        # gallery-call: demo_cislunar_plot_two_orbits.jpg
        orbit_plot(r=[r, r2], t=t, view="cislunar", save=f"demo_cislunar_plot_two_orbits")
        # gallery-end
        # gallery-call: demo_globe_two_orbits.jpg
        orbit_plot(r=[r, r2], t=t, view="globe", save=f"demo_globe_two_orbits", scale=16, c="black")
        # gallery-end

    # two orbits different lengths
    print("\nCalculating 2 different orbit.")
    diff_duration = (2, "day") if fast else (3, "day")
    r3, v3, t3 = ssapy_orbit(a=5 * RGEO, e=0.5, i=0.75, duration=diff_duration, t0="2024-1-1")
    print("Plotting two orbits different lengths.")

    if make_figures:
        # gallery-call: demo_orbit_plot_two_different_length_orbits.jpg
        orbit_plot(
            r=[r, r3],
            t=[t, t3],
            save=f"demo_orbit_plot_two_different_length_orbits",
        )
        # gallery-end
        # gallery-call: demo_orbit_plot_two_different_length_orbits_itrf.jpg
        orbit_plot(
            r=[r, r3],
            t=[t, t3],
            save=f"demo_orbit_plot_two_different_length_orbits_itrf",
            frame="itrf",
        )
        # gallery-end
        # gallery-call: demo_cislunar_plot_two_different_length_orbits.jpg
        orbit_plot(
            r=[r, r3],
            t=[t, t3],
            view="cislunar",
            save=f"demo_cislunar_plot_two_different_length_orbits",
        )
        # gallery-end
        # gallery-call: demo_globe_two_different_length_orbits.jpg
        orbit_plot(
            r=[r, r3],
            t=[t, t3],
            view="globe",
            save=f"demo_globe_two_different_length_orbits",
            scale=16,
            c="black",
        )
        # gallery-end

    # groundtrack dashboard
    # A full day in both gallery and standalone runs. Input samples retain
    # orbital curvature independently of the animation's playback frame count.
    r_dash, v_dash, t_dash = ssapy_orbit(
        a=RGEO, e=0.2, i=np.deg2rad(20), duration=(1, "day"),
        freq=(1, "min"), t0=epoch,
    )

    if make_figures:
        # gallery-call: demo_ground_dashboard_test.jpg
        orbit_plot(
            r_dash,
            t_dash,
            view="dashboard",
            save=f"demo_ground_dashboard_test",
        )
        # gallery-end
        # gallery-call: demo_ground_dashboard_two_different_length_orbits.jpg
        orbit_plot(
            r=[r_dash, r3],
            t=[t_dash, t3],
            view="dashboard",
            save=f"demo_ground_dashboard_two_different_length_orbits",
        )
        # gallery-end
        # gallery-call: demo_orbit_plot_dashboard_animation.gif
        orbit_plot(
            r_dash,
            t_dash,
            view=("groundtrack", "globe", "xy"),
            figsize=(12, 9),
            save=f"demo_orbit_plot_dashboard_animation.gif",
            fps=10,
            max_frames=300,
            tail=75,
            earth_resolution=120,
        )
        # gallery-end
        # gallery-call: demo_orbit_plot_xy_animation.mp4
        orbit_plot(
            r_dash,
            t_dash,
            view="xy",
            save=f"demo_orbit_plot_xy_animation.mp4",
            fps=12,
            max_frames=360,
            tail=90,
        )
        # gallery-end

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
