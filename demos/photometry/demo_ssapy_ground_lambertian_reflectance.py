"""
Save all plots with figpath (no interactive windows).
This script mirrors the original example but routes every figure to disk.

Pytest-safe behavior:
- figures are not saved by default under pytest
- runtime is reduced under pytest where practical
"""

import os
import sys

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from ssapy import Orbit, constants, get_body, rv
from ssapy.accel import AccelEarthRad, AccelKepler, AccelSolRad
from ssapy.gravity import AccelHarmonic, AccelThirdBody
from ssapy.propagator import SciPyPropagator

from ssatk.compute.lambertian_magnitude import lambertian_reflection
from ssatk.plots.figpath import figpath
from ssatk.plots.groundtrack_plot import groundtrack_plot
from ssatk.plots.orbit_plot import orbit_plot
from ssatk.plots.plotutils import figsave
from ssatk.time_functions.get_times import get_times

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def decimal_to_datetime_label(d):
    year = int(d)
    rem = d - year
    is_leap = (year % 4 == 0) and ((year % 100 != 0) or (year % 400 == 0))
    days_in_year = 366 if is_leap else 365
    total_seconds = rem * days_in_year * 24 * 3600

    day = int(total_seconds // (24 * 3600))
    seconds_in_day = total_seconds % (24 * 3600)
    hour = int(seconds_in_day // 3600)
    minute = int((seconds_in_day % 3600) // 60)

    base_date = np.datetime64(f"{year}-01-01") + np.timedelta64(day, "D")
    return f"{base_date} {hour:02d}:{minute:02d}"


def main(make_figures=None, fast=None):
    """
    Run the Lambertian reflectance demo.

    Parameters
    ----------
    make_figures : bool or None
        If None, defaults to False under pytest and True otherwise.
    fast : bool or None
        If None, defaults to True under pytest and False otherwise.

    Returns
    -------
    dict
        Computed orbit states and Lambertian reflectance.
    """
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    # ------------------------------------------------------------
    # Epoch
    # ------------------------------------------------------------
    t0 = Time("2024-01-01", scale="utc")
    print("t0:", t0)

    # ------------------------------------------------------------
    # Moon state (simple finite-difference velocity)
    # ------------------------------------------------------------
    moon_body = get_body("moon")
    r_moon = moon_body.position(t0).T
    r_moon_minus = moon_body.position(t0 - 1 * u.s).T
    r_moon_plus = moon_body.position(t0 + 1 * u.s).T
    v_moon = (r_moon_plus - r_moon_minus) / 2.0
    print("r_moon[0]:", r_moon[0], "v_moon[0]:", v_moon[0])

    # Example lunar-bound-ish initial state (not used below—kept for reference)
    r0 = r_moon[0] + (1000e3 * r_moon[0] / np.linalg.norm(r_moon[0]))
    v0 = v_moon[0] + 100.0
    print("Example r0:", r0, "\nExample v0:", v0)

    print("\nCalculating orbit...")

    # ------------------------------------------------------------
    # Keplerian elements → Orbit object at t0
    # ------------------------------------------------------------
    a = constants.RGEO
    e = 0.0
    i = np.radians(45.0)
    pa = np.radians(0.0)
    raan = np.radians(0.0)
    ta = np.radians(180.0)

    kElements = [a, e, i, pa, raan, ta]
    orbit = Orbit.fromKeplerianElements(*kElements, t0)

    # ------------------------------------------------------------
    # Spacecraft + force models
    # ------------------------------------------------------------
    if fast:
        accel = AccelKepler()
    else:
        moon = get_body("moon")
        sun = get_body("sun")
        Earth = get_body("earth", model="EGM2008")

        aEarth = AccelKepler() + AccelHarmonic(Earth, 140, 140)
        aSun = AccelThirdBody(sun)
        aMoon = AccelThirdBody(moon) + AccelHarmonic(moon, 20, 20)
        aSolRad = AccelSolRad(mass=100.0, area=1.0, CD=2.3, CR=1.3)
        aEarthRad = AccelEarthRad(mass=100.0, area=1.0, CD=2.3, CR=1.3)
        accel = aEarth + aMoon + aSun + aSolRad + aEarthRad

    prop = SciPyPropagator(accel)

    # ------------------------------------------------------------
    # Time vector
    # ------------------------------------------------------------
    duration = (20, "minute") if fast else (2, "day")
    freq = (20, "minute") if fast else (1, "minute")
    times = get_times(duration=duration, freq=freq, t0=t0)

    # ------------------------------------------------------------
    # Propagate
    # ------------------------------------------------------------
    r, v = rv(orbit=orbit, time=times, propagator=prop)

    # ------------------------------------------------------------
    # Plot: GCRF and lunar frames — save with figpath
    # ------------------------------------------------------------
    if make_figures:
        fig, _ax = orbit_plot(r, times, frame="gcrf")
        out_gcrf = figpath("ssapy_orbit_gcrf")
        figsave(fig, save_path=out_gcrf)

        fig, _ax = orbit_plot(r, times, frame="lunar")
        out_lunar = figpath("ssapy_orbit_lunar")
        figsave(fig, save_path=out_lunar)

        # Ground track
        groundtrack_plot(r, times, save_path=figpath("ssapy_ground_track"))

    # ------------------------------------------------------------
    # Lambertian reflectance (apparent magnitude)
    # ------------------------------------------------------------
    mv = lambertian_reflection(
        r, time=times, lon=-156.26, lat=20.71, elevation=3055.0
    )["ab_mag_observed"]

    if make_figures:
        xticks = np.linspace(times.decimalyear[0], times.decimalyear[-1], 4)
        xtick_labels = [decimal_to_datetime_label(t) for t in xticks]

        plt.figure(dpi=300)
        plt.plot(times.decimalyear, mv)
        plt.xlabel("Date")
        plt.ylabel("Lambertian Reflectance [Apparent Magnitude]")
        plt.xticks(xticks, xtick_labels, rotation=0)
        plt.tight_layout()

        out_mv = figpath("lambertian_reflectance")
        plt.savefig(out_mv, dpi=300, bbox_inches="tight")
        plt.close()
        print("Saved:", out_mv)

    return {
        "t0": t0,
        "orbit": orbit,
        "times": times,
        "r": r,
        "v": v,
        "mv": mv,
        "r_moon": r_moon,
        "v_moon": v_moon,
        "example_r0": r0,
        "example_v0": v0,
    }


if __name__ == "__main__":
    main(make_figures=True, fast=False)
