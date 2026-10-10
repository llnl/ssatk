import os
import sys

import numpy as np
from ssapy import Orbit, rv
from astropy.time import Time
from ssapy.accel import AccelKepler
from ssapy.propagator import SciPyPropagator

from ssatk.constants import EARTH_RADIUS
from ssatk.time_functions.get_times import get_times
from ssatk.plots.orbit_plot import orbit_plot
from ssatk.plots.figpath import figpath

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    print("Finished imports.")

    accel = AccelKepler()
    prop = SciPyPropagator(accel)

    t0 = Time("2025-1-1", scale="utc")
    ap = EARTH_RADIUS + 1000e3

    rs = []
    peris = np.linspace(10e3, EARTH_RADIUS, 4 if fast else 10)
    for peri in peris:
        a = (peri + ap) / 2
        e = (ap - peri) / (peri + ap)
        orbit = Orbit.fromKeplerianElements(a, e, 0, 0, 0, 0, t=t0)

        times = get_times(duration=(orbit.period if not fast else min(orbit.period, 3600.0), "s"), freq=(10 if fast else 1, "s"), t0=t0)
        r, v = rv(orbit=orbit, time=times, propagator=prop)
        rs.append(r)

    if make_figures:
        orbit_plot(
            rs,
            view="xy",
            save_path=figpath("testing_ellipses.jpg"),
            pad=500,
            title="Point source Earth",
        )

    return {"trajectories": rs}


if __name__ == "__main__":
    main(make_figures=True, fast=False)
