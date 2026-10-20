import os
import sys

import numpy as np
from ssapy import Orbit, rv
from astropy.time import Time
from ssapy.accel import AccelKepler
from ssapy.propagator import SciPyPropagator

from ssatk.constants import EARTH_RADIUS
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
    ts = []
    peris = EARTH_RADIUS + np.linspace(200e3, 900e3, 4 if fast else 10)
    for peri in peris:
        a = (peri + ap) / 2
        e = (ap - peri) / (peri + ap)
        orbit = Orbit.fromKeplerianElements(a, e, 0, 0, 0, 0, t=t0)

        times = t0 + np.linspace(0, orbit.period, 1441) / 86400.
        r, v = rv(orbit=orbit, time=times, propagator=prop)
        rs.append(r)
        ts.append(times)

    if make_figures:
        # gallery-call: testing_ellipses.jpg
        orbit_plot(
            rs,
            t=ts,
            view="xy",
            figsize=(10, 10),
            save_path=figpath("testing_ellipses.jpg"),
            pad=500,
            title="Exterior Earth orbits: 200–900 km perigee, 1,000 km apogee",
        )
        # gallery-end

    return {"trajectories": rs}


if __name__ == "__main__":
    main(make_figures=True, fast=False)
