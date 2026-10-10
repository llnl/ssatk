"""Minimal first workflow: propagate one SSAPy orbit and save an SSATK plot."""

import os
import sys

import numpy as np
from astropy.time import Time

import ssatk
from ssatk.plots.orbit_plot import orbit_plot

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None
FIGDIR = "figures"


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    t0 = Time("2025-01-01T00:00:00", scale="utc")
    orbit = ssatk.Orbit.fromKeplerianElements(
        a=ssatk.RGEO,
        e=0.05,
        i=np.radians(28.5),
        pa=np.radians(40.0),
        raan=np.radians(20.0),
        trueAnomaly=0.0,
        t=t0,
    )
    hours = 4 if fast else 12
    times = Time(t0.gps + np.linspace(0.0, hours * 3600.0, 80 if fast else 180), format="gps")
    r, v = ssatk.rv(orbit, times)
    r = np.asarray(r, dtype=float).reshape((-1, 3))

    if make_figures:
        orbit_plot(
            r,
            t=times,
            view=("xy", "groundtrack", "globe"),
            title="First SSATK workflow: orbit, ground track, and globe",
            save=f"{FIGDIR}/first_user_workflow.jpg",
            show=False,
        )

    return {"orbit": orbit, "times": times, "r": r, "v": v}


if __name__ == "__main__":
    main(make_figures=True, fast=False)
