import os
import sys
import numpy as np

from ssapy import Orbit, rv
from astropy.time import Time

from ssatk.plots.figpath import figpath
from ssatk.constants import RGEO
from ssatk.plots.orbit_plot import orbit_plot

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    print("True Anomalies")
    t0 = Time(0, format="gps")
    rs = []

    anomalies = np.arange(0, 360, 30 if fast else 5)
    for trueAnomaly in anomalies:
        orbit = Orbit.fromKeplerianElements(
            a=RGEO,
            e=0.4,
            i=0,
            pa=0,
            raan=0,
            trueAnomaly=np.radians(trueAnomaly),
            t=t0,
        )
        rs.append(orbit.r)

    if make_figures:
        orbit_plot(rs, view="xy", show=False, save_path=figpath("ssapy_orbit_sampling_trueAnomaly"))

    print("Time sampling")
    rs = []
    orbit = Orbit.fromKeplerianElements(a=RGEO, e=0.4, i=0, pa=0, raan=0, trueAnomaly=0.0, t=t0)
    for t in np.arange(0, orbit.period, 3600 if fast else 600):
        orbit_new = orbit.at(t)
        rs.append(orbit_new.r)

    if make_figures:
        orbit_plot(np.array(rs), show=False, save_path=figpath("ssapy_orbit_sampling_time"))

    print("Full orbit sampled.")
    rs, v = rv(orbit, np.arange(0, orbit.period, 3600 if fast else 600))
    if make_figures:
        orbit_plot(rs, show=False, save_path=figpath("ssapy_orbit_object"))
    print("Complete.")

    return {"samples_true_anomaly": rs, "orbit": orbit}


if __name__ == "__main__":
    main(make_figures=True, fast=False)
