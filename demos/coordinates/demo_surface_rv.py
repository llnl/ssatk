import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from astropy.time import Time

from ssatk.coordinates.geodetic import surface_rv
from ssatk.time_functions.get_times import get_times
from ssatk.plots.plotutils import figsave

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None

# Smoke/validation module: exercised by pytest, not rendered in the demo gallery.
GALLERY_INCLUDE = False


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    t0 = Time("2025-5-1")
    times = get_times(duration=(2 if fast else 1, "day"), freq=(10 if fast else 1, "min"))

    rs = []
    for t in times:
        r, v = surface_rv(lat=0, lon=0, t=t)
        rs.append(r)
    rs = np.array(rs)

    if make_figures:
        fig = plt.figure()
        plt.plot(rs[:, 0], rs[:, 1], linewidth=1.5)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Single site trajectory (lat=0, lon=0)")
        plt.tight_layout()
        figsave(fig, "surface_rv_xy.jpg")
        plt.close(fig)

    return {"rs": rs, "times": times, "t0": t0}


if __name__ == "__main__":
    main(make_figures=True, fast=False)
