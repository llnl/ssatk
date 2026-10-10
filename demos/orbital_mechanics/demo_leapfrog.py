import os
import sys

import numpy as np

from ssatk.plots.orbit_plot import orbit_plot
from ssatk.propagators_orbit.leap_frog import leapfrog
from ssatk.accelerations_orbit.accel_uniform_earth import accel_uniform_earth
from ssatk.plots.figpath import figpath
from ssatk.constants import RGEO, VGEO

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    t = np.arange(0, 3600 * (6 if fast else 24))

    r1, v1 = leapfrog(r0=[RGEO, 0, 0], v0=[0, VGEO, 0], t=t)
    r2, v2 = leapfrog(r0=[RGEO, 0, 0], v0=[0, VGEO, 0], t=t, velocity=(0, 600, -1))

    if make_figures:
        orbit_plot(
            r1,
            view="xy",
            save_path=figpath("testing_leapfrog_RGEO.jpg"),
            pad=0.1,
            title="GEO",
            show=False,
        )
        orbit_plot(
            r2,
            view="xy",
            save_path=figpath("testing_leapfrog_RGEO_velocity_burn.jpg"),
            pad=0.1,
            title="GEO",
            show=False,
        )

    return {"nominal": (r1, v1), "velocity_burn": (r2, v2)}


if __name__ == "__main__":
    main(make_figures=True, fast=False)
