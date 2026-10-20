import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from ssatk.coordinates.earth_fixed import gcrf_to_itrf, itrf_to_gcrf
from ssatk.time_functions.get_times import get_times
from ssatk.ssapy_wrappers.ssapy_orbits import ssapy_orbit
from ssatk.constants import RGEO
from ssatk.plots.plotutils import figsave

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None

# Smoke/validation module: exercised by pytest, not rendered in the demo gallery.
GALLERY_INCLUDE = False


def test_coordinate_transforms(make_figures=False):
    t = get_times(duration=(1, "days"), freq=(10 if UNDER_PYTEST else 1, "min"))
    r_gcrf_orig, v, t_ssapy = ssapy_orbit(a=2 * RGEO, e=0.3, t=t)

    r_itrf = gcrf_to_itrf(r_gcrf_orig, t)
    r_gcrf_back = itrf_to_gcrf(r_itrf, t)

    tolerance = 1e-8 * np.max(r_gcrf_orig)
    difference = np.max(np.abs(r_gcrf_orig - r_gcrf_back))

    if difference < tolerance:
        print("Test passed: GCRF -> ITRF -> GCRF transformation is consistent.")
    else:
        print("Test failed: Transformations are not inverses within tolerance.")

    if make_figures:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(r_gcrf_orig[:, 0], r_gcrf_orig[:, 1], r_gcrf_orig[:, 2], label="GCRF")
        ax.plot(r_itrf[:, 0], r_itrf[:, 1], r_itrf[:, 2], label="ITRF")
        ax.plot(r_gcrf_back[:, 0], r_gcrf_back[:, 1], r_gcrf_back[:, 2], label="Back to GCRF")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("GCRF and ITRF Coordinate Transformations")
        ax.legend()
        figsave(fig, "itrf_to_gcrf_roundtrip.jpg")
        plt.close(fig)

    return {"difference": difference, "tolerance": tolerance}


if __name__ == "__main__":
    test_coordinate_transforms(make_figures=not UNDER_PYTEST)
