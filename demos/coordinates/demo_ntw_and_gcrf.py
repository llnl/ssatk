import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from ssatk.coordinates.satellite_frames import ntw_to_gcrf
from ssatk.plots.plotutils import figsave

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None

# Smoke/validation module: exercised by pytest, not rendered in the demo gallery.
GALLERY_INCLUDE = False


def main(make_figures=None):
    """
    Demonstrate transforming a delta-v vector from NTW to GCRF.

    Parameters
    ----------
    make_figures : bool or None
        If None, defaults to False under pytest and True otherwise.

    Returns
    -------
    dict
        Position, velocity, input NTW delta-v, and converted GCRF delta-v.
    """
    if make_figures is None:
        make_figures = not UNDER_PYTEST

    # Define a simple circular orbit in GCRF (equatorial orbit)
    R_earth = 6378e3
    mu = 3.986e14
    r_magnitude = R_earth + 500e3

    r_center = np.array([r_magnitude, 0.0, 0.0])
    v_magnitude = np.sqrt(mu / r_magnitude)
    v_center = np.array([0.0, v_magnitude, 0.0])

    # Sample delta-v in NTW frame
    delta_v_ntw = np.array([0.0, 100.0, 0.0])

    # Convert delta-v from NTW to GCRF
    delta_v_gcrf = ntw_to_gcrf(delta_v_ntw, r_center, v_center)

    print("Position (GCRF, m):", r_center)
    print("Velocity (GCRF, m/s):", v_center)
    print("Delta-v (NTW, m/s):", delta_v_ntw)
    print("Delta-v (GCRF, m/s):", delta_v_gcrf)

    if make_figures:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter([0], [0], [0], color="blue", label="Earth Center", s=100)

        ax.quiver(
            0, 0, 0,
            r_center[0], r_center[1], r_center[2],
            color="green",
            label="Position (r)",
            arrow_length_ratio=0.1,
        )

        ax.quiver(
            r_center[0], r_center[1], r_center[2],
            v_center[0], v_center[1], v_center[2],
            color="red",
            label="Velocity (v)",
            arrow_length_ratio=0.1,
        )

        ax.quiver(
            r_center[0], r_center[1], r_center[2],
            delta_v_gcrf[0], delta_v_gcrf[1], delta_v_gcrf[2],
            color="purple",
            label="Delta-v (GCRF)",
            arrow_length_ratio=0.1,
        )

        scale = r_magnitude * 1.5
        ax.set_xlim([-scale, scale])
        ax.set_ylim([-scale, scale])
        ax.set_zlim([-scale, scale])

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("NTW to GCRF Delta-v Transformation")
        ax.legend()
        ax.set_box_aspect([1, 1, 1])

        figsave(fig, "ntw_to_gcrf_delta_v.jpg")
        plt.close(fig)

    return {
        "r_center": r_center,
        "v_center": v_center,
        "delta_v_ntw": delta_v_ntw,
        "delta_v_gcrf": delta_v_gcrf,
    }


if __name__ == "__main__":
    main(make_figures=True)
