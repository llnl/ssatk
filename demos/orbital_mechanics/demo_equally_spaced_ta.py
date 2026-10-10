import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import ssapy

from ssatk.orbital_mechanics.equally_spaced_ta import equally_spaced_ta
from ssatk.plots.plotutils import figsave

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def main(make_figures=None, verbose=None, fast=None):
    """
    Demonstrate equal arc-length sampling on an elliptical orbit.

    Parameters
    ----------
    make_figures : bool or None
        If None, defaults to False under pytest and True otherwise.
    verbose : bool or None
        If None, defaults to False under pytest and True otherwise.
    fast : bool or None
        If None, defaults to True under pytest and False otherwise.

    Returns
    -------
    dict
        Computed true anomalies and sampled coordinates.
    """
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if verbose is None:
        verbose = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    # Example elliptical orbit
    a = 12000e3          # meters
    e = 0.7
    i = 0.0              # radians
    pa = 0.0             # radians
    raan = 0.0           # radians
    ta0 = 0.0            # radians
    t0 = 0.0             # GPS seconds

    # Build SSAPy Orbit from Keplerian elements
    orbit = ssapy.Orbit.fromKeplerianElements(a, e, i, pa, raan, ta0, t0)

    # Request an even number of equal-arc-length samples
    n_samples = 8 if fast else 16
    ta = equally_spaced_ta(a=a, e=e, n_samples=n_samples, degrees=False)

    # Compute radius for each sampled true anomaly
    r = a * (1 - e ** 2) / (1 + e * np.cos(ta))

    # Focus-centered perifocal coordinates
    x = r * np.cos(ta)
    y = r * np.sin(ta)

    # Dense ellipse for plotting
    n_dense = 250 if fast else 1000
    ta_dense = np.linspace(0, 2 * np.pi, n_dense)
    r_dense = a * (1 - e ** 2) / (1 + e * np.cos(ta_dense))
    x_dense = r_dense * np.cos(ta_dense)
    y_dense = r_dense * np.sin(ta_dense)

    if make_figures:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(x_dense, y_dense, label="Orbit")
        ax.scatter(x, y, color="red", zorder=3, label="Equal arc-length samples")
        ax.scatter([0], [0], color="black", marker="*", s=120, label="Focus")

        # Highlight periapsis and apoapsis
        rp = a * (1 - e)
        ra = a * (1 + e)
        ax.scatter([rp], [0], s=150, color="green", label="Periapsis")
        ax.scatter([-ra], [0], s=150, color="purple", label="Apoapsis")

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title("Equal Arc-Length Sampling on an Elliptical Orbit")
        ax.legend(loc="upper left")
        ax.grid(True)

        plt.axis("equal")
        figsave(fig, "equally_spaced_ta.jpg")
        plt.close(fig)

    if verbose:
        print("Returned ta [deg]:")
        print(np.degrees(ta))

    return {
        "orbit": orbit,
        "ta": ta,
        "r": r,
        "x": x,
        "y": y,
        "ta_dense": ta_dense,
        "r_dense": r_dense,
        "x_dense": x_dense,
        "y_dense": y_dense,
    }


if __name__ == "__main__":
    main(make_figures=True, verbose=True, fast=False)
