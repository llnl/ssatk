import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ssatk.orbital_mechanics.rv_to_ellipse import rv_to_ellipse
from ssatk.io.pprint_utils import pprint
from ssatk.plots.figpath import figpath

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def main(make_figures=None, verbose=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if verbose is None:
        verbose = not UNDER_PYTEST

    r0 = [9000e3, 0.0, 0.0]
    v0 = [0.0, 9.0e3, 2.0e3]
    result = rv_to_ellipse(r0, v0, num=600)

    if verbose:
        pprint(result)

    x = None
    y = None

    if isinstance(result, dict):
        if "r" in result:
            r_arr = np.asarray(result["r"])
            if r_arr.ndim == 2 and r_arr.shape[1] >= 2:
                x, y = r_arr[:, 0], r_arr[:, 1]
        if x is None and "x" in result and "y" in result:
            x_arr = np.asarray(result["x"])
            y_arr = np.asarray(result["y"])
            if x_arr.shape == y_arr.shape:
                x, y = x_arr, y_arr
    else:
        arr = np.asarray(result)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            x, y = arr[:, 0], arr[:, 1]

    if make_figures and x is not None and y is not None:
        plt.figure()
        plt.plot(x, y, linewidth=1.5)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title("rv_to_ellipse: XY projection")

        outpath = Path(figpath("rv_to_ellipse_xy"))
        if outpath.suffix == "":
            outpath = outpath.with_suffix(".png")
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath, dpi=200, bbox_inches="tight")
        print("Saved plot to:", outpath)
        plt.close()
    elif verbose and x is None:
        print("Could not determine x,y from rv_to_ellipse output; skipping plot.")

    return {"result": result, "x": x, "y": y}


if __name__ == "__main__":
    main(make_figures=True, verbose=True)