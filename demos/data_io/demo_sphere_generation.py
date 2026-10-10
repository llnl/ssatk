import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from ssatk.compute.generate_sphere_of_vectors import generate_sphere_vectors
from ssatk.plots.plotutils import figsave

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    n = 1000 if fast else 10_000
    mag = 3.5
    seed = 0

    A_uniform = generate_sphere_vectors(n, mag, seed=seed, distribution="uniform")
    A_random = generate_sphere_vectors(n, mag, seed=seed, distribution="random")

    if make_figures:
        fig = plt.figure(figsize=(12, 6))

        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.scatter(A_uniform[:, 0], A_uniform[:, 1], A_uniform[:, 2], s=2, alpha=0.8)
        ax1.set_title("Uniform on S^2 (Gaussian-normalized)")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("z")
        ax1.set_xlim(-mag, mag)
        ax1.set_ylim(-mag, mag)
        ax1.set_zlim(-mag, mag)
        try:
            ax1.set_box_aspect((1.0, 1.0, 1.0))
        except Exception:
            pass

        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        ax2.scatter(A_random[:, 0], A_random[:, 1], A_random[:, 2], s=2, alpha=0.8)
        ax2.set_title("Area-uniform via z~U[-1,1], phi~U[0,2*pi)")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("z")
        ax2.set_xlim(-mag, mag)
        ax2.set_ylim(-mag, mag)
        ax2.set_zlim(-mag, mag)
        try:
            ax2.set_box_aspect((1.0, 1.0, 1.0))
        except Exception:
            pass

        fig.tight_layout()
        figsave(fig, "spheres_subplots.png")

    return {"uniform": A_uniform, "random": A_random}


if __name__ == "__main__":
    main(make_figures=True, fast=False)
