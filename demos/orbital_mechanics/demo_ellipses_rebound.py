import os
import sys

import rebound
import numpy as np
import matplotlib.pyplot as plt

from ssatk.plots.plotutils import figsave

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None

# Smoke/validation module: exercised by pytest, not rendered in the demo gallery.
GALLERY_INCLUDE = False


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    G = 6.67430e-11
    M_earth = 5.972e24
    R_earth = 6.371e6

    sim = rebound.Simulation()
    sim.add(m=M_earth, x=0, y=0, z=0)
    sim.add(m=1e-20, x=R_earth, y=0, z=0, vx=0, vy=8000, vz=0)
    sim.G = G
    sim.integrator = "ias15"

    times = np.linspace(0, 4000 if fast else 15000, 100 if fast else 1000)

    positions = []
    for t in times:
        sim.integrate(t)
        particle = sim.particles[1]
        positions.append([particle.x, particle.y, particle.z])

    positions = np.array(positions)

    if make_figures:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label="Path of Object", color="blue")

        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = R_earth * np.outer(np.cos(u), np.sin(v))
        y = R_earth * np.outer(np.sin(u), np.sin(v))
        z = R_earth * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_wireframe(x, y, z, color="gray", alpha=0.3)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("Path of Object in Free Fall Through Earth (Point Source)")
        ax.legend()
        plt.axis("equal")
        figsave(fig, "ellipses_rebound_path.jpg")
        plt.close(fig)

    return {"positions": positions}


if __name__ == "__main__":
    main(make_figures=True, fast=False)
