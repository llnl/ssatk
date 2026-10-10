"""
Demo for perturb_state_3d.

This script:
  * Defines a nominal 3D position + velocity
  * Applies perturb_state_3d many times with different distributions:
        - uniform   (in a solid ball)
        - normal    (component-wise Gaussian)
        - shell     (fixed-radius sphere surface)
        - laplace   (component-wise Laplace)
  * Prints basic stats of the perturbations
  * Plots a 2x2 figure per case:
        [0,0] position offsets (dx, dy)
        [0,1] velocity offsets (dvx, dvy)
        [1,0] position samples  (x,  y)  centered around r_nom
        [1,1] velocity samples  (vx, vy) centered around v_nom

Pytest-safe behavior:
- figures are not saved/shown by default under pytest
- sample count is reduced under pytest
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from ssatk.compute.sampling import perturb_state_3d
from ssatk.plots.plotutils import figsave

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def demo_single_distribution(
    r_nom,
    v_nom,
    pos_scale,
    vel_scale,
    pos_distribution,
    vel_distribution,
    n_samples=1000,
    rng_seed=42,
):
    """
    Draw many perturbed states and return offsets for analysis / plotting.
    """
    rng = np.random.default_rng(rng_seed)

    r_offsets = np.zeros((n_samples, 3))
    v_offsets = np.zeros((n_samples, 3))

    for i in range(n_samples):
        r_p, v_p = perturb_state_3d(
            r_nom,
            v_nom,
            pos_scale=pos_scale,
            vel_scale=vel_scale,
            pos_distribution=pos_distribution,
            vel_distribution=vel_distribution,
            rng=rng,
        )
        r_offsets[i] = r_p - r_nom
        v_offsets[i] = v_p - v_nom

    return r_offsets, v_offsets


def summarize_offsets(name, r_offsets, v_offsets, verbose=True):
    """
    Print basic statistics for position & velocity perturbations.
    """
    r_norm = np.linalg.norm(r_offsets, axis=1)
    v_norm = np.linalg.norm(v_offsets, axis=1)

    summary = {
        "position_mean_radius": float(np.mean(r_norm)),
        "position_std_radius": float(np.std(r_norm)),
        "position_max_radius": float(np.max(r_norm)),
        "velocity_mean_radius": float(np.mean(v_norm)),
        "velocity_std_radius": float(np.std(v_norm)),
        "velocity_max_radius": float(np.max(v_norm)),
    }

    if verbose:
        print(f"\n=== {name} ===")
        print(
            "Position offsets | "
            f"mean radius = {summary['position_mean_radius']:.3f}, "
            f"std radius  = {summary['position_std_radius']:.3f}, "
            f"max radius  = {summary['position_max_radius']:.3f}"
        )
        print(
            "Velocity offsets | "
            f"mean radius = {summary['velocity_mean_radius']:.3f}, "
            f"std radius  = {summary['velocity_std_radius']:.3f}, "
            f"max radius  = {summary['velocity_max_radius']:.3f}"
        )

    return summary


def plot_offsets_2d(name, r_offsets, v_offsets, r_nom, v_nom):
    """
    2x2 plots:
      [0,0] position offsets (dx, dy)
      [0,1] velocity offsets (dvx, dvy)
      [1,0] position samples  (x,  y)  centered around r_nom
      [1,1] velocity samples  (vx, vy) centered around v_nom
    """
    r_samples = r_nom[np.newaxis, :] + r_offsets
    v_samples = v_nom[np.newaxis, :] + v_offsets

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(name)

    ax = axes[0, 0]
    ax.scatter(r_offsets[:, 0], r_offsets[:, 1], s=5, alpha=0.5)
    ax.set_title("Position offsets (dx, dy)")
    ax.set_xlabel("dx")
    ax.set_ylabel("dy")
    ax.axis("equal")
    ax.grid(True)

    ax = axes[0, 1]
    ax.scatter(v_offsets[:, 0], v_offsets[:, 1], s=5, alpha=0.5)
    ax.set_title("Velocity offsets (dvx, dvy)")
    ax.set_xlabel("dvx")
    ax.set_ylabel("dvy")
    ax.axis("equal")
    ax.grid(True)

    ax = axes[1, 0]
    ax.scatter(r_samples[:, 0], r_samples[:, 1], s=5, alpha=0.5)
    ax.scatter([r_nom[0]], [r_nom[1]], marker="+", s=80)
    ax.set_title("Position samples (x, y)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True)

    ax = axes[1, 1]
    ax.scatter(v_samples[:, 0], v_samples[:, 1], s=5, alpha=0.5)
    ax.scatter([v_nom[0]], [v_nom[1]], marker="+", s=80)
    ax.set_title("Velocity samples (vx, vy)")
    ax.set_xlabel("vx")
    ax.set_ylabel("vy")
    ax.axis("equal")
    ax.grid(True)

    plt.tight_layout()
    return fig


def main(make_figures=None, fast=None, verbose=None):
    """
    Run the sampling demo.

    Parameters
    ----------
    make_figures : bool or None
        If None, defaults to False under pytest and True otherwise.
    fast : bool or None
        If None, defaults to True under pytest and False otherwise.
    verbose : bool or None
        If None, defaults to False under pytest and True otherwise.

    Returns
    -------
    dict
        Per-distribution offsets and summaries.
    """
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST
    if verbose is None:
        verbose = not UNDER_PYTEST

    r_nom = np.array([100.0, 100.0, 100.0])
    v_nom = np.array([0.0, 1.0, 0.0])

    pos_scale = 1.0
    vel_scale = 0.1

    distributions = [
        ("Uniform ball", "uniform", "uniform"),
        ("Gaussian", "normal", "normal"),
        ("Shell (fixed radius)", "shell", "shell"),
        ("Laplace", "laplace", "laplace"),
    ]

    if make_figures:
        os.makedirs("tests", exist_ok=True)

    n_samples = 300 if fast else 2000

    outputs = {}

    for label, pos_dist, vel_dist in distributions:
        r_offsets, v_offsets = demo_single_distribution(
            r_nom,
            v_nom,
            pos_scale=pos_scale,
            vel_scale=vel_scale,
            pos_distribution=pos_dist,
            vel_distribution=vel_dist,
            n_samples=n_samples,
            rng_seed=123,
        )

        summary = summarize_offsets(label, r_offsets, v_offsets, verbose=verbose)
        outputs[label] = {
            "r_offsets": r_offsets,
            "v_offsets": v_offsets,
            "summary": summary,
        }

        if make_figures:
            fig = plot_offsets_2d(label, r_offsets, v_offsets, r_nom, v_nom)
            safe_label = label.replace(" ", "_")
            figsave(fig, f"sampling_tests_{safe_label}")
            plt.close(fig)

    return outputs


if __name__ == "__main__":
    main(make_figures=True, fast=False, verbose=True)
