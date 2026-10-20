"""Small helpers shared by the 6-DoF demos."""

from __future__ import annotations

import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

from ssatk.constants import EARTH_MU, EARTH_RADIUS
from ssatk.propagators_6dof import rotate_vector
from ssatk.plots.plotutils import figsave

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None
FIGDIR = "six_dof"


def demo_flags(make_figures, fast):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST
    return bool(make_figures), bool(fast)


def circular_leo_state(altitude_m: float = 700_000.0):
    radius = EARTH_RADIUS + altitude_m
    r0 = np.array([radius, 0.0, 0.0])
    v0 = np.array([0.0, np.sqrt(EARTH_MU / radius), 0.0])
    return r0, v0


def hours(times):
    times = np.asarray(times, dtype=float)
    return (times - times[0]) / 3600.0


def body_axis_history(quaternions, axis_body=(1.0, 0.0, 0.0)):
    return np.array([rotate_vector(q, axis_body) for q in quaternions])


def quaternion_angle_deg(quaternions):
    quaternions = np.asarray(quaternions, dtype=float)
    return np.degrees(2.0 * np.arccos(np.clip(np.abs(quaternions[:, 0]), 0.0, 1.0)))


def vector_angle_deg(vectors, references):
    vectors = np.asarray(vectors, dtype=float)
    references = np.asarray(references, dtype=float)
    vectors = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    references = references / np.linalg.norm(references, axis=1)[:, None]
    return np.degrees(np.arccos(np.clip(np.sum(vectors * references, axis=1), -1.0, 1.0)))


def save_demo_figure(fig, filename, make_figures):
    if not make_figures:
        plt.close(fig)
        return None
    return figsave(fig, f"{FIGDIR}/{filename}", dpi=200)


def plot_orbit_km(ax, trajectory, *, label, color):
    r_km = np.asarray(trajectory.r, dtype=float) / 1000.0
    ax.plot(r_km[:, 0], r_km[:, 1], r_km[:, 2], color=color, lw=2.0, label=label)
    ax.scatter(r_km[0, 0], r_km[0, 1], r_km[0, 2], color=color, s=30)
    ax.scatter(r_km[-1, 0], r_km[-1, 1], r_km[-1, 2], color=color, s=60, marker="x")
    return r_km


def set_equal_3d(ax, *arrays_km):
    points = np.vstack([np.asarray(item, dtype=float).reshape((-1, 3)) for item in arrays_km])
    center = points.mean(axis=0)
    span = np.max(np.ptp(points, axis=0))
    span = max(float(span), 1.0)
    for setter, value in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        setter(value - span / 2.0, value + span / 2.0)
    ax.set_xlabel("GCRF x [km]")
    ax.set_ylabel("GCRF y [km]")
    ax.set_zlabel("GCRF z [km]")


def plot_earth(ax):
    radius_km = EARTH_RADIUS / 1000.0
    u = np.linspace(0.0, 2.0 * np.pi, 36)
    v = np.linspace(0.0, np.pi, 18)
    x = radius_km * np.outer(np.cos(u), np.sin(v))
    y = radius_km * np.outer(np.sin(u), np.sin(v))
    z = radius_km * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color="lightsteelblue", alpha=0.35, linewidth=0)
