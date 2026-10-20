"""Numeric plot contracts with literal/analytic references, not image snapshots."""

import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.quiver import Quiver
import numpy as np
from astropy.time import Time

from ssatk.constants import EARTH_MU
from ssatk.plots.groundtrack_dashboard_gamma_heading import (
    groundtrack_dashboard_gamma_heading,
)


def test_speed_plot_uses_seconds_and_supplied_orbit_velocities():
    """R1/C: straight track sqrt(14) km/s and circle sqrt(mu/r), 1e-8 km/s.

    Unequal time spacing catches displacement-per-sample units. Sparse circle
    samples with supplied velocities distinguish true speed from chord speed.
    """
    seconds = np.array([0.0, 10.0, 40.0, 90.0])
    times = Time(Time("2025-01-01").gps + seconds, format="gps")
    straight = [7e6, 0, 0] + seconds[:, None] * [1000, 2000, 3000]
    omega = np.sqrt(EARTH_MU / 7e6**3)
    angle = omega * seconds
    circle = 7e6 * np.column_stack((np.cos(angle), np.sin(angle), np.zeros(4)))
    velocity = (
        7e6 * omega * np.column_stack((-np.sin(angle), np.cos(angle), np.zeros(4)))
    )
    for r, v, expected in [
        (straight, None, np.sqrt(14)),
        (circle, velocity, 7e6 * omega / 1000),
    ]:
        fig = groundtrack_dashboard_gamma_heading(r, times, v=v, fontsize=10)
        try:
            ax = next(ax for ax in fig.axes if ax.get_ylabel() == "Speed (km/s)")
            np.testing.assert_allclose(
                ax.lines[0].get_ydata(), expected, atol=1e-8, rtol=0
            )
        finally:
            plt.close(fig)


def test_direction_panel_uses_radial_symbols_and_true_cardinal_arrows():
    """R1/C: radial +/-90 degrees has no arrow; E/W/N/S arrows have known signs."""
    from demos.orbital_mechanics.demo_gamma_and_heading import heading_gamma_panel

    fig = heading_gamma_panel()
    try:
        for ax, gamma in zip(fig.axes[:2], [90, -90]):
            assert not any(isinstance(c, Quiver) for c in ax.collections)
            points = next(c for c in ax.collections if isinstance(c, PathCollection))
            assert len(points.get_offsets()) == 30
            np.testing.assert_allclose(points.get_array(), gamma, atol=1e-10, rtol=0)
        for ax, east, north in zip(fig.axes[2:6], [18, -18, 0, 0], [0, 0, 18, -18]):
            arrow = next(c for c in ax.collections if isinstance(c, Quiver))
            np.testing.assert_allclose(arrow.U, east, atol=1e-10, rtol=0)
            np.testing.assert_allclose(arrow.V, north, atol=1e-10, rtol=0)
            np.testing.assert_allclose(arrow.get_array(), 0, atol=1e-10, rtol=0)
    finally:
        plt.close(fig)
