"""Propagated position-error clouds over one complete nominal GEO orbit.

Gallery mode uses 128 sample orbits and 721 epochs; the larger standalone
example uses 512 orbits and 1441 epochs. Both animate 361 evenly spaced epochs.
The initial 100 m / 1 m/s error balls are synthetic, not tracking measurements.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

from ssapy import Orbit, compute
from ssapy.accel import AccelKepler
from ssapy.body import get_body
from ssapy.propagator import SciPyPropagator

from ssatk.plots.orbit_plot import orbit_plot
from ssatk.plots.divergence_gif import divergence_gif
from ssatk.plots.figpath import ssatk_path

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None
GALLERY_CATEGORY = "analysis_dashboards"
TITLE = "Orbit divergence over a full period"
DESCRIPTION = (
    "Synthetic 100 m / 1 m/s initial errors propagated for one GEO orbit "
    "(23.934 hours): 128 sample orbits and 721 epochs in gallery mode. "
    "Animations show 361 frames with nominal or median position centers "
    "and the propagated nominal velocity; about 29 seconds of playback."
)
GIF_FRAME_DURATION_S = 0.08
GIF_FRAMES = 361


def _propagate_ensemble(fast):
    """Integrate point-mass trajectories, retaining both endpoints of a period."""
    n_samples = 128 if fast else 512
    n_snapshots = 721 if fast else 1441
    earth = get_body("earth")
    r_nominal = np.array([42164000.0, 0.0, 0.0])
    v_nominal = np.array([0.0, np.sqrt(earth.mu / r_nominal[0]), 0.0])

    t0 = Time("2026-01-01T00:00:00", scale="utc")
    t0_gps = t0.gps
    orbit_nominal = Orbit(r=r_nominal, v=v_nominal, t=t0_gps, mu=earth.mu)
    period_s = float(orbit_nominal.period)
    times_gps = t0_gps + np.linspace(0.0, period_s, n_snapshots)

    pos_radius = 100.0
    vel_radius = 1.0

    r_initial = []
    v_initial = []

    rng = np.random.default_rng(0)
    for _ in range(n_samples):
        pos_direction = rng.standard_normal(3)
        pos_direction /= np.linalg.norm(pos_direction)
        pos_magnitude = pos_radius * np.cbrt(rng.uniform(0, 1))
        pos_error = pos_direction * pos_magnitude

        vel_direction = rng.standard_normal(3)
        vel_direction /= np.linalg.norm(vel_direction)
        vel_magnitude = vel_radius * np.cbrt(rng.uniform(0, 1))
        vel_error = vel_direction * vel_magnitude

        r_initial.append(r_nominal + pos_error)
        v_initial.append(v_nominal + vel_error)

    accel = AccelKepler(earth.mu)
    # Keep numerical error well below the 100 m initial cloud at GEO.
    propagator = SciPyPropagator(
        accel, ode_kwargs=dict(method="DOP853", rtol=1e-11, atol=1e-8)
    )

    r_nominal_hist, v_nominal_hist = compute.rv(orbit_nominal, times_gps, propagator)

    print(f"Propagating {n_samples} orbits for one nominal period "
          f"({period_s / 3600:.6f} hours) with {n_snapshots} snapshots...")
    r_histories = []
    v_histories = []
    for i in range(n_samples):
        orbit = Orbit(r=r_initial[i], v=v_initial[i], t=t0_gps, mu=earth.mu)
        r_hist, v_hist = compute.rv(orbit, times_gps, propagator)
        r_histories.append(r_hist)
        v_histories.append(v_hist)

    return dict(
        orbit_nominal=orbit_nominal, period_s=period_s, times_gps=times_gps,
        r_nominal_hist=r_nominal_hist, v_nominal_hist=v_nominal_hist,
        r_histories=np.array(r_histories), v_histories=np.array(v_histories),
    )


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    result = _propagate_ensemble(fast)
    result.update(fig_median=None, fig_nominal=None, gif_results={})
    if not make_figures:
        return result

    r_histories = result["r_histories"]
    r_nominal_hist = result["r_nominal_hist"]
    v_nominal_hist = result["v_nominal_hist"]
    times_gps = result["times_gps"]
    span = f"One Orbital Period ({result['period_s'] / 3600:.3f} h)"

    r_final = r_histories[:, -1, :]
    # gallery-call: divergence_plot_final_median.jpg
    fig1 = orbit_plot(
        r_final,
        view="divergence",
        v_center=v_nominal_hist[-1],
        title=f"Position Errors after {span}\nMedian Center",
        show=False,
        save=f"divergence_plot_final_median",
    )
    # gallery-end
    # gallery-call: divergence_plot_final_explicit.jpg
    fig2 = orbit_plot(
        r_final,
        view="divergence",
        r_center=r_nominal_hist[-1],
        v_center=v_nominal_hist[-1],
        title=f"Position Errors after {span}\nNominal Center",
        show=False,
        save=f"divergence_plot_final_explicit",
    )
    # gallery-end

    # gallery-call: orbit_divergence_samples.jpg
    orbit_plot(
        np.transpose(r_histories, (1, 2, 0)),
        t=times_gps,
        view="orbit_divergence",
        title=f"{len(r_histories)} Sample Orbits — {span}",
        show=False,
        save=f"orbit_divergence_samples",
    )
    # gallery-end

    # GIF cadence is independent of integration/plot sampling. Include both
    # endpoints; supply the propagated velocity to avoid one-sided derivative
    # errors rotating the median-centered frame at the start and end.
    frames = np.linspace(0, len(times_gps) - 1, GIF_FRAMES, dtype=int)
    gif_results = {}
    for name, centers in (("explicit", r_nominal_hist),
                          ("median", np.median(r_histories, axis=0))):
        timing_errors = np.einsum("stj,tj->st", r_histories - centers, v_nominal_hist)
        timing_errors /= np.sum(v_nominal_hist**2, axis=1)
        color_limit = float(np.max(np.abs(timing_errors)))
        # gallery-call: divergence_test_*.gif
        gif_results[name] = divergence_gif(
            r_histories=r_histories[:, frames],
            times_gps=times_gps[frames],
            output_path=ssatk_path(f"divergence_test_{name}.gif"),
            r_nominal_hist=centers[frames],
            v_nominal_hist=v_nominal_hist[frames],
            duration=GIF_FRAME_DURATION_S,
            vmin=-color_limit,
            vmax=color_limit,
        )
        # gallery-end

    plt.close("all")
    result.update(fig_median=fig1, fig_nominal=fig2, gif_results=gif_results)
    return result


if __name__ == "__main__":
    main(make_figures=True, fast=False)
