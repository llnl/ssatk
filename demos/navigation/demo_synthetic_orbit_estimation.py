"""Noise-free synthetic ground-station observations and a batch orbit fit."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ssatk.constants import EARTH_MU
from ssatk.navigation import GroundStation, GroundStationSensor, solve_batch_orbit
from ssatk.plots.plotutils import figsave
from ssatk.propagators_orbit import propagate_orbit_state

GALLERY_CATEGORY = "navigation"


def main(make_figures=True):
    t0 = 1.4e9
    times = t0 + np.arange(0.0, 1801.0, 300.0)
    station = GroundStation(
        -104.9, 39.7, elevation_m=1600.0, min_elevation_rad=-np.pi / 2
    )
    truth = np.array([7_000_000.0, 0.0, 0.0, 0.0, np.sqrt(EARTH_MU / 7_000_000.0), 1200.0])
    trajectory = propagate_orbit_state(
        times=times, r0=truth[:3], v0=truth[3:], t0=t0, rtol=1e-11, atol=1e-7
    )
    sensors = [
        GroundStationSensor(station, 0.0, measurement="range", rng=0),
        GroundStationSensor(station, 0.0, measurement="range_rate", rng=0),
        GroundStationSensor(station, np.zeros((2, 2)), measurement="az_el", rng=0),
    ]
    observations = []
    for index, time in enumerate(times):
        state = np.r_[trajectory.r[index], trajectory.v[index]]
        for sensor in sensors:
            observation = sensor.measure(state, time)
            prediction = station.predict(state, time, sensor.measurement)
            # Zero sensor covariance makes the synthetic value noise-free;
            # report a tiny positive covariance so the batch whitener is valid.
            covariance = np.eye(np.atleast_1d(prediction.value).size) * (
                1e-4 if sensor.measurement == "range" else 1e-8
            )
            observations.append(
                (station, replace(observation, value=prediction.value, covariance=covariance))
            )
    perturbed = truth + np.array([20_000.0, -15_000.0, 12_000.0, 12.0, -8.0, 5.0])
    fit = solve_batch_orbit(
        observations,
        state0=perturbed,
        t0=t0,
        propagation_kwargs={"mu": EARTH_MU},
        max_nfev=80,
    )
    error = fit.state0 - truth
    print(
        f"batch orbit fit: success={fit.success}, rank={fit.numerical_rank}, "
        f"position_error={np.linalg.norm(error[:3]):.6g} m, "
        f"velocity_error={np.linalg.norm(error[3:]):.6g} m/s"
    )
    result = {"fit": fit, "truth": truth, "state_error": error}
    if make_figures:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
        axes[0].plot((times - t0) / 60, np.linalg.norm(fit.trajectory.r - trajectory.r, axis=1))
        axes[1].plot((times - t0) / 60, np.linalg.norm(fit.trajectory.v - trajectory.v, axis=1))
        axes[0].set(ylabel="Position error (m)", title="Noise-free synthetic batch fit")
        axes[1].set(xlabel="Elapsed time (min)", ylabel="Velocity error (m/s)")
        for axis in axes:
            axis.grid(alpha=0.3)
        figure.tight_layout()
        # gallery-call: demo_synthetic_orbit_estimation.png
        figsave(figure, "demo_synthetic_orbit_estimation.png")
        # gallery-end
        plt.close(figure)
    return result


if __name__ == "__main__":
    main(make_figures=True)
