"""Synthetic coherent radio observations, light-time effects and a batch fit."""

from dataclasses import replace

import numpy as np

from ssatk.constants import EARTH_MU, c
from ssatk.navigation import GroundStation, GroundStationSensor, RadioTrackingConfig, solve_batch_orbit
from ssatk.plots.plotutils import figsave
from ssatk.propagators_orbit import propagate_orbit_state_with_stm

GALLERY_CATEGORY = "navigation"


def main(make_figures=True):
    t0 = 1.4e9
    times = t0 + np.arange(60.0, 1801.0, 120.0)
    truth = np.array([7e6, 0, 0, 0, np.sqrt(EARTH_MU / 7e6), 1200.0])
    trajectory = propagate_orbit_state_with_stm(
        r0=truth[:3], v0=truth[3:], t0=t0, times=np.r_[t0, times],
        dense_output=True, rtol=1e-12, atol=1e-9,
    )
    # A synthetic all-sky mask keeps this example independent of pass scheduling.
    station = GroundStation(-104.9, 39.7, elevation_m=1600, min_elevation_rad=-np.pi / 2)
    radio = RadioTrackingConfig(7.2e9, 10.0, 880 / 749, 2e-6)
    observations, ranges, dopplers = [], [], []
    range_offsets, doppler_offsets = [], []
    for measurement, values, offsets in (
        ("two_way_range", ranges, range_offsets), ("two_way_doppler", dopplers, doppler_offsets),
    ):
        sensor = GroundStationSensor(station, 0.0, measurement=measurement, radio=radio, rng=0)
        for index, time in enumerate(times, start=1):
            observation = sensor.measure(trajectory, time)
            observations.append((station, replace(observation, covariance=np.array([[1.0]]))))
            state = np.r_[trajectory.r[index], trajectory.v[index]]
            geometric = station.predict(state, time, "range" if measurement == "two_way_range" else "range_rate").value
            if measurement == "two_way_doppler":
                geometric *= -2 * radio.turnaround_ratio * radio.uplink_frequency_hz / c
            values.append(observation.value)
            offsets.append(observation.value - geometric)
    fit = solve_batch_orbit(
        observations, state0=truth + [200, -150, 120, 0.2, -0.1, 0.3], t0=t0,
        propagation_kwargs={"rtol": 1e-12, "atol": 1e-9}, max_nfev=30,
    )
    error = fit.state0 - truth
    print(f"Vacuum synthetic radio fit: success={fit.success}, rank={fit.numerical_rank}, "
          f"position_error={np.linalg.norm(error[:3]):.6g} m, velocity_error={np.linalg.norm(error[3:]):.6g} m/s")
    if make_figures:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
        axes[0].plot((times - t0) / 60, range_offsets)
        axes[1].plot((times - t0) / 60, doppler_offsets)
        axes[0].set(ylabel="Range difference (m)", title="Vacuum radio versus instantaneous geometry")
        axes[1].set(xlabel="Receive epoch since t0 (min)", ylabel="Doppler difference (Hz)")
        for axis in axes:
            axis.grid(alpha=0.3)
        figure.tight_layout()
        # gallery-call: demo_radio_tracking.png
        figsave(figure, "demo_radio_tracking.png")
        # gallery-end
        plt.close(figure)
    return {"fit": fit, "truth": truth, "state_error": error, "ranges_m": np.array(ranges),
            "doppler_hz": np.array(dopplers), "radio": radio}


if __name__ == "__main__":
    main()
