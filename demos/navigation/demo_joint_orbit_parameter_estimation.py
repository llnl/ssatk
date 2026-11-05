"""Joint synthetic orbit, drag/SRP, finite-burn thrust and measurement-bias fit."""

from dataclasses import replace

import numpy as np

from ssatk.accelerations_6dof import SpacecraftAccelDrag, SpacecraftAccelSolRad, make_finite_burn_acceleration
from ssatk.constants import AU, EARTH_MU
from ssatk.navigation import GroundStation, GroundStationSensor, solve_batch_orbit
from ssatk.plots.plotutils import figsave
from ssatk.propagators_orbit import propagate_orbit_state

GALLERY_CATEGORY = "navigation"


def main(make_figures=True):
    t0 = 1.4e9
    times = t0 + np.arange(0., 1801., 120.)
    state = np.array([7e6, 1e5, 2e5, -100., np.sqrt(EARTH_MU / 7e6), 700.])
    truth = {"cd": 2.1, "cr": 1.4, "thrust_n": 0.2, "range_bias_m": 15., "rate_bias_m_s": 0.05}
    station = GroundStation(-104.9, 39.7, elevation_m=1600., min_elevation_rad=-np.pi / 2)

    def propagation(parameters):
        # Prescribed density/Sun and constant mass make this a controlled synthetic example.
        return {"acceleration": (
            SpacecraftAccelDrag(density=2e-11, area=1000., mass=100., cd=parameters["cd"]),
            SpacecraftAccelSolRad([AU, 0, 0], area=1000., mass=100., cr=parameters["cr"]),
            make_finite_burn_acceleration(parameters["thrust_n"], mass=100., direction=[0, 0, 1],
                                         frame="inertial", start=t0+600., stop=t0+660.),
        ), "rtol": 1e-12, "atol": 1e-9, "fd_step": 1e-4}

    def bias(station, observation, parameters):
        return {"range": parameters["range_bias_m"], "range_rate": parameters["rate_bias_m_s"]}.get(
            observation.measurement, np.zeros(2)
        )

    truth_options = propagation(truth)
    truth_options.pop("fd_step")
    trajectory = propagate_orbit_state(times=times, r0=state[:3], v0=state[3:], t0=t0, **truth_options)
    sensors = [GroundStationSensor(station, np.zeros((dimension, dimension)), measurement=kind, rng=0)
               for kind, dimension in (("range", 1), ("range_rate", 1), ("az_el", 2))]
    observations = []
    for time, r, v in zip(times, trajectory.r, trajectory.v):
        for sensor in sensors:
            observation = sensor.measure(np.r_[r, v], time)
            sigma = {"range": 1., "range_rate": 0.001, "az_el": 1e-6}[sensor.measurement]
            observations.append((station, replace(
                observation, value=observation.value + bias(station, observation, truth),
                covariance=np.eye(observation.covariance.shape[0]) * sigma**2,
            )))
    fit = solve_batch_orbit(
        observations, state + [50., -40., 30., 0.1, -0.05, 0.02], t0=t0,
        parameters0={name: value * 0.8 for name, value in truth.items()},
        parameter_scale=[2., 1., 0.2, 20., 0.1],
        parameter_step=[0.002, 0.002, 0.0002, 0.1, 0.001],
        parameter_bounds=([0.1, 0.1, 0., -100., -1.], [5., 3., 1., 100., 1.]),
        propagation_kwargs=propagation, measurement_bias=bias, max_nfev=30,
    )
    print(f"Joint fit: success={fit.success}, rank={fit.numerical_rank}/{len(fit.estimate_names)}")
    for index, (name, value) in enumerate(fit.parameters.items(), start=6):
        sigma = np.nan if fit.joint_covariance is None else np.sqrt(fit.joint_covariance[index, index])
        print(f"  {name}: fitted={value:.8g}, specified={truth[name]:.8g}, formal_sigma={sigma:.4g}")
    if not fit.success:
        raise RuntimeError(f"Joint synthetic fit failed: {fit.message}")
    if make_figures:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(9, 8))
        image = axis.imshow(fit.correlation, vmin=-1, vmax=1, cmap="coolwarm")
        ticks = np.arange(len(fit.estimate_names))
        axis.set(xticks=ticks, yticks=ticks, xticklabels=fit.estimate_names, yticklabels=fit.estimate_names,
                 title="Joint orbit and parameter correlations\nSynthetic observations; prescribed force models")
        plt.setp(axis.get_xticklabels(), rotation=60, ha="right")
        figure.colorbar(image, ax=axis, label="Correlation")
        figure.tight_layout()
        # gallery-call: demo_joint_orbit_parameter_estimation.png
        figsave(figure, "demo_joint_orbit_parameter_estimation.png")
        # gallery-end
        plt.close(figure)
    return {"fit": fit, "state_truth": state, "parameter_truth": truth}


if __name__ == "__main__":
    main()
