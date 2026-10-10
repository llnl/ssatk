"""Analytic Hohmann transfers between coplanar circular orbits."""

from __future__ import annotations

import numpy as np
from ssapy import Orbit
from ssapy.constants import EARTH_MU

from ssatk.coordinates import gcrf_to_ntw
from ssatk.orbital_mechanics._transfer_result import (
    maneuver_burn,
    trajectory_dict,
    transfer_result,
    transfer_state,
)
from ssatk.orbital_mechanics.misc import circular_velocity, hohmann_transfer_delta_v, vis_viva


def _circularity_error(state, mu):
    radius = np.linalg.norm(state["r"])
    expected_speed = circular_velocity(mu, radius)
    radial_speed = abs(float(np.dot(state["r"], state["v"]))) / radius
    speed_error = abs(np.linalg.norm(state["v"]) - expected_speed)
    return max(radial_speed, speed_error) / expected_speed


def _transfer_arc(radius1, radius2, mu, eccentric_anomaly):
    semi_major_axis = 0.5 * (radius1 + radius2)
    eccentricity = abs(radius2 - radius1) / (radius1 + radius2)
    if radius2 >= radius1:
        anomaly = eccentric_anomaly
        radius_periapsis = radius1
        sign = 1.0
    else:
        anomaly = eccentric_anomaly + np.pi
        radius_periapsis = radius2
        sign = -1.0
    mean_motion = np.sqrt(mu / semi_major_axis**3)
    denominator = 1.0 - eccentricity * np.cos(anomaly)
    x_coord = sign * semi_major_axis * (np.cos(anomaly) - eccentricity)
    y_coord = sign * semi_major_axis * np.sqrt(1.0 - eccentricity**2) * np.sin(anomaly)
    x_velocity = sign * -semi_major_axis * mean_motion * np.sin(anomaly) / denominator
    y_velocity = sign * (
        semi_major_axis
        * mean_motion
        * np.sqrt(1.0 - eccentricity**2)
        * np.cos(anomaly)
        / denominator
    )
    return x_coord, y_coord, x_velocity, y_velocity


def _rotate(x_coord, y_coord, x_velocity, y_velocity, radial_hat, tangent_hat):
    r = x_coord[:, None] * radial_hat[None, :] + y_coord[:, None] * tangent_hat[None, :]
    v = x_velocity[:, None] * radial_hat[None, :] + y_velocity[:, None] * tangent_hat[None, :]
    return r, v


def _parse_inputs(args, initial, target, orbit1, orbit2, r1, v1, r2, v2):
    if orbit1 is not None:
        if initial is not None:
            raise TypeError("Specify either orbit1 or initial, not both")
        initial = orbit1
    if orbit2 is not None:
        if target is not None:
            raise TypeError("Specify either orbit2 or target, not both")
        target = orbit2
    if args:
        if len(args) == 2:
            initial, target = args
        elif len(args) == 4:
            r1, v1, r2, v2 = args
        else:
            raise ValueError("Use transfer_hohmann(initial, target) or transfer_hohmann(r1, v1, r2, v2)")
    if initial is None and r1 is not None:
        initial = (r1, v1, 0.0) if v1 is not None else r1
    if target is None and r2 is not None:
        target = (r2, v2, 0.0) if v2 is not None else r2
    if initial is None or target is None:
        raise ValueError("initial and target are required")
    return initial, target


def transfer_hohmann(
    *args,
    orbit1=None,
    orbit2=None,
    initial=None,
    target=None,
    r1=None,
    v1=None,
    r2=None,
    v2=None,
    mu=EARTH_MU,
    samples=300,
    circular_tol=1e-3,
    burn_accel=None,
    thrust=None,
    mass=None,
    isp=None,
    plot=False,
    save_path=False,
):
    """Return the canonical result dict for a two-impulse Hohmann transfer.

    Inputs may be circular-orbit radii, SSAPy ``Orbit`` objects, ``(r, v, t)``
    tuples, mappings with ``r``/``v``/``t`` keys, or explicit ``r1``/``v1`` and
    ``r2``/``v2`` vectors. The solver intentionally treats the problem as an
    orbit-to-orbit transfer: target phasing is not solved, and the final burn is
    placed half a transfer period after departure on the opposite line of
    apsides.
    """
    if samples < 2:
        raise ValueError("samples must be at least 2")
    initial, target = _parse_inputs(args, initial, target, orbit1, orbit2, r1, v1, r2, v2)
    initial_state = transfer_state(state=initial, mu=mu)
    target_state = transfer_state(state=target, mu=mu)
    initial_error = _circularity_error(initial_state, mu)
    target_error = _circularity_error(target_state, mu)
    if initial_error > circular_tol or target_error > circular_tol:
        raise ValueError(
            "transfer_hohmann assumes circular boundary orbits; "
            f"relative errors were {initial_error:.3g} and {target_error:.3g}"
        )

    radius1 = float(np.linalg.norm(initial_state["r"]))
    radius2 = float(np.linalg.norm(target_state["r"]))
    radial_hat = initial_state["r"] / radius1
    tangent_hat = initial_state["v"] - np.dot(initial_state["v"], radial_hat) * radial_hat
    tangent_hat = tangent_hat / np.linalg.norm(tangent_hat)

    semi_major_axis = 0.5 * (radius1 + radius2)
    tof = np.pi * np.sqrt(semi_major_axis**3 / mu)
    v_circular1 = circular_velocity(mu, radius1)
    v_circular2 = circular_velocity(mu, radius2)
    v_transfer1 = vis_viva(mu, radius1, semi_major_axis)
    v_transfer2 = vis_viva(mu, radius2, semi_major_axis)

    r_depart = radius1 * radial_hat
    v_depart = v_circular1 * tangent_hat
    r_arrive = -radius2 * radial_hat
    v_transfer_arrive = -v_transfer2 * tangent_hat
    v_arrive = -v_circular2 * tangent_hat
    delta_v1 = (v_transfer1 - v_circular1) * tangent_hat
    delta_v2 = v_arrive - v_transfer_arrive

    eccentric_anomaly = np.linspace(0.0, np.pi, samples)
    x_coord, y_coord, x_velocity, y_velocity = _transfer_arc(radius1, radius2, mu, eccentric_anomaly)
    r_transfer, v_transfer = _rotate(x_coord, y_coord, x_velocity, y_velocity, radial_hat, tangent_hat)
    eccentricity = abs(radius2 - radius1) / (radius1 + radius2)
    mean_motion = np.sqrt(mu / semi_major_axis**3)
    if radius2 >= radius1:
        t_transfer = (eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly)) / mean_motion
    else:
        anomaly = eccentric_anomaly + np.pi
        t_transfer = (anomaly - eccentricity * np.sin(anomaly) - np.pi) / mean_motion
    t_transfer = t_transfer + initial_state["t"]

    transfer_orbit = Orbit(r_depart, v_depart + delta_v1, initial_state["t"], mu=mu)
    final_state = transfer_state(state={"r": r_arrive, "v": v_arrive, "t": initial_state["t"] + tof}, mu=mu)
    burn1 = maneuver_burn(
        name="departure",
        state={"r": r_depart, "v": v_depart, "t": initial_state["t"]},
        delta_v=delta_v1,
        t=initial_state["t"],
        burn_accel=burn_accel,
        thrust=thrust,
        mass=mass,
        isp=isp,
    )
    burn2 = maneuver_burn(
        name="arrival_circularization",
        state={"r": r_arrive, "v": v_transfer_arrive, "t": initial_state["t"] + tof},
        delta_v=delta_v2,
        t=initial_state["t"] + tof,
        burn_accel=burn_accel,
        thrust=thrust,
        mass=mass,
        isp=isp,
    )
    expected = hohmann_transfer_delta_v(radius1, radius2, mu)
    result = transfer_result(
        method="transfer_hohmann",
        initial=initial_state,
        target=final_state,
        final=final_state,
        burns=[burn1, burn2],
        trajectory=trajectory_dict(t=t_transfer, r=r_transfer, v=v_transfer),
        transfer_orbits=[transfer_orbit],
        tof=tof,
        assumptions=[
            "circular boundary orbits",
            "coplanar orbit-to-orbit transfer",
            "target phasing is not solved",
            "two-body impulsive Hohmann transfer",
        ],
        diagnostics={
            "arrival_mode": "insertion",
            "timing_constraint": "free",
            "arrival_velocity_match": True,
            "arrival_burn": True,
            "radius1": radius1,
            "radius2": radius2,
            "semi_major_axis": semi_major_axis,
            "analytic_delta_v": expected,
            "input_target_state": target_state,
            "delta_v_ntw": [gcrf_to_ntw(delta_v1, r_depart, v_depart), gcrf_to_ntw(delta_v2, r_arrive, v_transfer_arrive)],
            "circularity_error": [initial_error, target_error],
        },
    )

    if plot:
        from ssatk.plots import transfer_plot

        fig = transfer_plot(
            r_depart,
            v_depart,
            r_transfer,
            v_transfer,
            r_arrive,
            v_arrive,
            save_path=save_path,
            title=f"Hohmann transfer | Δv {result['delta_v_total'] / 1e3:.3f} km/s",
        )
        result["figure"] = fig
    return result
