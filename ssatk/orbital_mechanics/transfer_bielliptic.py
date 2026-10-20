"""Analytic bi-elliptic transfers between coplanar circular orbits."""

from __future__ import annotations

import numpy as np
from ssapy import Orbit
from astropy.time import Time

from ..constants import EARTH_MU
from ..time_functions import to_gps
from ..plots.plotutils import _pop_save_path_aliases, _raise_unrecognized_kwargs
from ._transfer_result import maneuver_burn, trajectory_dict, transfer_result, transfer_state
from .misc import bi_elliptic_transfer_delta_v, circular_velocity, vis_viva


def _norm(vector, name):
    vector = np.asarray(vector, dtype=float).reshape(3)
    magnitude = float(np.linalg.norm(vector))
    if magnitude <= 0.0:
        raise ValueError(f"{name} must be non-zero")
    return vector, magnitude


def _default_tangent(radius_hat):
    reference = np.array([0.0, 0.0, 1.0])
    tangent = np.cross(reference, radius_hat)
    if np.linalg.norm(tangent) <= 1e-12:
        tangent = np.cross(np.array([0.0, 1.0, 0.0]), radius_hat)
    return tangent / np.linalg.norm(tangent)


def _circularity_error(radius_vector, velocity, mu):
    radius_vector, radius = _norm(radius_vector, "radius_vector")
    velocity = np.asarray(velocity, dtype=float).reshape(3)
    expected_speed = circular_velocity(mu, radius)
    radial_speed = abs(float(np.dot(radius_vector, velocity))) / radius
    speed_error = abs(np.linalg.norm(velocity) - expected_speed)
    return max(radial_speed, speed_error) / expected_speed


def _resolve_transfer_inputs(
    *args,
    initial=None,
    target=None,
    r1=None,
    v1=None,
    r2=None,
    v2=None,
    orbit1=None,
    orbit2=None,
    radius1=None,
    radius2=None,
    t0=Time("2025-01-01"),
    mu=EARTH_MU,
):
    if args:
        if len(args) == 2 and all(hasattr(arg, attr) for arg in args for attr in ("r", "v", "t")):
            orbit1, orbit2 = args
        elif len(args) == 2 and all(np.isscalar(arg) for arg in args):
            radius1, radius2 = args
        elif len(args) == 4:
            r1, v1, r2, v2 = args
        else:
            raise ValueError(
                "Positional arguments must be (orbit1, orbit2), "
                "(radius1, radius2), or (r1, v1, r2, v2)."
            )

    if orbit1 is not None:
        r1 = orbit1.r
        v1 = orbit1.v
        t0 = orbit1.t
    if orbit2 is not None:
        r2 = orbit2.r
        v2 = orbit2.v

    if radius1 is not None or radius2 is not None:
        if radius1 is None or radius2 is None:
            raise ValueError("radius1 and radius2 must be supplied together")
        radius1 = float(radius1)
        radius2 = float(radius2)
        if radius1 <= 0.0 or radius2 <= 0.0:
            raise ValueError("radius1 and radius2 must be positive")
        r1 = np.array([radius1, 0.0, 0.0])
        v1 = np.array([0.0, circular_velocity(mu, radius1), 0.0])
        r2 = np.array([radius2, 0.0, 0.0])
        v2 = np.array([0.0, circular_velocity(mu, radius2), 0.0])

    if r1 is None:
        raise ValueError("Supply orbit1, radius1/radius2, or r1/v1/r2/v2")
    r1, radius1 = _norm(r1, "r1")
    if v1 is None:
        radius_hat = r1 / radius1
        v1 = circular_velocity(mu, radius1) * _default_tangent(radius_hat)
    v1 = np.asarray(v1, dtype=float).reshape(3)

    if r2 is None:
        raise ValueError("Supply orbit2, radius1/radius2, or r2")
    r2, radius2 = _norm(r2, "r2")
    if v2 is None:
        v2 = circular_velocity(mu, radius2) * _default_tangent(r2 / radius2)
    v2 = np.asarray(v2, dtype=float).reshape(3)

    return r1, v1, radius1, r2, v2, radius2, to_gps(t0)


def _transfer_arc(radius_periapsis, radius_apoapsis, mu, eccentric_anomaly):
    semi_major_axis = 0.5 * (radius_periapsis + radius_apoapsis)
    eccentricity = (radius_apoapsis - radius_periapsis) / (radius_apoapsis + radius_periapsis)
    mean_motion = np.sqrt(mu / semi_major_axis**3)
    denominator = 1.0 - eccentricity * np.cos(eccentric_anomaly)
    x_coord = semi_major_axis * (np.cos(eccentric_anomaly) - eccentricity)
    y_coord = semi_major_axis * np.sqrt(1.0 - eccentricity**2) * np.sin(eccentric_anomaly)
    x_velocity = -semi_major_axis * mean_motion * np.sin(eccentric_anomaly) / denominator
    y_velocity = (
        semi_major_axis
        * mean_motion
        * np.sqrt(1.0 - eccentricity**2)
        * np.cos(eccentric_anomaly)
        / denominator
    )
    return x_coord, y_coord, x_velocity, y_velocity


def _rotate_planar_state(x_coord, y_coord, x_velocity, y_velocity, radius_hat, tangent_hat):
    positions = x_coord[:, None] * radius_hat[None, :] + y_coord[:, None] * tangent_hat[None, :]
    velocities = x_velocity[:, None] * radius_hat[None, :] + y_velocity[:, None] * tangent_hat[None, :]
    return positions, velocities


def transfer_bielliptic(
    *args,
    initial=None,
    target=None,
    r1=None,
    v1=None,
    r2=None,
    v2=None,
    orbit1=None,
    orbit2=None,
    radius1=None,
    radius2=None,
    rb=None,
    intermediate_radius=None,
    apoapsis_radius=None,
    mu=EARTH_MU,
    t0=Time("2025-01-01"),
    samples_per_arc=300,
    check_circular=True,
    circular_tol=1e-3,
    burn_accel=None,
    thrust=None,
    mass=None,
    isp=None,
    plot=False,
    save_path=False,
    **save_kwargs,
):
    """Compute a three-impulse bi-elliptic transfer.

    This is the analytic circular, coplanar orbit-to-orbit transfer: burn from
    the departure circular orbit onto an ellipse with apoapsis ``rb``, burn at
    that apoapsis onto a second ellipse, then circularize at the target radius.
    It optimizes no phasing; input state vectors set the transfer plane and
    prograde tangent direction, while the final arrival point is the circular
    target radius on that same line of apsides.

    Parameters
    ----------
    orbit1, orbit2 : ssapy.Orbit, optional
        Circular departure and target orbits. Their radii are used; the first
        orbit sets the transfer plane and tangent direction.
    radius1, radius2 : float, optional
        Circular orbit radii in meters. Positional ``(radius1, radius2)`` is
        also accepted.
    r1, v1, r2, v2 : array_like, optional
        State-vector form. Positional ``(r1, v1, r2, v2)`` is also accepted.
    rb, intermediate_radius, apoapsis_radius : float
        Intermediate apoapsis radius in meters; must exceed both orbit radii.
    samples_per_arc : int
        Number of samples per half-ellipse transfer arc.
    check_circular : bool
        If True, reject state-vector inputs that are not close to circular.
    circular_tol : float
        Relative circularity tolerance used when ``check_circular=True``.
    plot : bool
        If True, attach a transfer plot as ``result["fig"]``.
    save_path, save, savefig, save_fig, save_figure, savepath, save_path
        Optional figure save path aliases passed to ``transfer_plot``.

    Returns
    -------
    dict
        Canonical transfer dictionary with three burns, sampled transfer arcs,
        time of flight, assumptions, diagnostics, and SSAPy transfer ``Orbit``
        objects.
    """
    save_path, save_kwargs = _pop_save_path_aliases(save_kwargs, save_path=save_path)
    _raise_unrecognized_kwargs(save_kwargs, "transfer_bielliptic")

    candidate_radii = [value for value in (rb, intermediate_radius, apoapsis_radius) if value is not None]
    if len(candidate_radii) != 1:
        raise ValueError("Specify exactly one of rb, intermediate_radius, or apoapsis_radius")
    rb = float(candidate_radii[0])

    if samples_per_arc < 2:
        raise ValueError("samples_per_arc must be at least 2")

    if initial is not None:
        initial_state = transfer_state(state=initial, mu=mu)
        r1, v1, t0 = initial_state["r"], initial_state["v"], initial_state["t"]
    if target is not None:
        target_state = transfer_state(state=target, mu=mu)
        r2, v2 = target_state["r"], target_state["v"]

    r1, v1, radius1, _r2_input, v2_input, radius2, t0 = _resolve_transfer_inputs(
        *args,
        r1=r1,
        v1=v1,
        r2=r2,
        v2=v2,
        orbit1=orbit1,
        orbit2=orbit2,
        radius1=radius1,
        radius2=radius2,
        t0=t0,
        mu=mu,
    )
    bi_elliptic_transfer_delta_v(radius1, radius2, rb, mu)

    if check_circular:
        error1 = _circularity_error(r1, v1, mu)
        error2 = _circularity_error(_r2_input, v2_input, mu)
        if error1 > circular_tol or error2 > circular_tol:
            raise ValueError(
                "transfer_bielliptic assumes circular boundary orbits; "
                f"relative errors were {error1:.3g} and {error2:.3g}"
            )

    radius_hat = r1 / radius1
    tangent_hat = v1 - np.dot(v1, radius_hat) * radius_hat
    if np.linalg.norm(tangent_hat) <= 1e-12:
        tangent_hat = _default_tangent(radius_hat)
    else:
        tangent_hat = tangent_hat / np.linalg.norm(tangent_hat)

    v_circular1 = circular_velocity(mu, radius1)
    v_circular2 = circular_velocity(mu, radius2)
    semi_major_axis1 = 0.5 * (radius1 + rb)
    semi_major_axis2 = 0.5 * (radius2 + rb)
    tof1 = np.pi * np.sqrt(semi_major_axis1**3 / mu)
    tof2 = np.pi * np.sqrt(semi_major_axis2**3 / mu)
    tof = tof1 + tof2

    v_periapsis1 = vis_viva(mu, radius1, semi_major_axis1)
    v_apoapsis1 = vis_viva(mu, rb, semi_major_axis1)
    v_apoapsis2 = vis_viva(mu, rb, semi_major_axis2)
    v_periapsis2 = vis_viva(mu, radius2, semi_major_axis2)

    delta_v1 = (v_periapsis1 - v_circular1) * tangent_hat
    delta_v2 = (v_apoapsis1 - v_apoapsis2) * tangent_hat
    delta_v3 = (v_circular2 - v_periapsis2) * tangent_hat

    eccentric_anomaly1 = np.linspace(0.0, np.pi, int(samples_per_arc))
    x1, y1, vx1, vy1 = _transfer_arc(radius1, rb, mu, eccentric_anomaly1)
    arc1_r, arc1_v = _rotate_planar_state(x1, y1, vx1, vy1, radius_hat, tangent_hat)
    mean_motion1 = np.sqrt(mu / semi_major_axis1**3)
    arc1_t = (eccentric_anomaly1 - (rb - radius1) / (rb + radius1) * np.sin(eccentric_anomaly1)) / mean_motion1

    eccentric_anomaly2 = np.linspace(np.pi, 2.0 * np.pi, int(samples_per_arc))
    x2, y2, vx2, vy2 = _transfer_arc(radius2, rb, mu, eccentric_anomaly2)
    arc2_r, arc2_v = _rotate_planar_state(x2, y2, vx2, vy2, radius_hat, tangent_hat)
    mean_motion2 = np.sqrt(mu / semi_major_axis2**3)
    eccentricity2 = (rb - radius2) / (rb + radius2)
    arc2_t = tof1 + (
        eccentric_anomaly2 - eccentricity2 * np.sin(eccentric_anomaly2) - np.pi
    ) / mean_motion2

    r_transfer = np.vstack((arc1_r, arc2_r[1:]))
    v_transfer = np.vstack((arc1_v, arc2_v[1:]))
    t_transfer = np.concatenate((arc1_t, arc2_t[1:]))

    r_depart = radius1 * radius_hat
    v_depart = v_circular1 * tangent_hat
    r_intermediate = -rb * radius_hat
    v_intermediate_before = -v_apoapsis1 * tangent_hat
    v_intermediate_after = -v_apoapsis2 * tangent_hat
    r_arrive = radius2 * radius_hat
    v_arrive_preburn = v_periapsis2 * tangent_hat
    v_arrive = v_circular2 * tangent_hat

    initial_orbit = Orbit(r=r_depart, v=v_depart, t=t0, mu=mu)
    final_orbit = Orbit(r=r_arrive, v=v_arrive, t=t0 + tof, mu=mu)
    transfer1 = Orbit(r=r_depart, v=v_depart + delta_v1, t=t0, mu=mu)
    transfer2 = Orbit(r=r_intermediate, v=v_intermediate_after, t=t0 + tof1, mu=mu)

    initial_state = transfer_state(state={"r": r_depart, "v": v_depart, "t": t0, "orbit": initial_orbit}, mu=mu)
    target_state = transfer_state(state={"r": r_arrive, "v": v_arrive, "t": t0 + tof, "orbit": final_orbit}, mu=mu)
    burns = [
        maneuver_burn(
            name="departure",
            state={"r": r_depart, "v": v_depart, "t": t0},
            delta_v=delta_v1,
            t=t0,
            burn_accel=burn_accel,
            thrust=thrust,
            mass=mass,
            isp=isp,
        ),
        maneuver_burn(
            name="intermediate_apoapsis",
            state={"r": r_intermediate, "v": v_intermediate_before, "t": t0 + tof1},
            delta_v=delta_v2,
            t=t0 + tof1,
            burn_accel=burn_accel,
            thrust=thrust,
            mass=mass,
            isp=isp,
        ),
        maneuver_burn(
            name="arrival_circularization",
            state={"r": r_arrive, "v": v_arrive_preburn, "t": t0 + tof},
            delta_v=delta_v3,
            t=t0 + tof,
            burn_accel=burn_accel,
            thrust=thrust,
            mass=mass,
            isp=isp,
        ),
    ]
    result = transfer_result(
        method="transfer_bielliptic",
        initial=initial_state,
        target=target_state,
        final=target_state,
        burns=burns,
        trajectory=trajectory_dict(t=t_transfer + t0, r=r_transfer, v=v_transfer),
        transfer_orbits=[transfer1, transfer2],
        tof=tof,
        assumptions=[
            "circular boundary orbits",
            "coplanar orbit-to-orbit transfer",
            "target phasing is not solved",
            "two-body impulsive bi-elliptic transfer",
        ],
        diagnostics={
            "arrival_mode": "insertion",
            "timing_constraint": "free",
            "arrival_velocity_match": True,
            "arrival_burn": True,
            "radius1": radius1,
            "radius2": radius2,
            "intermediate_radius": rb,
            "semi_major_axes": (semi_major_axis1, semi_major_axis2),
            "tof1": float(tof1),
            "tof2": float(tof2),
            "analytic_delta_v": bi_elliptic_transfer_delta_v(radius1, radius2, rb, mu),
            "intermediate_state": {
                "r": r_intermediate,
                "v_before": v_intermediate_before,
                "v_after": v_intermediate_after,
                "t": t0 + tof1,
            },
            "circularity_error": [error1 if check_circular else None, error2 if check_circular else None],
        },
    )

    if plot:
        from ..plots import transfer_plot

        fig = transfer_plot(
            r_depart,
            v_depart,
            r_transfer,
            v_transfer,
            r_arrive,
            v_arrive,
            show=False,
            save_path=save_path,
            title=(
                f"Bi-elliptic transfer via {rb / 1e3:.0f} km apoapsis\n"
                f"TOF {tof / 3600:.2f} h | Δv {result['delta_v_total'] / 1e3:.3f} km/s"
            ),
        )
        result["figure"] = fig

    return result
