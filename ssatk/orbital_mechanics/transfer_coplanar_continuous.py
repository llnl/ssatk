import numpy as np
from astropy.time import Time
from scipy.integrate import solve_ivp

from ..constants import EARTH_MU
from ._transfer_result import maneuver_burn, trajectory_dict, transfer_result, transfer_state


def transfer_coplanar_continuous(r1=None,
                                 v1=None,
                                 r2=None,
                                 v2=None,
                                 *,
                                 initial=None,
                                 target=None,
                                 a_thrust=1,
                                 mu=EARTH_MU,
                                 t0=None,
                                 max_time=2 * 3600.0,
                                 tol=1.0,
                                 plot=False):
    """
    Continuous‐thrust, coplanar transfer: thrust always lies in the initial
    orbital plane (normal = r1×v1), steering to rendezvous r2 (and optionally v2).

    The steering is a heuristic proportional law on the thrust angle with the
    thrust always on at ``a_thrust``; the event requires ``|r - r2| <= tol`` and
    ``|v - v2| <= 1 cm/s`` at the same instant, which this law rarely achieves, so
    the function usually raises ``ValueError``. ``v2`` defaults to the circular
    velocity at r2, in the initial plane and sense of motion. The burn's
    ``delta_v_mag`` is the propellant cost a_thrust * t (the thrust never
    throttles), not the norm of the net velocity change.

    All units SI: r (m), v (m/s), a_thrust (m/s²), t (s), mu (m³/s²).
    """
    # Epoch
    if t0 is None:
        t0 = Time("2025-01-01", format="iso")

    if initial is not None:
        state1 = transfer_state(state=initial, mu=mu)
        r1, v1, t0 = state1["r"], state1["v"], state1["t"]
    if target is not None:
        state2 = transfer_state(state=target, mu=mu)
        r2, v2 = state2["r"], state2["v"]
    if r1 is None or v1 is None or r2 is None:
        raise ValueError("transfer_coplanar_continuous requires initial/target or r1/v1/r2")

    # Build orthonormal basis in initial plane
    h_vec = np.cross(r1, v1)
    h_hat = h_vec/np.linalg.norm(h_vec)

    # Default circular target if v2 omitted: circular speed at r2, perpendicular
    # to r2 in the initial plane (previously always along +y).
    r2 = np.asarray(r2, dtype=float)
    if v2 is None:
        v2 = np.sqrt(mu/np.linalg.norm(r2)) * np.cross(h_hat, r2/np.linalg.norm(r2))
    r_hat1 = r1/np.linalg.norm(r1)
    p_hat1 = np.cross(h_hat, r_hat1)

    # y = [r(3), v(3), dv1(3), theta]
    # theta controls thrust direction: d_hat = cosθ·r_hat1 + sinθ·p_hat1
    def equations(t, y):
        r = y[0:3]
        v = y[3:6]
        dv1 = y[6:9]
        theta = y[9]

        # gravity
        r_norm = np.linalg.norm(r)
        a_grav = -mu * r / r_norm**3

        # thrust direction in plane
        d_hat = np.cos(theta)*r_hat1 + np.sin(theta)*p_hat1
        a_th = a_thrust * d_hat

        # steering law: point thrust at error vector in plane
        err = r - r2
        er = np.dot(err, r_hat1)
        ep = np.dot(err, p_hat1)
        # simple P-control on angle
        theta_dot = -(ep * np.cos(theta) - er * np.sin(theta)) / (np.linalg.norm(err)+1e-6)

        # accumulate dv1
        dv1_dot = a_th

        return np.hstack((v, a_grav + a_th, dv1_dot, theta_dot))

    # event: position error hits tol (and optionally velocity match)
    def rendezvous_event(t, y):
        r = y[0:3]
        v = y[3:6]
        err = np.linalg.norm(r - r2) - tol
        if np.linalg.norm(v - v2) > 1e-2:
            err = max(err, np.linalg.norm(v - v2) - 1e-2)
        return err

    rendezvous_event.terminal = True
    rendezvous_event.direction = -1

    # initial state: no accumulated dv, initial thrust angle = 0
    y0 = np.hstack((r1, v1, np.zeros(3), 0.0))
    sol = solve_ivp(
        fun=equations,
        t_span=(0.0, max_time),
        y0=y0,
        events=rendezvous_event,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
        dense_output=True,
    )

    if sol.status != 1 or len(sol.t_events[0]) == 0:
        raise ValueError("Failed to rendezvous within max_time")

    t_final = sol.t_events[0][0]
    yf = sol.sol(t_final)
    r_final = yf[0:3]
    v_final = yf[3:6]
    net_dv = yf[6:9]
    # The thrust is always on at a_thrust, so the cost is a_thrust * t_final;
    # |net_dv| is smaller whenever the steering turns the thrust.
    total_dv1 = float(a_thrust) * float(t_final)
    net_norm = np.linalg.norm(net_dv)
    dv1_vec = total_dv1 * (net_dv / net_norm if net_norm > 0.0 else np.zeros(3))
    initial_state = transfer_state(state={"r": r1, "v": v1, "t": t0}, mu=mu)
    sample_times = np.linspace(0.0, t_final, 300)
    sampled = sol.sol(sample_times)
    r_traj = sampled[0:3].T
    v_traj = sampled[3:6].T
    t_abs = initial_state["t"] + sample_times
    result = transfer_result(
        method="transfer_coplanar_continuous",
        initial=initial_state,
        target={"r": r2, "v": v2, "t": initial_state["t"] + t_final},
        final={"r": r_final, "v": v_final, "t": initial_state["t"] + t_final},
        burns=[
            maneuver_burn(
                name="steered_coplanar_continuous_burn",
                kind="continuous_steered_coplanar_burn",
                state=initial_state,
                delta_v=dv1_vec,
                t_start=initial_state["t"],
                t_end=initial_state["t"] + t_final,
                notes="Acceleration direction is steered inside the initial orbital plane.",
            )
        ],
        trajectory=trajectory_dict(t=t_abs, r=r_traj, v=v_traj),
        tof=t_final,
        success=True,
        assumptions=["continuous steered acceleration in initial orbital plane", "two-body gravity"],
        diagnostics={"position_tolerance": tol, "a_thrust": a_thrust, "max_time": max_time, "net_delta_v": net_dv},
    )

    if plot:
        from ..plots import transfer_plot
        fig = transfer_plot(r1, v1, r_traj, v_traj, r2, v2,
                            title=(f"t={t_final/60:.1f} min "
                                   f"|Δv₁|={total_dv1:.3f} m/s"),
                            show=False)
        result["figure"] = fig

    return result
