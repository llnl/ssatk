import numpy as np
from scipy.integrate import solve_ivp
import warnings
import matplotlib.pyplot as plt
from ..accelerations_orbit import accel_velocity
from ..plots.set_axes_equal import set_axes_equal
from ..constants import EARTH_MU, EARTH_RADIUS
from ._two_body import _keplerian_two_body_rhs
from ._transfer_result import maneuver_burn, trajectory_dict, transfer_result, transfer_state


def transfer_velocity_and_inclination_continuous(
    r0=None,
    v0=None,
    *,
    initial=None,
    i_target=None,
    a_thrust=1.0,
    mu=EARTH_MU,
    t0=0.0,
    max_time1=36000,
    max_time2=36000,
    body_radius=EARTH_RADIUS,
    plot=False
):
    """
    Two continuous burns: along the velocity for ``max_time1`` seconds, then
    normal to the orbit until the inclination reaches ``i_target`` (rad).

    ``max_time1`` is the duration of the velocity-aligned phase, not an upper
    bound. In the normal phase the thrust sign follows cos(u), the argument of
    latitude, so by Gauss's equation di/dt = r cos(u) a_W / h the inclination
    changes monotonically toward ``i_target`` (raising or lowering it).
    ``max_time2`` bounds that phase.
    """
    if initial is not None:
        state0 = transfer_state(state=initial, mu=mu)
        r0, v0, t0 = state0["r"], state0["v"], state0["t"]
    if r0 is None or v0 is None:
        raise ValueError("transfer_velocity_and_inclination_continuous requires initial or r0/v0")
    if i_target is None:
        raise ValueError("i_target is required")

    def equations_velocity_burn(t, y):
        r = y[:3]
        v = y[3:]
        r_norm = np.linalg.norm(r)
        a_grav = -mu * r / r_norm**3
        a_thrust_vec = accel_velocity(v, a_thrust)
        return np.hstack((v, a_grav + a_thrust_vec))

    def equations_normal_burn(t, y):
        r = y[:3]
        v = y[3:]
        r_norm = np.linalg.norm(r)
        a_grav = -mu * r / r_norm**3
        h = np.cross(r, v)
        h_norm = np.linalg.norm(h)
        n_vec = h / h_norm if h_norm > 0.0 else np.zeros(3)
        # Gauss: di/dt = r cos(u) a_W / h. Flip the normal thrust with cos(u) so
        # the inclination moves monotonically toward the target instead of
        # oscillating about its initial value.
        node = np.cross([0.0, 0.0, 1.0], h)
        node_norm = np.linalg.norm(node)
        cos_u = np.dot(node, r) / (node_norm * r_norm) if node_norm > 0.0 else 1.0
        steer = np.sign(cos_u) * direction if cos_u != 0.0 else direction
        return np.hstack((v, a_grav + steer * a_thrust * n_vec))

    def inclination_event(t, y):
        r = y[:3]
        v = y[3:]
        h = np.cross(r, v)
        h_norm = np.linalg.norm(h)
        if h_norm == 0.0:
            return -i_target
        inc = np.arccos(np.clip(h[2] / h_norm, -1.0, 1.0))
        return inc - i_target

    inclination_event.terminal = True

    y0 = np.hstack((r0, v0))
    sol1 = solve_ivp(
        fun=equations_velocity_burn,
        t_span=(t0, t0 + max_time1),
        y0=y0,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
        dense_output=True,
    )

    y1 = sol1.y[:, -1]
    t1 = sol1.t[-1]

    h1_vec = np.cross(y1[:3], y1[3:])
    i_start = np.arccos(np.clip(h1_vec[2] / np.linalg.norm(h1_vec), -1.0, 1.0))
    direction = 1.0 if i_target >= i_start else -1.0
    inclination_event.direction = direction

    sol2 = solve_ivp(
        fun=equations_normal_burn,
        t_span=(t1, t1 + max_time2),
        y0=y1,
        events=inclination_event,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
        dense_output=True,
    )

    if sol2.status != 1 or sol2.t_events[0].size == 0:
        raise ValueError("Target inclination not reached within max_time2")

    t_final = sol2.t_events[0][0]
    y_final = sol2.sol(t_final)
    r_final, v_final = y_final[:3], y_final[3:]

    energy = 0.5 * np.dot(v_final, v_final) - mu / np.linalg.norm(r_final)
    if energy > 0.0:
        warnings.warn("Final orbit is unbound (specific energy > 0)", RuntimeWarning)

    # Create concatenated full state vectors and time arrays for output
    # Sample sol1 and sol2 up to event time for smooth plotting and output
    t1_full = sol1.t
    r1_full = sol1.y[:3].T
    v1_full = sol1.y[3:].T

    # For sol2, sample from start to event time (t_final)
    t2_full = np.linspace(sol2.t[0], t_final, 300)
    y2_full = sol2.sol(t2_full).T
    r2_full = y2_full[:, :3]
    v2_full = y2_full[:, 3:]

    # Concatenate phase 1 and phase 2 trajectories (excluding overlap at t1)
    t_full = np.hstack((t1_full, t2_full[1:]))
    r_full = np.vstack((r1_full, r2_full[1:]))
    v_full = np.vstack((v1_full, v2_full[1:]))

    if plot:
        _plot_transfer(r0, v0, r_full, v_full, t_full, t0, t_final, mu, body_radius)

    duration1 = float(t1 - t0)
    duration2 = float(t_final - t1)
    vhat0 = np.asarray(v0, dtype=float) / np.linalg.norm(v0)
    h1 = np.cross(y1[:3], y1[3:])
    normal1 = h1 / np.linalg.norm(h1) if np.linalg.norm(h1) > 0.0 else np.array([0.0, 0.0, 1.0])
    burns = [
        maneuver_burn(
            name="velocity_aligned_phase",
            kind="continuous_velocity_aligned_burn",
            state={"r": r0, "v": v0, "t": t0},
            delta_v=a_thrust * duration1 * vhat0,
            t_start=t0,
            t_end=t1,
            notes="Phase 1 acceleration follows instantaneous velocity; delta_v is the integrated-thrust equivalent.",
        ),
        maneuver_burn(
            name="normal_phase",
            kind="continuous_orbit-normal_burn",
            state={"r": y1[:3], "v": y1[3:], "t": t1},
            delta_v=a_thrust * duration2 * normal1,
            t_start=t1,
            t_end=t_final,
            notes="Phase 2 acceleration follows instantaneous angular momentum; delta_v is the integrated-thrust equivalent.",
        ),
    ]
    return transfer_result(
        method="transfer_velocity_and_inclination_continuous",
        initial={"r": r0, "v": v0, "t": t0},
        target={"r": r_full[-1], "v": v_full[-1], "t": t_final},
        final={"r": r_full[-1], "v": v_full[-1], "t": t_final},
        burns=burns,
        trajectory=trajectory_dict(t=t_full, r=r_full, v=v_full),
        tof=t_final - t0,
        assumptions=["two sequential continuous burns", "two-body gravity"],
        diagnostics={"i_target": i_target, "a_thrust": a_thrust, "phase_split_time": t1},
    )


def _plot_transfer(r0, v0, r_full, v_full, t_full, t0, t_final, mu, body_radius):
    # Compute initial orbit parameters for plotting
    r0_mag = np.linalg.norm(r0)
    v0_mag = np.linalg.norm(v0)
    a0 = 1 / (2 / r0_mag - v0_mag**2 / mu)
    T0 = 2 * np.pi * np.sqrt(abs(a0)**3 / mu)

    # Compute final orbit parameters for plotting
    rf = r_full[-1]
    vf = v_full[-1]
    rf_mag = np.linalg.norm(rf)
    vf_mag = np.linalg.norm(vf)
    af = 1 / (2 / rf_mag - vf_mag**2 / mu)
    Tf = 2 * np.pi * np.sqrt(abs(af)**3 / mu)

    # Propagate initial orbit for one period
    sol_initial = solve_ivp(
        _keplerian_two_body_rhs,
        (t0, t0 + T0),
        np.concatenate((r0, v0)),
        t_eval=np.linspace(t0, t0 + T0, 200),
        args=(mu,),
        rtol=1e-8,
        atol=1e-10,
    )

    # Propagate final orbit for one period
    sol_final = solve_ivp(
        _keplerian_two_body_rhs,
        (t_final, t_final + Tf),
        np.concatenate((rf, vf)),
        t_eval=np.linspace(t_final, t_final + Tf, 200),
        args=(mu,),
        rtol=1e-8,
        atol=1e-10,
    )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Earth
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = body_radius * np.outer(np.cos(u), np.sin(v))
    y = body_radius * np.outer(np.sin(u), np.sin(v))
    z = body_radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color='blue', alpha=0.3)

    # Plot initial and final orbits
    ax.plot(sol_initial.y[0], sol_initial.y[1], sol_initial.y[2], 'g--', label='Initial Orbit')
    ax.plot(sol_final.y[0], sol_final.y[1], sol_final.y[2], 'b--', label='Final Orbit')

    # Plot transfer trajectory
    ax.plot(r_full[:, 0], r_full[:, 1], r_full[:, 2], 'r-', label='Transfer Trajectory')

    # Mark start and end points
    ax.scatter(*r0, color='green', s=50, label='Start')
    ax.scatter(*r_full[-1], color='blue', s=50, label='End')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Velocity Burn + Inclination Transfer')
    ax.legend()
    ax.set_box_aspect([1, 1, 1])
    set_axes_equal(ax)
    plt.show()
