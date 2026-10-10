import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from ..plots import set_axes_equal, figsave
from ..plots.plotutils import _pop_save_path_aliases, _raise_unrecognized_kwargs
from ..constants import EARTH_MU, EARTH_RADIUS
from ._two_body import _keplerian_two_body_rhs
from ._transfer_result import maneuver_burn, trajectory_dict, transfer_result, transfer_state


def transfer_inclination_continuous(
    r0=None,
    v0=None,
    *,
    initial=None,
    i_target=None,
    delta_v=None,
    a_thrust=1,
    mu=EARTH_MU,
    t0=0.0,
    max_time=36000,
    body_radius=EARTH_RADIUS,
    plot=False,
    save_path=False,
    **save_kwargs,
):
    save_path, save_kwargs = _pop_save_path_aliases(save_kwargs, save_path=save_path)
    _raise_unrecognized_kwargs(save_kwargs, "transfer_inclination_continuous")
    if initial is not None:
        state0 = transfer_state(state=initial, mu=mu)
        r0, v0, t0 = state0["r"], state0["v"], state0["t"]
    if r0 is None or v0 is None:
        raise ValueError("transfer_inclination_continuous requires initial or r0/v0")

    if (i_target is None) == (delta_v is None):
        raise ValueError("Specify exactly one of i_target or delta_v.")

    def equations(t, y):
        r, v = y[:3], y[3:]
        r_norm = np.linalg.norm(r)
        a_grav = -mu * r / r_norm**3
        h = np.cross(r, v)

        if np.linalg.norm(h) > 0:
            n_vec = h / np.linalg.norm(h)
            if delta_v is not None and delta_v < 0:
                n_vec = -n_vec
        else:
            n_vec = np.zeros(3)

        a = a_grav + a_thrust * n_vec
        return np.concatenate((v, a))

    def inclination_event(t, y):
        r, v = y[:3], y[3:]
        h = np.cross(r, v)
        h_norm = np.linalg.norm(h)
        if h_norm == 0:
            return np.inf
        inclination = np.arccos(np.clip(h[2] / h_norm, -1.0, 1.0))
        if h[1] < 0:
            inclination = -inclination
        return inclination - i_target

    inclination_event.terminal = True
    inclination_event.direction = 1

    def deltav_event(t, y):
        return a_thrust * (t - t0) - abs(delta_v)

    deltav_event.terminal = True
    deltav_event.direction = 1

    y0 = np.concatenate((r0, v0))
    event_fn = inclination_event if i_target is not None else deltav_event

    sol = solve_ivp(
        equations,
        (t0, t0 + max_time),
        y0,
        events=event_fn,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
        dense_output=True,
        max_step=10.0,
    )

    if sol.status != 1 or len(sol.t_events) == 0 or sol.t_events[0].size == 0:
        raise ValueError("Condition not reached within max_time.")

    # Extract full trajectory at the solver points
    r_traj = sol.y[:3].T  # shape (n,3)
    v_traj = sol.y[3:].T  # shape (n,3)
    t_traj = sol.t        # shape (n,)

    if plot:
        _plot_transfer(
            sol,
            r0,
            v0,
            sol.y[:3, -1],
            sol.y[3:, -1],
            t0,
            sol.t[-1],
            mu,
            body_radius,
            save_path,
        )

    duration = float(t_traj[-1] - t0)
    h0 = np.cross(r0, v0)
    h_norm = np.linalg.norm(h0)
    normal = h0 / h_norm if h_norm > 0.0 else np.array([0.0, 0.0, 1.0])
    sign = -1.0 if delta_v is not None and delta_v < 0 else 1.0
    integrated_delta_v = abs(delta_v) if delta_v is not None else abs(a_thrust) * duration
    burn = maneuver_burn(
        name="normal_continuous_burn",
        kind="continuous_orbit-normal_burn",
        state={"r": r0, "v": v0, "t": t0},
        delta_v=sign * integrated_delta_v * normal,
        t_start=t0,
        t_end=t_traj[-1],
        notes="Acceleration follows the instantaneous angular-momentum direction; delta_v is the integrated-thrust equivalent.",
    )
    return transfer_result(
        method="transfer_inclination_continuous",
        initial={"r": r0, "v": v0, "t": t0},
        target={"r": r_traj[-1], "v": v_traj[-1], "t": t_traj[-1]},
        final={"r": r_traj[-1], "v": v_traj[-1], "t": t_traj[-1]},
        burns=[burn],
        trajectory=trajectory_dict(t=t_traj, r=r_traj, v=v_traj),
        tof=duration,
        assumptions=["continuous thrust along instantaneous orbit normal", "two-body gravity"],
        diagnostics={"i_target": i_target, "delta_v_target": delta_v, "a_thrust": a_thrust},
    )


def _plot_transfer(
    sol, r0, v0, r_final, v_final, t0, t_final, mu, body_radius, save_path
):
    r0_mag = np.linalg.norm(r0)
    v0_mag = np.linalg.norm(v0)
    a0 = 1 / (2 / r0_mag - v0_mag**2 / mu)
    T0 = 2 * np.pi * np.sqrt(abs(a0) ** 3 / mu)

    rf_mag = np.linalg.norm(r_final)
    vf_mag = np.linalg.norm(v_final)
    af = 1 / (2 / rf_mag - vf_mag**2 / mu)
    Tf = 2 * np.pi * np.sqrt(abs(af) ** 3 / mu)

    sol_initial = solve_ivp(
        _keplerian_two_body_rhs,
        (t0, t0 + T0),
        np.concatenate((r0, v0)),
        t_eval=np.linspace(t0, t0 + T0, 200),
        args=(mu,),
        rtol=1e-8,
        atol=1e-10,
    )

    sol_final = solve_ivp(
        _keplerian_two_body_rhs,
        (t_final, t_final + Tf),
        np.concatenate((r_final, v_final)),
        t_eval=np.linspace(t_final, t_final + Tf, 200),
        args=(mu,),
        rtol=1e-8,
        atol=1e-10,
    )

    t_burn = np.linspace(t0, t_final, 300)
    r_burn = sol.sol(t_burn)[:3].T

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = body_radius * np.outer(np.cos(u), np.sin(v))
    y = body_radius * np.outer(np.sin(u), np.sin(v))
    z = body_radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color="blue", alpha=0.3)

    ax.plot(*sol_initial.y[:3], "g--", label="Initial Orbit")
    ax.plot(*sol_final.y[:3], "b--", label="Final Orbit")
    ax.plot(r_burn[:, 0], r_burn[:, 1], r_burn[:, 2], "r-", label="Burn Trajectory")
    ax.scatter(*r0, color="green", s=50, label="Start")
    ax.scatter(*r_final, color="blue", s=50, label="End")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Inclination Transfer with Normal Thrust")
    ax.legend()
    ax.set_box_aspect([1, 1, 1])
    set_axes_equal(ax)
    plt.show()

    if save_path:
        figsave(fig, save_path)
