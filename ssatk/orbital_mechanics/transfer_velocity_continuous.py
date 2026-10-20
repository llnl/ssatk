import numpy as np
from scipy.integrate import solve_ivp
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ..accelerations_orbit import accel_velocity
from ..plots import set_axes_equal, figsave
from ..plots.plotutils import _pop_save_path_aliases, _raise_unrecognized_kwargs
from ..constants import EARTH_MU, EARTH_RADIUS
from ._transfer_result import maneuver_burn, trajectory_dict, transfer_result, transfer_state


def transfer_velocity_continuous(
    r0=None,
    v0=None,
    *,
    initial=None,
    v_target=None,  # Target delta_v in m/s, optional
    a_thrust=1.0,
    mu=EARTH_MU,
    t0=0.0,
    max_time=36000,
    body_radius=EARTH_RADIUS,
    plot=False,
    save_path=False,
    **save_kwargs,
):
    """
    Burn continuously in velocity direction until a specific accumulated delta_v magnitude is reached.
    Accumulate delta_v as the integral of absolute thrust acceleration (always positive).
    If v_target is None, burn for full max_time.

    Returns:
    - r: (n,3) array of positions during burn
    - v: (n,3) array of velocities during burn
    - t: (n,) array of time points during burn
    """
    save_path, save_kwargs = _pop_save_path_aliases(save_kwargs, save_path=save_path)
    _raise_unrecognized_kwargs(save_kwargs, "transfer_velocity_continuous")
    if initial is not None:
        state0 = transfer_state(state=initial, mu=mu)
        r0, v0, t0 = state0["r"], state0["v"], state0["t"]
    if r0 is None or v0 is None:
        raise ValueError("transfer_velocity_continuous requires initial or r0/v0")

    def equations(t, y):
        r = y[:3]
        v = y[3:6]
        dv = y[6]  # accumulated delta-v (positive scalar)

        r_norm = np.linalg.norm(r)
        a_grav = -mu * r / r_norm**3

        # Thrust direction: sign from v_target (negative means slow down)
        thrust_sign = -1.0 if v_target is not None and v_target < 0 else 1.0
        a_thrust_vec = accel_velocity(v, thrust_sign * np.abs(a_thrust))

        a_total = a_grav + a_thrust_vec
        dv_dot = np.linalg.norm(a_thrust_vec)  # always positive, magnitude of thrust accel

        return np.hstack((v, a_total, dv_dot))


    if v_target is not None:
        def delta_v_event(t, y):
            return y[6] - abs(v_target)  # trigger on positive accumulated delta-v reaching abs(target)
        delta_v_event.terminal = True
        delta_v_event.direction = 1
        events = [delta_v_event]
    else:
        events = None

    y0 = np.hstack((r0, v0, 0.0))
    sol = solve_ivp(
        equations,
        (t0, t0 + max_time),
        y0,
        events=events,
        rtol=1e-8,
        atol=1e-10,
        max_step=1.0,
        dense_output=True,
    )

    if v_target is not None:
        if sol.status != 1 or sol.t_events[0].size == 0:
            raise ValueError("Target delta-v not reached within max_time")
        t_final = sol.t_events[0][0]
    else:
        t_final = sol.t[-1]

    # Extract full solution arrays up to t_final
    # Interpolate dense output at fine resolution
    t_vals = np.linspace(t0, t_final, 1000)
    y_vals = sol.sol(t_vals)
    r_vals = y_vals[:3].T
    v_vals = y_vals[3:6].T

    r_final = r_vals[-1]
    v_final = v_vals[-1]

    energy = 0.5 * np.dot(v_final, v_final) - mu / np.linalg.norm(r_final)
    if energy > 0.0:
        warnings.warn("Final orbit is unbound (specific energy > 0)", RuntimeWarning)

    if plot:
        _plot_transfer(sol, r0, v0, r_final, v_final, t0, t_final, mu, body_radius, save_path)

    thrust_sign = -1.0 if v_target is not None and v_target < 0 else 1.0
    duration = float(t_final - t0)
    integrated_delta_v = abs(v_target) if v_target is not None else abs(a_thrust) * duration
    delta_v_vec = thrust_sign * integrated_delta_v * np.asarray(v0, dtype=float) / np.linalg.norm(v0)
    burn = maneuver_burn(
        name="velocity_aligned_continuous_burn",
        kind="continuous_velocity_aligned_burn",
        state={"r": r0, "v": v0, "t": t0},
        delta_v=delta_v_vec,
        t_start=t0,
        t_end=t_final,
        notes="Acceleration follows the instantaneous velocity direction; delta_v is the integrated-thrust equivalent.",
    )
    return transfer_result(
        method="transfer_velocity_continuous",
        initial={"r": r0, "v": v0, "t": t0},
        target={"r": r_final, "v": v_final, "t": t_final},
        final={"r": r_final, "v": v_final, "t": t_final},
        burns=[burn],
        trajectory=trajectory_dict(t=t_vals, r=r_vals, v=v_vals),
        tof=duration,
        assumptions=["continuous thrust along instantaneous velocity", "two-body gravity"],
        diagnostics={"v_target": v_target, "a_thrust": a_thrust, "specific_energy_final": energy},
    )


def _plot_transfer(sol, r0, v0, r_final, v_final, t0, t_final, mu, body_radius, save_path):
    def orbital_period(r, v, mu):
        r_norm = np.linalg.norm(r)
        v_norm = np.linalg.norm(v)
        energy = 0.5 * v_norm**2 - mu / r_norm
        a = -mu / (2 * energy)
        period = 2 * np.pi * np.sqrt(a**3 / mu)
        return period

    period_initial = orbital_period(r0, v0, mu)
    period_final = orbital_period(r_final, v_final, mu)

    t_vals_burn = np.linspace(t0, t_final, 500)
    traj_burn = sol.sol(t_vals_burn)
    positions_burn = traj_burn[:3].T

    def gravity_only(t, y):
        r = y[:3]
        v = y[3:]
        r_norm = np.linalg.norm(r)
        a_grav = -mu * r / r_norm**3
        return np.hstack((v, a_grav))

    y_final = np.hstack((r_final, v_final))
    sol_post = solve_ivp(
        gravity_only,
        (t_final, t_final + period_final),
        y_final,
        rtol=1e-8,
        atol=1e-10,
        dense_output=True,
    )
    t_vals_post = np.linspace(t_final, t_final + period_final, 500)
    positions_post = sol_post.sol(t_vals_post)[:3].T

    y_initial = np.hstack((r0, v0))
    sol_noburn = solve_ivp(
        gravity_only,
        (t0, t0 + period_initial),
        y_initial,
        rtol=1e-8,
        atol=1e-10,
        dense_output=True,
    )
    t_vals_noburn = np.linspace(t0, t0 + period_initial, 500)
    positions_noburn = sol_noburn.sol(t_vals_noburn)[:3].T

    fig = plt.figure(figsize=(12, 9), dpi=120)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        positions_burn[:, 0],
        positions_burn[:, 1],
        positions_burn[:, 2],
        label="Burn phase",
        color="#d62728",
        linewidth=2.5,
    )
    ax.plot(
        positions_post[:, 0],
        positions_post[:, 1],
        positions_post[:, 2],
        linestyle="--",
        label="Coasting phase",
        color="#1f77b4",
        linewidth=2,
    )
    ax.plot(
        positions_noburn[:, 0],
        positions_noburn[:, 1],
        positions_noburn[:, 2],
        linestyle=":",
        color="gray",
        label="Original orbit",
        linewidth=1.8,
    )

    ax.scatter(*r0, color="green", s=80, label="Start", edgecolors='k', zorder=5)
    ax.scatter(*r_final, color="red", s=80, label="Burn end", edgecolors='k', zorder=5)

    u = np.linspace(0, 2 * np.pi, 150)
    v = np.linspace(0, np.pi, 150)
    x = body_radius * np.outer(np.cos(u), np.sin(v))
    y = body_radius * np.outer(np.sin(u), np.sin(v))
    z = body_radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(
        x,
        y,
        z,
        color="lightblue",
        alpha=0.25,
        edgecolor="none",
        linewidth=0,
        zorder=0,
    )

    set_axes_equal(ax)

    ax.set_title("Orbit Transfer: Burn, Coast, and Original Orbit (Full Periods)", fontsize=16, fontweight='semibold')
    ax.set_xlabel("X [m]", fontsize=14)
    ax.set_ylabel("Y [m]", fontsize=14)
    ax.set_zlabel("Z [m]", fontsize=14)

    leg = ax.legend(frameon=True, fontsize=12, loc='best')
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_alpha(0.9)

    plt.tight_layout()
    plt.show()

    if save_path:
        figsave(fig, save_path)
