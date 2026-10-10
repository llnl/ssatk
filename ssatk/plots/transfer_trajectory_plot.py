"""Plot a propagated transfer trajectory with annotated burns.

Draws the transfer arc, the departure/arrival orbits (reconstructed from
the trajectory's boundary states), the Earth, and a marker at each burn
location annotated with its strength: delta-v magnitude, duration, the
acceleration flown, and -- when an engine model was used -- the thrust
and propellant estimate.

Works with the canonical transfer dictionaries returned by
``transfer_ssapy`` and ``transfer_optimal``; the result must have been
produced with ``propagate=True`` so a trajectory exists.
"""

import numpy as np

from ssapy.orbit import Orbit
from ssapy.propagator import KeplerianPropagator
from ssapy.compute import rv
from ssapy.constants import EARTH_MU, EARTH_RADIUS

from .plotutils import _pop_save_path_aliases, _raise_unrecognized_kwargs


def _burn_get(b, name, default=None):
    return b.get(name, default) if isinstance(b, dict) else getattr(b, name, default)


def _burn_label(i, b):
    dv_mag = _burn_get(b, "delta_v_mag", _burn_get(b, "dv_mag", 0.0))
    t_start = _burn_get(b, "t_start", _burn_get(b, "t", 0.0))
    t_end = _burn_get(b, "t_end", t_start)
    duration = _burn_get(b, "duration", t_end - t_start)
    a = _burn_get(b, "acceleration_mag", None)
    if a is None:
        a = dv_mag / duration if duration else 0.0
    label = (f"burn {i}: {dv_mag:.1f} m/s\n"
             f"{a:.3f} m/s$^2$ x {duration:.0f} s")
    thrust = _burn_get(b, "thrust")
    propellant_mass = _burn_get(b, "propellant_mass")
    if thrust is not None:
        label += f"\nF = {thrust:.0f} N"
    if propellant_mass is not None:
        label += f", prop ~{propellant_mass:.1f} kg"
    return label


def _orbit_ring(r, v, n=361):
    orb = Orbit(np.asarray(r, float), np.asarray(v, float), t=0.0)
    period = 2 * np.pi * np.sqrt(abs(orb.a) ** 3 / EARTH_MU)
    rr, _ = rv(orb, np.linspace(0.0, period, n),
               propagator=KeplerianPropagator())
    return rr


def transfer_trajectory_plot(result, ax=None, three_d=False,
                             show_orbits=True, show_earth=True,
                             annotate_burns=True, title=None,
                             save_path=None, **save_kwargs):
    """Plot a transfer trajectory with burn locations and strengths.

    Parameters
    ----------
    result : dict
        A canonical propagated transfer dictionary (``propagate=True``).
    ax : matplotlib axes, optional
        Draw onto existing axes (e.g. a gallery panel); otherwise a new
        figure is created.  Must be a 3-D axes when ``three_d=True``.
    three_d : bool
        Render in 3-D (useful for plane-change transfers).
    show_orbits, show_earth, annotate_burns : bool
        Toggle the orbit rings, the Earth disk (2-D only), and the
        per-burn strength annotations.
    title : str, optional
        Axes title; a default with total delta-v and arrival error is
        used when omitted.
    save_path : str, optional
        If given, save the figure via ``ssatk.plots.figsave`` and
        close it; otherwise the axes are returned for further styling.

    Returns
    -------
    matplotlib axes (when ``save_path`` is None)
    """
    save_path, save_kwargs = _pop_save_path_aliases(save_kwargs, save_path=save_path)
    _raise_unrecognized_kwargs(save_kwargs, "transfer_trajectory_plot")
    should_save = save_path is not None and save_path is not False

    transfer = getattr(result, "transfer", result)
    trajectory = transfer.get("trajectory") if isinstance(transfer, dict) else transfer.trajectory
    if trajectory is None:
        raise ValueError("result has no trajectory; rerun the transfer "
                         "with propagate=True.")
    burns = transfer.get("burns") if isinstance(transfer, dict) else transfer.burns
    import matplotlib
    if should_save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = None
    if ax is None:
        if three_d:
            fig = plt.figure(figsize=(9, 8))
            ax = fig.add_subplot(projection="3d")
        else:
            fig, ax = plt.subplots(figsize=(7.5, 7.5))
    else:
        fig = ax.get_figure()

    tt = trajectory["t"]
    tr = trajectory["r"] / 1e3
    tv = trajectory["v"]

    def plot(xyz, *a, **kw):
        cols = (xyz[:, 0], xyz[:, 1], xyz[:, 2]) if three_d \
            else (xyz[:, 0], xyz[:, 1])
        return ax.plot(*cols, *a, **kw)

    if show_orbits:
        plot(_orbit_ring(tr[0] * 1e3, tv[0]) / 1e3, "C0--", lw=0.9,
             label="departure orbit")
        plot(_orbit_ring(tr[-1] * 1e3, tv[-1]) / 1e3, "C2--", lw=0.9,
             label="arrival orbit")
    if show_earth and not three_d:
        ang = np.linspace(0, 2 * np.pi, 181)
        ax.fill(EARTH_RADIUS / 1e3 * np.cos(ang),
                EARTH_RADIUS / 1e3 * np.sin(ang), color="0.85")
    plot(tr, "C3-", lw=2, label="transfer")

    for i, b in enumerate(burns, 1):
        # Burn location: trajectory position at the burn start.
        burn_time = _burn_get(b, "t_start", _burn_get(b, "t", tt[0]))
        rb = np.array([np.interp(burn_time, tt, tr[:, k])
                       for k in range(3)])
        if three_d:
            ax.scatter(*rb, color="k", marker="*", s=120, zorder=5)
            if annotate_burns:
                ax.text(*rb, "  " + _burn_label(i, b), fontsize=8)
        else:
            ax.plot(rb[0], rb[1], "k*", ms=13, zorder=5)
            if annotate_burns:
                ax.annotate(_burn_label(i, b), rb[:2],
                            textcoords="offset points", xytext=(10, 8),
                            fontsize=8)

    if not three_d:
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    if three_d:
        ax.set_zlabel("z [km]")
        lim = np.max(np.abs(tr)) * 1.1
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
    if title is None:
        dv_total = transfer.get("delta_v_total") if isinstance(transfer, dict) else transfer.dv_total
        diagnostics = transfer.get("diagnostics", {}) if isinstance(transfer, dict) else {}
        arrival_error = diagnostics.get("arrival_error", getattr(transfer, "arrival_error", None))
        title = f"dv {dv_total:.1f} m/s"
        if arrival_error is not None:
            title += f" | arrival err {arrival_error:.1f} m"
    ax.set_title(title, fontsize=10)

    if should_save:
        from ssatk.plots import figsave
        ax.legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        figsave(fig, save_path)
        plt.close(fig)
        return None
    return ax
