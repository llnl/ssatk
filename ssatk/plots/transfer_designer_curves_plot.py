"""Mission-designer curves for an optimal-transfer search.

Left: the porkchop contour (departure time x time of flight -> objective
delta-v, log color scale) with infeasible candidates greyed out and the
chosen transfer starred.  Right: the delta-v versus time-of-flight trade
(Pareto front) broken out per burn -- total, departure burn, and arrival
burn -- with the chosen transfer and the delta-v budget line.

Takes the canonical dictionary returned by ``transfer_optimal``; all
curves are recreated from the stored search grid in
``result["diagnostics"]``, so this plot can be regenerated at any time
from the result dictionary alone.
"""

import numpy as np

from .plotutils import _pop_save_path_aliases, _raise_unrecognized_kwargs


def transfer_designer_curves_plot(result, title=None, save_path=None, **save_kwargs):
    """Plot porkchop + per-burn Pareto curves from a transfer_optimal
    result.

    Parameters
    ----------
    result : dict
        Canonical ``transfer_optimal`` result dictionary.
    save_path : str, optional
        If given, save via ``ssatk.plots.figsave`` and close;
        otherwise the figure is returned.
    """
    save_path, save_kwargs = _pop_save_path_aliases(save_kwargs, save_path=save_path)
    _raise_unrecognized_kwargs(save_kwargs, "transfer_designer_curves_plot")
    should_save = save_path is not None and save_path is not False

    import matplotlib
    if should_save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    if isinstance(result, dict):
        diagnostics = result.get("diagnostics", {})
        g = diagnostics["grid"]
        pareto = diagnostics["pareto"]
        t_depart = result["initial"]["t"]
        tof = result["tof"]
        dv_total = result["delta_v_total"]
        dv_budget = diagnostics.get("dv_budget")
        objective = diagnostics.get("objective", "min_dv")
        delta_v_mode = diagnostics.get("delta_v_mode", "total")
        rendezvous = diagnostics.get("rendezvous", True)
        arrival_burn = diagnostics.get("arrival_burn", len(result.get("burns", [])) > 1)
        arrival_mode = diagnostics.get("arrival_mode")
    else:
        g = result.grid
        pareto = result.pareto
        t_depart = result.t_depart
        tof = result.tof
        dv_total = result.dv_total
        dv_budget = getattr(result, "dv_budget", None)
        objective = result.objective
        delta_v_mode = getattr(result, "delta_v_mode", "total")
        rendezvous = result.rendezvous
        arrival_burn = result.arrival_burn
        arrival_mode = getattr(result, "arrival_mode", None)
    dep_h = (g["t_dep"] - g["t_dep"][0]) / 3600.0
    tof_h = g["tof"] / 3600.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.5))

    cost = np.ma.masked_invalid(g["cost"])
    pc = ax1.pcolormesh(dep_h, tof_h, cost.T, shading="nearest",
                        norm=LogNorm(vmin=max(cost.min(), 1e-1),
                                     vmax=cost.max()), cmap="viridis")
    fig.colorbar(pc, ax=ax1, label="objective delta-v [m/s]")
    ax1.plot((t_depart - g["t_dep"][0]) / 3600.0,
             tof / 3600.0, "r*", ms=16, mec="w",
             label=f"chosen: {dv_total:.1f} m/s")
    ax1.set_xlabel("departure time into window [h]")
    ax1.set_ylabel("time of flight [h]")
    ax1.set_title("Porkchop (grey = infeasible: no 0-rev solution,\n"
                  "perigee below margin, or burns don't fit)")
    ax1.set_facecolor("0.85")
    ax1.legend(loc="upper right", fontsize=9)

    p = pareto
    ax2.plot(tof_h, p["dv"], "k.-", lw=2, label="total (best per TOF)")
    if "dv1" in p:
        ax2.plot(tof_h, p["dv1"], "C0.--", lw=1.2,
                 label="burn 1 (departure)")
    if "dv2" in p and arrival_burn:
        ax2.plot(tof_h, p["dv2"], "C1.--", lw=1.2,
                 label="burn 2 (arrival)")
    ax2.plot(tof / 3600.0, dv_total, "r*", ms=16, mec="w",
             label="chosen transfer")
    if dv_budget is not None:
        ax2.axhline(dv_budget, color="k", ls="--", lw=1,
                    label=f"delta-v budget ({dv_budget:.0f} m/s)")
    ax2.set_yscale("log")
    ax2.set_xlabel("time of flight [h]")
    ax2.set_ylabel("delta-v [m/s]")
    ax2.set_title("Delta-v vs time-of-flight trade, per burn")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=8)

    mode = arrival_mode or ("rendezvous" if rendezvous else ("insertion" if arrival_burn else "inject"))
    burns = "both burns" if arrival_burn else "first burn only"
    fig.suptitle(title or f"transfer_optimal: {objective}, {mode}, {burns}, {delta_v_mode} dv",
                 fontsize=12)
    fig.tight_layout()

    if should_save:
        from ssatk.plots import figsave
        figsave(fig, save_path)
        plt.close(fig)
        return None
    return fig
