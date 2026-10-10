"""Plot the burn timeline of a transfer: where each burn occurs in time
and how strong it is.

Top panel: commanded acceleration magnitude versus time -- a step
profile that is ``|dv|/duration`` inside each burn window and zero while
coasting -- with each burn block annotated by its delta-v, duration,
acceleration, and (when an engine model was used) thrust and propellant
estimate.  Bottom panel: cumulative delta-v expended along the transfer.

Works with the canonical transfer dictionaries returned by
``transfer_ssapy`` and ``transfer_optimal``.
"""

from .plotutils import _pop_save_path_aliases, _raise_unrecognized_kwargs


def _burn_get(burn, name, default=None):
    return burn.get(name, default) if isinstance(burn, dict) else getattr(burn, name, default)


def transfer_burn_profile_plot(result, title=None, save_path=None, **save_kwargs):
    """Plot acceleration-vs-time and cumulative delta-v for all burns.

    Parameters
    ----------
    result : dict
        Canonical transfer dictionary.
    title : str, optional
    save_path : str, optional
        If given, save via ``ssatk.plots.figsave`` and close;
        otherwise the figure is returned.
    """
    save_path, save_kwargs = _pop_save_path_aliases(save_kwargs, save_path=save_path)
    _raise_unrecognized_kwargs(save_kwargs, "transfer_burn_profile_plot")

    transfer = getattr(result, "transfer", result)
    burns = transfer.get("burns") if isinstance(transfer, dict) else transfer.burns
    should_save = save_path is not None and save_path is not False
    trajectory = transfer.get("trajectory") if isinstance(transfer, dict) else transfer.trajectory
    if trajectory is not None:
        t0 = float(trajectory["t"][0])
        t1 = float(trajectory["t"][-1])
    else:
        t0 = _burn_get(burns[0], "t_start", _burn_get(burns[0], "t"))
        t1 = _burn_get(burns[-1], "t_end", _burn_get(burns[-1], "t"))

    import matplotlib
    if should_save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6.5), sharex=True,
        gridspec_kw=dict(height_ratios=[2, 1]))

    th = lambda t: (t - t0) / 3600.0
    a_max = 0.0
    for i, b in enumerate(burns, 1):
        t_start = _burn_get(b, "t_start", _burn_get(b, "t", t0))
        t_end = _burn_get(b, "t_end", t_start)
        dur = _burn_get(b, "duration", t_end - t_start)
        dv_mag = _burn_get(b, "delta_v_mag", _burn_get(b, "dv_mag", 0.0))
        a = _burn_get(b, "acceleration_mag", None)
        if a is None:
            a = dv_mag / dur if dur else 0.0
        a_max = max(a_max, a)
        ax1.fill_between([th(t_start), th(t_end)], 0, a,
                         color=f"C{i - 1}", alpha=0.75, step="pre")
        label = (f"burn {i}: {dv_mag:.1f} m/s\n"
                 f"{a:.3f} m/s$^2$ x {dur:.0f} s")
        thrust = _burn_get(b, "thrust")
        propellant_mass = _burn_get(b, "propellant_mass")
        if thrust is not None:
            label += f"\nF = {thrust:.0f} N"
        if propellant_mass is not None:
            label += f"\nprop ~{propellant_mass:.1f} kg"
        ax1.annotate(label,
                     (th(0.5 * (t_start + t_end)), a),
                     textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=8)
    ax1.set_ylim(0, a_max * 1.45 if a_max > 0 else 1)
    ax1.set_xlim(th(t0), th(t1))
    ax1.set_ylabel("commanded acceleration [m/s$^2$]")
    ax1.grid(alpha=0.3)
    dv_total = transfer.get("delta_v_total") if isinstance(transfer, dict) else transfer.dv_total
    default_title = (
        f"Burn timeline: total dv {dv_total:.1f} m/s "
        f"across {len(burns)} burn(s)"
    )
    ax1.set_title(title or default_title)

    # Cumulative delta-v: piecewise-linear ramps inside burn windows.
    ts = [t0]
    dvs = [0.0]
    total = 0.0
    for b in burns:
        t_start = _burn_get(b, "t_start", _burn_get(b, "t", t0))
        t_end = _burn_get(b, "t_end", t_start)
        dv_mag = _burn_get(b, "delta_v_mag", _burn_get(b, "dv_mag", 0.0))
        ts += [t_start, t_end]
        dvs += [total, total + dv_mag]
        total += dv_mag
    ts.append(t1)
    dvs.append(total)
    ax2.plot([th(t) for t in ts], dvs, "C3-", lw=2)
    ax2.set_xlabel("time since departure [h]")
    ax2.set_ylabel("cumulative dv [m/s]")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    if should_save:
        from ssatk.plots import figsave
        figsave(fig, save_path)
        plt.close(fig)
        return None
    return fig
