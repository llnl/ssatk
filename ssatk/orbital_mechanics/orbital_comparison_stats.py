import numpy as np
from ..constants import EARTH_MU


def orbit_stats_dashboard(
    r_list,
    v_list=None,
    t_list=None,
    *,
    mu=EARTH_MU,
    reference=0,
    baseline="nominal",        # "nominal" | "mean" | "median"
    mode="population",         # "population" | "benchmark"
    resample="intersection",   # "intersection" | "union" | "ref" | None
    n_resample=2000,
    eps=1e-15,
    percentiles=(5, 25, 50, 75, 95),
    make_plots=False,
    plot_title="Orbit dashboard",
    time_unit="s",
    r_unit="m",
    v_unit="m/s",
    labels=None,
    show_legend=True,
    hist_bins=60,
    envelope_on_log_threshold=1e3,
):
    """
    Orbit ensemble dashboard with two modes.

    ``mode="population"`` shows position and velocity envelopes, split
    histograms, and an RTN stack. ``mode="benchmark"`` shows the corresponding
    per-orbit time series. When enabled, legends appear only in upper-left
    panels.
    """
    r_list, v_list, t_list = _normalize_inputs(r_list, v_list, t_list)

    M = len(r_list)
    if M < 2:
        raise ValueError("Need at least two orbits to analyze.")
    if labels is not None and len(labels) != M:
        raise ValueError(f"labels must have length M={M}, got {len(labels)}.")
    if baseline not in ("nominal", "mean", "median"):
        raise ValueError("baseline must be one of: 'nominal', 'mean', 'median'.")
    if mode not in ("population", "benchmark"):
        raise ValueError("mode must be one of: 'population', 'benchmark'.")
    if reference < 0 or reference >= M:
        raise ValueError("reference index out of range.")

    # Benchmark mode is always "vs nominal"
    if mode == "benchmark":
        baseline = "nominal"

    # Align to a common time grid
    t_grid, R, V = _align_all_to_grid(
        t_list=t_list,
        r_list=r_list,
        v_list=v_list,
        reference=int(reference),
        resample=resample,
        n_resample=n_resample,
    )

    # Baselines
    r_mean = np.nanmean(R, axis=0)
    v_mean = np.nanmean(V, axis=0) if V is not None else None

    r_med = np.nanmedian(R, axis=0)
    v_med = np.nanmedian(V, axis=0) if V is not None else None

    r_nom = R[int(reference)]
    v_nom = V[int(reference)] if V is not None else None

    if baseline == "nominal":
        r_base = r_nom
        v_base = v_nom
        base_label = f"nominal orbit (index {int(reference)})"
        pop_mask = (np.arange(M) != int(reference))
    elif baseline == "mean":
        r_base = r_mean
        v_base = v_mean
        base_label = "ensemble mean"
        pop_mask = np.ones((M,), dtype=bool)
    else:  # "median"
        r_base = r_med
        v_base = v_med
        base_label = "ensemble median"
        pop_mask = np.ones((M,), dtype=bool)

    # Spread from baseline
    dr = R - r_base[None, :, :]
    sep = np.linalg.norm(dr, axis=2)  # (M,N)

    if V is not None and v_base is not None:
        dv = V - v_base[None, :, :]
        vsep = np.linalg.norm(dv, axis=2)  # (M,N)
    else:
        vsep = None

    # Envelopes (population mode only)
    env_sep = _envelope_over_orbits(sep[pop_mask], percentiles=percentiles) if mode == "population" else None
    env_vsep = _envelope_over_orbits(vsep[pop_mask], percentiles=percentiles) if (mode == "population" and vsep is not None) else None

    # Per-orbit scalars
    per_orbit = {
        "sep_max": _nanmax_per_row(sep),
        "sep_final": _nanfinal_per_row(sep),
        "sep_rms": _nanrms_per_row(sep),
    }
    if vsep is not None:
        per_orbit.update(
            {
                "vsep_max": _nanmax_per_row(vsep),
                "vsep_final": _nanfinal_per_row(vsep),
                "vsep_rms": _nanrms_per_row(vsep),
            }
        )

    # RTN (baseline frame)
    rtn = None
    if V is not None and v_base is not None:
        dr_rtn = _to_rtn_series(r_base, v_base, dr)  # (M,N,3)
        rtn = {
            "dr_rtn": dr_rtn,  # raw (M,N,3)
            "env": {
                "R": _envelope_over_orbits(dr_rtn[pop_mask, :, 0], percentiles=percentiles),
                "T": _envelope_over_orbits(dr_rtn[pop_mask, :, 1], percentiles=percentiles),
                "N": _envelope_over_orbits(dr_rtn[pop_mask, :, 2], percentiles=percentiles),
            },
        }

    fig = None
    if make_plots:
        if mode == "population":
            fig = _make_population_dashboard(
                t=t_grid,
                sep=sep,
                vsep=vsep,
                env_sep=env_sep,
                env_vsep=env_vsep,
                per_orbit=per_orbit,
                pop_mask=pop_mask,
                baseline_label=base_label,
                title=plot_title,
                time_unit=time_unit,
                r_unit=r_unit,
                v_unit=v_unit,
                hist_bins=int(hist_bins),
                show_legend=bool(show_legend),
                envelope_on_log_threshold=float(envelope_on_log_threshold),
                rtn=rtn,
            )
        else:
            fig = _make_benchmark_dashboard(
                t=t_grid,
                sep=sep,
                vsep=vsep,
                per_orbit=per_orbit,
                pop_mask=pop_mask,
                labels=labels,
                baseline_label=base_label,
                title=plot_title,
                time_unit=time_unit,
                r_unit=r_unit,
                v_unit=v_unit,
                hist_bins=int(hist_bins),
                show_legend=bool(show_legend),
                envelope_on_log_threshold=float(envelope_on_log_threshold),
                rtn=rtn,
            )

    return {
        "population": {
            "t": t_grid,
            "baseline": {
                "kind": str(baseline),
                "label": str(base_label),
                "r": r_base,
                "v": v_base,
            },
            "r_mean": r_mean,
            "v_mean": v_mean,
            "r_median": r_med,
            "v_median": v_med,
            "sep": sep,
            "vsep": vsep,
            "envelope_sep": env_sep,
            "envelope_vsep": env_vsep,
            "per_orbit": per_orbit,
            "population_mask": pop_mask,
            "rtn": rtn,
        },
        "meta": {
            "M": int(M),
            "N_grid": int(t_grid.size),
            "mode": str(mode),
            "baseline": str(baseline),
            "reference": int(reference),
            "resample": resample,
            "n_resample": int(n_resample),
            "mu": None if mu is None else float(mu),
            "percentiles": tuple(float(p) for p in percentiles),
            "units": {"time": str(time_unit), "r": str(r_unit), "v": str(v_unit)},
            "labels_provided": labels is not None,
        },
        "figure": fig,
    }


# ----------------------------
# Plotting utilities
# ----------------------------

def _pad_ylim(ax, pad_frac=0.06):
    """
    Add a little headroom/footroom to the current y-limits so ticks/labels
    aren't pinned to the panel boundaries (helps stacked RTN plots).
    """
    y0, y1 = ax.get_ylim()
    if not (np.isfinite(y0) and np.isfinite(y1)):
        return
    if y1 == y0:
        span = 1.0 if y0 == 0.0 else abs(y0) * 0.1
        ax.set_ylim(y0 - span, y1 + span)
        return
    span = y1 - y0
    pad = float(pad_frac) * span
    ax.set_ylim(y0 - pad, y1 + pad)


def _round_down_sig1(x):
    x = float(x)
    if not np.isfinite(x) or x <= 0.0:
        return x
    p = 10.0 ** np.floor(np.log10(x))
    return np.floor(x / p) * p


def _round_up_sig1(x):
    x = float(x)
    if not np.isfinite(x) or x <= 0.0:
        return x
    p = 10.0 ** np.floor(np.log10(x))
    return np.ceil(x / p) * p


def _set_three_integer_xticks(ax, data, *, include_zero=True, xpad_frac=0.06):
    """
    Histogram x-range uses 1-sig-digit rounded min/max.
    Middle tick is the mean.
    Adds small x-limits padding so end ticks aren't pinned to the axes edge.
    """
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return

    xmin = float(np.min(x))
    xmax = float(np.max(x))
    meanv = float(np.mean(x))

    # If there are negatives, fall back to integer bounds
    if xmin < 0.0:
        lo_tick = float(np.floor(xmin))
        hi_tick = float(np.ceil(xmax))
        span = max(hi_tick - lo_tick, 1.0)
        pad = float(xpad_frac) * span
        ax.set_xlim(lo_tick - pad, hi_tick + pad)
        ticks = np.array([lo_tick, meanv, hi_tick], dtype=float)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{int(np.round(t))}" for t in ticks])
        return

    lo_tick = _round_down_sig1(xmin)
    hi_tick = _round_up_sig1(xmax)

    # Make sure mean is inside [lo_tick, hi_tick]
    mean_tick = float(np.clip(meanv, lo_tick, hi_tick))

    span = max(hi_tick - lo_tick, 1.0 if hi_tick == lo_tick else (hi_tick - lo_tick))
    pad = float(xpad_frac) * span
    ax.set_xlim(lo_tick - pad, hi_tick + pad)

    ticks = np.array([lo_tick, mean_tick, hi_tick], dtype=float)
    ax.set_xticks(ticks)

    def _fmt(v):
        v = float(v)
        if abs(v) >= 1.0:
            return f"{int(np.round(v))}"
        return np.format_float_positional(v, precision=3, unique=False, trim="-")

    ax.set_xticklabels([_fmt(t) for t in ticks])


def _hist_step(ax, x, *, bins, linestyle, linewidth, color):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    return ax.hist(
        x,
        bins=bins,
        histtype="step",
        linewidth=float(linewidth),
        linestyle=str(linestyle),
        color=str(color),
    )


def _add_mean_max_vlines(
    ax,
    x,
    *,
    mean_color="black",
    max_color="tab:green",
    mean_ls="--",
    max_ls=":",
    mean_lw=2.0,
    max_lw=2.4,
):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return
    m = float(np.mean(x))
    mx = float(np.max(x))
    ax.axvline(m, color=mean_color, linestyle=mean_ls, linewidth=mean_lw)
    ax.axvline(mx, color=max_color, linestyle=max_ls, linewidth=max_lw)


def _rtn_stack_axes(fig, cell, *, time_unit, shared_ylabel, title_text):
    """
    Create a 3x1 stack (R/T/N) inside `cell` with no vertical gap and shared x.
    Adds black divider lines between panels and places R/T/N labels as text.
    """
    axc = fig.add_subplot(cell)
    axc.set_frame_on(False)
    axc.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)

    # More padding for the shared ylabel (per request)
    axc.set_ylabel(shared_ylabel, labelpad=26)
    axc.set_title(title_text)

    subgs = cell.subgridspec(3, 1, hspace=0.0)
    axR = fig.add_subplot(subgs[0, 0])
    axT = fig.add_subplot(subgs[1, 0], sharex=axR)
    axN = fig.add_subplot(subgs[2, 0], sharex=axR)

    axR.tick_params(labelbottom=False)
    axT.tick_params(labelbottom=False)
    axN.set_xlabel(f"time ({time_unit})")

    for ax in (axR, axT, axN):
        ax.set_ylabel("")

    # Divider lines
    for a in (axR, axT):
        a.spines["bottom"].set_visible(True)
        a.spines["bottom"].set_color("black")
        a.spines["bottom"].set_linewidth(1.0)
    for a in (axT, axN):
        a.spines["top"].set_visible(True)
        a.spines["top"].set_color("black")
        a.spines["top"].set_linewidth(1.0)

    # Labels inside panels
    axR.text(0.02, 0.88, "R", transform=axR.transAxes, ha="left", va="top")
    axT.text(0.02, 0.88, "T", transform=axT.transAxes, ha="left", va="top")
    axN.text(0.02, 0.88, "N", transform=axN.transAxes, ha="left", va="top")

    return axR, axT, axN, axc


# ----------------------------
# Population plotting
# ----------------------------

def _make_population_dashboard(
    *,
    t,
    sep,
    vsep,
    env_sep,
    env_vsep,
    per_orbit,
    pop_mask,
    baseline_label,
    title,
    time_unit,
    r_unit,
    v_unit,
    hist_bins,
    show_legend,
    envelope_on_log_threshold,
    rtn,
):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    STYLE = {
        "median": {"color": "tab:blue",  "linestyle": "-",  "linewidth": 2.4},
        "mean":   {"color": "black",     "linestyle": "--", "linewidth": 2.0},
        "max":    {"color": "tab:green", "linestyle": ":",  "linewidth": 2.4},
        # Make percentile bands explicitly "orange" family (per request)
        "band_color": "tab:orange",
        "band_5_95": {"alpha": 0.20},
        "band_25_75": {"alpha": 0.35},
    }

    # Histogram style (consistent)
    HIST_MAX_COLOR = "tab:blue"
    HIST_FINAL_COLOR = "tab:orange"
    MEAN_COLOR = "black"
    MAX_COLOR = "tab:green"
    HIST_MAX_LS = "-"
    HIST_FINAL_LS = "--"
    HIST_LW = 2.4
    MEAN_LS = "--"
    MAX_LS = ":"
    MEAN_LW = 2.0
    MAX_LW = 2.4

    t = np.asarray(t, dtype=float)
    sep = np.asarray(sep, dtype=float)
    Y_sep = sep[pop_mask, :]

    def _should_log_y_from_env(env):
        ys = []
        for k, v in env.items():
            if k.startswith("p"):
                vv = np.asarray(v, dtype=float)
                vv = vv[np.isfinite(vv) & (vv > 0.0)]
                if vv.size:
                    ys.append(vv)
        if not ys:
            return False
        ycat = np.concatenate(ys)
        lo = float(np.min(ycat))
        hi = float(np.max(ycat))
        return (hi / max(lo, 1e-300)) >= float(envelope_on_log_threshold)

    def _safe_log(y):
        return np.maximum(np.asarray(y, dtype=float), 1e-300)

    def _plot_env_with_mean_max(ax, env, Y, *, title_, ylabel):
        ax.set_title(title_)
        ax.set_xlabel(f"time ({time_unit})")
        ax.set_ylabel(ylabel)

        use_logy = _should_log_y_from_env(env)
        if use_logy:
            ax.set_yscale("log")

        def _Yy(y):
            y = np.asarray(y, dtype=float)
            return _safe_log(y) if use_logy else y

        # Bands (explicit orange color)
        if ("p5" in env) and ("p95" in env):
            ax.fill_between(
                t, _Yy(env["p5"]), _Yy(env["p95"]),
                color=STYLE["band_color"],
                alpha=STYLE["band_5_95"]["alpha"],
            )
        if ("p25" in env) and ("p75" in env):
            ax.fill_between(
                t, _Yy(env["p25"]), _Yy(env["p75"]),
                color=STYLE["band_color"],
                alpha=STYLE["band_25_75"]["alpha"],
            )

        if "p50" in env:
            ax.plot(t, _Yy(env["p50"]), color=STYLE["median"]["color"], linestyle="-", linewidth=STYLE["median"]["linewidth"])

        mean_t = np.nanmean(Y, axis=0)
        max_t = np.nanmax(Y, axis=0)
        ax.plot(t, _Yy(mean_t), color=STYLE["mean"]["color"], linestyle="--", linewidth=STYLE["mean"]["linewidth"])
        ax.plot(t, _Yy(max_t), color=STYLE["max"]["color"], linestyle=":", linewidth=STYLE["max"]["linewidth"])

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(title)

    gs = fig.add_gridspec(2, 2)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.07, wspace=0.16, hspace=0.22)

    # (1) Position envelope (legend inside top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_env_with_mean_max(
        ax1,
        env_sep,
        Y_sep,
        title_=f"Spread from {baseline_label}: ||r - r_base||",
        ylabel=f"||r - r_base|| ({r_unit})",
    )
    if show_legend:
        ax1.legend(
            handles=[
                Patch(facecolor=STYLE["band_color"], alpha=STYLE["band_5_95"]["alpha"], label="p5–p95"),
                Patch(facecolor=STYLE["band_color"], alpha=STYLE["band_25_75"]["alpha"], label="p25–p75"),
                Line2D([0], [0], color=STYLE["median"]["color"], linestyle="-", linewidth=STYLE["median"]["linewidth"], label="median (p50)"),
                Line2D([0], [0], color=STYLE["mean"]["color"], linestyle="--", linewidth=STYLE["mean"]["linewidth"], label="mean"),
                Line2D([0], [0], color=STYLE["max"]["color"], linestyle=":", linewidth=STYLE["max"]["linewidth"], label="max"),
            ],
            loc="upper left",
            fontsize=9,
        )

    # (2) Velocity envelope (NO LEGEND)
    ax2 = fig.add_subplot(gs[0, 1])
    if vsep is not None and env_vsep is not None:
        Y_v = np.asarray(vsep, dtype=float)[pop_mask, :]
        _plot_env_with_mean_max(
            ax2,
            env_vsep,
            Y_v,
            title_=f"Velocity spread from {baseline_label}: ||v - v_base||",
            ylabel=f"||v - v_base|| ({v_unit})",
        )
    else:
        _plot_env_with_mean_max(
            ax2,
            env_sep,
            Y_sep,
            title_=f"(No v) Spread from {baseline_label}: ||r - r_base|| (duplicate)",
            ylabel=f"||r - r_base|| ({r_unit})",
        )

    # (3) Bottom-left: split into two side-by-side axes (share y)
    cell = gs[1, 0].subgridspec(1, 2, wspace=0.04)
    ax3_pos = fig.add_subplot(cell[0, 0])
    ax3_vel = fig.add_subplot(cell[0, 1], sharey=ax3_pos)

    ax3_pos.set_title("Distributions vs baseline (step)")
    ax3_pos.set_ylabel("count (-)")
    ax3_pos.set_xlabel(f"position spread ({r_unit})")

    ax3_vel.tick_params(labelleft=False)
    ax3_vel.set_xlabel(f"velocity spread ({v_unit})")

    sep_max = np.asarray(per_orbit["sep_max"], dtype=float)[pop_mask]
    sep_final = np.asarray(per_orbit["sep_final"], dtype=float)[pop_mask]

    bins_pos = int(hist_bins)
    _hist_step(ax3_pos, sep_max, bins=bins_pos, linestyle=HIST_MAX_LS, linewidth=HIST_LW, color=HIST_MAX_COLOR)
    _hist_step(ax3_pos, sep_final, bins=bins_pos, linestyle=HIST_FINAL_LS, linewidth=HIST_LW, color=HIST_FINAL_COLOR)
    _add_mean_max_vlines(
        ax3_pos, np.concatenate([sep_max, sep_final]),
        mean_color=MEAN_COLOR, max_color=MAX_COLOR,
        mean_ls=MEAN_LS, max_ls=MAX_LS,
        mean_lw=MEAN_LW, max_lw=MAX_LW,
    )
    # Extra x padding on the LEFT histogram to reduce label collision at the center seam
    _set_three_integer_xticks(ax3_pos, np.concatenate([sep_max, sep_final]), include_zero=True, xpad_frac=0.12)

    if vsep is not None and ("vsep_max" in per_orbit):
        v_max = np.asarray(per_orbit["vsep_max"], dtype=float)[pop_mask]
        v_final = np.asarray(per_orbit["vsep_final"], dtype=float)[pop_mask]
        bins_vel = int(hist_bins)
        _hist_step(ax3_vel, v_max, bins=bins_vel, linestyle=HIST_MAX_LS, linewidth=HIST_LW, color=HIST_MAX_COLOR)
        _hist_step(ax3_vel, v_final, bins=bins_vel, linestyle=HIST_FINAL_LS, linewidth=HIST_LW, color=HIST_FINAL_COLOR)
        _add_mean_max_vlines(
            ax3_vel, np.concatenate([v_max, v_final]),
            mean_color=MEAN_COLOR, max_color=MAX_COLOR,
            mean_ls=MEAN_LS, max_ls=MAX_LS,
            mean_lw=MEAN_LW, max_lw=MAX_LW,
        )
        # Slightly less padding on the RIGHT histogram (still not edge-pinned)
        _set_three_integer_xticks(ax3_vel, np.concatenate([v_max, v_final]), include_zero=True, xpad_frac=0.08)
    else:
        ax3_vel.text(0.5, 0.5, "velocity not provided", ha="center", va="center", transform=ax3_vel.transAxes)
        ax3_vel.set_xticks([])
        ax3_vel.set_yticks([])

    # Single legend (inside, upper-left) for bottom-left
    if show_legend:
        legend_handles = [
            Line2D([0], [0], color=HIST_MAX_COLOR, linestyle=HIST_MAX_LS, linewidth=HIST_LW, label="max"),
            Line2D([0], [0], color=HIST_FINAL_COLOR, linestyle=HIST_FINAL_LS, linewidth=HIST_LW, label="final"),
            Line2D([0], [0], color=MEAN_COLOR, linestyle=MEAN_LS, linewidth=MEAN_LW, label="mean"),
            Line2D([0], [0], color=MAX_COLOR, linestyle=MAX_LS, linewidth=MAX_LW, label="max"),
        ]
        ax3_pos.legend(handles=legend_handles, loc="upper left", fontsize=8)

    # (4) Bottom-right: RTN 3-stack (NO LEGEND)
    axR, axT, axN, axc = _rtn_stack_axes(
        fig,
        gs[1, 1],
        time_unit=time_unit,
        shared_ylabel=f"Δr in RTN ({r_unit})",
        title_text="RTN component bands vs baseline",
    )

    if rtn is None or ("env" not in rtn):
        for ax in (axR, axT, axN):
            ax.text(0.5, 0.5, "RTN unavailable (need velocities)", ha="center", va="center")
            ax.set_axis_off()
    else:
        dr_rtn = np.asarray(rtn["dr_rtn"], dtype=float)  # (M,N,3)
        YR = dr_rtn[pop_mask, :, 0]
        YT = dr_rtn[pop_mask, :, 1]
        YN = dr_rtn[pop_mask, :, 2]

        envR = rtn["env"]["R"]
        envT = rtn["env"]["T"]
        envN = rtn["env"]["N"]

        def _plot_comp(ax, env, Y):
            # Bands (orange)
            if ("p5" in env) and ("p95" in env):
                ax.fill_between(t, env["p5"], env["p95"], color=STYLE["band_color"], alpha=STYLE["band_5_95"]["alpha"])
            if ("p25" in env) and ("p75" in env):
                ax.fill_between(t, env["p25"], env["p75"], color=STYLE["band_color"], alpha=STYLE["band_25_75"]["alpha"])
            if "p50" in env:
                ax.plot(t, env["p50"], color=STYLE["median"]["color"], linestyle="-", linewidth=2.0)

            mean_t = np.nanmean(Y, axis=0)
            max_abs_t = np.nanmax(np.abs(Y), axis=0)
            ax.plot(t, mean_t, color=STYLE["mean"]["color"], linestyle="--", linewidth=STYLE["mean"]["linewidth"])
            ax.plot(t, max_abs_t, color=STYLE["max"]["color"], linestyle=":", linewidth=STYLE["max"]["linewidth"])

        _plot_comp(axR, envR, YR)
        _plot_comp(axT, envT, YT)
        _plot_comp(axN, envN, YN)

        # Per request: add headroom/footroom so y ticks aren't pinned against divider lines
        _pad_ylim(axR, pad_frac=0.07)
        _pad_ylim(axT, pad_frac=0.07)
        _pad_ylim(axN, pad_frac=0.07)

    return fig


# ----------------------------
# Benchmark plotting
# ----------------------------

def _make_benchmark_dashboard(
    *,
    t,
    sep,
    vsep,
    per_orbit,
    pop_mask,
    labels,
    baseline_label,
    title,
    time_unit,
    r_unit,
    v_unit,
    hist_bins,
    show_legend,
    envelope_on_log_threshold,
    rtn,
):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # Histogram style
    HIST_MAX_COLOR = "tab:blue"
    HIST_FINAL_COLOR = "tab:orange"
    MEAN_COLOR = "black"
    MAX_COLOR = "tab:green"
    HIST_MAX_LS = "-"
    HIST_FINAL_LS = "--"
    HIST_LW = 2.4
    MEAN_LS = "--"
    MAX_LS = ":"
    MEAN_LW = 2.0
    MAX_LW = 2.4

    t = np.asarray(t, dtype=float)
    sep = np.asarray(sep, dtype=float)

    def _safe_label(k):
        if labels is None:
            return f"orbit {int(k)}"
        return str(labels[int(k)])

    def _should_log_y_matrix(Y):
        y = np.asarray(Y, dtype=float)
        y = y[np.isfinite(y) & (y > 0.0)]
        if y.size < 2:
            return False
        lo = float(np.min(y))
        hi = float(np.max(y))
        return (hi / max(lo, 1e-300)) >= float(envelope_on_log_threshold)

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(title)

    gs = fig.add_gridspec(2, 2)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.07, wspace=0.16, hspace=0.22)

    # (1) Position distances vs nominal (legend inside upper-left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title(f"Distances vs {baseline_label}: ||r - r_base||")
    ax1.set_xlabel(f"time ({time_unit})")
    ax1.set_ylabel(f"||r - r_base|| ({r_unit})")

    Y_sep = sep[pop_mask, :]
    use_logy = _should_log_y_matrix(Y_sep)
    if use_logy:
        ax1.set_yscale("log")

    for k in np.where(pop_mask)[0]:
        yy = np.maximum(sep[int(k)], 1e-300) if use_logy else sep[int(k)]
        ax1.plot(t, yy, label=_safe_label(k))

    if show_legend:
        ax1.legend(loc="upper left", fontsize=8)

    # (2) Velocity distances vs nominal (NO LEGEND)
    ax2 = fig.add_subplot(gs[0, 1])
    if vsep is not None:
        ax2.set_title(f"Velocity vs {baseline_label}: ||v - v_base||")
        ax2.set_xlabel(f"time ({time_unit})")
        ax2.set_ylabel(f"||v - v_base|| ({v_unit})")

        Y_v = np.asarray(vsep, dtype=float)[pop_mask, :]
        use_logy_v = _should_log_y_matrix(Y_v)
        if use_logy_v:
            ax2.set_yscale("log")

        for k in np.where(pop_mask)[0]:
            yy = np.maximum(vsep[int(k)], 1e-300) if use_logy_v else vsep[int(k)]
            ax2.plot(t, yy, label=_safe_label(k))
    else:
        ax2.set_title("Velocity not provided")
        ax2.set_axis_off()

    # (3) Bottom-left: split into two side-by-side axes (share y)
    cell = gs[1, 0].subgridspec(1, 2, wspace=0.04)
    ax3_pos = fig.add_subplot(cell[0, 0])
    ax3_vel = fig.add_subplot(cell[0, 1], sharey=ax3_pos)

    ax3_pos.set_title("Distributions vs nominal (step)")
    ax3_pos.set_ylabel("count (-)")
    ax3_pos.set_xlabel(f"position spread ({r_unit})")

    ax3_vel.tick_params(labelleft=False)
    ax3_vel.set_xlabel(f"velocity spread ({v_unit})")

    sep_max = np.asarray(per_orbit["sep_max"], dtype=float)[pop_mask]
    sep_final = np.asarray(per_orbit["sep_final"], dtype=float)[pop_mask]

    bins_pos = int(hist_bins)
    _hist_step(ax3_pos, sep_max, bins=bins_pos, linestyle=HIST_MAX_LS, linewidth=HIST_LW, color=HIST_MAX_COLOR)
    _hist_step(ax3_pos, sep_final, bins=bins_pos, linestyle=HIST_FINAL_LS, linewidth=HIST_LW, color=HIST_FINAL_COLOR)
    _add_mean_max_vlines(
        ax3_pos, np.concatenate([sep_max, sep_final]),
        mean_color=MEAN_COLOR, max_color=MAX_COLOR,
        mean_ls=MEAN_LS, max_ls=MAX_LS,
        mean_lw=MEAN_LW, max_lw=MAX_LW,
    )
    _set_three_integer_xticks(ax3_pos, np.concatenate([sep_max, sep_final]), include_zero=True, xpad_frac=0.12)

    if vsep is not None and ("vsep_max" in per_orbit):
        v_max = np.asarray(per_orbit["vsep_max"], dtype=float)[pop_mask]
        v_final = np.asarray(per_orbit["vsep_final"], dtype=float)[pop_mask]
        bins_vel = int(hist_bins)
        _hist_step(ax3_vel, v_max, bins=bins_vel, linestyle=HIST_MAX_LS, linewidth=HIST_LW, color=HIST_MAX_COLOR)
        _hist_step(ax3_vel, v_final, bins=bins_vel, linestyle=HIST_FINAL_LS, linewidth=HIST_LW, color=HIST_FINAL_COLOR)
        _add_mean_max_vlines(
            ax3_vel, np.concatenate([v_max, v_final]),
            mean_color=MEAN_COLOR, max_color=MAX_COLOR,
            mean_ls=MEAN_LS, max_ls=MAX_LS,
            mean_lw=MEAN_LW, max_lw=MAX_LW,
        )
        _set_three_integer_xticks(ax3_vel, np.concatenate([v_max, v_final]), include_zero=True, xpad_frac=0.08)
    else:
        ax3_vel.text(0.5, 0.5, "velocity not provided", ha="center", va="center", transform=ax3_vel.transAxes)
        ax3_vel.set_xticks([])
        ax3_vel.set_yticks([])

    # Single shared legend for bottom-left only (inside upper-left)
    if show_legend:
        legend_handles = [
            Line2D([0], [0], color=HIST_MAX_COLOR, linestyle=HIST_MAX_LS, linewidth=HIST_LW, label="max"),
            Line2D([0], [0], color=HIST_FINAL_COLOR, linestyle=HIST_FINAL_LS, linewidth=HIST_LW, label="final"),
            Line2D([0], [0], color=MEAN_COLOR, linestyle=MEAN_LS, linewidth=MEAN_LW, label="mean"),
            Line2D([0], [0], color=MAX_COLOR, linestyle=MAX_LS, linewidth=MAX_LW, label="max"),
        ]
        ax3_pos.legend(handles=legend_handles, loc="upper left", fontsize=8)

    # (4) Bottom-right: RTN 3-stack (NO LEGEND)
    axR, axT, axN, axc = _rtn_stack_axes(
        fig,
        gs[1, 1],
        time_unit=time_unit,
        shared_ylabel=f"Δr in RTN ({r_unit})",
        title_text="RTN components vs nominal (stacked)",
    )

    if rtn is None or "dr_rtn" not in rtn:
        for ax in (axR, axT, axN):
            ax.text(0.5, 0.5, "RTN unavailable (need velocities)", ha="center", va="center")
            ax.set_axis_off()
    else:
        dr_rtn = np.asarray(rtn["dr_rtn"], dtype=float)  # (M,N,3)
        for k in np.where(pop_mask)[0]:
            axR.plot(t, dr_rtn[int(k), :, 0])
            axT.plot(t, dr_rtn[int(k), :, 1])
            axN.plot(t, dr_rtn[int(k), :, 2])

        _pad_ylim(axR, pad_frac=0.07)
        _pad_ylim(axT, pad_frac=0.07)
        _pad_ylim(axN, pad_frac=0.07)

    return fig


# ----------------------------
# Alignment + population helpers
# ----------------------------

def _align_all_to_grid(*, t_list, r_list, v_list, reference, resample, n_resample):
    M = len(r_list)

    if resample is None:
        t0 = t_list[0]
        for k in range(1, M):
            if t_list[k].shape != t0.shape or not np.allclose(t_list[k], t0, rtol=0.0, atol=0.0):
                raise ValueError("resample=None requires identical time arrays across all orbits.")
        t_grid = np.asarray(t0, dtype=float)
        R = np.stack([np.asarray(r, dtype=float) for r in r_list], axis=0)
        V = None if (v_list is None or all(v is None for v in v_list)) else np.stack([np.asarray(v, dtype=float) for v in v_list], axis=0)
        return t_grid, R, V

    t_starts = np.array([float(t[0]) for t in t_list], dtype=float)
    t_ends = np.array([float(t[-1]) for t in t_list], dtype=float)

    if resample == "intersection":
        t0 = float(np.max(t_starts))
        t1 = float(np.min(t_ends))
        if not (t1 > t0):
            raise ValueError("No overlapping time interval for intersection resampling.")
        t_grid = np.linspace(t0, t1, int(n_resample), dtype=float)
        fill = np.nan
    elif resample == "union":
        t0 = float(np.min(t_starts))
        t1 = float(np.max(t_ends))
        if not (t1 > t0):
            raise ValueError("Invalid union interval.")
        t_grid = np.linspace(t0, t1, int(n_resample), dtype=float)
        fill = np.nan
    elif resample == "ref":
        t_grid = np.asarray(t_list[int(reference)], dtype=float)
        fill = np.nan
    else:
        raise ValueError("resample must be one of: 'intersection', 'union', 'ref', None.")

    R = np.empty((M, t_grid.size, 3), dtype=float)
    if v_list is None or all(v is None for v in v_list):
        V = None
    else:
        V = np.empty((M, t_grid.size, 3), dtype=float)

    for k in range(M):
        R[k] = _interp_xyz_nan(t_list[k], r_list[k], t_grid, fill=fill)
        if V is not None:
            V[k] = _interp_xyz_nan(t_list[k], v_list[k], t_grid, fill=fill)

    return t_grid, R, V


def _interp_xyz_nan(t_src, x_src, t_dst, *, fill=np.nan):
    t_src = np.asarray(t_src, dtype=float)
    x_src = np.asarray(x_src, dtype=float)
    t_dst = np.asarray(t_dst, dtype=float)

    out = np.empty((t_dst.size, 3), dtype=float)
    for c in range(3):
        out[:, c] = np.interp(t_dst, t_src, x_src[:, c])

    m = (t_dst < t_src[0]) | (t_dst > t_src[-1])
    if np.any(m):
        out[m, :] = fill
    return out


def _envelope_over_orbits(Y, *, percentiles):
    Y = np.asarray(Y, dtype=float)
    out = {}
    for p in percentiles:
        out[f"p{int(p)}"] = np.nanpercentile(Y, p, axis=0)
    return out


def _nanmax_per_row(Y):
    Y = np.asarray(Y, dtype=float)
    return np.nanmax(Y, axis=1)


def _nanrms_per_row(Y):
    Y = np.asarray(Y, dtype=float)
    return np.sqrt(np.nanmean(Y * Y, axis=1))


def _nanfinal_per_row(Y):
    Y = np.asarray(Y, dtype=float)
    M, N = Y.shape
    out = np.full((M,), np.nan, dtype=float)
    finite = np.isfinite(Y)
    idx = np.where(finite, np.arange(N, dtype=int)[None, :], -1)
    last = np.max(idx, axis=1)
    ok = last >= 0
    out[ok] = Y[np.arange(M, dtype=int)[ok], last[ok]]
    return out


# ----------------------------
# RTN projection helper
# ----------------------------

def _to_rtn_series(r_base, v_base, d_series):
    """
    Project vectors d_series (M,N,3) into the RTN frame defined by baseline (r_base, v_base).
    Returns (M,N,3) with components [R, T, N].
    """
    r_base = np.asarray(r_base, dtype=float)  # (N,3)
    v_base = np.asarray(v_base, dtype=float)  # (N,3)
    d = np.asarray(d_series, dtype=float)     # (M,N,3)

    r_norm = np.linalg.norm(r_base, axis=1)
    Rhat = r_base / np.maximum(r_norm[:, None], 1e-300)

    h = np.cross(r_base, v_base)
    h_norm = np.linalg.norm(h, axis=1)
    Nhat = h / np.maximum(h_norm[:, None], 1e-300)

    That = np.cross(Nhat, Rhat)
    t_norm = np.linalg.norm(That, axis=1)
    That = That / np.maximum(t_norm[:, None], 1e-300)

    dR = np.sum(d * Rhat[None, :, :], axis=2)
    dT = np.sum(d * That[None, :, :], axis=2)
    dN = np.sum(d * Nhat[None, :, :], axis=2)

    return np.stack([dR, dT, dN], axis=2)


# ----------------------------
# Input normalization
# ----------------------------

def _normalize_inputs(r_list, v_list, t_list):
    if isinstance(r_list, np.ndarray) and r_list.ndim == 3:
        r_list = [np.asarray(r_list[k], dtype=float) for k in range(r_list.shape[0])]
    else:
        r_list = [np.asarray(r, dtype=float) for r in r_list]

    for k, r in enumerate(r_list):
        if r.ndim != 2 or r.shape[1] != 3:
            raise ValueError(f"r_list[{k}] must have shape (N,3). Got {r.shape}.")

    if v_list is None:
        v_list = [None] * len(r_list)
    else:
        if isinstance(v_list, np.ndarray) and v_list.ndim == 3:
            v_list = [np.asarray(v_list[k], dtype=float) for k in range(v_list.shape[0])]
        else:
            v_list = [np.asarray(v, dtype=float) for v in v_list]
        for k, (r, v) in enumerate(zip(r_list, v_list)):
            if v is None:
                continue
            if v.ndim != 2 or v.shape[1] != 3:
                raise ValueError(f"v_list[{k}] must have shape (N,3). Got {v.shape}.")
            if v.shape[0] != r.shape[0]:
                raise ValueError(f"r/v length mismatch at {k}: {r.shape[0]} vs {v.shape[0]}.")

    if t_list is None:
        t_list = [np.arange(r.shape[0], dtype=float) for r in r_list]
    else:
        if isinstance(t_list, np.ndarray) and t_list.ndim == 2:
            t_list = [np.asarray(t_list[k], dtype=float) for k in range(t_list.shape[0])]
        else:
            t_list = [np.asarray(t, dtype=float) for t in t_list]
        for k, (t, r) in enumerate(zip(t_list, r_list)):
            if t.ndim != 1:
                raise ValueError(f"t_list[{k}] must be 1D. Got {t.shape}.")
            if t.shape[0] != r.shape[0]:
                raise ValueError(f"t/r length mismatch at {k}: {t.shape[0]} vs {r.shape[0]}.")

    return r_list, v_list, t_list
