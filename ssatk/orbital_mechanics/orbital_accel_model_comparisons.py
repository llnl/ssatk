"""
SSAPy acceleration-ladder runner + accel-ladder-specific divergence dashboard.

Design
------

- ``calculate_accel_comparisons()`` performs propagation and comparison math.
- ``make_accel_ladder_dashboard_figures()`` performs plotting.
- ``compare_models()`` calls both functions; plotting is always enabled.

Early-stop behavior
-------------------

If SSAPy terminates early, the runner truncates all rungs to their common
prefix and reports per-rung stop indices and epochs.

Dashboard
---------

- Figure 1 shows time-domain reference, incremental, and NTW comparisons.
- Figure 2 summarizes final errors and a reference-error heatmap.

Notes
-----

- Log plots use a floor ``epsilon_m`` (default 1 mm).
- The heatmap colorbar is capped at ``max_error_m`` (default 1e7 m).
"""

from __future__ import annotations

import numpy as np
from ..constants import EARTH_MU

# =============================================================================
# Time helpers (used only by calculate_accel_comparisons)
# =============================================================================


def _is_astropy_time(x):
    return x.__class__.__module__.startswith("astropy.time") and x.__class__.__name__ == "Time"


def _to_astropy_time(x):
    from astropy.time import Time
    return Time(x)


def _orbit_epoch_gps(orbit):
    t0 = getattr(orbit, "t", None)
    if t0 is None:
        raise ValueError("Orbit object is missing attribute `.t` (epoch).")
    if _is_astropy_time(t0):
        return float(t0.gps)
    return float(np.asarray(t0).reshape(()))


def _coerce_times_for_ssapy(times, orbit_epoch_gps_s, assume="auto"):
    """
    Return SSAPy-compatible times:
      - astropy Time, or
      - float GPS seconds (since 1980-01-06)

    If numeric and assume="auto":
      - treat as offsets if values look "small" relative to GPS seconds.
    """
    if _is_astropy_time(times):
        return times

    if isinstance(times, (list, tuple)):
        times = np.array(times, dtype=object)

    if isinstance(times, np.ndarray) and times.dtype.kind in {"M", "m"}:
        return _to_astropy_time(times)

    if isinstance(times, np.ndarray) and times.dtype == object:
        sample = times.flat[0]
        if _is_astropy_time(sample):
            return sample.__class__(times)
        if hasattr(sample, "year") or isinstance(sample, np.datetime64):
            if isinstance(sample, np.datetime64):
                times = np.asarray(times, dtype="datetime64[ns]")
            return _to_astropy_time(times)

    t = np.asarray(times, dtype=float).ravel()

    if assume == "gps":
        return t
    if assume == "offset":
        return float(orbit_epoch_gps_s) + t

    if (np.nanmax(np.abs(t)) < 1e8) and (orbit_epoch_gps_s > 1e8):
        return float(orbit_epoch_gps_s) + t
    return t


def _times_to_relative_seconds(times_ssapy):
    if _is_astropy_time(times_ssapy):
        dt = (times_ssapy - times_ssapy[0]).to_value("s")
        return np.asarray(dt, dtype=float).ravel()
    t = np.asarray(times_ssapy, dtype=float).ravel()
    return t - float(t[0])


# =============================================================================
# Math helpers (used only by calculate_accel_comparisons)
# =============================================================================
def _keplerian_elements_from_rv(r_m, v_mps, mu_m3s2=3.986004418e14):
    """
    Classical Keplerian elements from inertial r,v. Returns degrees for angles.
    """
    r = np.asarray(r_m, dtype=float).reshape(3)
    v = np.asarray(v_mps, dtype=float).reshape(3)
    mu = float(mu_m3s2)

    tiny = 1e-14

    rnorm = float(np.linalg.norm(r))
    vnorm = float(np.linalg.norm(v))

    h = np.cross(r, v)
    hnorm = float(np.linalg.norm(h))

    k = np.array([0.0, 0.0, 1.0], dtype=float)
    n = np.cross(k, h)
    nnorm = float(np.linalg.norm(n))

    e_vec = (np.cross(v, h) / mu) - (r / rnorm)
    e = float(np.linalg.norm(e_vec))

    energy = 0.5 * (vnorm * vnorm) - mu / rnorm
    if abs(energy) > tiny:
        a = -mu / (2.0 * energy)
    else:
        a = np.inf

    if hnorm > tiny:
        inc = float(np.arccos(np.clip(h[2] / hnorm, -1.0, 1.0)))
    else:
        inc = 0.0

    if nnorm > tiny:
        raan = float(np.arccos(np.clip(n[0] / nnorm, -1.0, 1.0)))
        if n[1] < 0.0:
            raan = 2.0 * np.pi - raan
    else:
        raan = 0.0

    if nnorm > tiny and e > 1e-10:
        argp = float(np.arccos(np.clip(np.dot(n, e_vec) / (nnorm * e), -1.0, 1.0)))
        if e_vec[2] < 0.0:
            argp = 2.0 * np.pi - argp
    else:
        argp = 0.0

    if e > 1e-10:
        nu = float(np.arccos(np.clip(np.dot(e_vec, r) / (e * rnorm), -1.0, 1.0)))
        if np.dot(r, v) < 0.0:
            nu = 2.0 * np.pi - nu
    else:
        if nnorm > tiny:
            u = float(np.arccos(np.clip(np.dot(n, r) / (nnorm * rnorm), -1.0, 1.0)))
            if r[2] < 0.0:
                u = 2.0 * np.pi - u
            nu = u
        else:
            lam = float(np.arccos(np.clip(r[0] / rnorm, -1.0, 1.0)))
            if r[1] < 0.0:
                lam = 2.0 * np.pi - lam
            nu = lam

    if np.isfinite(a) and a > 0.0:
        period_s = float(2.0 * np.pi * np.sqrt((a * a * a) / mu))
    else:
        period_s = np.nan

    return {
        "a_m": float(a),
        "e": float(e),
        "i_deg": float(np.degrees(inc)),
        "raan_deg": float(np.degrees(raan)),
        "argp_deg": float(np.degrees(argp)),
        "nu_deg": float(np.degrees(nu)),
        "period_s": period_s,
    }


def _format_oe_text(oe):
    a_km = oe["a_m"] / 1e3 if np.isfinite(oe["a_m"]) else np.nan
    e = oe["e"]
    inc = oe["i_deg"]
    raan = oe["raan_deg"]
    argp = oe["argp_deg"]
    nu = oe["nu_deg"]
    T = oe["period_s"]

    if np.isfinite(T):
        T_min = T / 60.0
        tline = f"T = {T_min:.2f} min"
    else:
        tline = "T = n/a"

    return (
        "Initial Keplerian OE\n"
        f"a = {a_km:.3f} km\n"
        f"e = {e:.6f}\n"
        f"i = {inc:.3f} deg\n"
        f"RAAN = {raan:.3f} deg\n"
        f"argp = {argp:.3f} deg\n"
        f"nu = {nu:.3f} deg\n"
        f"{tline}"
    )


def _ntw_components(r_ref, v_ref, dr):
    """
    Project dr into an NTW basis built from reference (r_ref, v_ref).
      T along velocity
      W along (r x v) (orbit normal)
      N along v x (r x v) (in-plane, perpendicular to T)
    Returns (N,T,W) components in meters.
    """
    r_ref = np.asarray(r_ref, dtype=float)
    v_ref = np.asarray(v_ref, dtype=float)
    dr = np.asarray(dr, dtype=float)

    def _normed(x):
        n = np.linalg.norm(x, axis=-1, keepdims=True)
        n = np.where(n == 0.0, 1.0, n)
        return x / n

    t_hat = _normed(v_ref)
    w_hat = _normed(np.cross(r_ref, v_ref))
    n_hat = _normed(np.cross(v_ref, np.cross(r_ref, v_ref)))

    n = np.einsum("...i,...i->...", n_hat, dr)
    t = np.einsum("...i,...i->...", t_hat, dr)
    w = np.einsum("...i,...i->...", w_hat, dr)
    return np.stack([n, t, w], axis=-1)


def _small_accel_ladder(max_rungs):
    from ssapy.accel import AccelKepler
    from ssapy.body import get_body
    from ssapy.gravity import AccelHarmonic, AccelThirdBody

    a_kepler = AccelKepler()
    suite = {"Kep": a_kepler}
    if max_rungs <= 1:
        return suite

    moon = get_body("moon")
    a_moon_point = AccelThirdBody(moon)
    suite["Kep+Moon"] = a_kepler + a_moon_point
    if max_rungs <= 2:
        return suite

    earth = get_body("Earth", model="EGM2008")
    a_earth_j2 = AccelHarmonic(earth, 2, 0)
    suite["Kep+Moon+J2"] = suite["Kep+Moon"] + a_earth_j2
    return suite
# =============================================================================
# Model-comparison calculation
# =============================================================================


def calculate_accel_comparisons(
    orbit=None,
    r=None,
    v=None,
    t0=None,
    times=None,
    assume_times="auto",
    ode_kwargs=None,
    reference=None,
    mu_m3s2=3.986004418e14,
    max_rungs=None,
):
    """
    All propagation + all divergence math in ONE function.

    Returns a dict containing:
      - labels, reference
      - times_ssapy, t_rel_s (aligned to common prefix), t_rel_s_full
      - r_list, v_list (aligned)
      - early-stop metadata: stop_idx, stop_t_s, common_len
      - orbit elements + text
      - divergence arrays/metrics: drn_vs_ref, drn_inc, final_drn_vs_ref, final_drn_inc, worst_idx, ntw_worst, final_ntw_abs
    """
    if times is None:
        raise ValueError("You must provide `times`.")

    import ssapy
    from ..ssapy_wrappers.accel_ladder import ssapy_accel_ladder
    from ssapy.compute import rv as ssapy_rv
    from ssapy.propagator import SciPyPropagator

    # Orbit construction if needed
    if orbit is None:
        if r is None or v is None or t0 is None:
            raise ValueError("Provide either `orbit` OR (`r`, `v`, `t0`).")
        orbit = ssapy.Orbit(r=np.asarray(r, dtype=float), v=np.asarray(v, dtype=float), t=t0)

    r0 = getattr(orbit, "r", None)
    v0 = getattr(orbit, "v", None)
    if r0 is None or v0 is None:
        raise ValueError("Orbit object is missing `.r` and/or `.v` needed for initial elements.")

    # Initial elements
    oe = _keplerian_elements_from_rv(r0, v0, mu_m3s2=mu_m3s2)
    oe_text = _format_oe_text(oe)

    # Ladder models
    if max_rungs is not None:
        max_rungs = int(max_rungs)
        if max_rungs < 2:
            raise ValueError("max_rungs must be at least 2 when provided.")
        if max_rungs <= 3:
            ladder = _small_accel_ladder(max_rungs)
        else:
            ladder = dict(list(ssapy_accel_ladder().items())[:max_rungs])
    else:
        ladder = ssapy_accel_ladder()
    labels = list(ladder.keys())

    if reference is None:
        reference = max(0, len(labels) - 1)
    reference = int(reference)
    if reference < 0 or reference >= len(labels):
        raise ValueError("reference index out of range.")

    # Time coercion
    epoch_gps = _orbit_epoch_gps(orbit)
    times_ssapy = _coerce_times_for_ssapy(times, epoch_gps, assume=assume_times)
    t_rel_s_full = _times_to_relative_seconds(times_ssapy)

    # Propagate each rung
    r_list = []
    v_list = []
    for name in labels:
        accel = ladder[name]
        prop = SciPyPropagator(accel, ode_kwargs=None if ode_kwargs is None else dict(ode_kwargs))
        r_hist, v_hist = ssapy_rv(orbit, times_ssapy, prop)

        r_hist = np.asarray(r_hist, dtype=float)
        v_hist = np.asarray(v_hist, dtype=float)

        if r_hist.ndim == 3:
            r_hist = r_hist[0]
            v_hist = v_hist[0]

        r_list.append(r_hist)
        v_list.append(v_hist)

    # Early-stop alignment (common prefix)
    lengths = np.array([int(np.asarray(rh).shape[0]) for rh in r_list], dtype=int)
    if np.any(lengths <= 0):
        raise ValueError("One or more rungs returned empty histories (length 0).")

    common_len = int(np.min(lengths))

    r_list = [np.asarray(rh, dtype=float)[:common_len] for rh in r_list]
    v_list = [np.asarray(vh, dtype=float)[:common_len] for vh in v_list]
    t_rel_s = np.asarray(t_rel_s_full, dtype=float)[:common_len]

    stop_idx = lengths - 1
    stop_idx_clip = np.clip(stop_idx, 0, t_rel_s_full.size - 1)
    stop_t_s = np.asarray(t_rel_s_full, dtype=float)[stop_idx_clip]

    # Divergence metrics
    n_models = len(r_list)
    r_ref = r_list[reference]
    v_ref = v_list[reference]

    drn_vs_ref = np.zeros((n_models, t_rel_s.size), dtype=float)
    for i in range(n_models):
        drn_vs_ref[i] = np.linalg.norm(r_list[i] - r_ref, axis=-1)

    drn_inc = np.zeros_like(drn_vs_ref)
    for i in range(1, n_models):
        drn_inc[i] = np.linalg.norm(r_list[i] - r_list[i - 1], axis=-1)

    final_drn_vs_ref = drn_vs_ref[:, -1]
    final_drn_inc = drn_inc[:, -1]

    idx_candidates = np.arange(n_models)
    idx_candidates = idx_candidates[idx_candidates != reference]
    worst_idx = int(idx_candidates[np.argmax(final_drn_vs_ref[idx_candidates])]) if idx_candidates.size else reference

    ntw_worst = _ntw_components(r_ref, v_ref, (r_list[worst_idx] - r_ref))

    final_ntw_abs = np.zeros((n_models, 3), dtype=float)
    for i in range(n_models):
        ntw_i = _ntw_components(r_ref, v_ref, (r_list[i] - r_ref))
        final_ntw_abs[i] = np.abs(ntw_i[-1])

    return {
        # Core ladder outputs
        "labels": labels,
        "reference": reference,
        "times_ssapy": times_ssapy,
        "t_rel_s": t_rel_s,
        "t_rel_s_full": t_rel_s_full,
        "r_list": r_list,
        "v_list": v_list,
        # Early-stop metadata
        "stop_idx": stop_idx,
        "stop_t_s": stop_t_s,
        "common_len": common_len,
        # Orbit elements
        "orbit_elements": oe,
        "orbit_elements_text": oe_text,
        # Divergence products (for any plot)
        "drn_vs_ref": drn_vs_ref,
        "drn_inc": drn_inc,
        "final_drn_vs_ref": final_drn_vs_ref,
        "final_drn_inc": final_drn_inc,
        "worst_idx": worst_idx,
        "ntw_worst": ntw_worst,
        "final_ntw_abs": final_ntw_abs,
    }


# =============================================================================
# Plotting (plot-only)
# =============================================================================
def _nice_vivid_colors(n):
    import matplotlib.pyplot as plt

    if n <= 10:
        cmap = plt.get_cmap("tab10")
        return cmap(np.arange(n))
    if n <= 20:
        cmap = plt.get_cmap("tab20")
        return cmap(np.arange(n))

    hsv = plt.get_cmap("hsv")
    return hsv(np.linspace(0.0, 1.0, n, endpoint=False))


def _draw_ladder_key_two_columns(
    fig,
    labels,
    colors,
    header_x,
    header_y,
    x_left,
    x_right,
    y_top,
    line_height,
    idx_dx=0.045,
    header_fontsize=14,
    idx_fontsize=12,
    label_fontsize=11,
    title_gap=0.030,
):
    labels = list(labels)
    n = len(labels)
    half = int(np.ceil(n / 2.0))

    fig.text(
        header_x,
        header_y,
        "Accel ladder key (rung index -> model)",
        ha="left",
        va="top",
        fontsize=header_fontsize,
    )

    def _draw_column(start_i, end_i, x0):
        for row, i in enumerate(range(start_i, end_i)):
            y = (y_top - title_gap) - row * line_height
            idx_txt = f"{i:2d}: "
            fig.text(x0, y, idx_txt, ha="left", va="top", fontsize=idx_fontsize, color=colors[i])
            fig.text(x0 + idx_dx, y, str(labels[i]), ha="left", va="top", fontsize=label_fontsize, color="black")

    _draw_column(0, half, x_left)
    _draw_column(half, n, x_right)


def make_accel_ladder_dashboard_figures(
    calc,
    plot_title="SSAPy accel ladder divergences",
    show_legend=True,
    epsilon_m=1e-3,
    max_error_m=1e7,
):
    """
    Plot-only: builds the two dashboard figures from calculate_accel_comparisons() output.
    """
    import matplotlib.pyplot as plt

    labels = calc["labels"]
    reference = int(calc["reference"])
    worst_idx = int(calc["worst_idx"])
    t_rel_s = np.asarray(calc["t_rel_s"], dtype=float).ravel()

    drn_vs_ref = np.asarray(calc["drn_vs_ref"], dtype=float)
    drn_inc = np.asarray(calc["drn_inc"], dtype=float)
    final_drn_vs_ref = np.asarray(calc["final_drn_vs_ref"], dtype=float)
    final_drn_inc = np.asarray(calc["final_drn_inc"], dtype=float)
    ntw_worst = np.asarray(calc["ntw_worst"], dtype=float)
    orbit_elements_text = calc.get("orbit_elements_text", None)

    n_models = len(labels)
    colors = _nice_vivid_colors(n_models)

    def _alt_ls(i):
        return "--" if (i % 2 == 1) else "-"

    oe_bbox = dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.90)

    # -------------------------
    # Figure 1: time-domain view
    # -------------------------
    fig1, ax = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)
    fig1.suptitle(plot_title, fontsize=16)

    for a in ax:
        a.tick_params(labelsize=11)

    ax[0].set_title("Divergence vs reference rung", fontsize=13)
    for i in range(n_models):
        if i == reference:
            continue
        y = np.maximum(drn_vs_ref[i], float(epsilon_m))
        ax[0].plot(t_rel_s, y, label=labels[i], color=colors[i], linestyle=_alt_ls(i))
    ax[0].set_yscale("log")
    ax[0].set_ylabel("||dr|| vs ref (m)", fontsize=12)
    ax[0].grid(True, which="both", alpha=0.3)
    if show_legend:
        ax[0].legend(fontsize=8, ncol=2, loc="upper left")

    if orbit_elements_text:
        ax[0].text(
            0.98,
            0.98,
            str(orbit_elements_text),
            transform=ax[0].transAxes,
            ha="right",
            va="top",
            fontsize=11,
            bbox=oe_bbox,
        )

    ax[1].set_title("Incremental effect per rung (rung i minus rung i-1)", fontsize=13)
    for i in range(1, n_models):
        y = np.maximum(drn_inc[i], float(epsilon_m))
        ax[1].plot(t_rel_s, y, label=labels[i], color=colors[i], linestyle=_alt_ls(i))
    ax[1].set_yscale("log")
    ax[1].set_ylabel("||dr_inc|| (m)", fontsize=12)
    ax[1].grid(True, which="both", alpha=0.3)

    ax[2].set_title("Worst rung NTW components (worst = %s)" % labels[worst_idx], fontsize=13)
    ax[2].plot(t_rel_s, ntw_worst[:, 0], label="N")
    ax[2].plot(t_rel_s, ntw_worst[:, 1], label="T")
    ax[2].plot(t_rel_s, ntw_worst[:, 2], label="W")
    ax[2].set_xlabel("t (s)", fontsize=12)
    ax[2].set_ylabel("NTW(dr) (m)", fontsize=12)
    ax[2].grid(True, alpha=0.3)
    if show_legend:
        ax[2].legend(fontsize=10, loc="upper left")

    ax[0].tick_params(axis="x", labelbottom=False)
    ax[1].tick_params(axis="x", labelbottom=False)

    # -------------------------
    # Figure 2: rung-summary + heatmap + compact header
    # -------------------------
    idx = np.arange(n_models, dtype=int)
    fig2, ax2 = plt.subplots(3, 1, figsize=(13, 12))

    rows = int(np.ceil(n_models / 2.0))
    target_line_h = 0.021
    header_pad = 0.060
    header_height = header_pad + rows * target_line_h
    top_axes = float(np.clip(0.985 - header_height, 0.62, 0.86))

    fig2.subplots_adjust(left=0.10, right=0.98, top=top_axes, bottom=0.10, hspace=0.45)

    for a in ax2:
        a.tick_params(labelsize=11)

    ax2[0].set_title("Final divergence vs reference by rung", fontsize=13, pad=10)
    y0 = np.maximum(final_drn_vs_ref, float(epsilon_m))
    ax2[0].set_yscale("log")
    ax2[0].scatter(idx, y0, s=55, c=colors, edgecolors="black", linewidths=0.5)
    ax2[0].set_ylabel("final ||dr|| (m)", fontsize=12)
    ax2[0].grid(True, which="both", alpha=0.3)
    for i in range(n_models):
        ax2[0].annotate(
            str(i),
            xy=(idx[i], y0[i]),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color=colors[i],
        )
    ax2[0].tick_params(axis="x", labelbottom=False)

    ax2[1].set_title("Final incremental effect by rung (rung i minus rung i-1)", fontsize=13, pad=10)
    y1 = np.maximum(final_drn_inc, float(epsilon_m))
    ax2[1].set_yscale("log")
    ax2[1].scatter(idx, y1, s=55, c=colors, edgecolors="black", linewidths=0.5)
    ax2[1].set_ylabel("final ||dr_inc|| (m)", fontsize=12)
    ax2[1].grid(True, which="both", alpha=0.3)
    for i in range(n_models):
        ax2[1].annotate(
            str(i),
            xy=(idx[i], y1[i]),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color=colors[i],
        )
    ax2[1].tick_params(axis="x", labelbottom=False)

    ax2[2].set_title("log10(||dr|| vs ref) heatmap", fontsize=13, pad=12)
    log_drn = np.log10(np.maximum(drn_vs_ref, float(epsilon_m)))
    vmin = float(np.log10(float(epsilon_m)))
    vmax = float(np.log10(float(max_error_m)))

    extent = [float(t_rel_s[0]), float(t_rel_s[-1]), -0.5, float(n_models - 0.5)]
    im = ax2[2].imshow(
        log_drn,
        aspect="auto",
        origin="lower",
        extent=extent,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    ax2[2].set_xlabel("t (s)", fontsize=12)
    ax2[2].set_ylabel("rung index", fontsize=12)

    step = int(max(1, np.ceil(n_models / 25.0)))
    y_ticks = np.arange(0, n_models, step, dtype=int)
    ax2[2].set_yticks(y_ticks)
    ax2[2].set_yticklabels([str(i) for i in y_ticks])
    for tick_text, i in zip(ax2[2].get_yticklabels(), y_ticks):
        tick_text.set_color(colors[int(i)])
        tick_text.set_fontsize(10)

    cbar = fig2.colorbar(im, ax=ax2[2], pad=0.02)
    cbar.set_label("log10(||dr||) [m]", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    # Header: key LEFT, OE RIGHT
    header_bottom = top_axes + 0.004
    y_top = 0.975
    available = max(0.04, y_top - header_bottom)
    line_height = min(target_line_h, available / float(max(1, rows)))

    _draw_ladder_key_two_columns(
        fig2,
        labels=labels,
        colors=colors,
        header_x=0.10,
        header_y=0.985,
        x_left=0.10,
        x_right=0.34,
        y_top=y_top,
        line_height=line_height,
        idx_dx=0.045,
        header_fontsize=14,
        idx_fontsize=12,
        label_fontsize=11,
        title_gap=0.030,
    )

    if orbit_elements_text:
        fig2.text(
            0.98,
            0.985,
            str(orbit_elements_text),
            ha="right",
            va="top",
            fontsize=12,
            bbox=oe_bbox,
        )

    return {
        "figure_time_domain": fig1,
        "figure_rung_summary": fig2,
        "figures": [fig1, fig2],
    }


# =============================================================================
# compare_models: calls calculation + plot (non-optional)
# =============================================================================
def compare_models(
    orbit=None,
    r=None,
    v=None,
    t0=None,
    times=None,
    assume_times="auto",
    ode_kwargs=None,
    reference=None,
    plot_title="SSAPy accel ladder divergences",
    show_legend=True,
    epsilon_m=1e-3,
    max_error_m=1e7,
    mu_m3s2=EARTH_MU,
    max_rungs=None,
):
    """
    Workflow wrapper:
      - always computes
      - always plots
      - only calls:
          1) calculate_accel_comparisons()
          2) make_accel_ladder_dashboard_figures()
    """
    calc = calculate_accel_comparisons(
        orbit=orbit,
        r=r,
        v=v,
        t0=t0,
        times=times,
        assume_times=assume_times,
        ode_kwargs=ode_kwargs,
        reference=reference,
        mu_m3s2=mu_m3s2,
        max_rungs=max_rungs,
    )

    figs = make_accel_ladder_dashboard_figures(
        calc=calc,
        plot_title=plot_title,
        show_legend=show_legend,
        epsilon_m=epsilon_m,
        max_error_m=max_error_m,
    )

    dashboard = {"calc": calc}
    dashboard.update(figs)

    return {
        "labels": calc["labels"],
        "times_ssapy": calc["times_ssapy"],
        "t_rel_s": calc["t_rel_s"],
        "t_rel_s_full": calc["t_rel_s_full"],
        "r_list": calc["r_list"],
        "v_list": calc["v_list"],
        "orbit_elements": calc["orbit_elements"],
        "orbit_elements_text": calc["orbit_elements_text"],
        "stop_idx": calc["stop_idx"],
        "stop_t_s": calc["stop_t_s"],
        "common_len": calc["common_len"],
        "dashboard": dashboard,
    }
