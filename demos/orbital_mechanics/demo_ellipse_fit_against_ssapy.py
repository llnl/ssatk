#!/usr/bin/env python3
"""
Demo: ellipse_fit vs multiple SSAPy propagations using ssatk props.

Pytest-safe mode:
- smaller sample count
- plotting disabled by default
"""

import os
import sys
import inspect

import numpy as np
import matplotlib.pyplot as plt

from ssatk.constants import RGEO
from ssatk.orbital_mechanics.ellipse_fit import ellipse_fit
from ssatk.plots.figpath import figpath
from ssatk.io.pprint_utils import pprint
from ssatk.plots.orbit_plot import orbit_plot
from ssatk.plots.groundtrack_dashboard import groundtrack_dashboard
from ssatk.ssapy_wrappers.ssapy_orbits import ssapy_orbit
from ssatk.ssapy_wrappers.ssapy_props import keplerian_prop, ssapy_prop, best_prop

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


def resample_cartesian(time_src_s, r_src, time_dst_s):
    time_src_s = np.asarray(time_src_s, dtype=float)
    r_src = np.asarray(r_src, dtype=float)
    time_dst_s = np.asarray(time_dst_s, dtype=float)

    if time_src_s.ndim != 1 or time_dst_s.ndim != 1:
        raise ValueError("time arrays must be 1-D")
    if r_src.ndim != 2 or r_src.shape[1] != 3 or r_src.shape[0] != time_src_s.shape[0]:
        raise ValueError("r_src must be shape (N,3) matching time_src_s length")
    if np.any(np.diff(time_src_s) < 0.0):
        raise ValueError("time_src_s must be ascending")

    r_dst = np.empty((time_dst_s.shape[0], 3), dtype=float)
    for k in range(3):
        r_dst[:, k] = np.interp(time_dst_s, time_src_s, r_src[:, k])
    return r_dst


def trim_or_pad_to(arr, n_samples):
    arr = np.asarray(arr, dtype=float)
    if arr.shape[0] == n_samples:
        return arr
    if arr.shape[0] > n_samples:
        return arr[:n_samples]
    pad = np.repeat(arr[-1][None, :], n_samples - arr.shape[0], axis=0)
    return np.vstack([arr, pad])


def _supports_kwarg(func, kw):
    try:
        sig = inspect.signature(func)
        if kw in sig.parameters:
            return True
        for p in sig.parameters.values():
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                return True
    except Exception:
        pass
    return False


def main(make_figures=None, fast=None, verbose=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST
    if verbose is None:
        verbose = not UNDER_PYTEST

    P1 = np.array([RGEO, 0.0, 0.0], dtype=float)
    P2 = np.array([0.0, -1.1 * RGEO, 0.1 * RGEO], dtype=float)

    n_pts = 120 if fast else 400

    # This installed ellipse_fit version does not accept ccw=...
    res = ellipse_fit(P1, P2, n_pts=n_pts, plot=False, inc=0.0)

    if verbose:
        pprint(res)

    r_arc = np.asarray(res["r"], dtype=float)
    v_arc = np.asarray(res["v"], dtype=float)
    t_rel = np.asarray(res["t_rel"], dtype=float)

    r0 = np.asarray(res["r0"], dtype=float)
    v0 = np.asarray(res["v0"], dtype=float)
    r2 = r_arc[-1].copy()
    v2 = v_arc[-1].copy()

    if t_rel[0] != 0.0:
        t_rel = t_rel - t_rel[0]

    T_flight = float(t_rel[-1])
    N_native = int(t_rel.shape[0])

    if N_native > 1:
        t_grid = np.linspace(0.0, T_flight, N_native, dtype=float)
        dt_s = T_flight / float(N_native - 1)
    else:
        t_grid = np.array([0.0], dtype=float)
        dt_s = 0.0

    r_efit = resample_cartesian(t_rel, r_arc, t_grid)

    prop_kep = keplerian_prop()
    prop_nom = ssapy_prop()
    prop_best = best_prop()

    def run_with_prop(prop):
        r, v, t = ssapy_orbit(
            r=r0,
            v=v0,
            duration=(T_flight, "s"),
            freq=((dt_s, "s") if N_native > 1 else (0.0, "s")),
            prop=prop,
        )
        return trim_or_pad_to(r, N_native), t

    r_kep, t_kep = run_with_prop(prop_kep)
    r_nom, t_nom = run_with_prop(prop_nom)
    r_best, t_best = run_with_prop(prop_best)

    r_back, _, t_back = ssapy_orbit(
        r=r2,
        v=-v2,
        duration=(T_flight, "s"),
        freq=((dt_s, "s") if N_native > 1 else (0.0, "s")),
        prop=prop_nom,
    )
    r_back = trim_or_pad_to(r_back, N_native)

    t_fit = t_kep
    t_fit_desc = t_back[::-1] if hasattr(t_back, "__getitem__") else t_fit

    r_list = [r_efit, r_kep, r_nom, r_best, r_back]
    t_list = [t_fit, t_kep, t_nom, t_best, t_fit_desc]
    labels = [
        "ellipse_fit",
        "keplerian_prop",
        "ssapy_prop",
        "best_prop",
        "backward(ssapy_prop)",
    ]

    if make_figures:
        save_path = figpath("testing_ellipse_fit_vs_ssapy.jpg")
        if _supports_kwarg(orbit_plot, "labels"):
            orbit_plot(
                r_list,
                t_list,
                title="Ellipse-fit vs SSAPy (keplerian, ssapy_prop, best_prop, backward)",
                save_path=save_path,
                frame="gcrf",
                labels=labels,
            )
        else:
            orbit_plot(
                r_list,
                t_list,
                title="Ellipse-fit vs SSAPy (keplerian, ssapy_prop, best_prop, backward)",
                save_path=save_path,
                frame="gcrf",
            )

        save_dash = figpath("testing_ellipse_fit_vs_ssapy_dashboard.jpg")
        if _supports_kwarg(groundtrack_dashboard, "labels"):
            groundtrack_dashboard(
                r_list,
                t_list,
                save_path=save_dash,
                labels=labels,
            )
        else:
            groundtrack_dashboard(
                r_list,
                t_list,
                save_path=save_dash,
            )

        t_minutes = t_grid / 60.0
        curves = [
            np.linalg.norm(r_kep - r_efit, axis=1) / 1e3,
            np.linalg.norm(r_nom - r_efit, axis=1) / 1e3,
            np.linalg.norm(r_best - r_efit, axis=1) / 1e3,
            np.linalg.norm(r_back - r_efit[::-1], axis=1) / 1e3,
        ]
        names = [
            "keplerian_prop vs ellipse_fit",
            "ssapy_prop vs ellipse_fit",
            "best_prop vs ellipse_fit",
            "backward(ssapy_prop) vs reverse(ellipse_fit)",
        ]

        dash_styles = [
            (0, (1, 1)),
            (0, (4, 2)),
            (0, (8, 2)),
            (0, (4, 1, 1, 1)),
        ]

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        for i, (d, n) in enumerate(zip(curves, names)):
            ls = dash_styles[i % len(dash_styles)]
            ax.plot(t_minutes, d, linewidth=2.5, linestyle=ls, label=n)

        ax.set_xlabel("Time since start (minutes)")
        ax.set_ylabel("Distance to reference (km)")
        ax.set_title("Distance to ellipse_fit (or reversed) vs time")
        ax.grid(True, alpha=0.3)
        ax.legend()

        save_dist = figpath("testing_ellipse_fit_distance.jpg")
        fig.savefig(save_dist, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return {
        "ellipse_fit": r_efit,
        "keplerian_prop": r_kep,
        "ssapy_prop": r_nom,
        "best_prop": r_best,
        "backward": r_back,
        "result": res,
    }


if __name__ == "__main__":
    main(make_figures=True, fast=False, verbose=True)
