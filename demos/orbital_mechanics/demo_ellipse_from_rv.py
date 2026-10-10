#!/usr/bin/env python3
"""
Demo: rigorous validation driver for ssatk ellipse fitting.

Baseline:
  - Runs ellipse_fit twice (A/B) using v_pref to pick direction
  - Validates shapes, finiteness, required keys
  - Checks endpoints: r[0]≈P1, r[-1]≈P2
  - Checks invariants from returned r,v
  - Reconstructs with SSAPy (SV + Kepler) and checks internal consistency
  - Saves plots when SAVE_PLOTS=True

Inclination sweep:
  - Start: equatorial GEO position at +X
  - End: point on a sphere of radius 2*RGEO
  - For each inclination:
      P2(inc) = R2 * [0, cos(inc), sin(inc)]
  - Computes transfer arcs for A/B at each inclination
  - Does not compare to SSAPy in the sweep
  - Failing sweep cases are logged and skipped, not fatal
"""

import os
import sys
import traceback
import numpy as np

from ssatk.constants import RGEO
from ssatk.orbital_mechanics.ellipse_fit import ellipse_fit
from ssatk.ssapy_wrappers.ssapy_orbits import ssapy_orbit
from ssatk.plots.orbit_plot import orbit_plot
from ssatk.plots.plotutils import figsave

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _norm(x, axis=None):
    return np.linalg.norm(x, axis=axis)


def _unit(x, eps=1e-15):
    x = np.asarray(x, float)
    n = float(np.linalg.norm(x))
    if n < eps:
        return None
    return x / n


def _finite(x, name):
    x = np.asarray(x)
    if not np.all(np.isfinite(x)):
        bad = np.where(~np.isfinite(x))
        raise AssertionError(f"Non-finite values in {name} at indices {bad}")


def _assert_keys(d, keys):
    missing = [k for k in keys if k not in d]
    if missing:
        raise AssertionError(f"Missing keys from ellipse_fit result: {missing}")


def _assert_close(a, b, atol, msg):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    err = float(np.linalg.norm(a - b))
    if err > float(atol):
        raise AssertionError(f"{msg} |err|={err:.6e} > atol={float(atol):.6e}")
    return err


def _rel_std(x, eps=1e-30):
    x = np.asarray(x, float)
    return float(np.std(x) / max(float(np.mean(np.abs(x))), eps))


def _invariants_from_rv(mu, r_m, v_m_s):
    r_m = np.asarray(r_m, float)
    v_m_s = np.asarray(v_m_s, float)

    h_vec = np.cross(r_m, v_m_s)
    h_mag = _norm(h_vec, axis=1)

    rmag = _norm(r_m, axis=1)
    vmag = _norm(v_m_s, axis=1)

    energy = 0.5 * (vmag ** 2) - float(mu) / rmag

    e_vec = (np.cross(v_m_s, h_vec) / float(mu)) - (r_m / rmag[:, None])
    e_mag = _norm(e_vec, axis=1)

    return {
        "h_mag": h_mag,
        "energy": energy,
        "e_mag": e_mag,
        "h_rel_std": _rel_std(h_mag),
        "E_rel_std": _rel_std(energy),
        "e_rel_std": _rel_std(e_mag),
        "h_mean": float(np.mean(h_mag)),
        "E_mean": float(np.mean(energy)),
        "e_mean": float(np.mean(e_mag)),
    }


def _cos_sim(a, b, eps=1e-15):
    ah = _unit(a, eps=eps)
    bh = _unit(b, eps=eps)
    if ah is None or bh is None:
        return -np.inf
    return float(np.dot(ah, bh))


def _compute_t_hat(P1_m, P2_m):
    """
    Approximate tangential direction at P1 using:
      w_hat ~ unit(P1 x P2)
      t_hat = unit(w_hat x r_hat) at P1
    Robust fallback if nearly colinear.
    """
    P1_m = np.asarray(P1_m, float)
    P2_m = np.asarray(P2_m, float)

    r_hat = _unit(P1_m)
    if r_hat is None:
        raise AssertionError("P1 has near-zero norm; cannot form tangential direction.")

    w_hat = _unit(np.cross(P1_m, P2_m))
    if w_hat is None:
        cand = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(r_hat, cand))) > 0.99:
            cand = np.array([1.0, 0.0, 0.0])
        w_hat = _unit(np.cross(r_hat, cand))
        if w_hat is None:
            raise AssertionError("Could not form stable plane normal for tangential direction.")

    t_hat = _unit(np.cross(w_hat, r_hat))
    if t_hat is None:
        raise AssertionError("Could not form tangential unit vector at P1.")
    return t_hat


def _pos_err(r_ref_m, r_test_m, label=""):
    r_ref_m = np.asarray(r_ref_m, float)
    r_test_m = np.asarray(r_test_m, float)
    if r_ref_m.shape != r_test_m.shape:
        raise AssertionError(f"{label}shape mismatch: {r_ref_m.shape} vs {r_test_m.shape}")
    err = _norm(r_ref_m - r_test_m, axis=1)
    rms = float(np.sqrt(np.mean(err ** 2)))
    mx = float(np.max(err))
    print(f"{label}position error: RMS={rms:.6e} m, max={mx:.6e} m")
    return rms, mx, err


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
REQUIRED_KEYS = [
    "r", "v", "t_rel", "t_abs",
    "a", "e", "i", "raan", "pa", "ta", "L",
    "rp", "ra", "rp_alt", "ra_alt", "b", "p",
    "mean_motion", "eta", "period",
    "h_vec", "h", "Energy", "e_vec", "n_vec",
    "r0", "v0", "F2",
    "plane_basis", "rot_dir",
    "mu",
]

ENDPOINT_ATOL_M = 5.0e-3
TIME_MONO_TOL_S = 1.0e-9

INV_REL_STD_H = 2.0e-9
INV_REL_STD_E = 2.0e-9
INV_REL_STD_e = 2.0e-9

SSAPY_INTERNAL_RMS_M = 1.0e-3
SSAPY_INTERNAL_MAX_M = 1.0e-2

SSAPY_FIT_RMS_M = 2.0e4
SSAPY_FIT_MAX_M = 5.0e4

DIR_COS_MIN = 0.90

SAVE_PLOTS = not UNDER_PYTEST

INC_SWEEP_DEG = np.arange(0.0, 90.0 + 1e-9, 15.0)
R1_SWEEP = 1.0 * RGEO
R2_SWEEP = 2.0 * RGEO


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def _cube_limits_from_points(pts3d, pad=0.05):
    pts = np.asarray(pts3d, float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        return None

    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    spans = maxs - mins
    half = 0.5 * float(np.max(spans))
    if not np.isfinite(half) or half <= 0.0:
        half = 1.0
    half *= (1.0 + float(pad))
    return center, half


def _apply_cube_limits(ax, cube_limits):
    if cube_limits is None:
        return
    center, half = cube_limits
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def _set_equal_3d_cube(ax, pts3d, pad=0.05):
    _apply_cube_limits(ax, _cube_limits_from_points(pts3d, pad=pad))


def _plot_error_vs_time(results, recon_cache, prefix):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for name in ["A", "B"]:
        t = np.asarray(results[name]["t_rel"], float)
        r_fit = np.asarray(results[name]["r"], float)
        r_ke = np.asarray(recon_cache[name]["ke"][0], float)
        ax.plot(t, _norm(r_fit - r_ke, axis=1), label=f"Kepler err {name}")
    ax.set_xlabel("t_rel [s]")
    ax.set_ylabel("|r_fit - r_ssapy| [m]")
    ax.set_title("ellipse_fit vs SSAPy position error (Kepler reconstruction)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    figsave(fig, f"{prefix}_error_kepler.png")
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for name in ["A", "B"]:
        t = np.asarray(results[name]["t_rel"], float)
        r_fit = np.asarray(results[name]["r"], float)
        r_sv = np.asarray(recon_cache[name]["sv"][0], float)
        ax.plot(t, _norm(r_fit - r_sv, axis=1), label=f"SV err {name}")
    ax.set_xlabel("t_rel [s]")
    ax.set_ylabel("|r_fit - r_ssapy| [m]")
    ax.set_title("ellipse_fit vs SSAPy position error (SV reconstruction)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    figsave(fig, f"{prefix}_error_sv.png")
    plt.close(fig)


def _plot_radius_and_speed(results, recon_cache, prefix):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for name in ["A", "B"]:
        t = np.asarray(results[name]["t_rel"], float)
        r_fit = np.asarray(results[name]["r"], float)
        r_ke = np.asarray(recon_cache[name]["ke"][0], float)
        ax.plot(t, _norm(r_fit, axis=1), label=f"|r| fit {name}")
        ax.plot(t, _norm(r_ke, axis=1), label=f"|r| ssapy {name}")
    ax.set_xlabel("t_rel [s]")
    ax.set_ylabel("|r| [m]")
    ax.set_title("Radius vs time (ellipse_fit vs SSAPy)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    figsave(fig, f"{prefix}_radius.png")
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for name in ["A", "B"]:
        t = np.asarray(results[name]["t_rel"], float)
        v_fit = np.asarray(results[name]["v"], float)
        v_ke = np.asarray(recon_cache[name]["ke"][1], float)
        ax.plot(t, _norm(v_fit, axis=1), label=f"|v| fit {name}")
        ax.plot(t, _norm(v_ke, axis=1), label=f"|v| ssapy {name}")
    ax.set_xlabel("t_rel [s]")
    ax.set_ylabel("|v| [m/s]")
    ax.set_title("Speed vs time (ellipse_fit vs SSAPy)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    figsave(fig, f"{prefix}_speed.png")
    plt.close(fig)


def _plot_arcs_3d(results, P1_m, P2_m, prefix, title="ellipse_fit arcs", cube_limits=None):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    all_pts = [np.asarray(P1_m, float), np.asarray(P2_m, float)]

    for name in results:
        r = np.asarray(results[name]["r"], float)
        ax.plot(r[:, 0], r[:, 1], r[:, 2], label=f"fit {name}")
        all_pts.append(r)

    ax.scatter([P1_m[0]], [P1_m[1]], [P1_m[2]], s=40, label="P1")
    ax.scatter([P2_m[0]], [P2_m[1]], [P2_m[2]], s=40, label="P2")

    pts = np.vstack([p.reshape(-1, 3) for p in all_pts])
    if cube_limits is None:
        _set_equal_3d_cube(ax, pts, pad=0.08)
    else:
        _apply_cube_limits(ax, cube_limits)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    figsave(fig, f"{prefix}_arcs.png")
    plt.close(fig)


def _plot_sep_AB(results, prefix):
    import matplotlib.pyplot as plt

    rA = np.asarray(results["A"]["r"], float)
    rB = np.asarray(results["B"]["r"], float)
    if rA.shape != rB.shape:
        return
    sep = _norm(rA - rB, axis=1)
    t = np.asarray(results["A"]["t_rel"], float)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t, sep)
    ax.set_xlabel("t_rel [s]")
    ax.set_ylabel("|r_A - r_B| [m]")
    ax.set_title("A vs B separation vs time")
    ax.grid(True)
    fig.tight_layout()
    figsave(fig, f"{prefix}_separation_AB.png")
    plt.close(fig)


def _plot_recon_overlays(results, recon_cache, prefix):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    all_pts = []
    for name in ["A", "B"]:
        r_fit = np.asarray(results[name]["r"], float)
        r_sv = np.asarray(recon_cache[name]["sv"][0], float)
        ax.plot(r_fit[:, 0], r_fit[:, 1], r_fit[:, 2], label=f"fit {name}")
        ax.plot(r_sv[:, 0], r_sv[:, 1], r_sv[:, 2], alpha=0.7, label=f"sv {name}")
        all_pts.extend([r_fit, r_sv])
    pts = np.vstack([p.reshape(-1, 3) for p in all_pts])
    _set_equal_3d_cube(ax, pts, pad=0.08)
    ax.set_title("SSAPy reconstruction via state vectors")
    ax.legend()
    fig.tight_layout()
    figsave(fig, f"{prefix}_recons_sv.png")
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    all_pts = []
    for name in ["A", "B"]:
        r_fit = np.asarray(results[name]["r"], float)
        r_ke = np.asarray(recon_cache[name]["ke"][0], float)
        ax.plot(r_fit[:, 0], r_fit[:, 1], r_fit[:, 2], label=f"fit {name}")
        ax.plot(r_ke[:, 0], r_ke[:, 1], r_ke[:, 2], alpha=0.7, label=f"ke {name}")
        all_pts.extend([r_fit, r_ke])
    pts = np.vstack([p.reshape(-1, 3) for p in all_pts])
    _set_equal_3d_cube(ax, pts, pad=0.08)
    ax.set_title("SSAPy reconstruction via Kepler elements")
    ax.legend()
    fig.tight_layout()
    figsave(fig, f"{prefix}_recons_kepler.png")
    plt.close(fig)


def _plot_incsweep_all_transfers_3d(sweep_store, P1_m, prefix):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    labeledA = False
    labeledB = False
    all_pts = [np.asarray(P1_m, float)]

    for item in sweep_store:
        all_pts.append(np.asarray(item["P2"], float))

        if item["A"] is not None:
            rA = np.asarray(item["A"]["r"], float)
            if not labeledA:
                ax.plot(rA[:, 0], rA[:, 1], rA[:, 2], alpha=0.25, linewidth=1.0, label="A (all inc)")
                labeledA = True
            else:
                ax.plot(rA[:, 0], rA[:, 1], rA[:, 2], alpha=0.25, linewidth=1.0)
            all_pts.append(rA)

        if item["B"] is not None:
            rB = np.asarray(item["B"]["r"], float)
            if not labeledB:
                ax.plot(rB[:, 0], rB[:, 1], rB[:, 2], alpha=0.25, linewidth=1.0, label="B (all inc)")
                labeledB = True
            else:
                ax.plot(rB[:, 0], rB[:, 1], rB[:, 2], alpha=0.25, linewidth=1.0)
            all_pts.append(rB)

    ax.scatter([P1_m[0]], [P1_m[1]], [P1_m[2]], s=45, label="P1 (start)")
    P2s = np.array([it["P2"] for it in sweep_store], float)
    ax.scatter(P2s[:, 0], P2s[:, 1], P2s[:, 2], s=18, alpha=0.9, label="P2 (ends)")

    pts = np.vstack([p.reshape(-1, 3) for p in all_pts])
    _set_equal_3d_cube(ax, pts, pad=0.08)

    ax.set_title("Inclination sweep transfers: 1*RGEO equatorial -> 2*RGEO sphere (0..90° step 15°)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend()
    fig.tight_layout()
    figsave(fig, f"{prefix}_all_transfers.png")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Core runner
# -----------------------------------------------------------------------------
def _run_one_case(P1_m, P2_m, inc_rad, v_pref_m_s, tag, dir_cos_min=DIR_COS_MIN, compare_ssapy=True):
    """
    If compare_ssapy is False, SSAPy reconstruction and comparisons are skipped.
    """
    res = ellipse_fit(
        P1_m,
        P2_m,
        n_pts=400,
        plot=False,
        inc=float(inc_rad),
        v_pref_m_s=v_pref_m_s,
    )

    _assert_keys(res, REQUIRED_KEYS)

    r = np.asarray(res["r"], float)
    v = np.asarray(res["v"], float)
    t_rel = np.asarray(res["t_rel"], float)

    if r.ndim != 2 or r.shape[1] != 3:
        raise AssertionError(f"{tag}: r must be (N,3), got {r.shape}")
    if v.shape != r.shape:
        raise AssertionError(f"{tag}: v must match r shape, got {v.shape}")
    if t_rel.ndim != 1 or t_rel.shape[0] != r.shape[0]:
        raise AssertionError(f"{tag}: t_rel must be (N,), got {t_rel.shape}")

    _finite(r, f"r {tag}")
    _finite(v, f"v {tag}")
    _finite(t_rel, f"t_rel {tag}")

    if np.any(np.diff(t_rel) < -TIME_MONO_TOL_S):
        raise AssertionError(f"{tag}: t_rel is not non-decreasing.")

    d0 = _assert_close(r[0], P1_m, ENDPOINT_ATOL_M, f"{tag}: r[0] != P1")
    d1 = _assert_close(r[-1], P2_m, ENDPOINT_ATOL_M, f"{tag}: r[-1] != P2")
    print(f"{tag}: endpoints |r[0]-P1|={d0:.3e} m, |r[-1]-P2|={d1:.3e} m")

    inv = _invariants_from_rv(float(res["mu"]), r, v)
    print(
        f"{tag}: invariants "
        f"h_rel_std={inv['h_rel_std']:.3e}, "
        f"E_rel_std={inv['E_rel_std']:.3e}, "
        f"e_rel_std={inv['e_rel_std']:.3e}"
    )
    if inv["h_rel_std"] > INV_REL_STD_H:
        raise AssertionError(f"{tag}: |h| rel_std too large: {inv['h_rel_std']:.3e}")
    if inv["E_rel_std"] > INV_REL_STD_E:
        raise AssertionError(f"{tag}: Energy rel_std too large: {inv['E_rel_std']:.3e}")
    if inv["e_rel_std"] > INV_REL_STD_e:
        raise AssertionError(f"{tag}: |e| rel_std too large: {inv['e_rel_std']:.3e}")

    cs = _cos_sim(v[0], v_pref_m_s)
    print(f"{tag}: v_pref cos_sim(v0, v_pref)={cs:.3f}")
    if dir_cos_min is not None and cs < float(dir_cos_min):
        raise AssertionError(f"{tag}: v_pref not respected: cos_sim={cs:.3f} < {float(dir_cos_min):.3f}")

    recon = {"sv": (None, None, None), "ke": (None, None, None)}

    if compare_ssapy:
        r_sv, v_sv, t_sv = ssapy_orbit(
            r=np.asarray(res["r0"], float),
            v=np.asarray(res["v0"], float),
            t=t_rel,
        )
        rms_fit_sv, mx_fit_sv, _ = _pos_err(r, r_sv, label=f"{tag} FIT vs SV: ")

        r_ke, v_ke, t_ke = ssapy_orbit(
            a=float(res["a"]),
            e=float(res["e"]),
            i=float(res["i"]),
            raan=float(res["raan"]),
            pa=float(res["pa"]),
            ta=float(res["ta"]),
            t=t_rel,
        )
        rms_fit_ke, mx_fit_ke, _ = _pos_err(r, r_ke, label=f"{tag} FIT vs KE: ")

        rms_int, mx_int, _ = _pos_err(r_sv, r_ke, label=f"{tag} SSAPY SV vs KE: ")
        if rms_int > SSAPY_INTERNAL_RMS_M or mx_int > SSAPY_INTERNAL_MAX_M:
            raise AssertionError(
                f"{tag}: SSAPy internal mismatch too large (RMS={rms_int:.3e}, max={mx_int:.3e})"
            )

        if rms_fit_sv > SSAPY_FIT_RMS_M or mx_fit_sv > SSAPY_FIT_MAX_M:
            raise AssertionError(
                f"{tag}: Fit vs SSAPy(SV) too far (RMS={rms_fit_sv:.3e}, max={mx_fit_sv:.3e})"
            )
        if rms_fit_ke > SSAPY_FIT_RMS_M or mx_fit_ke > SSAPY_FIT_MAX_M:
            raise AssertionError(
                f"{tag}: Fit vs SSAPy(KE) too far (RMS={rms_fit_ke:.3e}, max={mx_fit_ke:.3e})"
            )

        recon = {"sv": (r_sv, v_sv, t_sv), "ke": (r_ke, v_ke, t_ke)}
    else:
        print(f"{tag}: SSAPy comparison skipped (compare_ssapy=False).")

    return res, recon


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("=== demo_ellipse_from_rv.py ===")
    print(f"SAVE_PLOTS={SAVE_PLOTS}")

    # Baseline scenario
    P1_m = np.array([RGEO, 0.0, 0.0], dtype=float)
    P2_m = np.array([0.0, -1.1 * RGEO, 0.1 * RGEO], dtype=float)

    t_hat = _compute_t_hat(P1_m, P2_m)
    v_pref_A_m_s = 1.0 * t_hat
    v_pref_B_m_s = -1.0 * t_hat

    results = {}
    recon_cache = {}

    for name, v_pref in [("A", v_pref_A_m_s), ("B", v_pref_B_m_s)]:
        print(f"\n----- RUN {name} (baseline) -----")
        res, recon = _run_one_case(
            P1_m,
            P2_m,
            inc_rad=0.0,
            v_pref_m_s=v_pref,
            tag=f"RUN {name}",
            dir_cos_min=DIR_COS_MIN,
            compare_ssapy=True,
        )
        results[name] = res
        recon_cache[name] = recon

    if SAVE_PLOTS:
        prefix = "figures/testing_ellipse_fit"
        _plot_arcs_3d(results, P1_m, P2_m, prefix, title="ellipse_fit arcs (baseline)")
        try:
            orbit_plot(
                [results["A"]["r"], results["B"]["r"]],
                t=[results["A"]["t_rel"], results["B"]["t_rel"]],
                title="ellipse_fit arcs",
                show=False,
                save_path=None,
            )
        except Exception:
            pass
        _plot_recon_overlays(results, recon_cache, prefix)
        _plot_error_vs_time(results, recon_cache, prefix)
        _plot_radius_and_speed(results, recon_cache, prefix)
        _plot_sep_AB(results, prefix)

    # Inclination sweep
    print("\n=== Inclination sweep: 1*RGEO equatorial -> 2*RGEO sphere, inc=0..90 step 15 ===")

    P1s_m = np.array([R1_SWEEP, 0.0, 0.0], dtype=float)

    sweep_store = []
    sweep_failures = []

    for inc_deg in INC_SWEEP_DEG:
        inc_rad = np.deg2rad(float(inc_deg))

        P2s_m = np.array([0.0, R2_SWEEP * np.cos(inc_rad), R2_SWEEP * np.sin(inc_rad)], dtype=float)

        t_hat_s = _compute_t_hat(P1s_m, P2s_m)
        v_pref_s_A = 1.0 * t_hat_s
        v_pref_s_B = -1.0 * t_hat_s

        resA = None
        resB = None

        print(f"\n----- SWEEP inc={inc_deg:+.0f} deg (A) -----")
        try:
            resA, _ = _run_one_case(
                P1s_m,
                P2s_m,
                inc_rad,
                v_pref_s_A,
                tag=f"SWEEP inc={inc_deg:+.0f} A",
                dir_cos_min=None,
                compare_ssapy=False,
            )
        except Exception as e:
            err = traceback.format_exc()
            print(f"SWEEP inc={inc_deg:+.0f} A FAILED: {e}")
            sweep_failures.append(
                {"inc_deg": float(inc_deg), "branch": "A", "error": str(e), "traceback": err}
            )

        print(f"\n----- SWEEP inc={inc_deg:+.0f} deg (B) -----")
        try:
            resB, _ = _run_one_case(
                P1s_m,
                P2s_m,
                inc_rad,
                v_pref_s_B,
                tag=f"SWEEP inc={inc_deg:+.0f} B",
                dir_cos_min=None,
                compare_ssapy=False,
            )
        except Exception as e:
            err = traceback.format_exc()
            print(f"SWEEP inc={inc_deg:+.0f} B FAILED: {e}")
            sweep_failures.append(
                {"inc_deg": float(inc_deg), "branch": "B", "error": str(e), "traceback": err}
            )

        sweep_store.append({"inc_deg": float(inc_deg), "P2": P2s_m, "A": resA, "B": resB})

    if sweep_failures:
        print("\nSweep failures summary:")
        for item in sweep_failures:
            print(f"  inc={item['inc_deg']:+.0f} branch={item['branch']}: {item['error']}")

    if SAVE_PLOTS:
        prefix = "figures/testing_ellipse_fit_incsweep_geo2_polar15"
        _plot_incsweep_all_transfers_3d(sweep_store, P1s_m, prefix)

        sweep_pts = [np.asarray(P1s_m, float)]
        for item in sweep_store:
            sweep_pts.append(np.asarray(item["P2"], float))
            if item["A"] is not None:
                sweep_pts.append(np.asarray(item["A"]["r"], float))
            if item["B"] is not None:
                sweep_pts.append(np.asarray(item["B"]["r"], float))
        sweep_cube_limits = _cube_limits_from_points(
            np.vstack([p.reshape(-1, 3) for p in sweep_pts]),
            pad=0.08,
        )

        for item in sweep_store:
            inc_deg = item["inc_deg"]
            rdict = {}
            if item["A"] is not None:
                rdict["A"] = item["A"]
            if item["B"] is not None:
                rdict["B"] = item["B"]
            if not rdict:
                continue

            P2s_m = item["P2"]
            pp = f"{prefix}_{inc_deg:+.0f}deg"
            _plot_arcs_3d(
                rdict,
                P1s_m,
                P2s_m,
                pp,
                title=f"Transfer arcs to 2*RGEO sphere, inc={inc_deg:+.0f} deg",
                cube_limits=sweep_cube_limits,
            )

    print("\nDONE! (baseline checks passed; sweep processed)")
    return {
        "baseline": results,
        "recon": recon_cache,
        "sweep": sweep_store,
        "sweep_failures": sweep_failures,
    }


if __name__ == "__main__":
    main()
