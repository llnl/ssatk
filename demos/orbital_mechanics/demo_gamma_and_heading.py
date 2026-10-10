import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from astropy.time import Time
from astropy.coordinates import ITRS, GCRS, CartesianRepresentation
import astropy.units as u

from ssatk.ssapy_wrappers.ssapy_orbits import ssapy_orbit
from ssatk.orbital_mechanics.gamma_and_heading import calc_gamma_and_heading
from ssatk.plots.figpath import figpath
from ssatk.constants import EARTH_RADIUS
from ssatk.plots.groundtrack_dashboard_gamma_heading import groundtrack_dashboard_gamma_heading

try:
    from ssapy import groundTrack
    SSAPY_AVAILABLE = True
except Exception:
    SSAPY_AVAILABLE = False

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None

FS_BASE = 20
FS_TITLE = FS_BASE + 2
FS_LABEL = FS_BASE
FS_TICKS = FS_BASE - 4

a_m = EARTH_RADIUS + 7000e3
e = 0.2
i_deg = 0.0
t0_str = "2025-01-01T00:00:00"
duration = (1, "day")
freq = (30, "s")

GRID_AZ_STEP_DEG = 45.0
GRID_LAT_STEP_DEG = 45.0
D_DISP_M = 10000.0
DT_S = 1.0
ARROW_LEN_DEG = 22.5
ARROW_WIDTH = 0.035
ARROW_EDGE_LW = 1.4
HEADLENGTH = 13
HEADWIDTH = 12
EQUAL_ONSCREEN_LEN = True
DRAW_RADIAL_MARKERS = True

GRID_LON_STEP_DEG = 45.0
GRID_LAT_STEP_DEG3 = 45.0
ALTITUDE_M = 700e3
ARROW_LEN_KM_3D = 1200.0
ARROW_LINEWIDTH_3D = 3.0
ARROW_HEAD_RATIO_3D = 0.20
FRAME_FOR_3D = "ITRF"
TITLE_3D = "3D arrows — categories colored (Up/Down/East/West/North/South)"


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v if n == 0.0 else v / n


def _enu_basis_from_r(vec):
    u_hat = _unit(vec)
    k = np.array([0.0, 0.0, 1.0])
    e_hat = np.cross(k, u_hat)
    if np.linalg.norm(e_hat) < 1e-12:
        e_hat = np.cross(np.array([1.0, 0.0, 0.0]), u_hat)
    e_hat = _unit(e_hat)
    n_hat = _unit(np.cross(u_hat, e_hat))
    return e_hat, n_hat, u_hat


def _heading_gamma_from_pair(r0, r1, dt_s):
    e_hat, n_hat, u_hat = _enu_basis_from_r(r0)
    v = (np.asarray(r1, float) - np.asarray(r0, float)) / float(dt_s)
    ve = float(np.dot(v, e_hat))
    vn = float(np.dot(v, n_hat))
    vu = float(np.dot(v, u_hat))
    vt = np.sqrt(ve * ve + vn * vn)
    gamma = np.degrees(np.arctan2(vu, vt))
    if vt <= 1e-12:
        heading = np.nan
    else:
        heading = (np.degrees(np.arctan2(ve, vn)) + 360.0) % 360.0
    return gamma, heading


def _ok(msg):
    print("[PASS]", msg)


def _fail(msg):
    print("[FAIL]", msg)


def _rms(x):
    return float(np.sqrt(np.nanmean(np.asarray(x, float) ** 2)))


def heading_gamma_panel(
    frame_label="ITRF",
    altitude_m=700e3,
    az_step_deg=GRID_AZ_STEP_DEG,
    lat_step_deg=GRID_LAT_STEP_DEG,
    d_disp_m=D_DISP_M,
    dt_s=DT_S,
    t0=t0_str,
    arrow_len_deg=ARROW_LEN_DEG,
    arrow_width=ARROW_WIDTH,
    arrow_edge_lw=ARROW_EDGE_LW,
    headlength=HEADLENGTH,
    headwidth=HEADWIDTH,
    equal_onscreen_length=EQUAL_ONSCREEN_LEN,
    draw_radial_markers=DRAW_RADIAL_MARKERS,
    show_points=True,
):
    frame_label_u = str(frame_label).upper().strip()
    if frame_label_u not in ("ITRF", "GCRF"):
        frame_label_u = "ITRF"

    R = float(EARTH_RADIUS) + float(altitude_m)
    lons = np.arange(0.0, 360.0, float(az_step_deg), dtype=float)
    lats = np.arange(-90.0, 90.0 + 1e-9, float(lat_step_deg), dtype=float)

    dirs = ["Up", "Down", "East", "West", "North", "South"]
    data = {d: {"lon": [], "lat": [], "heading": [], "gamma": []} for d in dirs}

    for lat_deg in lats:
        phi = np.radians(lat_deg)
        for lon_deg in lons:
            lam = np.radians(lon_deg)
            r0 = R * np.array([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)])
            e_hat, n_hat, u_hat = _enu_basis_from_r(r0)
            dirs_local = {"Up": +u_hat, "Down": -u_hat, "East": +e_hat, "West": -e_hat, "North": +n_hat, "South": -n_hat}
            for name, dhat in dirs_local.items():
                r1 = r0 + float(d_disp_m) * dhat
                g0, h0 = _heading_gamma_from_pair(r0, r1, dt_s)
                data[name]["lon"].append(lon_deg)
                data[name]["lat"].append(lat_deg)
                data[name]["heading"].append(h0)
                data[name]["gamma"].append(g0)

    fig, axes = plt.subplots(2, 3, figsize=(24, 14), constrained_layout=True)
    axes = axes.ravel()
    vmin, vmax = -90.0, 90.0
    gamma_norm = Normalize(vmin=vmin, vmax=vmax)

    for ax, name in zip(axes, dirs):
        lon_arr = np.asarray(data[name]["lon"], float)
        lat_arr = np.asarray(data[name]["lat"], float)
        hdg_arr = np.asarray(data[name]["heading"], float)
        gam_arr = np.asarray(data[name]["gamma"], float)

        lat_rad = np.radians(lat_arr)
        hrad = np.radians(hdg_arr)
        coslat = np.cos(lat_rad)
        coslat = np.where(np.abs(coslat) < 1e-12, 1e-12, coslat)

        u_dir = np.sin(hrad) / coslat
        v_dir = np.cos(hrad)

        norm = np.sqrt(u_dir * u_dir + v_dir * v_dir)
        norm = np.where(norm == 0.0, 1.0, norm)
        u_plot = (u_dir / norm) * arrow_len_deg
        v_plot = (v_dir / norm) * arrow_len_deg

        mask = ~np.isfinite(u_plot) | ~np.isfinite(v_plot) | ~np.isfinite(gam_arr)

        q = ax.quiver(
            lon_arr[~mask], lat_arr[~mask], u_plot[~mask], v_plot[~mask], np.clip(gam_arr[~mask], vmin, vmax),
            cmap="coolwarm", norm=gamma_norm,
            angles="xy", scale_units="xy", scale=1.0,
        )
        ax.set_xlim(-2, 362)
        ax.set_ylim(-95, 95)
        ax.set_xticks(np.arange(0, 361, 45))
        ax.set_yticks(np.arange(-90, 91, 45))
        ax.grid(True, alpha=0.35)
        ax.set_title(f"{name} velocity — {frame_label_u}", fontsize=FS_TITLE)
        cb = plt.colorbar(q, ax=ax, pad=0.01, fraction=0.046)
        cb.set_label("Gamma (deg)", fontsize=FS_LABEL)

    return fig


def test_equatorial_local_only():
    r, v, t = ssapy_orbit(
        a=EARTH_RADIUS + 400e3, e=0.0, i=0.0, pa=0.0, raan=0.0, ta=0.0,
        duration=(2, "hour"), freq=(60, "s"), t0=t0_str
    )
    r_arr = np.asarray(r, float)
    t_unx = np.asarray(t.unix if hasattr(t, "unix") else t, float)
    dt = np.diff(t_unx)
    r0 = r_arr[:-1]
    r1 = r_arr[1:]
    dt = np.where(dt == 0.0, 1e-6, dt)

    g_loc, h_loc = [], []
    for i in range(len(dt)):
        g_i, h_i = _heading_gamma_from_pair(r0[i], r1[i], dt[i])
        g_loc.append(g_i)
        h_loc.append(h_i)
    g_loc = np.asarray(g_loc, float)
    h_loc = np.asarray(h_loc, float)

    g_rms = _rms(g_loc)
    if g_rms <= 2.5:
        _ok(f"local equatorial: gamma RMS {g_rms:.3f} deg <= 2.5 deg")
    else:
        _fail(f"local equatorial: gamma RMS {g_rms:.3f} deg > 2.5 deg")
        return False
    return True


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    run_duration = (2, "hour") if fast else duration
    run_freq = (300, "s") if fast else freq

    r, v, t = ssapy_orbit(
        a=a_m, e=e, i=np.radians(i_deg), pa=0.0, raan=0.0, ta=0.0,
        duration=run_duration, freq=run_freq, t0=t0_str
    )

    fig_dash = groundtrack_dashboard_gamma_heading([r], [t], show=False, save_path=None, fontsize=FS_BASE)
    if make_figures:
        tag = f"a{int(round(a_m / 1000))}km_e{e:.2f}_i{int(round(i_deg))}deg_equatorial"
        out_dash = Path(figpath(f"dashboard_gamma_heading_{tag}")).with_suffix(".png")
        out_dash.parent.mkdir(parents=True, exist_ok=True)
        fig_dash.savefig(out_dash, dpi=220, bbox_inches="tight")
        print("Saved dashboard:", out_dash)
    plt.close(fig_dash)

    fig_panel = heading_gamma_panel(frame_label="ITRF")
    if make_figures:
        out_itrf_panel = Path(figpath("heading_gamma_panel_ITRF")).with_suffix(".png")
        out_itrf_panel.parent.mkdir(parents=True, exist_ok=True)
        fig_panel.savefig(out_itrf_panel, dpi=240, bbox_inches="tight")
        print("Saved:", out_itrf_panel)
    plt.close(fig_panel)

    ok = test_equatorial_local_only()
    if not ok:
        raise AssertionError("One or more validation tests FAILED.")

    return {"r": r, "v": v, "t": t, "ok": ok}


if __name__ == "__main__":
    main(make_figures=True, fast=False)