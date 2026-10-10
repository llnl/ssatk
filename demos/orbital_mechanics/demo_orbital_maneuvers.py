"""All-in-one orbital maneuver demo for SSATK.

This gallery demo consolidates the previous small transfer, burn-conversion,
and rendezvous maneuver demos into one user-facing workflow.  It compares
analytic impulsive transfers, fixed-time Lambert solves, optimized searches,
explicit staged optimal transfers, continuous low-thrust burns, and
finite-burn/impulse conversions, then renders summary figures under
``~/ssatk_output/figures``.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from ssapy import Orbit
from ssapy.compute import rv

from ssatk.constants import EARTH_MU, EARTH_RADIUS
from ssatk.orbital_mechanics.burn_to_deltav import burn_to_deltav
from ssatk.orbital_mechanics.deltav_to_burn import deltav_to_burn
from ssatk.orbital_mechanics.transfer_bielliptic import transfer_bielliptic
from ssatk.orbital_mechanics.transfer_hohmann import transfer_hohmann
from ssatk.orbital_mechanics.transfer_inclination_continuous import transfer_inclination_continuous
from ssatk.orbital_mechanics.transfer_optimal_function import transfer_optimal
from ssatk.orbital_mechanics.transfer_ssapy_function import transfer_ssapy
from ssatk.orbital_mechanics.transfer_velocity_and_inclination_continuous import (
    transfer_velocity_and_inclination_continuous,
)
from ssatk.orbital_mechanics.transfer_velocity_continuous import transfer_velocity_continuous
from ssatk.plots.plotutils import figsave


UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None
FIGDIR = "figures"
INITIAL_ORBIT_COLOR = "#1f77b4"
FINAL_ORBIT_COLOR = "#2ca02c"
MANEUVER_COLOR = "#d62728"
IMPULSE_COLOR = "#ff7f0e"


def _circular_state(radius=7000e3, theta=0.0, inclination=0.0, t=0.0):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_inc = np.cos(inclination)
    sin_inc = np.sin(inclination)
    r = radius * np.array([cos_theta, sin_theta * cos_inc, sin_theta * sin_inc])
    v = np.sqrt(EARTH_MU / radius) * np.array([-sin_theta, cos_theta * cos_inc, cos_theta * sin_inc])
    return r, v, t


def _rotation_x(angle):
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, cos_angle, -sin_angle], [0.0, sin_angle, cos_angle]])


def _rotation_z(angle):
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return np.array([[cos_angle, -sin_angle, 0.0], [sin_angle, cos_angle, 0.0], [0.0, 0.0, 1.0]])


def _elliptical_state(rp, ra, true_anomaly=0.0, inclination=0.0, raan=0.0, arg_perigee=0.0, t=0.0):
    semimajor_axis = 0.5 * (rp + ra)
    eccentricity = (ra - rp) / (ra + rp)
    semilatus_rectum = semimajor_axis * (1.0 - eccentricity**2)
    radius = semilatus_rectum / (1.0 + eccentricity * np.cos(true_anomaly))
    r_perifocal = radius * np.array([np.cos(true_anomaly), np.sin(true_anomaly), 0.0])
    v_perifocal = np.sqrt(EARTH_MU / semilatus_rectum) * np.array([-np.sin(true_anomaly), eccentricity + np.cos(true_anomaly), 0.0])
    rotation = _rotation_z(raan) @ _rotation_x(inclination) @ _rotation_z(arg_perigee)
    return rotation @ r_perifocal, rotation @ v_perifocal, t, eccentricity


def _period_from_rpra(rp, ra):
    semimajor_axis = 0.5 * (rp + ra)
    return 2.0 * np.pi * np.sqrt(semimajor_axis**3 / EARTH_MU)


def _inclined_hohmann_result(r1, r2, inclination, split_inclination, *, samples=240, label="inclined_hohmann"):
    transfer_semimajor = 0.5 * (r1 + r2)
    transfer_eccentricity = (r2 - r1) / (r1 + r2)
    transfer_semilatus = transfer_semimajor * (1.0 - transfer_eccentricity**2)
    transfer_tof = np.pi * np.sqrt(transfer_semimajor**3 / EARTH_MU)

    v_initial = np.sqrt(EARTH_MU / r1)
    v_final = np.sqrt(EARTH_MU / r2)
    v_transfer_depart = np.sqrt(EARTH_MU * (2.0 / r1 - 1.0 / transfer_semimajor))
    v_transfer_arrive = np.sqrt(EARTH_MU * (2.0 / r2 - 1.0 / transfer_semimajor))

    depart_direction = np.array([0.0, np.cos(split_inclination), np.sin(split_inclination)])
    arrive_transfer_direction = np.array([0.0, -np.cos(split_inclination), -np.sin(split_inclination)])
    arrive_final_direction = np.array([0.0, -np.cos(inclination), -np.sin(inclination)])

    r_depart = np.array([r1, 0.0, 0.0])
    v_depart_initial = np.array([0.0, v_initial, 0.0])
    v_depart_transfer = v_transfer_depart * depart_direction
    r_arrive = np.array([-r2, 0.0, 0.0])
    v_arrive_transfer = v_transfer_arrive * arrive_transfer_direction
    v_arrive_final = v_final * arrive_final_direction

    delta_v0 = v_depart_transfer - v_depart_initial
    delta_vf = v_arrive_final - v_arrive_transfer

    true_anomaly = np.linspace(0.0, np.pi, samples)
    transfer_radius = transfer_semilatus / (1.0 + transfer_eccentricity * np.cos(true_anomaly))
    transfer_path = (_rotation_x(split_inclination) @ np.vstack((
        transfer_radius * np.cos(true_anomaly),
        transfer_radius * np.sin(true_anomaly),
        np.zeros_like(true_anomaly),
    ))).T

    return {
        "method": label,
        "initial": {"r": r_depart, "v": v_depart_initial, "t": 0.0},
        "target": {"r": r_arrive, "v": v_arrive_final, "t": transfer_tof},
        "final": {"r": r_arrive, "v": v_arrive_final, "t": transfer_tof},
        "trajectory": {"r": transfer_path, "t": np.linspace(0.0, transfer_tof, samples)},
        "burns": [
            {"state": {"r": r_depart, "v": v_depart_initial, "t": 0.0}, "delta_v": delta_v0, "delta_v_mag": float(np.linalg.norm(delta_v0))},
            {"state": {"r": r_arrive, "v": v_arrive_transfer, "t": transfer_tof}, "delta_v": delta_vf, "delta_v_mag": float(np.linalg.norm(delta_vf))},
        ],
        "delta_v_magnitudes": [float(np.linalg.norm(delta_v0)), float(np.linalg.norm(delta_vf))],
        "delta_v_total": float(np.linalg.norm(delta_v0) + np.linalg.norm(delta_vf)),
        "tof": float(transfer_tof),
        "diagnostics": {
            "inclination_change": float(inclination),
            "split_inclination": float(split_inclination),
            "remaining_inclination": float(inclination - split_inclination),
            "r1": float(r1),
            "r2": float(r2),
        },
    }


def _optimized_inclined_hohmann(r1, r2, inclination, *, samples=240):
    from scipy.optimize import minimize_scalar

    result = minimize_scalar(
        lambda split: _inclined_hohmann_result(r1, r2, inclination, split, samples=32)["delta_v_total"],
        bounds=(0.0, inclination),
        method="bounded",
    )
    return _inclined_hohmann_result(r1, r2, inclination, float(result.x), samples=samples, label="inclined_hohmann_split")


def _trajectory(result, samples=300):
    trajectory = result.get("trajectory") if isinstance(result, dict) else None
    if trajectory and trajectory.get("r") is not None:
        return np.asarray(trajectory["r"], dtype=float)
    path = _coast_path_from_first_burn(result, samples=samples)
    if path is not None:
        return path
    if isinstance(result, dict) and "initial" in result and "target" in result:
        return np.vstack((result["initial"]["r"], result["target"]["r"]))
    return None


def _coast_path_from_first_burn(result, samples=300):
    if not isinstance(result, dict) or not result.get("burns") or result.get("tof") is None:
        return None
    initial = result.get("initial", {})
    r0 = np.asarray(initial.get("r"), dtype=float)
    v0 = np.asarray(initial.get("v"), dtype=float) + np.asarray(result["burns"][0]["delta_v"], dtype=float)
    t0 = float(initial.get("t", 0.0))
    times = np.linspace(t0, t0 + float(result["tof"]), samples)
    try:
        r_path, _ = rv(Orbit(r0, v0, t=t0), times)
    except Exception:
        return None
    return np.asarray(r_path, dtype=float)


def _orbit_period(r, v):
    radius = np.linalg.norm(r)
    speed_squared = float(np.dot(v, v))
    specific_energy = 0.5 * speed_squared - EARTH_MU / radius
    if specific_energy >= 0.0:
        return 2.0 * np.pi * np.sqrt(radius**3 / EARTH_MU)
    semimajor_axis = -EARTH_MU / (2.0 * specific_energy)
    return 2.0 * np.pi * np.sqrt(semimajor_axis**3 / EARTH_MU)


def _orbit_path_from_state(state, samples=360):
    r0 = np.asarray(state["r"], dtype=float)
    v0 = np.asarray(state["v"], dtype=float)
    t0 = float(state.get("t", 0.0))
    period = _orbit_period(r0, v0)
    times = np.linspace(t0, t0 + period, samples)
    r_path, _ = rv(Orbit(r0, v0, t=t0), times)
    return np.asarray(r_path, dtype=float)


def _project(points, view="xy"):
    points = np.asarray(points, dtype=float)
    axes = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[view]
    labels = {"xy": ("x [km]", "y [km]"), "xz": ("x [km]", "z [km]"), "yz": ("y [km]", "z [km]")}[view]
    return points[:, axes[0]] / 1e3, points[:, axes[1]] / 1e3, labels


def _plot_projected(ax, points, *, view, color, label, linestyle="-", linewidth=2.0, alpha=1.0, zorder=2):
    points = np.asarray(points, dtype=float)
    if view == "3d":
        ax.plot(points[:, 0] / 1e3, points[:, 1] / 1e3, points[:, 2] / 1e3, color=color, label=label, linestyle=linestyle, linewidth=linewidth, alpha=alpha, zorder=zorder)
        return
    x, y, _ = _project(points, view=view)
    ax.plot(x, y, color=color, label=label, linestyle=linestyle, linewidth=linewidth, alpha=alpha, zorder=zorder)


def _scatter_projected(ax, points, *, view, color, marker, label, size=42, zorder=5):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if view == "3d":
        ax.scatter(points[:, 0] / 1e3, points[:, 1] / 1e3, points[:, 2] / 1e3, color=color, marker=marker, s=size, label=label, edgecolor="black", linewidth=0.4, zorder=zorder)
        return
    x, y, _ = _project(points, view=view)
    ax.scatter(x, y, color=color, marker=marker, s=size, label=label, edgecolor="black", linewidth=0.4, zorder=zorder)


def _plot_earth(ax, view="xy"):
    radius_km = EARTH_RADIUS / 1e3
    if view == "3d":
        theta = np.linspace(0.0, 2.0 * np.pi, 48)
        phi = np.linspace(0.0, np.pi, 24)
        x = radius_km * np.outer(np.cos(theta), np.sin(phi))
        y = radius_km * np.outer(np.sin(theta), np.sin(phi))
        z = radius_km * np.outer(np.ones_like(theta), np.cos(phi))
        ax.plot_surface(x, y, z, color="#4f9bd9", alpha=0.35, linewidth=0.0, shade=True, zorder=0)
        ax.plot_wireframe(x, y, z, color="#1f5f99", linewidth=0.2, alpha=0.25, zorder=0)
        return
    from matplotlib.patches import Circle

    ax.add_patch(Circle((0.0, 0.0), radius_km, facecolor="#4f9bd9", edgecolor="#1f5f99", alpha=0.35, linewidth=1.0, zorder=0, label="Earth"))


def _set_equal_3d(ax, point_sets):
    arrays = [np.asarray(points, dtype=float).reshape(-1, 3) / 1e3 for points in point_sets if points is not None]
    arrays.append(np.array([[-EARTH_RADIUS, -EARTH_RADIUS, -EARTH_RADIUS], [EARTH_RADIUS, EARTH_RADIUS, EARTH_RADIUS]], dtype=float) / 1e3)
    combined = np.vstack(arrays)
    center = 0.5 * (np.nanmax(combined, axis=0) + np.nanmin(combined, axis=0))
    span = np.nanmax(np.nanmax(combined, axis=0) - np.nanmin(combined, axis=0))
    radius = 0.55 * span if np.isfinite(span) and span > 0.0 else EARTH_RADIUS / 1e3
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


def _style_axis(ax, title, view="xy", point_sets=None):
    _plot_earth(ax, view=view)
    if view == "3d":
        ax.set_title(title)
        ax.set_xlabel("x [km]")
        ax.set_ylabel("y [km]")
        ax.set_zlabel("z [km]")
        _set_equal_3d(ax, point_sets or [])
        ax.grid(alpha=0.25)
        ax.view_init(elev=22, azim=-55)
        return
    _, _, labels = _project(np.zeros((1, 3)), view=view)
    ax.set_title(title)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.axis("equal")
    ax.grid(alpha=0.25)


def _format_dv(result):
    return f"Δv={result['delta_v_total']:.0f} m/s"


def _format_tof(result):
    tof = result.get("tof")
    if tof is None:
        return ""
    if tof < 3600.0:
        return f"TOF={tof / 60.0:.1f} min"
    return f"TOF={tof / 3600.0:.1f} h"


def _maneuver_title(name, result):
    parts = [_format_dv(result)]
    tof = _format_tof(result)
    if tof:
        parts.append(tof)
    return f"{name}\n" + ", ".join(parts)


def _burn_label(index, count):
    if count == 1:
        return r"$\Delta v_0$"
    if count == 2:
        return (r"$\Delta v_0$", r"$\Delta v_f$")[index]
    if count == 3:
        return (r"$\Delta v_0$", r"$\Delta v_m$", r"$\Delta v_f$")[index]
    if index == 0:
        return r"$\Delta v_0$"
    if index == count - 1:
        return r"$\Delta v_f$"
    return rf"$\Delta v_{{s{index}}}$"


def _annotate_burns(ax, result, view="xy"):
    burns = result.get("burns") or []
    if not burns:
        return
    offsets = [(16, 14), (18, -22), (-58, 18), (-58, -26), (10, 34), (-66, 36), (22, -42), (-72, -44)]
    for index, burn in enumerate(burns):
        state = burn.get("state") or {}
        if state.get("r") is None:
            continue
        label = _burn_label(index, len(burns))
        text = f"{label}\n{burn['delta_v_mag']:.0f} m/s"
        if view == "3d":
            r = np.asarray(state["r"], dtype=float).reshape(3) / 1e3
            ax.scatter(r[0], r[1], r[2], color=IMPULSE_COLOR, marker="*", s=130, edgecolor="black", linewidth=0.6, zorder=6)
            ax.text(r[0], r[1], r[2], text, fontsize=7, ha="center", va="center", zorder=7)
            continue
        x, y, _ = _project(np.asarray(state["r"], dtype=float).reshape(1, 3), view=view)
        ax.scatter(x[0], y[0], color=IMPULSE_COLOR, marker="*", s=130, edgecolor="black", linewidth=0.6, zorder=6)
        ax.annotate(
            text,
            xy=(x[0], y[0]),
            xytext=offsets[index % len(offsets)],
            textcoords="offset points",
            fontsize=8,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": IMPULSE_COLOR, "alpha": 0.9},
            arrowprops={"arrowstyle": "->", "color": IMPULSE_COLOR, "lw": 0.8},
            zorder=7,
        )


def _plot_transfer_panel(ax, result, title, *, view="xy", trajectory=None, extra_paths=None):
    initial = result["initial"]
    final = result.get("final") or result.get("target")
    start_orbit = _orbit_path_from_state(initial)
    final_orbit = _orbit_path_from_state(final)
    transfer_path = _trajectory(result) if trajectory is None else np.asarray(trajectory, dtype=float)
    bounds = [start_orbit, final_orbit, transfer_path]

    _plot_projected(ax, start_orbit, view=view, color=INITIAL_ORBIT_COLOR, label="initial orbit", linewidth=1.7, alpha=0.85, zorder=1)
    _plot_projected(ax, final_orbit, view=view, color=FINAL_ORBIT_COLOR, label="final orbit", linewidth=1.7, alpha=0.85, zorder=1)
    if transfer_path is not None:
        _plot_projected(ax, transfer_path, view=view, color=MANEUVER_COLOR, label="maneuver trajectory", linewidth=2.8, zorder=3)
    for path_label, path_points, path_color, path_style in extra_paths or []:
        bounds.append(path_points)
        _plot_projected(ax, path_points, view=view, color=path_color, label=path_label, linestyle=path_style, linewidth=2.0, alpha=0.95, zorder=3)
    _scatter_projected(ax, initial["r"], view=view, color=INITIAL_ORBIT_COLOR, marker="o", label="start", size=44)
    _scatter_projected(ax, final["r"], view=view, color=FINAL_ORBIT_COLOR, marker="s", label="target/final", size=44)
    _annotate_burns(ax, result, view=view)
    _style_axis(ax, title, view=view, point_sets=bounds)
    ax.legend(fontsize=7, loc="best")


def _stage_waits(result):
    legs = result.get("stage_legs") or []
    waits = []
    for first_leg, next_leg in zip(legs[:-1], legs[1:]):
        first_arrive = first_leg.get("diagnostics", {}).get("t_arrive", first_leg.get("final", {}).get("t", 0.0))
        next_depart = next_leg.get("diagnostics", {}).get("t_depart", next_leg.get("initial", {}).get("t", first_arrive))
        waits.append(max(0.0, float(next_depart) - float(first_arrive)))
    return waits


def _format_waits(result):
    waits = _stage_waits(result)
    if not waits:
        return "waits=none"
    return "waits=" + "/".join(f"{wait / 60.0:.0f} min" for wait in waits)


def _staged_title(name, result):
    parts = [_format_dv(result)]
    tof = _format_tof(result)
    if tof:
        parts.append(tof)
    parts.append(_format_waits(result))
    return f"{name}\n" + ", ".join(parts)


def _stage_savings_text(result, baseline):
    savings = float(baseline["delta_v_total"]) - float(result["delta_v_total"])
    if abs(savings) < 1.0:
        return "same Δv as direct"
    if savings > 0.0:
        return f"saves {savings:.0f} m/s vs direct"
    return f"costs {-savings:.0f} m/s vs direct"


def _best_delta_v_result(results):
    return min(results, key=lambda result: float(result["delta_v_total"]))


def _staged_title_with_savings(name, result, baseline):
    return _staged_title(name, result) + "\n" + _stage_savings_text(result, baseline)


def _plot_staged_panel(ax, result, title, *, view="xy"):
    legs = result.get("stage_legs") or []
    if not legs:
        _plot_transfer_panel(ax, result, title, view=view)
        return

    initial = result["initial"]
    final = result.get("final") or result.get("target")
    initial_orbit = _orbit_path_from_state(initial)
    final_orbit = _orbit_path_from_state(final)
    bounds = [initial_orbit, final_orbit]
    _plot_projected(ax, initial_orbit, view=view, color=INITIAL_ORBIT_COLOR, label="initial orbit", linewidth=1.7, alpha=0.85, zorder=1)
    _plot_projected(ax, final_orbit, view=view, color=FINAL_ORBIT_COLOR, label="target orbit", linewidth=1.7, alpha=0.85, zorder=1)

    leg_colors = ["#d62728", "#ff7f0e", "#9467bd", "#8c564b"]
    for index, leg in enumerate(legs):
        path = _trajectory(leg)
        if path is not None:
            bounds.append(path)
            _plot_projected(
                ax,
                path,
                view=view,
                color=leg_colors[index % len(leg_colors)],
                label=f"transfer leg {index + 1}",
                linewidth=2.6,
                alpha=0.95,
                zorder=3,
            )
        if index < len(legs) - 1:
            stage_state = leg.get("final") or leg.get("target")
            if stage_state and stage_state.get("r") is not None and stage_state.get("v") is not None:
                stage_orbit = _orbit_path_from_state(stage_state)
                bounds.append(stage_orbit)
                _plot_projected(
                    ax,
                    stage_orbit,
                    view=view,
                    color=leg_colors[index % len(leg_colors)],
                    label=f"staging orbit {index + 1}",
                    linestyle="--",
                    linewidth=1.3,
                    alpha=0.55,
                    zorder=1,
                )
                _scatter_projected(ax, stage_state["r"], view=view, color=leg_colors[index % len(leg_colors)], marker="D", label=f"stage {index + 1}", size=38)

    _scatter_projected(ax, initial["r"], view=view, color=INITIAL_ORBIT_COLOR, marker="o", label="start", size=44)
    _scatter_projected(ax, final["r"], view=view, color=FINAL_ORBIT_COLOR, marker="s", label="target/final", size=44)
    _annotate_burns(ax, result, view=view)
    _style_axis(ax, title, view=view, point_sets=bounds)
    ax.legend(fontsize=6.5, loc="best")


def _burn_event_time(result, burn, index):
    state = burn.get("state") or {}
    if state.get("t") is not None:
        return float(state["t"])
    if index == 0:
        return float(result.get("diagnostics", {}).get("t_depart", result.get("initial", {}).get("t", 0.0)))
    if index == len(result.get("burns", [])) - 1:
        return float(result.get("diagnostics", {}).get("t_arrive", result.get("final", {}).get("t", 0.0)))
    return float(result.get("initial", {}).get("t", 0.0))


def _burn_conversion_panel_result(result):
    r_cont = np.asarray(result["r_continuous"], dtype=float)
    r_inst = np.asarray(result["r_instantaneous"], dtype=float)
    v_cont = np.asarray(result["v_continuous"], dtype=float)
    final_state = {
        "r": r_cont[-1],
        "v": v_cont[-1],
        "t": float(r_cont.shape[0] - 1),
    }
    return r_cont, r_inst, final_state


def _summarize_delta_v(results_by_family):
    rows = []
    for family, entries in results_by_family.items():
        for name, result in entries.items():
            rows.append(
                {
                    "family": family,
                    "case": name,
                    "delta_v_total": float(result["delta_v_total"]),
                    "burn_count": len(result.get("burns", [])),
                    "tof": result.get("tof"),
                    "method": result.get("method"),
                }
            )
    return rows


def _build_maneuver_results(fast):
    samples = 32 if fast else 180
    r0, v0, _ = _circular_state(7000e3)
    transfer_angle = np.deg2rad(150.0)
    r_target, v_target, _ = _circular_state(9000e3, transfer_angle)
    tof = (150.0 / 180.0) * np.pi * np.sqrt(((7000e3 + 9000e3) / 2.0) ** 3 / EARTH_MU)

    impulsive = {
        "Hohmann raise": transfer_hohmann(7000e3, 9000e3, samples=samples),
        "Hohmann lower": transfer_hohmann(9000e3, 7000e3, samples=samples),
        "Bi-elliptic raise": transfer_bielliptic(7000e3, 9000e3, intermediate_radius=20_000e3, samples_per_arc=samples // 2),
        "Bi-elliptic lower": transfer_bielliptic(9000e3, 7000e3, intermediate_radius=20_000e3, samples_per_arc=samples // 2),
    }

    departure = (r0, v0, 0.0)
    arrival = (r_target, v_target, tof)
    fixed_time = {
        "Fixed-time Lambert": transfer_ssapy(
            departure,
            arrival,
            propagate=False,
            refine=False,
            burn_duration=1.0,
        ),
    }

    velocity_delta_v = 10.0 if fast else 300.0
    velocity_lower_delta_v = -5.0 if fast else -180.0
    inclination_delta_v = 5.0 if fast else 300.0
    inclination_lower_delta_v = -2.0 if fast else -180.0
    continuous_accel = 1.0 if fast else 2.0
    continuous_time = 30.0 if fast else 900.0
    continuous = {
        f"Velocity +{velocity_delta_v:.0f} m/s": transfer_velocity_continuous(r0, v0, v_target=velocity_delta_v, a_thrust=continuous_accel, max_time=continuous_time),
        f"Velocity {velocity_lower_delta_v:.0f} m/s": transfer_velocity_continuous(r0, v0, v_target=velocity_lower_delta_v, a_thrust=continuous_accel, max_time=continuous_time),
        f"Inclination +{inclination_delta_v:.0f} m/s": transfer_inclination_continuous(r0, v0, delta_v=inclination_delta_v, a_thrust=continuous_accel, max_time=continuous_time),
        f"Inclination {inclination_lower_delta_v:.0f} m/s": transfer_inclination_continuous(r0, v0, delta_v=inclination_lower_delta_v, a_thrust=continuous_accel, max_time=continuous_time),
        "Velocity then inclination": transfer_velocity_and_inclination_continuous(
            r0,
            v0,
            i_target=np.deg2rad(0.01),
            a_thrust=1.0,
            max_time1=2.0,
            max_time2=200.0,
        ),
    }

    optimized_radius = 9000e3
    optimized_angle = 0.4
    r_short, v_short, _ = _circular_state(optimized_radius, optimized_angle)
    optimal = {
        "Optimal total Δv": transfer_optimal(
            (r0, v0, 0.0),
            (r_short, v_short, 0.0),
            delta_v_mode="total",
            t_window=(0.0, 1000.0),
            tof_range=(1000.0, 6000.0),
            n_grid=(2, 2) if fast else (8, 8),
            polish=False,
            propagate=False,
            refine=False,
            burn_duration=1.0,
        ),
        "Optimal first burn": transfer_optimal(
            (r0, v0, 0.0),
            (r_short, v_short, 0.0),
            delta_v_mode="first",
            arrival_burn=False,
            t_window=(0.0, 1000.0),
            tof_range=(1000.0, 6000.0),
            n_grid=(2, 2) if fast else (8, 8),
            polish=False,
            propagate=False,
            refine=False,
            burn_duration=1.0,
        ),
        "Optimal last burn": transfer_optimal(
            (r0, v0, 0.0),
            (r_short, v_short, 0.0),
            delta_v_mode="last",
            t_window=(0.0, 1000.0),
            tof_range=(1000.0, 6000.0),
            n_grid=(2, 2) if fast else (8, 8),
            polish=False,
            propagate=False,
            refine=False,
            burn_duration=1.0,
        ),
        "Min time under cap": transfer_optimal(
            (r0, v0, 0.0),
            (r_short, v_short, 0.0),
            objective="time",
            dv_budget=5000.0,
            t_window=(0.0, 1000.0),
            tof_range=(1000.0, 6000.0),
            n_grid=(2, 2) if fast else (8, 8),
            polish=False,
            propagate=False,
            refine=False,
            burn_duration=1.0,
        ),
        "Rendezvous mode": transfer_optimal(
            (r0, v0, 0.0),
            (r_short, v_short, 0.0),
            arrival_mode="rendezvous",
            t_window=(0.0, 1000.0),
            tof_range=(1000.0, 6000.0),
            n_grid=(2, 2) if fast else (8, 8),
            polish=False,
            propagate=False,
            refine=False,
            burn_duration=1.0,
        ),
    }

    staged_angle = 0.2
    staged_inclination = np.deg2rad(90.0)
    r_staged_target, v_staged_target, _ = _circular_state(15_000e3, staged_angle, staged_inclination)
    staged_departure = (r0, v0, 0.0)
    staged_arrival = (r_staged_target, v_staged_target, 0.0)
    staged_optimal = {
        "Direct leave-now": transfer_optimal(
            staged_departure,
            staged_arrival,
            departure_mode="now",
            tof_range=(1800.0, 15_000.0),
            n_grid=(3, 3) if fast else (5, 5),
            polish=False,
            propagate=False,
            refine=False,
            burn_duration=1.0,
        ),
        "Immediate one-stop": transfer_optimal(
            staged_departure,
            staged_arrival,
            stage_mode="immediate",
            n_stage_stops=1,
            departure_mode="now",
            tof_range=(1800.0, 15_000.0),
            n_grid=(3, 3) if fast else (5, 5),
            polish=False,
            propagate=False,
            refine=False,
            burn_duration=1.0,
            stage_radii=[20_000e3, 40_000e3, 80_000e3],
            stage_plane_fractions=[0.0, 0.5, 1.0],
            n_stage_phase=2,
            stage_beam_width=3 if fast else 5,
            stage_wait_window=25_000.0,
        ),
        "Timed one-stop": transfer_optimal(
            staged_departure,
            staged_arrival,
            stage_mode="timed",
            stage_timing="appropriately timed",
            n_stage_stops=1,
            departure_mode="now",
            tof_range=(1800.0, 15_000.0),
            n_grid=(3, 3) if fast else (5, 5),
            polish=False,
            propagate=False,
            refine=False,
            burn_duration=1.0,
            stage_radii=[20_000e3, 40_000e3, 80_000e3],
            stage_plane_fractions=[0.0, 0.5, 1.0],
            n_stage_phase=2,
            stage_beam_width=3 if fast else 5,
            stage_wait_window=25_000.0,
        ),
        "Timed two-stop min-time": transfer_optimal(
            staged_departure,
            staged_arrival,
            stage_mode="timed",
            stage_timing="appropriately timed",
            n_stage_stops=2,
            objective="time",
            dv_budget=50_000.0,
            departure_mode="now",
            tof_range=(1800.0, 15_000.0),
            n_grid=(3, 3) if fast else (5, 5),
            polish=False,
            propagate=False,
            refine=False,
            burn_duration=1.0,
            stage_radii=[20_000e3, 40_000e3, 80_000e3],
            stage_plane_fractions=[0.0, 0.5, 1.0],
            n_stage_phase=2,
            stage_beam_width=3 if fast else 5,
            stage_wait_window=25_000.0,
        ),
    }

    samples_for_split_plane = 96 if fast else 240
    split_plane_change = {}

    # Plane-change example 1: LEO to GEO with a 28.5 degree inclination change.
    leo_geo_r1 = 7000e3
    leo_geo_r2 = 42_164e3
    leo_geo_inclination = np.deg2rad(28.5)
    split_plane_change["LEO to GEO, 28.5° plane change all departure"] = _inclined_hohmann_result(
        leo_geo_r1,
        leo_geo_r2,
        leo_geo_inclination,
        leo_geo_inclination,
        samples=samples_for_split_plane,
        label="inclined_hohmann_all_departure",
    )
    split_plane_change["LEO to GEO, 28.5° plane change all arrival"] = _inclined_hohmann_result(
        leo_geo_r1,
        leo_geo_r2,
        leo_geo_inclination,
        0.0,
        samples=samples_for_split_plane,
        label="inclined_hohmann_all_arrival",
    )
    split_plane_change["LEO to GEO, 28.5° plane change split"] = _optimized_inclined_hohmann(
        leo_geo_r1,
        leo_geo_r2,
        leo_geo_inclination,
        samples=samples_for_split_plane,
    )

    # Plane-change example 2: LEO to MEO with a 20 degree inclination change.
    leo_meo_r1 = 7000e3
    leo_meo_r2 = 15_000e3
    leo_meo_inclination = np.deg2rad(20.0)
    split_plane_change["LEO to MEO, 20° plane change all departure"] = _inclined_hohmann_result(
        leo_meo_r1,
        leo_meo_r2,
        leo_meo_inclination,
        leo_meo_inclination,
        samples=samples_for_split_plane,
        label="inclined_hohmann_all_departure",
    )
    split_plane_change["LEO to MEO, 20° plane change all arrival"] = _inclined_hohmann_result(
        leo_meo_r1,
        leo_meo_r2,
        leo_meo_inclination,
        0.0,
        samples=samples_for_split_plane,
        label="inclined_hohmann_all_arrival",
    )
    split_plane_change["LEO to MEO, 20° plane change split"] = _optimized_inclined_hohmann(
        leo_meo_r1,
        leo_meo_r2,
        leo_meo_inclination,
        samples=samples_for_split_plane,
    )

    elliptical_two_burn = {}

    # Elliptical example 1: aligned sub-GEO ellipses.
    aligned_initial_rp = 7000e3
    aligned_initial_ra = 11_000e3
    aligned_initial_true_anomaly = 0.0
    aligned_initial_inclination = 0.0
    aligned_initial_raan = 0.0
    aligned_initial_arg_perigee = 0.0
    aligned_target_rp = 9000e3
    aligned_target_ra = 16_000e3
    aligned_target_true_anomaly = np.pi
    aligned_target_inclination = 0.0
    aligned_target_raan = 0.0
    aligned_target_arg_perigee = 0.0
    r_aligned_initial, v_aligned_initial, _, e_aligned_initial = _elliptical_state(
        aligned_initial_rp,
        aligned_initial_ra,
        aligned_initial_true_anomaly,
        aligned_initial_inclination,
        aligned_initial_raan,
        aligned_initial_arg_perigee,
    )
    r_aligned_target, v_aligned_target, _, e_aligned_target = _elliptical_state(
        aligned_target_rp,
        aligned_target_ra,
        aligned_target_true_anomaly,
        aligned_target_inclination,
        aligned_target_raan,
        aligned_target_arg_perigee,
    )
    aligned_max_period = max(
        _period_from_rpra(aligned_initial_rp, aligned_initial_ra),
        _period_from_rpra(aligned_target_rp, aligned_target_ra),
    )
    aligned_stage_radii = sorted(
        {
            float(np.sqrt(min(aligned_initial_rp, aligned_target_rp) * max(aligned_initial_ra, aligned_target_ra))),
            float(0.5 * (min(aligned_initial_rp, aligned_target_rp) + max(aligned_initial_ra, aligned_target_ra))),
            float(min(42_164e3, 1.2 * max(aligned_initial_ra, aligned_target_ra))),
            42_164e3,
        }
    )
    aligned_departure = (r_aligned_initial, v_aligned_initial, 0.0)
    aligned_arrival = (r_aligned_target, v_aligned_target, 0.0)
    aligned_direct = transfer_optimal(
        aligned_departure,
        aligned_arrival,
        departure_mode="now",
        tof_range=(0.10 * aligned_max_period, 1.20 * aligned_max_period),
        n_grid=(4, 4) if fast else (6, 6),
        polish=False,
        propagate=False,
        refine=False,
        burn_duration=1.0,
    )
    aligned_staged_candidates = [
        transfer_optimal(
            aligned_departure,
            aligned_arrival,
            stage_mode="immediate",
            n_stage_stops=1,
            departure_mode="now",
            tof_range=(0.10 * aligned_max_period, 1.20 * aligned_max_period),
            n_grid=(4, 4) if fast else (6, 6),
            polish=False,
            propagate=False,
            refine=False,
            burn_duration=1.0,
            stage_radii=aligned_stage_radii,
            stage_plane_fractions=[0.0, 0.5, 1.0],
            n_stage_phase=1 if fast else 2,
            stage_beam_width=2 if fast else 4,
            stage_wait_window=0.5 * aligned_max_period,
        ),
        transfer_optimal(
            aligned_departure,
            aligned_arrival,
            stage_mode="timed",
            n_stage_stops=1,
            departure_mode="now",
            tof_range=(0.10 * aligned_max_period, 1.20 * aligned_max_period),
            n_grid=(4, 4) if fast else (6, 6),
            polish=False,
            propagate=False,
            refine=False,
            burn_duration=1.0,
            stage_radii=aligned_stage_radii,
            stage_plane_fractions=[0.0, 0.5, 1.0],
            n_stage_phase=1 if fast else 2,
            stage_beam_width=2 if fast else 4,
            stage_wait_window=0.5 * aligned_max_period,
        ),
    ]
    aligned_best_staged = _best_delta_v_result(aligned_staged_candidates)
    aligned_direct["case_description"] = f"e₀={e_aligned_initial:.2f}, e_f={e_aligned_target:.2f}; both apogees below GEO"
    aligned_best_staged["case_description"] = aligned_direct["case_description"]
    elliptical_two_burn["Aligned sub-GEO ellipses direct"] = aligned_direct
    elliptical_two_burn["Aligned sub-GEO ellipses best staged"] = aligned_best_staged

    # Elliptical example 2: mildly inclined MEO ellipses.
    meo_initial_rp = 9000e3
    meo_initial_ra = 18_000e3
    meo_initial_true_anomaly = 0.0
    meo_initial_inclination = np.deg2rad(5.0)
    meo_initial_raan = 0.0
    meo_initial_arg_perigee = 0.0
    meo_target_rp = 12_000e3
    meo_target_ra = 26_000e3
    meo_target_true_anomaly = np.pi
    meo_target_inclination = np.deg2rad(7.0)
    meo_target_raan = 0.0
    meo_target_arg_perigee = 0.0
    r_meo_initial, v_meo_initial, _, e_meo_initial = _elliptical_state(
        meo_initial_rp,
        meo_initial_ra,
        meo_initial_true_anomaly,
        meo_initial_inclination,
        meo_initial_raan,
        meo_initial_arg_perigee,
    )
    r_meo_target, v_meo_target, _, e_meo_target = _elliptical_state(
        meo_target_rp,
        meo_target_ra,
        meo_target_true_anomaly,
        meo_target_inclination,
        meo_target_raan,
        meo_target_arg_perigee,
    )
    meo_max_period = max(
        _period_from_rpra(meo_initial_rp, meo_initial_ra),
        _period_from_rpra(meo_target_rp, meo_target_ra),
    )
    meo_stage_radii = sorted(
        {
            float(np.sqrt(min(meo_initial_rp, meo_target_rp) * max(meo_initial_ra, meo_target_ra))),
            float(0.5 * (min(meo_initial_rp, meo_target_rp) + max(meo_initial_ra, meo_target_ra))),
            float(min(42_164e3, 1.2 * max(meo_initial_ra, meo_target_ra))),
            42_164e3,
        }
    )
    meo_departure = (r_meo_initial, v_meo_initial, 0.0)
    meo_arrival = (r_meo_target, v_meo_target, 0.0)
    meo_direct = transfer_optimal(
        meo_departure,
        meo_arrival,
        departure_mode="now",
        tof_range=(0.10 * meo_max_period, 1.20 * meo_max_period),
        n_grid=(4, 4) if fast else (6, 6),
        polish=False,
        propagate=False,
        refine=False,
        burn_duration=1.0,
    )
    meo_staged_candidates = [
        transfer_optimal(
            meo_departure,
            meo_arrival,
            stage_mode="immediate",
            n_stage_stops=1,
            departure_mode="now",
            tof_range=(0.10 * meo_max_period, 1.20 * meo_max_period),
            n_grid=(4, 4) if fast else (6, 6),
            polish=False,
            propagate=False,
            refine=False,
            burn_duration=1.0,
            stage_radii=meo_stage_radii,
            stage_plane_fractions=[0.0, 0.5, 1.0],
            n_stage_phase=1 if fast else 2,
            stage_beam_width=2 if fast else 4,
            stage_wait_window=0.5 * meo_max_period,
        ),
        transfer_optimal(
            meo_departure,
            meo_arrival,
            stage_mode="timed",
            n_stage_stops=1,
            departure_mode="now",
            tof_range=(0.10 * meo_max_period, 1.20 * meo_max_period),
            n_grid=(4, 4) if fast else (6, 6),
            polish=False,
            propagate=False,
            refine=False,
            burn_duration=1.0,
            stage_radii=meo_stage_radii,
            stage_plane_fractions=[0.0, 0.5, 1.0],
            n_stage_phase=1 if fast else 2,
            stage_beam_width=2 if fast else 4,
            stage_wait_window=0.5 * meo_max_period,
        ),
    ]
    meo_best_staged = _best_delta_v_result(meo_staged_candidates)
    meo_direct["case_description"] = f"e₀={e_meo_initial:.2f}, e_f={e_meo_target:.2f}; both apogees below GEO"
    meo_best_staged["case_description"] = meo_direct["case_description"]
    elliptical_two_burn["Slightly inclined MEO ellipses direct"] = meo_direct
    elliptical_two_burn["Slightly inclined MEO ellipses best staged"] = meo_best_staged

    # Elliptical example 3: near-GEO aligned ellipses.
    geo_initial_rp = 12_000e3
    geo_initial_ra = 26_000e3
    geo_initial_true_anomaly = 0.0
    geo_initial_inclination = 0.0
    geo_initial_raan = 0.0
    geo_initial_arg_perigee = 0.0
    geo_target_rp = 22_000e3
    geo_target_ra = 42_164e3
    geo_target_true_anomaly = np.pi
    geo_target_inclination = 0.0
    geo_target_raan = 0.0
    geo_target_arg_perigee = 0.0
    r_geo_initial, v_geo_initial, _, e_geo_initial = _elliptical_state(
        geo_initial_rp,
        geo_initial_ra,
        geo_initial_true_anomaly,
        geo_initial_inclination,
        geo_initial_raan,
        geo_initial_arg_perigee,
    )
    r_geo_target, v_geo_target, _, e_geo_target = _elliptical_state(
        geo_target_rp,
        geo_target_ra,
        geo_target_true_anomaly,
        geo_target_inclination,
        geo_target_raan,
        geo_target_arg_perigee,
    )
    geo_max_period = max(
        _period_from_rpra(geo_initial_rp, geo_initial_ra),
        _period_from_rpra(geo_target_rp, geo_target_ra),
    )
    geo_stage_radii = sorted(
        {
            float(np.sqrt(min(geo_initial_rp, geo_target_rp) * max(geo_initial_ra, geo_target_ra))),
            float(0.5 * (min(geo_initial_rp, geo_target_rp) + max(geo_initial_ra, geo_target_ra))),
            float(min(42_164e3, 1.2 * max(geo_initial_ra, geo_target_ra))),
            42_164e3,
        }
    )
    geo_departure = (r_geo_initial, v_geo_initial, 0.0)
    geo_arrival = (r_geo_target, v_geo_target, 0.0)
    geo_direct = transfer_optimal(
        geo_departure,
        geo_arrival,
        departure_mode="now",
        tof_range=(0.10 * geo_max_period, 1.20 * geo_max_period),
        n_grid=(4, 4) if fast else (6, 6),
        polish=False,
        propagate=False,
        refine=False,
        burn_duration=1.0,
    )
    geo_staged_candidates = [
        transfer_optimal(
            geo_departure,
            geo_arrival,
            stage_mode="immediate",
            n_stage_stops=1,
            departure_mode="now",
            tof_range=(0.10 * geo_max_period, 1.20 * geo_max_period),
            n_grid=(4, 4) if fast else (6, 6),
            polish=False,
            propagate=False,
            refine=False,
            burn_duration=1.0,
            stage_radii=geo_stage_radii,
            stage_plane_fractions=[0.0, 0.5, 1.0],
            n_stage_phase=1 if fast else 2,
            stage_beam_width=2 if fast else 4,
            stage_wait_window=0.5 * geo_max_period,
        ),
        transfer_optimal(
            geo_departure,
            geo_arrival,
            stage_mode="timed",
            n_stage_stops=1,
            departure_mode="now",
            tof_range=(0.10 * geo_max_period, 1.20 * geo_max_period),
            n_grid=(4, 4) if fast else (6, 6),
            polish=False,
            propagate=False,
            refine=False,
            burn_duration=1.0,
            stage_radii=geo_stage_radii,
            stage_plane_fractions=[0.0, 0.5, 1.0],
            n_stage_phase=1 if fast else 2,
            stage_beam_width=2 if fast else 4,
            stage_wait_window=0.5 * geo_max_period,
        ),
    ]
    geo_best_staged = _best_delta_v_result(geo_staged_candidates)
    geo_direct["case_description"] = f"e₀={e_geo_initial:.2f}, e_f={e_geo_target:.2f}; both apogees below GEO"
    geo_best_staged["case_description"] = geo_direct["case_description"]
    elliptical_two_burn["Near-GEO aligned ellipses direct"] = geo_direct
    elliptical_two_burn["Near-GEO aligned ellipses best staged"] = geo_best_staged

    orbit = Orbit(r0, v0, t=0.0)
    burn_times = np.arange(0.0, 30.0 if fast else 600.0, 1.0)
    burn_accel = np.array([0.01, 0.02, 0.002]) if fast else np.array([0.10, 0.50, 0.05])
    impulse_delta_v = np.array([0.2, 0.4, 0.04]) if fast else np.array([60.0, 300.0, 30.0])
    burn_conversion = {
        "burn_to_deltav": burn_to_deltav(orbit, burn_times, burn_accel),
        "deltav_to_burn": deltav_to_burn(orbit, burn_times, impulse_delta_v),
    }

    return {
        "impulsive": impulsive,
        "fixed_time": fixed_time,
        "continuous": continuous,
        "optimal": optimal,
        "staged_optimal": staged_optimal,
        "split_plane_change": split_plane_change,
        "elliptical_two_burn": elliptical_two_burn,
        "burn_conversion": burn_conversion,
    }


def _first_key(entries, prefix):
    return next(key for key in entries if key.startswith(prefix))


def _make_summary_figure(results):
    fig, axes = plt.subplots(3, 3, figsize=(20, 18), constrained_layout=True, subplot_kw={"projection": "3d"})
    axes = axes.ravel()

    panels = [
        ("Hohmann raise", results["impulsive"]["Hohmann raise"], "3d", None),
        ("Hohmann lower", results["impulsive"]["Hohmann lower"], "3d", None),
        ("Bi-elliptic raise", results["impulsive"]["Bi-elliptic raise"], "3d", None),
        ("Bi-elliptic lower", results["impulsive"]["Bi-elliptic lower"], "3d", None),
        ("Fixed-time Lambert", results["fixed_time"]["Fixed-time Lambert"], "3d", None),
        ("Optimized transfer", results["optimal"]["Optimal total Δv"], "3d", None),
        (
            "Continuous tangential burn",
            results["continuous"][_first_key(results["continuous"], "Velocity +")],
            "3d",
            None,
        ),
        (
            "Continuous plane-change burn",
            results["continuous"][_first_key(results["continuous"], "Inclination +")],
            "3d",
            None,
        ),
    ]

    for ax, (name, result, view, trajectory) in zip(axes[:8], panels):
        _plot_transfer_panel(ax, result, _maneuver_title(name, result), view=view, trajectory=trajectory)

    burn_result = results["burn_conversion"]["burn_to_deltav"]
    r_cont, r_inst, final_state = _burn_conversion_panel_result(burn_result)
    burn_as_transfer = {
        "initial": {"r": r_cont[0], "v": burn_result["v_continuous"][0], "t": 0.0},
        "final": final_state,
        "target": final_state,
        "trajectory": {"r": r_cont},
        "burns": [
            {
                "state": {"r": r_cont[0], "v": burn_result["v_continuous"][0], "t": 0.0},
                "delta_v_mag": float(np.linalg.norm(burn_result["delta_v_ntw"])),
            }
        ],
        "tof": float(r_cont.shape[0] - 1),
        "delta_v_total": float(np.linalg.norm(burn_result["delta_v_ntw"])),
    }
    _plot_transfer_panel(
        axes[8],
        burn_as_transfer,
        _maneuver_title("Finite burn vs impulse", burn_as_transfer),
        view="3d",
        extra_paths=[("impulse approximation", r_inst, IMPULSE_COLOR, "--")],
    )

    fig.suptitle(
        "SSATK orbital maneuver overview\n"
        "Earth is rendered to scale at the origin; blue is the starting orbit, green is the target/final orbit, and red is the maneuver path.",
        fontsize=16,
    )
    return fig


def _make_staged_optimal_figure(results):
    fig, axes = plt.subplots(2, 2, figsize=(18, 15), constrained_layout=True, subplot_kw={"projection": "3d"})
    axes = axes.ravel()
    staged = results["staged_optimal"]
    baseline = staged["Direct leave-now"]
    panels = [
        ("Direct leave-now baseline", staged["Direct leave-now"], "3d"),
        ("Immediate one-stop staging", staged["Immediate one-stop"], "3d"),
        ("Timed one-stop staging", staged["Timed one-stop"], "3d"),
        ("Timed two-stop min-time", staged["Timed two-stop min-time"], "3d"),
    ]
    for ax, (name, result, view) in zip(axes, panels):
        title = _staged_title(name, result) if result is baseline else _staged_title_with_savings(name, result, baseline)
        _plot_staged_panel(ax, result, title, view=view)
    fig.suptitle(
        "When staged transfers help: fixed departure, orbit raise, and large plane change\n"
        "Earth is rendered to scale; blue is the starting orbit, green is the final orbit, colored dashed curves are staging orbits, and colored solid curves are sequential transfer legs.",
        fontsize=15,
    )
    return fig


def _make_staged_timeline_figure(results):
    staged = results["staged_optimal"]
    names = list(staged)
    baseline = staged["Direct leave-now"]
    fig, ax = plt.subplots(figsize=(16, 7), constrained_layout=True)
    leg_colors = ["#d62728", "#ff7f0e", "#9467bd", "#8c564b"]
    wait_label_added = False

    for row, name in enumerate(names):
        result = staged[name]
        legs = result.get("stage_legs") or [result]
        y = len(names) - row - 1
        for leg_index, leg in enumerate(legs):
            depart = float(leg.get("diagnostics", {}).get("t_depart", leg.get("initial", {}).get("t", 0.0))) / 3600.0
            arrive = float(leg.get("diagnostics", {}).get("t_arrive", leg.get("final", {}).get("t", 0.0))) / 3600.0
            ax.plot(
                [depart, arrive],
                [y, y],
                color=leg_colors[leg_index % len(leg_colors)],
                linewidth=4.0,
                solid_capstyle="round",
                label="transfer leg" if row == 0 and leg_index == 0 else None,
            )
            if leg_index < len(legs) - 1:
                next_depart = float(legs[leg_index + 1].get("diagnostics", {}).get("t_depart", arrive * 3600.0)) / 3600.0
                if next_depart > arrive:
                    ax.broken_barh(
                        [(arrive, next_depart - arrive)],
                        (y - 0.24, 0.48),
                        facecolors="#9e9e9e",
                        alpha=0.28,
                        label="optimized wait" if not wait_label_added else None,
                    )
                    wait_label_added = True
        burns = result.get("burns") or []
        for burn_index, burn in enumerate(burns):
            event_time = _burn_event_time(result, burn, burn_index) / 3600.0
            delta_v = float(burn.get("delta_v_mag", 0.0))
            marker_size = 45.0 + 0.03 * delta_v
            ax.scatter(
                event_time,
                y,
                s=marker_size,
                color=IMPULSE_COLOR,
                marker="*",
                edgecolor="black",
                linewidth=0.5,
                zorder=4,
                label="impulse" if row == 0 and burn_index == 0 else None,
            )
            ax.annotate(
                f"{_burn_label(burn_index, len(burns))}\n{delta_v:.0f} m/s",
                xy=(event_time, y),
                xytext=(0, 18 if burn_index % 2 == 0 else -24),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": IMPULSE_COLOR, "alpha": 0.88},
                arrowprops={"arrowstyle": "-", "color": IMPULSE_COLOR, "lw": 0.7},
            )

        ax.text(
            float(result.get("diagnostics", {}).get("t_arrive", result.get("tof", 0.0))) / 3600.0 + 0.25,
            y,
            f"{result['delta_v_total']:.0f} m/s, {_format_waits(result)}, {_stage_savings_text(result, baseline)}",
            va="center",
            fontsize=9,
        )

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(list(reversed(names)))
    ax.set_xlabel("mission elapsed time [hours]")
    ax.set_title("Staged optimal transfer burn timing: waiting and splitting plane changes can reduce total Δv")
    ax.grid(axis="x", alpha=0.25)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.legend(loc="upper right", fontsize=9)
    return fig


def _make_elliptical_two_burn_figure(results):
    elliptical = results["elliptical_two_burn"]
    case_names = [key.removesuffix(" direct") for key in elliptical if key.endswith(" direct")]
    fig, axes = plt.subplots(len(case_names), 2, figsize=(18, 8 * len(case_names)), constrained_layout=True, subplot_kw={"projection": "3d"})
    axes = np.asarray(axes).reshape(len(case_names), 2)

    for row, case_name in enumerate(case_names):
        direct = elliptical[f"{case_name} direct"]
        staged = elliptical[f"{case_name} best staged"]
        description = direct.get("case_description", "elliptical boundary orbits")
        direct_title = f"{case_name}: direct two-burn\n{_format_dv(direct)}, {_format_tof(direct)}, {description}"
        staged_title = (
            f"{case_name}: lowest-Δv staged alternative\n"
            f"{_format_dv(staged)}, {_format_tof(staged)}, {_stage_savings_text(staged, direct)}"
        )
        _plot_transfer_panel(axes[row, 0], direct, direct_title, view="3d")
        _plot_staged_panel(axes[row, 1], staged, staged_title, view="3d")

    fig.suptitle(
        "Elliptical GEO-or-below cases where direct two-burn transfers stay cheaper\n"
        "Right panels show the lowest-Δv staged alternative found; its higher Δv is the reason direct two-burn is preferred.",
        fontsize=15,
    )
    return fig


def _make_split_plane_change_figure(results):
    split_results = results["split_plane_change"]
    case_names = [key.removesuffix(" all departure") for key in split_results if key.endswith(" all departure")]
    fig, axes = plt.subplots(len(case_names), 3, figsize=(21, 7 * len(case_names)), constrained_layout=True, subplot_kw={"projection": "3d"})
    axes = np.asarray(axes).reshape(len(case_names), 3)

    for row, case_name in enumerate(case_names):
        all_departure = split_results[f"{case_name} all departure"]
        all_arrival = split_results[f"{case_name} all arrival"]
        optimized = split_results[f"{case_name} split"]
        inclination = np.rad2deg(optimized["diagnostics"]["inclination_change"])
        split = np.rad2deg(optimized["diagnostics"]["split_inclination"])
        remaining = np.rad2deg(optimized["diagnostics"]["remaining_inclination"])
        arrival_savings = all_arrival["delta_v_total"] - optimized["delta_v_total"]
        departure_savings = all_departure["delta_v_total"] - optimized["delta_v_total"]

        _plot_transfer_panel(
            axes[row, 0],
            all_departure,
            f"{case_name}\nall {inclination:.1f}° plane change in departure burn\n{_format_dv(all_departure)}, {_format_tof(all_departure)}",
            view="3d",
        )
        _plot_transfer_panel(
            axes[row, 1],
            all_arrival,
            f"{case_name}\nall {inclination:.1f}° plane change in arrival burn\n{_format_dv(all_arrival)}, {_format_tof(all_arrival)}",
            view="3d",
        )
        _plot_transfer_panel(
            axes[row, 2],
            optimized,
            f"{case_name}\nsplit plane change: {split:.1f}° + {remaining:.1f}°\n{_format_dv(optimized)}, saves {arrival_savings:.0f} m/s vs arrival-only and {departure_savings:.0f} m/s vs departure-only",
            view="3d",
        )

    fig.suptitle(
        "When two burns beat one concentrated large burn\n"
        "A complete circular-to-circular radius and inclination transfer needs two burns; splitting the plane-change component can lower total Δv.",
        fontsize=15,
    )
    return fig


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    results = _build_maneuver_results(fast=fast)
    summary_delta_v = _summarize_delta_v({key: value for key, value in results.items() if key != "burn_conversion"})
    output_path = None
    staged_output_path = None
    staged_timeline_output_path = None
    elliptical_output_path = None
    split_plane_change_output_path = None
    fig = None
    staged_fig = None
    staged_timeline_fig = None
    elliptical_fig = None
    split_plane_change_fig = None
    if make_figures:
        fig = _make_summary_figure(results)
        output_path = figsave(fig, f"{FIGDIR}/orbital_maneuvers_overview.jpg")
        staged_fig = _make_staged_optimal_figure(results)
        staged_output_path = figsave(staged_fig, f"{FIGDIR}/orbital_maneuvers_staged_optimal.jpg")
        staged_timeline_fig = _make_staged_timeline_figure(results)
        staged_timeline_output_path = figsave(staged_timeline_fig, f"{FIGDIR}/orbital_maneuvers_staged_timeline.jpg")
        elliptical_fig = _make_elliptical_two_burn_figure(results)
        elliptical_output_path = figsave(elliptical_fig, f"{FIGDIR}/orbital_maneuvers_elliptical_two_burn.jpg")
        split_plane_change_fig = _make_split_plane_change_figure(results)
        split_plane_change_output_path = figsave(split_plane_change_fig, f"{FIGDIR}/orbital_maneuvers_split_plane_change.jpg")

    return {
        "title": "Orbital Maneuvers Overview",
        "description": "Consolidated SSATK demo for impulsive, continuous, fixed-time, optimized, staged, and finite-burn maneuvers.",
        "results": results,
        "summary_delta_v": summary_delta_v,
        "figure": fig,
        "staged_figure": staged_fig,
        "staged_timeline_figure": staged_timeline_fig,
        "elliptical_figure": elliptical_fig,
        "split_plane_change_figure": split_plane_change_fig,
        "output_path": output_path,
        "staged_output_path": staged_output_path,
        "staged_timeline_output_path": staged_timeline_output_path,
        "elliptical_output_path": elliptical_output_path,
        "split_plane_change_output_path": split_plane_change_output_path,
        "output_paths": [path for path in (output_path, staged_output_path, staged_timeline_output_path, elliptical_output_path, split_plane_change_output_path) if path is not None],
    }


if __name__ == "__main__":
    main(make_figures=True, fast=False)
