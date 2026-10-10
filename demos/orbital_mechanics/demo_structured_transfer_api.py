"""Readable transfer_optimal API walkthrough for SSATK.

This demo is intentionally written like a help document.  The important
``transfer_optimal(...)`` calls are spelled out directly in ``main`` from top to
bottom.  Plotting and summary helpers are kept below the imports so they do not
hide how the solver is called.

The examples cover:

1. A nested ``problem={...}`` call with SSAPy ``Orbit`` objects.
2. ``inject``, ``intercept``, ``rendezvous``, and ``insertion`` arrival modes.
3. Raw ``r1, v1, r2, v2`` vector inputs.
4. The section-keyword form using ``boundary=``, ``objective=``,
   ``constraints=``, ``route=``, and ``solver=``.
5. Direct, immediate staged, timed staged, best-route, and engine-constrained
   searches.

Each run saves one three-panel overview image under
``~/ssatk_output/figures`` when figures are enabled.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from textwrap import fill


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from ssapy import Orbit, rv
from ssapy.propagator import KeplerianPropagator

from ssatk.constants import EARTH_MU
from ssatk.orbital_mechanics.transfer_optimal_function import transfer_optimal
from ssatk.plots.figpath import figpath
from ssatk.plots.plotutils import figsave


TITLE = "Structured Transfer API"
DESCRIPTION = (
    "Demonstrates readable transfer_optimal calls for structured boundary, "
    "objective, constraint, route, and solver controls."
)
UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None
FIGDIR = "figures"

COMPARISON_TOF = 3600.0
STAGED_LEG_TOF = COMPARISON_TOF / 2.0
DIRECT_TOF_RANGE = (COMPARISON_TOF, COMPARISON_TOF)
STAGED_LEG_TOF_RANGE = (STAGED_LEG_TOF, STAGED_LEG_TOF)

TITLE_FONTSIZE = 26
SUBTITLE_FONTSIZE = 20
AXIS_LABEL_FONTSIZE = 17
TICK_LABEL_FONTSIZE = 14
LEGEND_FONTSIZE = 13
CASE_LABEL_FONTSIZE = 12.5
ANNOTATION_FONTSIZE = 12.5


def _circular_state(radius=7000e3, theta=0.0, inclination=0.0, t=0.0):
    """Return a simple circular state vector in an inclined plane."""
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_inc = np.cos(inclination)
    sin_inc = np.sin(inclination)
    r = radius * np.array([cos_theta, sin_theta * cos_inc, sin_theta * sin_inc])
    v = np.sqrt(EARTH_MU / radius) * np.array([-sin_theta, cos_theta * cos_inc, cos_theta * sin_inc])
    return r, v, t


def _orbit(state):
    """Convert an ``(r, v, t)`` tuple into an SSAPy Orbit."""
    r, v, t = state
    return Orbit(r, v, t=t)


def _record_case(cases, results, rows, *, slug, title, result, designer=True):
    """Store one result for the overview plot and printed summary."""
    diagnostics = result.get("diagnostics", {})
    cases.append({"slug": slug, "title": title, "designer": designer})
    results[slug] = result
    rows.append(
        {
            "case": slug,
            "title": title,
            "method": result.get("method"),
            "arrival_mode": diagnostics.get("arrival_mode"),
            "route_mode": diagnostics.get("route_mode"),
            "delta_v_total_m_s": float(result.get("delta_v_total", np.nan)),
            "burn_count": len(result.get("burns", [])),
            "tof_s": float(result.get("tof", np.nan)),
            "problem_schema": diagnostics.get("problem_schema"),
        }
    )


def _burn_value(burn, key, default=None):
    return burn.get(key, default) if isinstance(burn, dict) else getattr(burn, key, default)


def _transfer_time_bounds(result):
    trajectory = result.get("trajectory")
    if trajectory is not None and trajectory.get("t") is not None:
        times = np.asarray(trajectory["t"], dtype=float)
        return float(times[0]), float(times[-1])
    burns = result.get("burns", [])
    if not burns:
        return 0.0, float(result.get("tof", 0.0))
    starts = [_burn_value(burn, "t_start", _burn_value(burn, "t", 0.0)) for burn in burns]
    ends = [_burn_value(burn, "t_end", start) for burn, start in zip(burns, starts)]
    return float(min(starts)), float(max(ends))


def _burn_duration(burn):
    duration = _burn_value(burn, "duration", None)
    if duration is not None:
        return float(duration)
    start = _burn_value(burn, "t_start", _burn_value(burn, "t", 0.0))
    end = _burn_value(burn, "t_end", start)
    return max(0.0, float(end) - float(start))


def _burn_acceleration(burn):
    acceleration = _burn_value(burn, "acceleration_mag", None)
    if acceleration is not None:
        return float(acceleration)
    duration = _burn_duration(burn)
    delta_v = float(_burn_value(burn, "delta_v_mag", _burn_value(burn, "dv_mag", 0.0)))
    return delta_v / duration if duration > 0.0 else 0.0


def _case_number(case):
    return case["slug"].split("_", 1)[0]


def _wrapped_case_label(case, width=24):
    wrapped = fill(case["title"], width=width)
    return f"{_case_number(case)} {wrapped.replace(chr(10), chr(10) + '   ')}"


def _sample_transfer_leg(leg, samples=90):
    transfer_orbits = leg.get("transfer_orbits") or []
    if not transfer_orbits:
        trajectory = leg.get("trajectory")
        if trajectory is None or trajectory.get("r") is None or trajectory.get("t") is None:
            return None, None
        return np.asarray(trajectory["t"], dtype=float), np.asarray(trajectory["r"], dtype=float)

    t0 = float(leg["initial"]["t"])
    t1 = float(leg["final"]["t"])
    times = np.linspace(t0, t1, max(2, int(samples)))
    positions, _velocities = rv(transfer_orbits[0], times, propagator=KeplerianPropagator())
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    positions[0] = np.asarray(leg["initial"]["r"], dtype=float).reshape(3)
    positions[-1] = np.asarray(leg["final"]["r"], dtype=float).reshape(3)
    return times, positions


def _planned_trajectory_samples(result, samples_per_leg=90):
    stage_legs = result.get("stage_legs") or []
    if stage_legs:
        times = []
        positions = []
        leg_samples = max(16, int(samples_per_leg) // max(1, len(stage_legs)))
        for leg_index, leg in enumerate(stage_legs):
            leg_times, leg_positions = _sample_transfer_leg(leg, leg_samples)
            if leg_times is None:
                continue
            if leg_index and len(leg_times) > 1:
                leg_times = leg_times[1:]
                leg_positions = leg_positions[1:]
            times.append(leg_times)
            positions.append(leg_positions)
        if times:
            return np.concatenate(times), np.vstack(positions)

    return _sample_transfer_leg(result, samples_per_leg)


def _draw_trajectory_overview(ax, results, cases, colors):
    for index, case in enumerate(cases):
        result = results[case["slug"]]
        times, positions = _planned_trajectory_samples(result, samples_per_leg=120)
        if positions is None:
            continue
        color = colors(index % 10)
        ax.plot(positions[:, 0] / 1e3, positions[:, 1] / 1e3, color=color, lw=2.0, label=_case_number(case))
        ax.scatter(positions[0, 0] / 1e3, positions[0, 1] / 1e3, color="blue", s=55, zorder=4)
        ax.scatter(positions[-1, 0] / 1e3, positions[-1, 1] / 1e3, color="green", s=55, zorder=4)
        mid_index = len(positions) // 2
        ax.annotate(
            _case_number(case),
            (positions[mid_index, 0] / 1e3, positions[mid_index, 1] / 1e3),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=ANNOTATION_FONTSIZE,
            color=color,
            weight="bold",
        )

    from matplotlib.patches import Circle

    earth = Circle((0.0, 0.0), 6378.137, color="0.70", alpha=0.35)
    ax.add_patch(earth)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("GCRF x [km]", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("GCRF y [km]", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Transfer Trajectories\nblue=start, green=end", fontsize=SUBTITLE_FONTSIZE)
    ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(alpha=0.25)
    ax.legend(title="case", title_fontsize=LEGEND_FONTSIZE, fontsize=LEGEND_FONTSIZE, ncol=3, loc="lower left")

def _draw_burn_event_overview(ax, results, cases, colors):
    all_burns = [burn for case in cases for burn in results[case["slug"]].get("burns", [])]
    max_delta_v = max((float(_burn_value(burn, "delta_v_mag", 0.0)) for burn in all_burns), default=1.0)
    max_accel = max((_burn_acceleration(burn) for burn in all_burns), default=1.0)
    y_positions = np.arange(len(cases))[::-1]
    y_labels = []

    for index, (case, y_position) in enumerate(zip(cases, y_positions)):
        result = results[case["slug"]]
        color = colors(index % 10)
        t0, t1 = _transfer_time_bounds(result)
        duration_hours = max((t1 - t0) / 3600.0, 1e-6)
        y_labels.append(_wrapped_case_label(case))
        ax.hlines(y_position, 0.0, duration_hours, color="0.82", lw=4.0, zorder=0)
        for burn_index, burn in enumerate(result.get("burns", []), 1):
            start = float(_burn_value(burn, "t_start", _burn_value(burn, "t", t0)))
            end = float(_burn_value(burn, "t_end", start))
            middle_hours = (0.5 * (start + end) - t0) / 3600.0
            delta_v = float(_burn_value(burn, "delta_v_mag", _burn_value(burn, "dv_mag", 0.0)))
            acceleration = _burn_acceleration(burn)
            marker_size = 80.0 + 360.0 * np.sqrt(delta_v / max_delta_v) if max_delta_v > 0.0 else 120.0
            alpha = 0.35 + 0.55 * (acceleration / max_accel) if max_accel > 0.0 else 0.75
            ax.scatter(
                middle_hours,
                y_position,
                s=marker_size,
                color=color,
                alpha=min(alpha, 0.95),
                edgecolor="k",
                linewidth=0.45,
                zorder=3,
            )
            ax.annotate(
                f"{burn_index}: {delta_v:.0f}",
                (middle_hours, y_position),
                textcoords="offset points",
                xytext=(0, 8 if burn_index % 2 else -13),
                ha="center",
                va="bottom" if burn_index % 2 else "top",
                fontsize=ANNOTATION_FONTSIZE,
                color="0.15",
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=CASE_LABEL_FONTSIZE, linespacing=1.15)
    ax.set_xlabel("time since departure [h]", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Burn Events\n(size = delta-v; opacity = acceleration)", fontsize=SUBTITLE_FONTSIZE)
    ax.tick_params(axis="x", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(axis="x", alpha=0.25)
    ax.set_ylim(-0.75, len(cases) - 0.25)


def _draw_objective_overview(ax, results, cases, colors):
    for index, case in enumerate(cases):
        result = results[case["slug"]]
        diagnostics = result.get("diagnostics", {})
        color = colors(index % 10)
        pareto = diagnostics.get("pareto") if case.get("designer") else None
        if pareto is not None:
            tof = np.asarray(pareto.get("tof", []), dtype=float) / 3600.0
            delta_v = np.asarray(pareto.get("dv", []), dtype=float)
            if tof.size and delta_v.size:
                ax.plot(tof, delta_v, color=color, lw=2.0, alpha=0.55)
        selected_tof = float(result.get("tof", 0.0)) / 3600.0
        selected_delta_v = float(diagnostics.get("objective_delta_v", result.get("delta_v_total", 0.0)))
        marker = "s" if result.get("method") == "transfer_optimal_staged" else "o"
        ax.scatter(selected_tof, selected_delta_v, color=color, marker=marker, s=135, edgecolor="k", linewidth=0.8, zorder=4)
        ax.annotate(
            _case_number(case),
            (selected_tof, selected_delta_v),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=ANNOTATION_FONTSIZE + 1,
            color=color,
            weight="bold",
        )
        dv_budget = diagnostics.get("dv_budget")
        if dv_budget is not None:
            ax.axhline(float(dv_budget), color=color, ls=":", lw=1.4, alpha=0.45)

    ax.set_xlabel("time of flight [h]", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("objective delta-v [m/s]", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(
        "Objective / Design Trade\n"
        "Pareto lines = direct cases\n"
        "markers = selected solutions",
        fontsize=SUBTITLE_FONTSIZE,
    )
    ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(alpha=0.3)


def _save_three_panel_overview(results, cases, *, make_figures=True):
    if not make_figures:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(34, 14.5),
        gridspec_kw={"width_ratios": [1.1, 1.9, 1.1]},
    )
    _draw_trajectory_overview(axes[0], results, cases, colors)
    _draw_burn_event_overview(axes[1], results, cases, colors)
    _draw_objective_overview(axes[2], results, cases, colors)
    fig.suptitle("Structured Transfer API Overview", fontsize=TITLE_FONTSIZE)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), w_pad=3.0)

    save_rel = f"{FIGDIR}/structured_transfer_api_overview.jpg"
    figsave(fig, save_rel)
    return figpath(save_rel)


def main(make_figures=None, fast=None):
    if make_figures is None:
        make_figures = not UNDER_PYTEST
    if fast is None:
        fast = UNDER_PYTEST

    solver_samples = 40 if fast else 90
    direct_grid = (1, 2) if fast else (1, 3)
    direct_single_cell_grid = (1, 1)
    staged_grid = (2, 2)

    # ------------------------------------------------------------------
    # Build one common scenario used by every example below.
    # ------------------------------------------------------------------
    target_radius = 9000e3
    target_inclination = np.deg2rad(8.0)
    target_initial_theta = 0.35
    target_arrival_theta = target_initial_theta + np.sqrt(EARTH_MU / target_radius**3) * COMPARISON_TOF

    initial_state = _circular_state(7000e3)
    target_state = _circular_state(
        target_radius,
        theta=target_arrival_theta,
        inclination=target_inclination,
        t=COMPARISON_TOF,
    )

    initial_orbit = _orbit(initial_state)
    target_orbit = _orbit(target_state)
    r1, v1, _t1 = initial_state
    r2, v2, _t2 = target_state

    stage_candidates = {
        "radii": [8500e3],
        "plane_fractions": [0.0, 0.5, 1.0],
        "phase_count": 32,
    }

    cases = []
    results = {}
    rows = []
    output_paths = []

    # ------------------------------------------------------------------
    # 01. Nested problem schema with SSAPy Orbit objects.
    # ------------------------------------------------------------------
    direct_rendezvous = transfer_optimal(
        problem={
            "boundary": {
                "initial": initial_orbit,
                "target": target_orbit,
                "departure_mode": "leave now",
                "arrival_mode": "rendezvous",
            },
            "objective": {
                "minimize": "delta_v",
                "delta_v_mode": "total",
            },
            "constraints": {
                "tof_range": DIRECT_TOF_RANGE,
                "perigee_altitude_min": 100e3,
                "max_burns": 2,
            },
            "route": "direct",
            "solver": {
                "n_grid": direct_grid,
                "polish": False,
                "refine": False,
                "propagate": False,
                "samples": solver_samples,
                "burn_duration": 1.0,
            },
        }
    )
    _record_case(
        cases,
        results,
        rows,
        slug="01_direct_rendezvous_orbits",
        title="Orbit objects: direct rendezvous, total delta-v",
        result=direct_rendezvous,
        designer=True,
    )

    # ------------------------------------------------------------------
    # 02. Intercept: reach target position at the chosen time, but do not
    # match target velocity.  This is a one-burn result.
    # ------------------------------------------------------------------
    intercept_first_burn = transfer_optimal(
        problem={
            "boundary": {
                "initial": initial_orbit,
                "target": target_orbit,
                "departure_mode": "leave now",
                "arrival_mode": "intercept",
            },
            "objective": {
                "minimize": "first_burn",
            },
            "constraints": {
                "tof_range": DIRECT_TOF_RANGE,
                "max_burns": 1,
            },
            "route": "direct",
            "solver": {
                "n_grid": direct_grid,
                "polish": False,
                "refine": False,
                "propagate": False,
                "samples": solver_samples,
                "burn_duration": 1.0,
            },
        }
    )
    _record_case(
        cases,
        results,
        rows,
        slug="02_intercept_first_burn",
        title="Intercept: first burn only, no arrival match",
        result=intercept_first_burn,
        designer=True,
    )

    # ------------------------------------------------------------------
    # 02b. Inject: search a free target-orbit phase for the departure burn
    # that places the spacecraft onto a transfer. No final burn is included.
    # ------------------------------------------------------------------
    inject_first_burn = transfer_optimal(
        problem={
            "boundary": {
                "initial": initial_orbit,
                "target": target_orbit,
                "departure_mode": "leave now",
                "arrival_mode": "inject",
            },
            "objective": {
                "minimize": "first_burn",
            },
            "constraints": {
                "tof_range": DIRECT_TOF_RANGE,
                "max_burns": 1,
            },
            "route": "direct",
            "solver": {
                "n_grid": direct_grid,
                "n_phase": 3,
                "polish": False,
                "refine": False,
                "propagate": False,
                "samples": solver_samples,
                "burn_duration": 1.0,
            },
        }
    )
    _record_case(
        cases,
        results,
        rows,
        slug="02b_inject_free_phase_first_burn",
        title="Inject: free-phase first burn only",
        result=inject_first_burn,
        designer=True,
    )

    # ------------------------------------------------------------------
    # 03. Insertion: search the target-orbit phase and include the final burn
    # that matches target-orbit velocity.
    # ------------------------------------------------------------------
    insertion_arrival_burn = transfer_optimal(
        problem={
            "boundary": {
                "initial": initial_orbit,
                "target": target_orbit,
                "departure_mode": "leave now",
                "arrival_mode": "insertion",
            },
            "objective": {
                "minimize": "arrival_burn",
            },
            "constraints": {
                "tof_range": DIRECT_TOF_RANGE,
                "max_burns": 2,
            },
            "route": "direct",
            "solver": {
                "n_grid": direct_grid,
                "n_phase": 1,
                "polish": False,
                "refine": False,
                "propagate": False,
                "samples": solver_samples,
                "burn_duration": 1.0,
            },
        }
    )
    _record_case(
        cases,
        results,
        rows,
        slug="03_insertion_arrival_burn",
        title="Insertion: free-phase arrival burn",
        result=insertion_arrival_burn,
        designer=True,
    )

    # ------------------------------------------------------------------
    # 04. Time objective with a delta-v budget.  The fixed TOF range keeps this
    # demo comparable with the other cases; widen it in real design trades.
    # ------------------------------------------------------------------
    min_time_budget = transfer_optimal(
        problem={
            "boundary": {
                "initial": initial_orbit,
                "target": target_orbit,
                "departure_mode": "leave now",
                "arrival_mode": "rendezvous",
            },
            "objective": {
                "minimize": "time",
                "delta_v_mode": "total",
            },
            "constraints": {
                "tof_range": DIRECT_TOF_RANGE,
                "dv_budget": 10000.0,
                "max_burns": 2,
            },
            "route": "direct",
            "solver": {
                "n_grid": direct_grid,
                "polish": False,
                "refine": False,
                "propagate": False,
                "samples": solver_samples,
                "burn_duration": 1.0,
            },
        }
    )
    _record_case(
        cases,
        results,
        rows,
        slug="04_min_time_budget",
        title="Time objective at the fixed comparison arrival",
        result=min_time_budget,
        designer=True,
    )

    # ------------------------------------------------------------------
    # 05. Raw vector input.  Use this when you have state vectors instead of
    # SSAPy Orbit objects.
    # ------------------------------------------------------------------
    raw_vectors_arrival_window = transfer_optimal(
        problem={
            "r1": r1,
            "v1": v1,
            "r2": r2,
            "v2": v2,
            "t2": COMPARISON_TOF,
            "departure_mode": "now",
            "arrival_mode": "rendezvous",
            "objective": {
                "minimize": "delta_v",
            },
            "constraints": {
                "tof_range": DIRECT_TOF_RANGE,
                "arrival_window": (COMPARISON_TOF, COMPARISON_TOF),
                "max_burns": 2,
            },
            "route": "direct",
            "solver": {
                "n_grid": direct_single_cell_grid,
                "polish": False,
                "refine": False,
                "propagate": False,
                "samples": solver_samples,
                "burn_duration": 1.0,
            },
        }
    )
    _record_case(
        cases,
        results,
        rows,
        slug="05_raw_vectors_arrival_window",
        title="Raw vectors: fixed arrival window constraint",
        result=raw_vectors_arrival_window,
        designer=True,
    )

    # ------------------------------------------------------------------
    # 06. Section-keyword form.  This is equivalent to the nested problem
    # schema, but each section is passed as its own top-level keyword.
    # ------------------------------------------------------------------
    immediate_staged_sections = transfer_optimal(
        boundary={
            "initial": initial_orbit,
            "target": target_orbit,
            "departure_mode": "now",
            "arrival_mode": "rendezvous",
        },
        objective={
            "minimize": "delta_v",
            "delta_v_mode": "total",
        },
        constraints={
            "tof_range": STAGED_LEG_TOF_RANGE,
            "max_burns": 4,
        },
        route={
            "mode": "immediate",
            "n_stage_stops": 1,
            "stage_candidates": stage_candidates,
        },
        solver={
            "n_grid": staged_grid,
            "polish": False,
            "refine": False,
            "propagate": False,
            "samples": solver_samples,
            "burn_duration": 1.0,
        },
    )
    _record_case(
        cases,
        results,
        rows,
        slug="06_immediate_staged_sections",
        title="Section keywords: immediate staged transfer",
        result=immediate_staged_sections,
        designer=False,
    )

    # ------------------------------------------------------------------
    # 07. Timed staged route.  The route section lets the post-stage leg wait
    # for an optimized departure phase instead of leaving immediately.
    # ------------------------------------------------------------------
    timed_multistage = transfer_optimal(
        problem={
            "boundary": {
                "initial": initial_orbit,
                "target": target_orbit,
                "departure_mode": "now",
                "arrival_mode": "rendezvous",
            },
            "objective": {
                "minimize": "delta_v",
                "delta_v_mode": "total",
            },
            "constraints": {
                "tof_range": STAGED_LEG_TOF_RANGE,
                "max_burns": 4,
            },
            "route": {
                "mode": "multi_stage",
                "timing": "optimized",
                "n_stage_stops": 1,
                "stage_candidates": stage_candidates,
            },
            "solver": {
                "n_grid": staged_grid,
                "polish": False,
                "refine": False,
                "propagate": False,
                "samples": solver_samples,
                "burn_duration": 1.0,
            },
        }
    )
    _record_case(
        cases,
        results,
        rows,
        slug="07_timed_multistage",
        title="Problem schema: timed multi-stage route",
        result=timed_multistage,
        designer=False,
    )

    # ------------------------------------------------------------------
    # 08. Best-route mode.  Here max_burns=2 prevents a four-burn staged route,
    # so the solver falls back to a direct transfer.
    # ------------------------------------------------------------------
    best_route_burn_limit = transfer_optimal(
        problem={
            "boundary": {
                "initial": initial_orbit,
                "target": target_orbit,
                "departure_mode": "leave now",
                "arrival_mode": "rendezvous",
            },
            "objective": {
                "minimize": "delta_v",
                "delta_v_mode": "total",
            },
            "constraints": {
                "tof_range": DIRECT_TOF_RANGE,
                "max_burns": 2,
            },
            "route": {
                "mode": "best",
                "n_stage_stops": 1,
            },
            "solver": {
                "n_grid": staged_grid,
                "polish": False,
                "refine": False,
                "propagate": False,
                "samples": solver_samples,
                "burn_duration": 1.0,
            },
        }
    )
    _record_case(
        cases,
        results,
        rows,
        slug="08_best_route_burn_limit",
        title="Best route: direct fallback under max_burns",
        result=best_route_burn_limit,
        designer=True,
    )

    # ------------------------------------------------------------------
    # 09. Engine constraints.  Supplying thrust/mass/isp sizes finite burns and
    # adds propellant estimates while preserving the same boundary format.
    # ------------------------------------------------------------------
    engine_constraints = transfer_optimal(
        problem={
            "boundary": {
                "initial": initial_orbit,
                "target": target_orbit,
                "departure_mode": "leave now",
                "arrival_mode": "rendezvous",
            },
            "objective": {
                "minimize": "delta_v",
                "delta_v_mode": "total",
            },
            "constraints": {
                "tof_range": DIRECT_TOF_RANGE,
                "max_burns": 2,
                "thrust": 10000.0,
                "mass": 1000.0,
                "isp": 300.0,
            },
            "route": "direct",
            "solver": {
                "n_grid": direct_grid,
                "polish": False,
                "refine": False,
                "propagate": False,
                "samples": solver_samples,
                "burn_duration": 1.0,
            },
        }
    )
    _record_case(
        cases,
        results,
        rows,
        slug="09_engine_constraints",
        title="Engine constraints: thrust, mass, and specific impulse",
        result=engine_constraints,
        designer=True,
    )

    overview_path = _save_three_panel_overview(results, cases, make_figures=make_figures)
    if overview_path is not None:
        output_paths.append(overview_path)

    print("Structured transfer API cases:")
    for row in rows:
        print(
            f"  {row['case']}: {row['method']} | {row['arrival_mode']} | "
            f"route={row['route_mode']} | burns={row['burn_count']} | "
            f"dv={row['delta_v_total_m_s']:.1f} m/s | tof={row['tof_s']:.1f} s"
        )

    return {
        "title": TITLE,
        "description": DESCRIPTION,
        "results": results,
        "summary": rows,
        "output_paths": output_paths,
    }


if __name__ == "__main__":
    main(make_figures=True, fast=False)
