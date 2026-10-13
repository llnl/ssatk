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

The dashboard separates total burn budgets, planned burn timing and two
representative routes. A separate fixed-departure rendezvous search samples
35--120 minutes of flight time; it is not a comparison of unlike objectives.
The plotted coast arcs are Keplerian plans, not finite-thrust propagation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from ssapy import Orbit, rv
from ssapy.propagator import KeplerianPropagator

from ssatk.constants import EARTH_MU, EARTH_RADIUS
from ssatk.orbital_mechanics.transfer_optimal_function import transfer_optimal
from ssatk.plots.figpath import figpath
from ssatk.plots.plotutils import figsave


TITLE = "Transfer Planning: Routes, Burns, and Flight Time"
DESCRIPTION = (
    "Compare direct and staged routes, total burn budgets, and planned burn "
    "timing; explore a sampled 35–120 minute rendezvous trade with transfer_optimal."
)
UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None
FIGDIR = "figures"

COMPARISON_TOF = 3600.0
STAGED_LEG_TOF = COMPARISON_TOF / 2.0
DIRECT_TOF_RANGE = (COMPARISON_TOF, COMPARISON_TOF)
STAGED_LEG_TOF_RANGE = (STAGED_LEG_TOF, STAGED_LEG_TOF)

BURN_COLORS = ("#087e9b", "#c07824", "#7560aa")
INK = "#20324b"
MUTED = "#5b6b80"
CASE_LABELS = {
    "01": "Direct rendezvous",
    "02": "Intercept",
    "02b": "Inject, free phase",
    "03": "Insertion, arrival objective",
    "04": "Time objective, fixed time",
    "05": "Raw-vector input",
    "06": "Staged: immediate",
    "07": "Staged: timed",
    "08": "Best route, ≤2 burns",
    "09": "Engine sizing",
}


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
    # A single departure burn does not mark the end of the transfer coast.
    if "initial" in result and "final" in result:
        return float(result["initial"]["t"]), float(result["final"]["t"])
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


def _case_number(case):
    return case["slug"].split("_", 1)[0]


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


def _flight_time_study(initial_orbit, target_orbit, *, fast):
    """One comparable objective at every sample: direct rendezvous total delta-v."""
    problem = {
        "boundary": {"initial": initial_orbit, "target": target_orbit,
                     "departure_mode": "now", "arrival_mode": "rendezvous"},
        "objective": {"minimize": "delta_v", "delta_v_mode": "total"},
        "constraints": {"tof_range": (35 * 60.0, 120 * 60.0),
                        "perigee_altitude_min": 100e3, "max_burns": 2},
        "route": "direct",
        "solver": {"n_grid": (1, 69 if fast else 137), "polish": False,
                   "refine": False, "propagate": False, "samples": 40,
                   "burn_duration": 1.0},
    }
    search = transfer_optimal(problem=problem)
    grid = search["diagnostics"]["pareto"]
    tofs = np.asarray(grid["tof"], dtype=float)
    costs = np.full(len(tofs), np.nan)
    best = None
    # The search grid uses a coarser Lambert tolerance than a finalized plan.
    # Finalize each accepted sample so the curve and the 60-min bars report
    # the same burn-cost definition, rather than mixing the two tolerances.
    for i in np.flatnonzero(np.isfinite(grid["dv"])):
        point = dict(problem,
                     constraints=dict(problem["constraints"], tof_range=(tofs[i], tofs[i])),
                     solver=dict(problem["solver"], n_grid=(1, 1)))
        result = transfer_optimal(problem=point)
        costs[i] = result["delta_v_total"]
        if best is None or costs[i] < best["delta_v_total"]:
            best = result
    best["diagnostics"]["flight_time_study"] = dict(tof=tofs, dv=costs)
    return best


def _burn_budget(result):
    """Departure / intermediate / final matching burns, all in m/s."""
    values = [float(_burn_value(b, "delta_v_mag", 0.0)) for b in result["burns"]]
    return np.array([values[0], sum(values[1:-1]), values[-1] if len(values) > 1 else 0.0])


def _schedule_groups(results, cases):
    """Group matching schedules to 1 microsecond and 1 micrometre/s precision."""
    groups = {}
    for case in cases:
        result = results[case["slug"]]
        key = (_transfer_time_bounds(result), tuple(
            tuple(round(float(_burn_value(b, k, 0.0)), 6)
                  for k in ("t_start", "t_end", "delta_v_mag"))
            for b in result["burns"]))
        if key not in groups:
            groups[key] = {"result": result, "numbers": []}
        groups[key]["numbers"].append(_case_number(case))
    return list(groups.values())


def _style_axis(ax):
    ax.set_facecolor("white")
    ax.tick_params(labelsize=10, colors=MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    for name in ("left", "bottom"):
        ax.spines[name].set_color("#cbd5e1")
    ax.set_axisbelow(True)
    ax.grid(color="#e5eaf0", lw=0.7)


def _draw_routes(ax, results):
    from matplotlib.patches import Circle

    direct = results["01_direct_rendezvous_orbits"]
    staged = results["06_immediate_staged_sections"]
    for result, color, label in ((direct, BURN_COLORS[0], "Direct · case 01"),
                                 (staged, BURN_COLORS[1], "Staged · case 06")):
        _, positions = _planned_trajectory_samples(result, samples_per_leg=480)
        ax.plot(*positions[:, :2].T / 1e3, color=color, lw=2.6, label=label)
    for state, label in ((direct["initial"], "Initial orbit"), (direct["final"], "Target orbit")):
        orbit = _orbit((state["r"], state["v"], state["t"]))
        period = 2 * np.pi * np.sqrt(np.linalg.norm(state["r"])**3 / EARTH_MU)
        positions, _ = rv(orbit, np.linspace(state["t"], state["t"] + period, 240),
                          propagator=KeplerianPropagator())
        ax.plot(*positions[:, :2].T / 1e3, color="#9cabbc", lw=0.9, ls=":")
    ax.add_patch(Circle((0, 0), EARTH_RADIUS / 1e3, facecolor="#d9e9f2", edgecolor="#9db9cb"))
    ax.text(0, -900, "Earth", ha="center", color="#52738b", fontsize=13)
    for point, text, offset, align in (
        (direct["initial"]["r"], "Departure", (-6, -30), "right"),
        (direct["final"]["r"], "Arrival", (4, 20), "left"),
        (staged["stage_legs"][0]["final"]["r"], "Stage stop", (22, 7), "left"),
    ):
        xy = np.asarray(point)[:2] / 1e3
        ax.scatter(*xy, color=INK, s=26, zorder=5)
        ax.annotate(text, xy, xytext=offset, textcoords="offset points", ha=align,
                    fontsize=11, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.set(xlim=(-10400, 10400), ylim=(-9600, 10300), xlabel="GCRF x [km]", ylabel="GCRF y [km]")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower center", frameon=False, fontsize=10, ncol=2)


def _draw_budgets(ax, results, cases):
    from matplotlib.patches import Patch

    y = np.arange(len(cases))[::-1]
    budgets = np.array([_burn_budget(results[c["slug"]]) for c in cases])
    for row in y[::2]:
        ax.axhspan(row - .48, row + .48, color="#f4f7fa", zorder=0)
    left = np.zeros(len(cases))
    for component, color in enumerate(BURN_COLORS):
        values = budgets[:, component]
        ax.barh(y, values, left=left, height=.57, color=color, zorder=2)
        for yi, x, width in zip(y, left, values):
            if width > 200:
                ax.text(x + width / 2, yi, f"{width:,.0f}", ha="center", va="center",
                        fontsize=10, color="white")
        left += values
    for yi, total in zip(y, left):
        ax.text(total + 55, yi, f"{total:,.0f}", va="center", fontsize=11, weight="bold", color=INK)
    ax.set_yticks(y, [f"{_case_number(c)}   {CASE_LABELS[_case_number(c)]}" for c in cases], fontsize=11)
    ax.tick_params(axis="y", length=0)
    ax.set(xlim=(0, max(left) * 1.21), ylim=(-.6, len(cases) - .4), xlabel="Total planned Δv [m/s]")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(color=color, label=label) for color, label in zip(
        BURN_COLORS, ("Departure", "Stage burns", "Arrival match"))],
        loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False, fontsize=10, ncol=3,
        borderaxespad=0, columnspacing=1)


def _draw_timeline(ax, results, cases):
    groups = _schedule_groups(results, cases)
    labels = []
    for row, group in enumerate(groups):
        y = len(groups) - row - 1
        result = group["result"]
        burns = result["burns"]
        start, stop = _transfer_time_bounds(result)
        engine = any(_burn_value(b, "thrust") is not None for b in burns)
        title = ("Engine sizing" if engine else "Departure only" if len(burns) == 1
                 else "Staged route" if len(burns) > 2 else "Two burns")
        labels.append(f"{title}\n{' · '.join(group['numbers'])}")
        ax.hlines(y, 0, (stop - start) / 60, color="#c9d3df", lw=2)
        for i, burn in enumerate(burns):
            t0 = (float(_burn_value(burn, "t_start")) - start) / 60
            t1 = (float(_burn_value(burn, "t_end")) - start) / 60
            dv = float(_burn_value(burn, "delta_v_mag"))
            color = BURN_COLORS[0 if i == 0 else 2 if i == len(burns) - 1 else 1]
            ax.plot([t0, t1], [y, y], color=color, lw=7, solid_capstyle="butt")
            x = (t0 + t1) / 2
            ax.scatter(x, y, color=color, s=28, zorder=3)
            # Two stage burns occur at almost the same epoch: label opposite sides.
            lower = len(burns) > 2 and i == 1
            align = "left" if i == 0 else "right" if i == len(burns) - 1 else "center"
            ax.annotate(f"B{i + 1} · {dv:,.0f}", (x, y), xytext=(0, -16 if lower else 13),
                        textcoords="offset points", ha=align, va="top" if lower else "bottom",
                        fontsize=10, color=INK)
            if engine:
                ax.annotate(f"{_burn_duration(burn):.0f} s", (x, y), xytext=(0, -13),
                            textcoords="offset points", ha=align, va="top", fontsize=10, color=MUTED)
    ax.set_yticks(np.arange(len(groups))[::-1], labels, fontsize=10)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.grid(axis="y", visible=False)
    ax.set(xlim=(-3, 63), ylim=(-.6, len(groups) - .35), xlabel="Minutes since departure")


def _draw_flight_time_trade(ax, trade, direct):
    pareto = trade["diagnostics"]["flight_time_study"]
    minutes = np.asarray(pareto["tof"]) / 60
    dv = np.asarray(pareto["dv"])
    valid = np.isfinite(dv)
    ax.plot(minutes, dv, color=BURN_COLORS[0], lw=2, label="Direct rendezvous")
    best = int(np.nanargmin(dv))
    ax.scatter(minutes[best], dv[best], marker="D", s=55, color=BURN_COLORS[1], zorder=4,
               label="Lowest sampled Δv")
    ax.scatter(direct["tof"] / 60, direct["delta_v_total"], s=70, facecolor="white",
               edgecolor=INK, linewidth=1.5, zorder=5, label="60 min API comparison")
    # Show rejected samples without inventing an interpolated feasible boundary.
    rejected = minutes[~valid]
    if len(rejected):
        ax.scatter(rejected, np.full(len(rejected), .965), transform=ax.get_xaxis_transform(),
                   marker="x", s=20, color="#9ca6b5", label="No feasible grid solution")
    ax.set(xlim=(minutes[0] - 1, minutes[-1] + 1), xlabel="Flight time [min]", ylabel="Total Δv [m/s]")
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)


def _build_overview(results, cases, trade):
    """Build four panels with separate geometry, budgets, timing and comparable trade."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": INK, "axes.labelcolor": MUTED}):
        fig = plt.figure(figsize=(20, 15), facecolor="#f5f7fb")
        fig.text(.05, .962, "Transfer planning: what changes, and what costs more?", fontsize=26, weight="bold")
        fig.text(.05, .936, "SSATK structured transfer API  •  7,000 → 9,000 km orbital radius  •  8° target-plane inclination", fontsize=13, color=MUTED)
        direct = results["01_direct_rendezvous_orbits"]
        staged = results["06_immediate_staged_sections"]
        extra = 100 * (staged["delta_v_total"] / direct["delta_v_total"] - 1)
        cards = [
            ("DIRECT RENDEZVOUS", f"{direct['delta_v_total']:,.0f} m/s", "Departure + arrival matching, at 60 min"),
            ("STAGED ROUTE, SAME ARRIVAL TIME", f"{extra:+.1f}% Δv", f"{staged['delta_v_total']:,.0f} m/s across four planned burns"),
            ("SEPARATE FLIGHT-TIME STUDY", "35–120 min", "Same departure; total rendezvous Δv at each sample"),
        ]
        for x, (label, value, detail) in zip((.05, .36, .67), cards):
            fig.add_artist(FancyBboxPatch((x, .85), .28, .069, boxstyle="round,pad=0.008",
                                         transform=fig.transFigure, facecolor="white", edgecolor="#dde4ee"))
            fig.text(x + .01, .9, label, fontsize=10, color=MUTED)
            fig.text(x + .01, .873, value, fontsize=23, weight="bold", color=BURN_COLORS[0])
            fig.text(x + .01, .852, detail, fontsize=10, color=MUTED)
        for x, y, title, subtitle in (
            (.05, .817, "A   Two routes, one arrival", "Geocentric celestial reference frame (GCRF), x–y projection"),
            (.53, .817, "B   Where the Δv is spent", "All ten API cases on the same total-burn-cost basis"),
            (.05, .375, "C   When the burns happen", "Equivalent schedules grouped; burn labels give Δv in m/s"),
            (.56, .375, "D   Does more flight time help?", "Fixed departure · rendezvous · perigee altitude ≥100 km"),
        ):
            fig.text(x, y, title, fontsize=18, weight="bold")
            fig.text(x, y - .02, subtitle, fontsize=11, color=MUTED)
        axes = [fig.add_axes(bounds) for bounds in (
            (.05, .483, .40, .292), (.645, .483, .30, .292),
            (.16, .126, .32, .20), (.61, .126, .335, .20))]
        for ax in axes:
            _style_axis(ax)
        _draw_routes(axes[0], results)
        _draw_budgets(axes[1], results, cases)
        _draw_timeline(axes[2], results, cases)
        _draw_flight_time_trade(axes[3], trade, direct)
        fig.text(.05, .424, "Representative cases 01 and 06; coincident API routes are not overplotted.\nDotted curves show the initial and target orbits. Earth uses its physical radius.", fontsize=10, color=MUTED, linespacing=1.6)
        fig.text(.53, .424, "02 / 02b omit arrival matching; 03 optimizes only the arrival burn.\n04 holds time fixed, so its API example does not demonstrate a faster arrival.", fontsize=10, color=MUTED, linespacing=1.6)
        fig.text(.05, .072, "Dots: nominal 1 s burns. Thick segments: engine-sized windows.\nStage arrival and departure burns are labelled on opposite sides.", fontsize=10, color=MUTED, linespacing=1.6)
        p = trade["diagnostics"]["flight_time_study"]
        best = int(np.nanargmin(p["dv"]))
        step = np.diff(p["tof"])[0] / 60
        fig.text(.56, .072, f"Lowest sampled cost: {p['dv'][best]:,.0f} m/s at {p['tof'][best] / 60:.2f} min.\nGrid spacing: {step:g} min; this is a sampled result, not a global optimum.", fontsize=10, color=MUTED, linespacing=1.6)
        fig.text(.05, .027, "MODEL: two-body Lambert / Kepler planning. Coast arcs are not finite-thrust propagation; engine durations are sizing estimates.\nCase numbers map to the explicit transfer_optimal calls in demo_structured_transfer_api.py. Synthetic scenario; no flight data.", fontsize=10, color=MUTED, linespacing=1.6)
    return fig


def _save_overview(results, cases, trade, *, make_figures=True):
    if not make_figures:
        return None
    fig = _build_overview(results, cases, trade)
    save_rel = f"{FIGDIR}/structured_transfer_api_overview.jpg"
    return figsave(fig, figpath(save_rel))


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

    flight_time_study = _flight_time_study(initial_orbit, target_orbit, fast=fast)
    overview_path = _save_overview(results, cases, flight_time_study, make_figures=make_figures)
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
        "cases": cases,
        "flight_time_study": flight_time_study,
        "summary": rows,
        "output_paths": output_paths,
    }


if __name__ == "__main__":
    main(make_figures=True, fast=False)
