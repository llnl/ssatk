"""R1/R2 transfer references and C display contracts for the planning dashboard."""

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from demos.orbital_mechanics import demo_structured_transfer_api as demo
from ssatk.constants import EARTH_MU


@pytest.fixture(scope="module")
def scenario():
    return demo.main(make_figures=False, fast=True)


def test_single_departure_burn_preserves_the_full_transfer_interval():
    """C: a literal one-hour coast does not end when its one-second burn ends."""
    result = dict(initial={"t": 100}, final={"t": 3700}, tof=3600,
                  burns=[dict(t_start=100, t_end=101, delta_v_mag=100)])
    assert demo._transfer_time_bounds(result) == (100, 3700)


def test_selected_plans_reach_analytic_target_under_independent_dop853(scenario):
    """R1/R2: circular target ephemeris and independently integrated coast.

    Converge DOP853 by 10x in tolerance: reference endpoints within 1 mm and
    1 um/s. Lambert endpoints must reach the analytic target within 5 cm;
    the independently reconstructed burn cost budget is 5e-5 m/s. These are
    synthetic two-body orbits, not observations or finite-thrust validation.
    """
    def rhs(_t, state):
        r = state[:3]
        return np.r_[state[3:], -EARTH_MU * r / np.linalg.norm(r)**3]

    for result in (scenario["results"]["01_direct_rendezvous_orbits"],
                   scenario["flight_time_study"]):
        orbit = result["transfer_orbits"][0]
        state = np.r_[orbit.r, orbit.v]
        tof = result["tof"]
        integrations = [solve_ivp(rhs, [0, tof], state, method="DOP853", rtol=tol,
                                 atol=np.r_[np.full(3, 1e-6), np.full(3, 1e-9)] * factor)
                        for tol, factor in ((3e-12, 1), (3e-13, .1))]
        assert all(sol.success for sol in integrations)
        reference = integrations[-1].y[:, -1]
        np.testing.assert_allclose(integrations[0].y[:3, -1], reference[:3], atol=.001, rtol=0)
        np.testing.assert_allclose(integrations[0].y[3:, -1], reference[3:], atol=1e-6, rtol=0)
        angle = .35 + np.sqrt(EARTH_MU / (9e6)**3) * tof
        inc = np.deg2rad(8)
        target_r = 9e6 * np.array([np.cos(angle), np.sin(angle)*np.cos(inc), np.sin(angle)*np.sin(inc)])
        target_v = np.sqrt(EARTH_MU / 9e6) * np.array([-np.sin(angle), np.cos(angle)*np.cos(inc), np.cos(angle)*np.sin(inc)])
        np.testing.assert_allclose(reference[:3], target_r, atol=.05, rtol=0)
        departure_v = [0, np.sqrt(EARTH_MU / 7e6), 0]
        cost = np.linalg.norm(orbit.v - departure_v) + np.linalg.norm(target_v - reference[3:])
        assert result["delta_v_total"] == pytest.approx(cost, abs=5e-5)
    p = scenario["flight_time_study"]["diagnostics"]["flight_time_study"]
    # The curve has one comparable total-rendezvous cost at each flight time.
    at_hour = np.flatnonzero(np.asarray(p["tof"]) == 3600)
    assert len(at_hour) == 1
    assert p["dv"][at_hour[0]] == pytest.approx(
        scenario["results"]["01_direct_rendezvous_orbits"]["delta_v_total"], abs=1e-6)


def test_budget_plot_uses_total_burns_even_for_an_arrival_only_objective(scenario):
    """C: plotted widths are burn magnitudes, not incomparable objective values."""
    fig = demo._build_overview(scenario["results"], scenario["cases"], scenario["flight_time_study"])
    try:
        bars = fig.axes[1].containers
        for i, case in enumerate(scenario["cases"]):
            result = scenario["results"][case["slug"]]
            expected = sum(np.linalg.norm(b["delta_v"]) for b in result["burns"])
            assert sum(group.patches[i].get_width() for group in bars) == pytest.approx(expected, abs=1e-9)
        insertion = scenario["results"]["03_insertion_arrival_burn"]
        assert insertion["delta_v_total"] > 3 * insertion["diagnostics"]["objective_delta_v"]
    finally:
        plt.close(fig)


def test_dashboard_grid_and_text_at_export_resolutions(scenario):
    """C: aligned panels, equal route scaling and disjoint, contained text.

    Check the PNG preview and JPEG export DPI. No pixel snapshot encodes a
    preferred style, and annotation arrows are excluded from text rectangles.
    """
    from matplotlib.text import Text

    fig = demo._build_overview(scenario["results"], scenario["cases"], scenario["flight_time_study"])
    try:
        for dpi in (140, 200):
            fig.set_dpi(dpi)
            fig.canvas.draw()
            panels = [ax.get_window_extent() for ax in fig.axes]
            for a, b in ((0, 2), (1, 3)):
                np.testing.assert_allclose(
                    [panels[a].x0, panels[a].x1], [panels[b].x0, panels[b].x1],
                    atol=1, rtol=0)  # one pixel at each export resolution
            for a, b in ((0, 1), (2, 3)):
                np.testing.assert_allclose(
                    [panels[a].y0, panels[a].y1], [panels[b].y0, panels[b].y1],
                    atol=1, rtol=0)
            np.testing.assert_allclose(
                [[p.width, p.height] for p in panels],
                np.tile([panels[0].width, panels[0].height], (4, 1)), atol=1, rtol=0)
            origin, dx, dy = fig.axes[0].transData.transform([(0, 0), (1000, 0), (0, 1000)])
            assert np.linalg.norm(dx - origin) == pytest.approx(np.linalg.norm(dy - origin), abs=1e-8)
            texts = list(fig.texts)
            for ax in fig.axes:
                texts += list(ax.texts) + [ax.xaxis.label, ax.yaxis.label]
                for axis, bounds in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
                    texts += [tick.label1 for tick in axis.get_major_ticks()
                              if bounds[0] <= tick.get_loc() <= bounds[1]]
                if ax.get_legend() is not None:
                    texts += ax.get_legend().get_texts()
            renderer = fig.canvas.get_renderer()
            boxes = [(t.get_text(), Text.get_window_extent(t, renderer)) for t in texts
                     if t.get_visible() and t.get_text()]
            for label, box in boxes:
                assert fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1), label
            for (a, ab), (b, bb) in itertools.combinations(boxes, 2):
                assert not ab.overlaps(bb), f"Overlapping labels: {a!r} / {b!r}"
    finally:
        plt.close(fig)
