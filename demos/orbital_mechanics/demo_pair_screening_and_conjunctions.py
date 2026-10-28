"""Contrast angular/velocity screening with trajectory conjunction analysis."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ssatk.plots.plotutils import figsave
from ssatk.ssa.conjunction import coarse_conjunction_screen, refine_closest_approach
from ssatk.ssa.pair_geometry import pair_metrics
from ssatk.ssa.pair_screening import screen_pairs

GALLERY_CATEGORY = "orbital_mechanics"


def _tracks():
    times = np.linspace(-300.0, 300.0, 61)
    # Two consistent rectilinear synthetic tracks pass 20 km apart at t=0.
    # They illustrate the conjunction tools, not a force-model propagation.
    position_a = np.array([0.0, 0.0, 7_000_000.0])
    position_b = np.array([0.0, 20_000.0, 7_000_000.0])
    velocity_a = np.array([7000.0, 0.0, 0.0])
    velocity_b = np.array([6900.0, 0.0, 0.0])
    first = SimpleNamespace(
        t=times,
        r=position_a + times[:, None] * velocity_a,
        v=np.broadcast_to(velocity_a, (len(times), 3)).copy(),
    )
    second = SimpleNamespace(
        t=times,
        r=position_b + times[:, None] * velocity_b,
        v=np.broadcast_to(velocity_b, (len(times), 3)).copy(),
    )
    return times, first, second


def main(make_figures=True):
    times, first, second = _tracks()
    angular = screen_pairs(
        np.array([first.r[30], second.r[30]]),
        np.array([first.v[30], second.v[30]]),
        np.zeros(3),
        angular_limit_rad=np.radians(1.0),
        velocity_threshold=-1.0,
        strategy="brute_force",
    )
    geometry = pair_metrics(
        first.r[30], first.v[30], second.r[30], second.v[30], np.zeros(3)
    )
    candidates = coarse_conjunction_screen(first, second, threshold=20_000.0)
    closest = tuple(refine_closest_approach(first, second, candidate) for candidate in candidates)
    print(
        f"angular/velocity candidates={len(angular.pairs)}, "
        f"trajectory brackets={len(candidates)}, "
        f"refined miss distances={[round(item.miss_distance, 3) for item in closest]}"
    )
    result = {
        "angular_screen": angular,
        "geometry": geometry,
        "candidates": candidates,
        "closest_approaches": closest,
    }
    if make_figures:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(7, 3.5))
        axis.plot(times, np.linalg.norm(second.r - first.r, axis=1) / 1e3)
        axis.axhline(20.0, color="tab:red", linestyle="--", label="conjunction threshold")
        axis.set(xlabel="Time from encounter (s)", ylabel="B-minus-A distance (km)")
        axis.set_title("Trajectory conjunction refinement, not angular risk")
        axis.legend()
        axis.grid(alpha=0.3)
        # gallery-call: demo_pair_screening_and_conjunctions.png
        figsave(figure, "demo_pair_screening_and_conjunctions.png")
        # gallery-end
        plt.close(figure)
    return result


if __name__ == "__main__":
    main(make_figures=True)
