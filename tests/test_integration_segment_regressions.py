import unittest

import numpy as np

from ssatk.propagators_6dof.high_accuracy import propagate_spacecraft_segments
from ssatk.propagators_6dof.sixdof import (
    Spacecraft,
    _piecewise_solution_sequence,
)


class SegmentEventTests(unittest.TestCase):
    def run_segments(self, first, second):
        return propagate_spacecraft_segments(
            Spacecraft(r=[7e6, 0, 0], v=[0, 7500, 0], inertia=np.eye(3)),
            [{"times": [0, 1], "events": first}, {"times": [1, 2], "events": second}],
            mu=0,
        )


    def test_flat_dense_grouping_matches_independent_segment_evaluation(self):
        rng = np.random.default_rng(218)
        breaks = np.arange(1., 50.)
        coefficients = rng.normal(size=(50, 13, 2))

        def factory(coefficient):
            def solution(t):
                t = np.asarray(t)
                return (coefficient[:, 0, None] + coefficient[:, 1, None] * t
                        if t.ndim else coefficient[:, 0] + coefficient[:, 1] * t)
            return solution

        solutions = [factory(coefficient) for coefficient in coefficients]
        combined = _piecewise_solution_sequence(solutions, breaks)
        queries = np.concatenate((rng.uniform(0., 50., 500), breaks, breaks))
        rng.shuffle(queries)
        expected = np.column_stack([solutions[sum(t > breaks)](t) for t in queries])
        np.testing.assert_array_equal(combined(queries), expected)
