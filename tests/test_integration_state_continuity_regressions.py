import unittest

import numpy as np

from ssatk.propagators_6dof.high_accuracy import (
    _propagate_spacecraft_segment,
    propagate_spacecraft_segments,
)
from ssatk.propagators_6dof.sixdof import Spacecraft
from ssatk.satellites import SpacecraftBody, Tank


class StateContinuityTests(unittest.TestCase):
    def craft(self, **kwargs):
        return Spacecraft(r=[7e6, 0, 0], v=[0, 7500, 0], inertia=np.eye(3), **kwargs)

    def test_direct_burn_after_coast_preserves_mass_in_every_segment(self):
        def acceleration(t, r, v, q, omega):
            return np.zeros(3)
        acceleration.mass_flow_rate = lambda **kwargs: 1.
        tr = propagate_spacecraft_segments(
            self.craft(mass=10.),
            [dict(times=[0, 1]), dict(times=[1, 2], acceleration=acceleration),
             dict(times=[2, 3])],
            mu=0, dense_output=True,
        )
        np.testing.assert_allclose(tr.mass, [10., 10., 9., 9.], atol=1e-10)
        np.testing.assert_allclose(tr.solution([0.5, 1.5, 2.5])[13], [10., 9.5, 9.], atol=1e-10)

    def test_segment_handoff_preserves_selected_tank_and_updated_inertia(self):
        body = SpacecraftBody.box(name="bus", mass=10., size=(1., 1., 1.)).with_tanks(
            Tank(propellant_mass=2., dry_mass=1., name="main", position_body=[2., 0, 0]),
            Tank(propellant_mass=3., dry_mass=1., name="aux", position_body=[0, 2., 0]),
        )
        craft = Spacecraft(r=[7e6, 0, 0], v=[0, 7500, 0], body=body)

        def mass_flow(t, r, v, q, omega):
            return 1.
        mass_flow.tank_name = "aux"
        _, final, _ = _propagate_spacecraft_segment(
            craft, dict(times=[0, 1], mass_flow_rate=mass_flow, mu=0), tracks_mass=True,
        )
        expected = body.with_tank_propellant_mass("aux", 2.)
        np.testing.assert_allclose(
            [tank.propellant_mass for tank in final.body.tanks], [2., 2.], atol=1e-10,
        )
        np.testing.assert_allclose(final.inertia, expected.current_inertia, atol=1e-10)
        np.testing.assert_allclose(final.inertia, final.body.current_inertia, atol=1e-10)
