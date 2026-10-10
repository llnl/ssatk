import numpy as np
import pytest

from ssatk.constants import G0
from ssatk.engines.catalog import thruster_spec
from ssatk.engines.fuel_usage import estimate_fuel_usage
from ssatk.engines.rescale_burn import rescale_burn


@pytest.mark.parametrize("steps", [10, 1000])
def test_fuel_for_a_constant_acceleration_follows_the_rocket_equation(steps):
    # R1 (Tsiolkovsky): at constant acceleration a for time T the mass falls to
    # m0 exp(-a T / (Isp g0)), so the propellant is m0 (1 - exp(-a T / (Isp g0))),
    # independent of the sampling. Here a T = 0.5 Isp g0. 1e-12 relative.
    isp = thruster_spec("biprop_400n").nominal_isp_s
    duration = 600.0
    acceleration = 0.5 * isp * G0 / duration
    accels = np.full(steps, acceleration)
    positions = np.zeros((steps, 3))
    fuel = estimate_fuel_usage(accels, duration / steps, positions, engine="biprop_400n", initial_mass_kg=1000.0)
    assert fuel == pytest.approx(1000.0 * (1.0 - np.exp(-0.5)), rel=1e-12)


def test_rescale_burn_closed_forms():
    # R1: thrust F = a0 m0. Constant thrust gives a = F/m and I = F t;
    # constant impulse gives F = a0 m0 t0 / t. Exact.
    a, t, dv, force, impulse = rescale_burn(2.0, 500.0, 30.0, m=250.0, t=60.0, mode="constant_thrust")
    assert (a, t, dv, force, impulse) == (4.0, 60.0, 240.0, 1000.0, 60000.0)
    a, t, dv, force, impulse = rescale_burn(2.0, 500.0, 30.0, m=250.0, t=60.0, mode="constant_impulse")
    assert (a, t, dv, force, impulse) == (2.0, 60.0, 120.0, 500.0, 30000.0)
