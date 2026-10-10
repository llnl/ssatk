import pytest

from ssatk.constants import G0
from ssatk.engines import (
    mass_flow_rate,
    propellant_mass_for_delta_v,
    thruster_spec,
)


def test_solid_motor_constraints_and_rocket_equation_helpers():
    solid = thruster_spec("solid", scale="small")
    with pytest.raises(ValueError, match="not throttleable"):
        solid.thrust_profile(start=0.0, burn_time=10.0, throttle=0.5)
    with pytest.raises(ValueError, match="finite"):
        solid.thrust_profile(start=0.0)

    profile = solid.thrust_profile(start=0.0, burn_time=10.0)
    assert profile(5.0) == pytest.approx(solid.nominal_thrust_n)
    assert solid.acceleration_for_mass(100.0) == pytest.approx(solid.nominal_thrust_n / 100.0)
    assert solid.mass_flow_rate() == pytest.approx(solid.nominal_thrust_n / (solid.nominal_isp_s * G0))
    assert mass_flow_rate(10.0, 250.0) == pytest.approx(10.0 / (250.0 * G0))
    assert propellant_mass_for_delta_v(100.0, wet_mass_kg=1000.0, isp_s=300.0) > 0.0
