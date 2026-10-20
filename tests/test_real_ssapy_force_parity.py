"""Final-behavior parity checks against the installed SSAPy force models."""

import numpy as np
from ssapy.accel import AccelDrag, AccelSolRad
from ssapy.body import get_body
from ssapy.gravity import AccelHarmonic, AccelThirdBody

from ssatk.accelerations_6dof import SpacecraftAccelSSAPy
from ssatk.propagators_6dof.sixdof import Spacecraft

R = np.array([7.0e6, -1.2e6, 2.5e6])
V = np.array([1.1e3, 7.2e3, -0.4e3])
T = 1.2e9


def test_real_ssapy_solrad_adapter_matches_explicit_force():
    kwargs = {"area": 12.0, "mass": 850.0, "CR": 1.6}
    force = AccelSolRad(**kwargs)
    adapter = SpacecraftAccelSSAPy(force, kwargs=kwargs, spacecraft_kwargs=False)

    expected = force(R, V, T, **kwargs)
    actual = adapter.acceleration(t=T, r=R, v=V, q=None, omega=None)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_real_ssapy_drag_adapter_matches_fixed_atmosphere_force():
    class FixedAtmosphere:
        def density(self, *args):
            return 2.5e-12

    kwargs = {"area": 8.0, "mass": 500.0, "CD": 2.25}
    force = AccelDrag(**kwargs)
    force.atm = FixedAtmosphere()
    adapter = SpacecraftAccelSSAPy(force, kwargs=kwargs, spacecraft_kwargs=False)

    expected = force(R, V, T, **kwargs)
    actual = adapter.acceleration(t=T, r=R, v=V, q=None, omega=None)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_real_ssapy_adapter_forwards_spacecraft_mass_area_cd_cr():
    solrad = AccelSolRad()
    adapter = SpacecraftAccelSSAPy(solrad)
    spacecraft = Spacecraft(r=R, v=V, t=T, mass=850.0, area=12.0, cd=2.25, cr=1.6)

    actual = adapter(spacecraft)
    expected = solrad(R, V, T, mass=spacecraft.mass, area=spacecraft.area, CD=spacecraft.cd, CR=spacecraft.cr)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_real_ssapy_gravity_adapters_match_harmonic_and_third_body_models():
    models = (
        AccelHarmonic(get_body("Earth", model="EGM2008"), 4, 4),
        AccelThirdBody(get_body("moon")),
        AccelThirdBody(get_body("Sun")),
    )

    for model in models:
        np.testing.assert_allclose(
            SpacecraftAccelSSAPy(model)(R, V, T), model(R, V, T), rtol=0.0, atol=0.0
        )
