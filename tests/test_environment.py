import numpy as np
import pytest

from ssatk.constants import (
    AU,
    EARTH_DIPOLE_EQUATOR_FIELD,
    EARTH_GEOMAGNETIC_REFERENCE_RADIUS,
    EARTH_RADIUS,
    MOON_RADIUS,
)
from ssatk.propagators_6dof import Spacecraft
from ssatk.environment import (
    SpaceEnvironment,
    cylindrical_eclipse_fraction,
    earth_dipole_magnetic_field,
    exponential_atmosphere,
    solar_disk_visible_fraction,
    solar_occultation_fraction,
)
from ssatk.satellites import Facet, SpacecraftBody


def test_eclipse_models_return_full_partial_and_zero_sun_fraction():
    sun = np.array([AU, 0.0, 0.0])

    assert cylindrical_eclipse_fraction([2.0 * EARTH_RADIUS, 0.0, 0.0], sun) == pytest.approx(1.0)
    assert cylindrical_eclipse_fraction([-2.0 * EARTH_RADIUS, 0.0, 0.0], sun) == pytest.approx(0.0)
    assert solar_disk_visible_fraction([2.0 * EARTH_RADIUS, 0.0, 0.0], sun) == pytest.approx(1.0)
    assert solar_disk_visible_fraction([-2.0 * EARTH_RADIUS, 0.0, 0.0], sun) == pytest.approx(0.0)

    partial = solar_disk_visible_fraction([-2.0 * EARTH_RADIUS, EARTH_RADIUS, 0.0], sun)
    assert 0.0 < partial < 1.0

    assert solar_occultation_fraction(
        [0.0, 0.0, 0.0],
        sun,
        [10.0 * MOON_RADIUS, 0.0, 0.0],
        MOON_RADIUS,
    ) == pytest.approx(0.0)
    assert solar_occultation_fraction(
        [0.0, 0.0, 0.0],
        sun,
        [10.0 * MOON_RADIUS, 10.0 * MOON_RADIUS, 0.0],
        MOON_RADIUS,
    ) == pytest.approx(1.0)


def test_environment_conical_eclipse_includes_moon_shadow_for_srp():
    body = SpacecraftBody.box(name="bus", mass=10.0, size=(1.0, 1.0, 1.0)).with_facets(
        Facet(area=2.0, normal_body=[1.0, 0.0, 0.0], cr=1.0),
        append=False,
    )
    spacecraft = Spacecraft(r=[0.0, 0.0, 0.0], v=[0.0, 0.0, 0.0], body=body)
    shadowed_environment = SpaceEnvironment(
        sun_position_model=[AU, 0.0, 0.0],
        moon_position_model=[10.0 * MOON_RADIUS, 0.0, 0.0],
    )
    earth_only_environment = SpaceEnvironment(
        sun_position_model=[AU, 0.0, 0.0],
        moon_position_model=[10.0 * MOON_RADIUS, 0.0, 0.0],
        solar_occulting_bodies=("earth",),
    )

    assert shadowed_environment.eclipse_fraction(0.0, spacecraft.r) == pytest.approx(0.0)
    assert earth_only_environment.eclipse_fraction(0.0, spacecraft.r) == pytest.approx(1.0)

    shadowed_srp = shadowed_environment.force_models(solar_radiation=True, body=body)[0]
    illuminated_srp = earth_only_environment.force_models(solar_radiation=True, body=body)[0]
    np.testing.assert_allclose(shadowed_srp(spacecraft), 0.0, atol=1e-20)
    assert np.linalg.norm(illuminated_srp(spacecraft)) > 0.0


def test_earth_dipole_magnetic_field_matches_equator_and_pole_limits():
    reference_radius = EARTH_GEOMAGNETIC_REFERENCE_RADIUS

    np.testing.assert_allclose(
        earth_dipole_magnetic_field([reference_radius, 0.0, 0.0]),
        [0.0, 0.0, -EARTH_DIPOLE_EQUATOR_FIELD],
    )
    np.testing.assert_allclose(
        earth_dipole_magnetic_field([0.0, 0.0, reference_radius]),
        [0.0, 0.0, 2.0 * EARTH_DIPOLE_EQUATOR_FIELD],
    )
    np.testing.assert_allclose(
        earth_dipole_magnetic_field([2.0 * reference_radius, 0.0, 0.0]),
        [0.0, 0.0, -EARTH_DIPOLE_EQUATOR_FIELD / 8.0],
    )

    environment = SpaceEnvironment()
    np.testing.assert_allclose(
        environment.magnetic_field(0.0, [reference_radius, 0.0, 0.0]),
        [0.0, 0.0, -EARTH_DIPOLE_EQUATOR_FIELD],
    )
    np.testing.assert_allclose(
        SpaceEnvironment(magnetic_field_model="zero").magnetic_field(
            0.0,
            [reference_radius, 0.0, 0.0],
        ),
        0.0,
    )
    with pytest.raises(ValueError, match="r_inertial"):
        environment.magnetic_field(0.0)


def test_exponential_atmosphere_validates_and_decays():
    density = exponential_atmosphere(
        reference_density=1.0e-12,
        reference_altitude=400_000.0,
        scale_height=50_000.0,
    )

    assert density(400_000.0) == pytest.approx(1.0e-12)
    assert density(450_000.0) < density(400_000.0)
    with pytest.raises(ValueError, match="positive"):
        exponential_atmosphere(reference_density=0.0, reference_altitude=0.0, scale_height=1.0)
