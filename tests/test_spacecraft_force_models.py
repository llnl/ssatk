"""Closed-form checks of the 6-DoF surface-force and torque models."""

import numpy as np
import pytest

from ssatk.accelerations_6dof.spacecraft import (
    drag_acceleration,
    exponential_density_model,
    facet_drag_acceleration_torque,
    facet_srp_acceleration_torque,
    flat_plate_drag_acceleration_torque,
    flat_plate_srp_acceleration_torque,
    gravity_gradient_torque,
    srp_acceleration,
)
from ssatk.constants import AU, EARTH_MU, SOLAR_FLUX_1_AU, c
from ssatk.satellites import Facet

IDENTITY_Q = np.array([1.0, 0.0, 0.0, 0.0])
R_LEO = np.array([6_778e3, 0.0, 0.0])
SUN_ON_MINUS_X = np.array([-AU, 0.0, 0.0])  # photons travel along +x at the spacecraft
P_1AU = SOLAR_FLUX_1_AU / c  # 4.54e-6 N/m^2


def test_gravity_gradient_torque_matches_the_rigid_body_closed_form():
    # R1: tau = 3 mu / r^3 r_hat x (I r_hat). For I = diag(Ixx, Iyy, Izz) and
    # r_hat = (cos t, sin t, 0) in the body frame, tau = (0, 0, 3 mu/r^3 (Iyy - Ixx) sin t cos t).
    # 1e-12 relative.
    inertia = np.diag([10.0, 25.0, 40.0])
    angle = np.radians(30.0)
    radius = 7_000e3
    r = radius * np.array([np.cos(angle), np.sin(angle), 0.0])
    torque = gravity_gradient_torque(r, IDENTITY_Q, inertia)
    expected = 3 * EARTH_MU / radius**3 * (25.0 - 10.0) * np.sin(angle) * np.cos(angle)
    np.testing.assert_allclose(torque, [0.0, 0.0, expected], rtol=1e-12, atol=1e-25)


def test_cannonball_drag_and_srp_match_their_closed_forms():
    # R1: a_drag = -1/2 rho Cd A / m |v_rel| v_rel; a_srp = P(1 AU) Cr A / m along
    # the photon direction, P = 1361 W m^-2 / c. 1e-12 relative.
    v = np.array([0.0, 7_670.0, 0.0])
    zero_wind = np.zeros(3)
    a_drag = drag_acceleration(R_LEO, v, density=3e-12, area=2.0, mass=500.0, cd=2.2, atmosphere_velocity=zero_wind)
    np.testing.assert_allclose(a_drag, -0.5 * 3e-12 * 2.2 * 2.0 / 500.0 * 7_670.0 * v, rtol=1e-12, atol=0)

    a_srp = srp_acceleration(R_LEO, SUN_ON_MINUS_X, area=2.0, mass=500.0, cr=1.3)
    distance = AU + R_LEO[0]
    np.testing.assert_allclose(
        a_srp, [P_1AU * (AU / distance) ** 2 * 1.3 * 2.0 / 500.0, 0.0, 0.0], rtol=1e-12, atol=0
    )


def test_exponential_density_falls_by_e_per_scale_height():
    # R1: rho(h0 + k H) = rho0 e^-k. 1e-12 relative.
    density = exponential_density_model(reference_density=3.6e-10, reference_altitude=300e3, scale_height=58.5e3)
    for k in (-1.0, 0.0, 1.0, 2.5):
        assert density(300e3 + k * 58.5e3) == pytest.approx(3.6e-10 * np.exp(-k), rel=1e-12, abs=0)


@pytest.mark.parametrize(
    "specular, diffuse, normal_factor",
    [(1.0, 0.0, 2.0), (0.0, 1.0, 1.0 + 2.0 / 3.0), (0.0, 0.0, 1.0)],
    ids=["mirror", "lambertian", "absorber"],
)
def test_flat_plate_srp_at_normal_incidence(specular, diffuse, normal_factor):
    # R1 (Montenbruck & Gill eq. 3.73): at normal incidence the force is
    # P A [(1 - s) + 2 s + 2 d / 3] along the photon direction: 2 P A for a
    # mirror, (5/3) P A for a Lambertian diffuser, P A for a black plate.
    # 1e-12 relative.
    area, mass = 3.0, 100.0
    acceleration, _torque = flat_plate_srp_acceleration_torque(
        R_LEO, IDENTITY_Q, SUN_ON_MINUS_X, area=area, mass=mass,
        specular_reflectivity=specular, diffuse_reflectivity=diffuse,
        normal_body=(-1.0, 0.0, 0.0),
    )
    pressure = P_1AU * (AU / (AU + R_LEO[0])) ** 2
    np.testing.assert_allclose(acceleration, [normal_factor * pressure * area / mass, 0.0, 0.0], rtol=1e-12, atol=0)


def test_flat_plate_drag_at_incidence_angle_and_its_torque():
    # R1: a plate at angle t to the flow presents A cos t; force -q Cd A cos t
    # v_hat with q = rho v^2 / 2, and torque r_cp x F in the body frame.
    # 1e-12 relative.
    t = np.radians(40.0)
    normal = np.array([np.cos(t), np.sin(t), 0.0])
    v = np.array([7_500.0, 0.0, 0.0])
    cp = np.array([0.0, 0.0, 0.5])
    acceleration, torque = flat_plate_drag_acceleration_torque(
        R_LEO, v, IDENTITY_Q, density=1e-11, area=4.0, mass=200.0, cd=2.2,
        normal_body=normal, center_of_pressure=cp, atmosphere_velocity=np.zeros(3),
    )
    force = -0.5 * 1e-11 * 7_500.0**2 * 2.2 * 4.0 * np.cos(t) * np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(acceleration, force / 200.0, rtol=1e-12, atol=0)
    np.testing.assert_allclose(torque, np.cross(cp, force), rtol=1e-12, atol=1e-30)


def test_facet_models_on_a_cube_see_only_the_exposed_faces():
    # R1: a unit cube moving along +x exposes one 1 m^2 face to the flow and,
    # with photons along +x, one face to the Sun, so facet drag and facet SRP
    # equal the single-plate closed forms (1e-12 relative), with zero torque
    # about the centre.
    faces = [Facet(area=1.0, normal_body=n, center_of_pressure=0.5 * np.asarray(n, float), cd=2.2, cr=1.0)
             for n in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1])]
    v = np.array([7_500.0, 0.0, 0.0])
    a_drag, tau_drag = facet_drag_acceleration_torque(
        R_LEO, v, IDENTITY_Q, faces, density=1e-11, mass=10.0, atmosphere_velocity=np.zeros(3)
    )
    np.testing.assert_allclose(a_drag, [-0.5 * 1e-11 * 7_500.0**2 * 2.2 / 10.0, 0.0, 0.0], rtol=1e-12, atol=0)
    np.testing.assert_allclose(tau_drag, 0.0, atol=1e-30)

    a_srp, tau_srp = facet_srp_acceleration_torque(R_LEO, IDENTITY_Q, SUN_ON_MINUS_X, faces, mass=10.0)
    pressure = P_1AU * (AU / (AU + R_LEO[0])) ** 2
    np.testing.assert_allclose(a_srp, [pressure * 1.0 / 10.0, 0.0, 0.0], rtol=1e-12, atol=0)
    np.testing.assert_allclose(tau_srp, 0.0, atol=1e-30)


def test_facet_srp_self_shadowing_hides_a_plate_behind_another():
    # R1: two parallel 1 m^2 squares facing the Sun, the rear one directly
    # behind the front one, receive one plate's worth of force with
    # self-shadowing and two without. 1e-12 relative.
    def square(x):
        return Facet(
            area=1.0, normal_body=(-1.0, 0.0, 0.0), center_of_pressure=(x, 0.0, 0.0), cr=1.0,
            vertices_body=[(x, -0.5, -0.5), (x, 0.5, -0.5), (x, 0.5, 0.5), (x, -0.5, 0.5)],
        )

    plates = [square(-1.0), square(1.0)]
    pressure = P_1AU * (AU / (AU + R_LEO[0])) ** 2
    shadowed, _ = facet_srp_acceleration_torque(R_LEO, IDENTITY_Q, SUN_ON_MINUS_X, plates, mass=1.0, self_shadowing=True)
    unshadowed, _ = facet_srp_acceleration_torque(R_LEO, IDENTITY_Q, SUN_ON_MINUS_X, plates, mass=1.0)
    np.testing.assert_allclose(shadowed, [pressure, 0.0, 0.0], rtol=1e-12, atol=0)
    np.testing.assert_allclose(unshadowed, [2 * pressure, 0.0, 0.0], rtol=1e-12, atol=0)
