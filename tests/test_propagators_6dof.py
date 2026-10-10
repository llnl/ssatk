
import numpy as np
import pytest

from ssatk.accelerations_6dof import (
    SpacecraftAttitudePD,
    SpacecraftMagneticTorque,
    SpacecraftManeuverAccel,
    SpacecraftReactionWheelTorque,
    SpacecraftThrusterAccel,
    attitude_error_quaternion,
    co_rotating_atmosphere_velocity,
    drag_acceleration,
    facet_drag_acceleration_torque,
    facet_srp_acceleration_torque,
    flat_plate_drag_acceleration_torque,
    flat_plate_srp_acceleration_torque,
    j2_acceleration,
    magnetic_dipole_torque,
    reaction_wheel_torque,
    reaction_wheel_torque_commands,
    srp_acceleration,
    third_body_acceleration,
    thrust_profile_constant,
    wrap_ssapy_acceleration,
)
from ssatk.constants import (
    AU,
    EARTH_MU,
    EARTH_RADIUS,
    SOLAR_FLUX_1_AU,
    STANDARD_GRAVITY,
    WGS84_EARTH_OMEGA,
    J2_wgs,
    c,
)
from ssatk.propagators_6dof import (
    Spacecraft,
    altitude_crossing_event,
    gravity_gradient_torque,
    mass_floor_event,
    normalize_quaternion,
    propagate_6dof,
    propellant_empty_event,
    radius_crossing_event,
    rotate_vector,
    sixdof_rhs,
)
from ssatk.propagators_6dof.high_accuracy import (
    ImpulseManeuver,
    propagate_spacecraft_segments,
)
from ssatk.satellites import (
    Component,
    Facet,
    MagneticDipole,
    ReactionWheel,
    SpacecraftBody,
    Tank,
    Thruster,
    cubesat_1u,
    cubesat_6u,
    load_obj_facets,
    mesh_facets,
    point_mass_inertia,
    reaction_wheel_triplet,
)


def test_ssapy_acceleration_adapter_matches_base_accel_const_ntw():
    from ssapy.accel import AccelConstNTW

    r = np.array([7_000_000.0, 0.0, 0.0])
    v = np.array([0.0, 7_500.0, 0.0])
    t = 12.0
    raw = AccelConstNTW([1e-8, 2e-8, 3e-8])
    adapter = wrap_ssapy_acceleration(raw)

    np.testing.assert_allclose(adapter(r, v, t), raw(r, v, t))


def test_spacecraft_propagation_uses_wrapped_ssapy_acceleration():
    class ConstantSSAPyAccel:
        def __call__(self, r, v, t, **kwargs):
            return [0.0, 1.0e-3, 0.0]

    spacecraft = Spacecraft(
        r=[0.0, 0.0, 0.0],
        v=[0.0, 0.0, 0.0],
        inertia=np.eye(3),
        mass=10.0,
    )
    trajectory = spacecraft.propagate(
        times=[0.0, 10.0],
        acceleration=wrap_ssapy_acceleration(ConstantSSAPyAccel()),
        mu=0.0,
    )

    np.testing.assert_allclose(trajectory.v[-1], [0.0, 0.01, 0.0], rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(trajectory.r[-1], [0.0, 0.05, 0.0], rtol=0.0, atol=1e-10)


def test_spacecraft_body_components_update_mass_center_and_inertia():
    body = SpacecraftBody.box(name="bus", mass=10.0, size=(1.0, 1.0, 1.0)).with_components(
        Component(mass=2.0, position_body=[1.0, 0.0, 0.0], name="payload")
    ).with_tanks(Tank(propellant_mass=3.0, dry_mass=1.0, position_body=[0.0, 2.0, 0.0], name="tank"))

    expected_mass = 16.0
    expected_center = np.array([2.0, 8.0, 0.0]) / expected_mass

    assert body.current_mass == pytest.approx(expected_mass)
    np.testing.assert_allclose(body.current_center_of_mass, expected_center)
    np.testing.assert_allclose(
        point_mass_inertia(2.0, [1.0, 0.0, 0.0]),
        np.diag([0.0, 2.0, 2.0]),
    )
    assert np.min(np.linalg.eigvalsh(body.current_inertia)) > 0.0

    spacecraft = Spacecraft(r=[0, 0, 0], v=[0, 0, 0], body=body)
    assert spacecraft.mass == pytest.approx(expected_mass)
    np.testing.assert_allclose(spacecraft.inertia, body.current_inertia)

    depleted = body.with_current_mass(14.0)
    assert depleted.current_mass == pytest.approx(14.0)
    assert depleted.propellant_mass == pytest.approx(1.0)
    assert depleted.current_center_of_mass[1] < body.current_center_of_mass[1]
    assert np.min(np.linalg.eigvalsh(depleted.current_inertia)) > 0.0

    with pytest.raises(ValueError, match="below dry mass"):
        body.with_current_mass(body.dry_mass_total - 1.0)
    with pytest.raises(ValueError, match="capacity"):
        body.with_current_mass(body.current_mass + 1.0)


def test_named_tank_burn_depletes_only_selected_tank_and_updates_body_properties():
    body = SpacecraftBody.box(name="bus", mass=10.0, size=(1.0, 1.0, 1.0)).with_tanks(
        Tank(propellant_mass=2.0, dry_mass=1.0, position_body=[2.0, 0.0, 0.0], name="main"),
        Tank(propellant_mass=3.0, dry_mass=1.0, position_body=[0.0, 2.0, 0.0], name="aux"),
    )
    spacecraft = Spacecraft(r=[7.0e6, 0.0, 0.0], v=[0.0, 7500.0, 0.0], body=body)
    burn = SpacecraftManeuverAccel(10.0, frame="gcrf", direction=[1.0, 0.0, 0.0], isp=100.0, tank_name="aux")
    trajectory = spacecraft.propagate(times=[0.0, 1.0], acceleration=burn, mu=0.0)
    final_body = body.with_tank_propellant_mass(
        "aux", 3.0 - (body.current_mass - trajectory.mass[-1])
    )
    assert final_body.tanks[0].propellant_mass == pytest.approx(2.0)
    assert final_body.tanks[1].propellant_mass < 3.0
    assert final_body.current_mass < body.current_mass
    assert final_body.current_center_of_mass[0] > body.current_center_of_mass[0]
    assert np.min(final_body.current_inertia.diagonal()) > 0.0


def test_named_tank_burn_stops_thrust_when_selected_tank_is_empty():
    body = SpacecraftBody.box(name="bus", mass=10.0, size=(1.0, 1.0, 1.0)).with_tanks(
        Tank(propellant_mass=0.02, dry_mass=1.0, name="main", position_body=[2.0, 0.0, 0.0]),
        Tank(propellant_mass=0.5, dry_mass=1.0, name="aux", position_body=[0.0, 2.0, 0.0]),
    )
    spacecraft = Spacecraft(r=[7.0e6, 0.0, 0.0], v=[0.0, 0.0, 0.0], body=body)
    burn = SpacecraftManeuverAccel(
        10.0, frame="gcrf", direction=[1.0, 0.0, 0.0], isp=100.0, tank_name="main"
    )
    trajectory = spacecraft.propagate(
        times=np.linspace(0.0, 4.0, 9), models=[burn], mu=0.0
    )

    assert trajectory.mass[-1] == pytest.approx(body.current_mass - 0.02, abs=1e-8)
    np.testing.assert_allclose(trajectory.v[5:, 0], trajectory.v[4, 0], atol=1e-8)


def test_quaternion_helpers_rotate_body_to_inertial():
    q_z90 = normalize_quaternion([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    np.testing.assert_allclose(
        rotate_vector(q_z90, [1.0, 0.0, 0.0]),
        [0.0, 1.0, 0.0],
        atol=1e-12,
    )
    with pytest.raises(ValueError, match="non-zero"):
        normalize_quaternion([0.0, 0.0, 0.0, 0.0])


def test_segment_impulse_applies_body_frame_delta_v_at_exact_epoch(gps_epoch):
    import ssatk

    assert ssatk.ImpulseManeuver is ImpulseManeuver
    spacecraft = Spacecraft(
        r=[1.0, 0.0, 0.0], v=[0.0, 0.0, 0.0], t=gps_epoch, inertia=np.eye(3),
        q=[np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
        mass=10.0,
    )
    trajectory = propagate_spacecraft_segments(spacecraft, [
        {"times": [gps_epoch + 0.0, gps_epoch + 1.0], "mu": 0.0},
        {"times": [gps_epoch + 1.0, gps_epoch + 2.0], "mu": 0.0,
         "impulses": ImpulseManeuver(
             [1.0, 0.0, 0.0], frame="body", mass_change=-2.0,
             q_reset=[1.0, 0.0, 0.0, 0.0], omega_reset=[0.0, 0.0, 0.25]),},
    ])
    boundary = np.flatnonzero(trajectory.t == gps_epoch + 1.0)
    assert boundary.size == 2
    np.testing.assert_allclose(trajectory.r[boundary[0]], trajectory.r[boundary[1]], atol=1e-12)
    np.testing.assert_allclose(trajectory.v[boundary[0]], [0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(trajectory.v[boundary[1]], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(trajectory.mass[boundary], [10.0, 8.0], atol=1e-12)
    np.testing.assert_allclose(trajectory.q[boundary[1]], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(trajectory.omega[boundary[1]], [0.0, 0.0, 0.25])
    np.testing.assert_allclose(trajectory.r[-1], [1.0, 1.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(trajectory.v[-1], [0.0, 1.0, 0.0], atol=1e-10)
    with pytest.raises(ValueError, match="unsupported satellite frame"):
        ImpulseManeuver([1.0, 0.0, 0.0], frame="bad").apply(spacecraft)


def test_cubesat_preset_bodies_expose_valid_mass_properties():
    for make_body in (cubesat_1u, cubesat_6u):
        body = make_body()
        assert body.current_mass == pytest.approx(body.mass)
        assert body.inertia.shape == (3, 3)
        assert np.all(np.linalg.eigvalsh(body.inertia) > 0.0)
        assert body.area > 0.0


def test_mass_only_impulse_updates_tank_body_and_preserves_mass_jump(gps_epoch):
    body = SpacecraftBody.box(name="bus", mass=10.0, size=(1.0, 1.0, 1.0)).with_tanks(
        Tank(propellant_mass=2.0, dry_mass=1.0, name="main")
    )
    spacecraft = Spacecraft(
        r=[1.0, 0.0, 0.0], v=[0.0, 0.0, 0.0], t=gps_epoch, body=body
    )
    trajectory = propagate_spacecraft_segments(spacecraft, [
        {"times": [gps_epoch + 0.0, gps_epoch + 1.0], "mu": 0.0},
        {"times": [gps_epoch + 1.0, gps_epoch + 2.0], "mu": 0.0,
         "impulses": ImpulseManeuver([0.0, 0.0, 0.0], mass_change=-1.0)},
    ])
    boundary = np.flatnonzero(trajectory.t == gps_epoch + 1.0)
    assert boundary.size == 2
    np.testing.assert_allclose(trajectory.mass[boundary], [13.0, 12.0])
    assert trajectory.spacecraft(boundary[1], body=body).body.current_mass == pytest.approx(12.0)
    assert trajectory.spacecraft(boundary[1], body=body).inertia is not None
    with pytest.raises(ValueError, match="mass_change must be finite"):
        ImpulseManeuver([0.0, 0.0, 0.0], mass_change=np.nan).apply(spacecraft)


def test_torque_free_principal_axis_spin_preserves_rate_and_norm():
    traj = propagate_6dof(
        r0=[7_000_000.0, 0.0, 0.0],
        v0=[0.0, 0.0, 0.0],
        times=np.linspace(0.0, 20.0, 5),
        mu=0.0,
        inertia=np.diag([10.0, 12.0, 8.0]),
        omega0=[0.0, 0.0, 0.01],
    )

    np.testing.assert_allclose(
        traj.omega,
        np.tile([0.0, 0.0, 0.01], (5, 1)),
        atol=1e-12,
    )
    np.testing.assert_allclose(np.linalg.norm(traj.q, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(traj.r, np.tile([7_000_000.0, 0.0, 0.0], (5, 1)), atol=1e-10)


def test_gravity_gradient_torque_matches_rigid_body_formula():
    inertia = np.diag([10.0, 20.0, 30.0])
    assert np.linalg.norm(
        gravity_gradient_torque(
            [7_000_000.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            inertia,
        )
    ) == pytest.approx(0.0)

    r = np.array([7_000_000.0, 7_000_000.0, 0.0])
    r_hat = r / np.linalg.norm(r)
    expected = (
        3.0
        * EARTH_MU
        / np.linalg.norm(r) ** 3
        * np.cross(r_hat, inertia @ r_hat)
    )
    np.testing.assert_allclose(
        gravity_gradient_torque(r, [1.0, 0.0, 0.0, 0.0], inertia),
        expected,
    )


def test_j2_acceleration_matches_standard_equatorial_formula():
    radius = 7_000_000.0
    r = np.array([radius, 0.0, 0.0])
    expected = np.array([
        -1.5 * J2_wgs * EARTH_MU * EARTH_RADIUS**2 / radius**4,
        0.0,
        0.0,
    ])

    np.testing.assert_allclose(j2_acceleration(r), expected)


def test_third_body_acceleration_is_earth_centered_perturbation():
    assert np.linalg.norm(third_body_acceleration([0, 0, 0], [10, 0, 0], 1.0)) == 0.0
    expected = np.array([1.0 / 81.0 - 1.0 / 100.0, 0.0, 0.0])

    np.testing.assert_allclose(
        third_body_acceleration([1, 0, 0], [10, 0, 0], 1.0),
        expected,
    )


def test_drag_and_srp_acceleration_have_expected_directions_and_magnitudes():
    r = np.array([EARTH_RADIUS + 400_000.0, 0.0, 0.0])
    atmosphere_velocity = co_rotating_atmosphere_velocity(r)
    np.testing.assert_allclose(
        atmosphere_velocity,
        np.cross([0.0, 0.0, WGS84_EARTH_OMEGA], r),
    )
    np.testing.assert_allclose(
        drag_acceleration(r, atmosphere_velocity, density=1e-12, area=2.0, mass=100.0),
        0.0,
        atol=1e-20,
    )

    v = atmosphere_velocity + np.array([0.0, 100.0, 0.0])
    drag = drag_acceleration(r, v, density=1e-12, area=2.0, mass=100.0)
    assert np.dot(drag, v - atmosphere_velocity) < 0.0
    np.testing.assert_allclose(
        drag_acceleration(
            r,
            [0.0, 0.0, 0.0],
            density=1e-12,
            area=2.0,
            mass=100.0,
            atmosphere_velocity=[0.0, 0.0, 0.0],
        ),
        0.0,
        atol=1e-20,
    )

    srp = srp_acceleration([0, 0, 0], [AU, 0, 0], area=2.0, mass=100.0, cr=1.5)
    expected_srp = np.array([-1361.0 / c * 1.5 * 2.0 / 100.0, 0.0, 0.0])
    np.testing.assert_allclose(srp, expected_srp)


def test_flat_plate_drag_acceleration_torque_and_attitude_shadowing():
    r = np.array([7_000_000.0, 0.0, 0.0])
    v = np.array([3.0, 0.0, 0.0])
    q = [1.0, 0.0, 0.0, 0.0]

    acceleration, torque = flat_plate_drag_acceleration_torque(
        r,
        v,
        q,
        density=1.0,
        area=2.0,
        mass=10.0,
        cd=2.0,
        normal_body=[1.0, 0.0, 0.0],
        center_of_pressure=[0.0, 1.0, 0.0],
        earth_radius=0.0,
        earth_rotation_rate=0.0,
    )

    np.testing.assert_allclose(acceleration, [-1.8, 0.0, 0.0])
    np.testing.assert_allclose(torque, [0.0, 0.0, 18.0])

    hidden_accel, hidden_torque = flat_plate_drag_acceleration_torque(
        r,
        v,
        q,
        density=1.0,
        area=2.0,
        mass=10.0,
        cd=2.0,
        normal_body=[-1.0, 0.0, 0.0],
        earth_radius=0.0,
        earth_rotation_rate=0.0,
    )
    np.testing.assert_allclose(hidden_accel, 0.0)
    np.testing.assert_allclose(hidden_torque, 0.0)

    facet_accel, facet_torque = facet_drag_acceleration_torque(
        r,
        v,
        q,
        [Facet(area=2.0, normal_body=[1.0, 0.0, 0.0], center_of_pressure=[0.0, 1.0, 0.0], cd=2.0)],
        density=1.0,
        mass=10.0,
        earth_radius=0.0,
        earth_rotation_rate=0.0,
    )
    np.testing.assert_allclose(facet_accel, acceleration)
    np.testing.assert_allclose(facet_torque, torque)


def test_flat_plate_and_facet_lift_are_projected_and_signed():
    r = np.array([7_000_000.0, 0.0, 0.0])
    v = np.array([3.0, 4.0, 0.0])
    q = [1.0, 0.0, 0.0, 0.0]
    kwargs = {
        "density": 1.0,
        "area": 2.0,
        "mass": 10.0,
        "cd": 2.0,
        "normal_body": [1.0, 0.0, 0.0],
        "center_of_pressure": [0.0, 1.0, 0.0],
        "atmosphere_velocity": [0.0, 0.0, 0.0],
        "earth_radius": 0.0,
        "earth_rotation_rate": 0.0,
    }

    legacy_accel, legacy_torque = flat_plate_drag_acceleration_torque(r, v, q, **kwargs)
    zero_lift_accel, zero_lift_torque = flat_plate_drag_acceleration_torque(r, v, q, cl=0.0, **kwargs)
    np.testing.assert_allclose(zero_lift_accel, legacy_accel)
    np.testing.assert_allclose(zero_lift_torque, legacy_torque)

    lift_accel, lift_torque = flat_plate_drag_acceleration_torque(r, v, q, cl=1.0, **kwargs)
    np.testing.assert_allclose(lift_accel, [-0.6, -3.3, 0.0])
    np.testing.assert_allclose(lift_torque, [0.0, 0.0, 6.0])

    facet_accel, facet_torque = facet_drag_acceleration_torque(
        r,
        v,
        q,
        [Facet(area=2.0, normal_body=[1.0, 0.0, 0.0], center_of_pressure=[0.0, 1.0, 0.0], cd=2.0, cl=1.0)],
        density=1.0,
        mass=10.0,
        atmosphere_velocity=[0.0, 0.0, 0.0],
        earth_radius=0.0,
        earth_rotation_rate=0.0,
    )
    np.testing.assert_allclose(facet_accel, lift_accel)
    np.testing.assert_allclose(facet_torque, lift_torque)

    with pytest.raises(ValueError, match="cl must be finite"):
        flat_plate_drag_acceleration_torque(r, v, q, cl=np.nan, **kwargs)
    with pytest.raises(ValueError, match="cl must be finite"):
        Facet(area=1.0, normal_body=[1.0, 0.0, 0.0], cl=np.inf)


def test_drag_models_include_local_surface_velocity_from_body_rotation():
    r = np.array([7_000_000.0, 0.0, 0.0])
    v = np.zeros(3)
    q = [1.0, 0.0, 0.0, 0.0]
    omega_body = [0.0, 0.0, 2.0]
    center_of_pressure = [0.0, 1.0, 0.0]

    static_accel, static_torque = flat_plate_drag_acceleration_torque(
        r,
        v,
        q,
        density=1.0,
        area=1.0,
        mass=10.0,
        cd=2.0,
        normal_body=[-1.0, 0.0, 0.0],
        center_of_pressure=center_of_pressure,
        earth_radius=0.0,
        earth_rotation_rate=0.0,
    )
    np.testing.assert_allclose(static_accel, 0.0)
    np.testing.assert_allclose(static_torque, 0.0)

    spinning_accel, spinning_torque = flat_plate_drag_acceleration_torque(
        r,
        v,
        q,
        density=1.0,
        area=1.0,
        mass=10.0,
        cd=2.0,
        normal_body=[-1.0, 0.0, 0.0],
        center_of_pressure=center_of_pressure,
        omega_body=omega_body,
        earth_radius=0.0,
        earth_rotation_rate=0.0,
    )
    np.testing.assert_allclose(spinning_accel, [0.4, 0.0, 0.0])
    np.testing.assert_allclose(spinning_torque, [0.0, 0.0, -4.0])

    facet_accel, facet_torque = facet_drag_acceleration_torque(
        r,
        v,
        q,
        [Facet(area=1.0, normal_body=[-1.0, 0.0, 0.0], center_of_pressure=center_of_pressure, cd=2.0)],
        density=1.0,
        mass=10.0,
        omega_body=omega_body,
        earth_radius=0.0,
        earth_rotation_rate=0.0,
    )
    np.testing.assert_allclose(facet_accel, spinning_accel)
    np.testing.assert_allclose(facet_torque, spinning_torque)


def test_flat_plate_srp_acceleration_torque_and_attitude_shadowing():
    acceleration, torque = flat_plate_srp_acceleration_torque(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [AU, 0.0, 0.0],
        area=2.0,
        mass=10.0,
        cr=1.0,
        normal_body=[1.0, 0.0, 0.0],
        center_of_pressure=[0.0, 1.0, 0.0],
    )

    assert acceleration[0] < 0.0
    np.testing.assert_allclose(acceleration[1:], 0.0)
    assert torque[2] > 0.0

    hidden_accel, hidden_torque = flat_plate_srp_acceleration_torque(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [AU, 0.0, 0.0],
        area=2.0,
        mass=10.0,
        cr=1.0,
        normal_body=[-1.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(hidden_accel, 0.0)
    np.testing.assert_allclose(hidden_torque, 0.0)

    facet_accel, facet_torque = facet_srp_acceleration_torque(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [AU, 0.0, 0.0],
        [Facet(area=2.0, normal_body=[1.0, 0.0, 0.0], center_of_pressure=[0.0, 1.0, 0.0], cr=1.0)],
        mass=10.0,
    )
    np.testing.assert_allclose(facet_accel, acceleration)
    np.testing.assert_allclose(facet_torque, torque)


def test_optical_srp_coefficients_match_flat_plate_limits():
    pressure = SOLAR_FLUX_1_AU / c

    absorber, _ = flat_plate_srp_acceleration_torque(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [AU, 0.0, 0.0],
        area=2.0,
        mass=10.0,
        cr=9.0,
        specular_reflectivity=0.0,
        diffuse_reflectivity=0.0,
        normal_body=[1.0, 0.0, 0.0],
    )
    specular, _ = flat_plate_srp_acceleration_torque(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [AU, 0.0, 0.0],
        area=2.0,
        mass=10.0,
        specular_reflectivity=1.0,
        diffuse_reflectivity=0.0,
        normal_body=[1.0, 0.0, 0.0],
    )
    diffuse, _ = flat_plate_srp_acceleration_torque(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [AU, 0.0, 0.0],
        area=2.0,
        mass=10.0,
        specular_reflectivity=0.0,
        diffuse_reflectivity=1.0,
        normal_body=[1.0, 0.0, 0.0],
    )

    np.testing.assert_allclose(absorber, [-pressure * 2.0 / 10.0, 0.0, 0.0])
    np.testing.assert_allclose(specular, 2.0 * absorber)
    np.testing.assert_allclose(diffuse, (5.0 / 3.0) * absorber)
    with pytest.raises(ValueError, match="must be <= 1"):
        flat_plate_srp_acceleration_torque(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [AU, 0.0, 0.0],
            area=1.0,
            mass=1.0,
            specular_reflectivity=0.8,
            diffuse_reflectivity=0.4,
        )


def test_facet_srp_self_shadowing_and_mesh_facets(tmp_path):
    vertices = np.array([
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5],
    ])
    blocker = mesh_facets(vertices, [(0, 1, 2, 3)], specular_reflectivity=0.0, diffuse_reflectivity=0.0)[0]
    shaded = Facet(
        area=1.0,
        normal_body=[1.0, 0.0, 0.0],
        center_of_pressure=[0.0, 0.0, 0.0],
        specular_reflectivity=0.0,
        diffuse_reflectivity=0.0,
        vertices_body=[[-0.01, -0.25, -0.25], [-0.01, 0.25, -0.25], [-0.01, 0.25, 0.25], [-0.01, -0.25, 0.25]],
    )

    no_shadow, _ = facet_srp_acceleration_torque(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [AU, 0.0, 0.0],
        [shaded, blocker],
        mass=10.0,
    )
    with_shadow, _ = facet_srp_acceleration_torque(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [AU, 0.0, 0.0],
        [shaded, blocker],
        mass=10.0,
        self_shadowing=True,
    )
    assert np.linalg.norm(with_shadow) < np.linalg.norm(no_shadow)
    np.testing.assert_allclose(np.linalg.norm(with_shadow), 0.5 * np.linalg.norm(no_shadow), rtol=1e-12)

    obj_path = tmp_path / "plate.obj"
    obj_path.write_text("v 0 -0.5 -0.5\nv 0 0.5 -0.5\nv 0 0.5 0.5\nv 0 -0.5 0.5\nf 1 2 3 4\n")
    facets = load_obj_facets(obj_path)
    assert len(facets) == 1
    assert facets[0].area == pytest.approx(1.0)
    assert facets[0].vertices_body is not None


def test_magnetic_dipole_torque_uses_body_frame_field():
    dipole = MagneticDipole(moment_body=[1.0, 0.0, 0.0], name="x_rod")
    body = SpacecraftBody.cubesat(1, mass=10.0).with_magnetic_dipoles(dipole)
    spacecraft = Spacecraft(
        r=[7_000_000.0, 0.0, 0.0],
        v=[0.0, 7_500.0, 0.0],
        q=[1.0, 0.0, 0.0, 0.0],
        omega=[0.0, 0.0, 0.0],
        body=body,
    )

    np.testing.assert_allclose(
        magnetic_dipole_torque([dipole], [0.0, 2.0e-5, 0.0]),
        [0.0, 0.0, 2.0e-5],
    )
    torque = SpacecraftMagneticTorque([0.0, 2.0e-5, 0.0])
    np.testing.assert_allclose(torque(spacecraft), [0.0, 0.0, 2.0e-5])
    np.testing.assert_allclose(
        SpacecraftMagneticTorque([0.0, 2.0e-5, 0.0], dipole_names=["missing"])(spacecraft),
        [0.0, 0.0, 0.0],
    )


def test_reaction_wheel_torque_allocates_and_saturates_body_torque():
    wheels = reaction_wheel_triplet(max_torque=0.02, name_prefix="wheel")
    body = SpacecraftBody.cubesat(1, mass=10.0).with_reaction_wheels(*wheels)

    assert all(isinstance(wheel, ReactionWheel) for wheel in wheels)
    assert len(body.reaction_wheels) == 3
    np.testing.assert_allclose(
        reaction_wheel_torque(body.reaction_wheels, [0.01, -0.03, 0.0]),
        [0.01, -0.02, 0.0],
    )
    np.testing.assert_allclose(
        reaction_wheel_torque_commands(body.reaction_wheels, [0.01, -0.03, 0.0]),
        [0.01, -0.02, 0.0],
    )
    np.testing.assert_allclose(
        reaction_wheel_torque(body.reaction_wheels, {"wheel_z": 0.05}),
        [0.0, 0.0, 0.02],
    )
    np.testing.assert_allclose(
        SpacecraftReactionWheelTorque([0.0, 0.0, -0.05])(Spacecraft(r=[0, 0, 0], v=[0, 0, 0], body=body)),
        [0.0, 0.0, -0.02],
    )
    with pytest.raises(ValueError, match="reaction-wheel command"):
        reaction_wheel_torque(body.reaction_wheels, [1.0, 2.0])


def test_reaction_wheel_momentum_state_conserves_internal_angular_momentum():
    body = SpacecraftBody(
        name="wheel_test",
        mass=10.0,
        inertia=np.diag([2.0, 3.0, 4.0]),
    ).with_reaction_wheels(
        *reaction_wheel_triplet(max_torque=0.1, wheel_inertia=0.01)
    )
    spacecraft = Spacecraft(
        r=[0.0, 0.0, 0.0],
        v=[0.0, 0.0, 0.0],
        q=[1.0, 0.0, 0.0, 0.0],
        omega=[0.0, 0.0, 0.0],
        body=body,
    )

    traj = spacecraft.propagate(
        times=[0.0, 1.0],
        mu=0.0,
        models=[SpacecraftReactionWheelTorque([0.0, 0.0, 0.02])],
    )

    assert traj.wheel_momentum.shape == (2, 3)
    np.testing.assert_allclose(traj.wheel_momentum[-1], [0.0, 0.0, -0.02], atol=1e-10)
    np.testing.assert_allclose(traj.omega[-1], [0.0, 0.0, 0.005], atol=1e-10)
    wheel_axes = np.column_stack([wheel.axis_body for wheel in body.reaction_wheels])
    total_h0 = body.current_inertia @ traj.omega[0] + wheel_axes @ traj.wheel_momentum[0]
    total_h1 = body.current_inertia @ traj.omega[-1] + wheel_axes @ traj.wheel_momentum[-1]
    np.testing.assert_allclose(total_h1, total_h0, atol=1e-10)


def test_attitude_pd_torque_uses_shortest_quaternion_error():
    q_z_error = normalize_quaternion([np.cos(0.05), 0.0, 0.0, np.sin(0.05)])
    controller = SpacecraftAttitudePD(kp=2.0, kd=0.5, max_torque=0.05)
    torque = controller(
        t=0.0,
        r=np.zeros(3),
        v=np.zeros(3),
        q=q_z_error,
        omega=np.array([0.0, 0.0, 0.02]),
    )

    assert attitude_error_quaternion(q_z_error)[3] > 0.0
    assert torque[2] < 0.0
    assert np.linalg.norm(torque) <= 0.05
    np.testing.assert_allclose(
        attitude_error_quaternion(-q_z_error),
        attitude_error_quaternion(q_z_error),
    )


def test_spacecraft_maneuver_accel_supports_operational_frames():
    spacecraft = Spacecraft(
        r=[7_000_000.0, 0.0, 0.0],
        v=[0.0, 7_500.0, 0.0],
        q=[np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
        omega=[0.0, 0.0, 0.0],
        inertia=np.eye(3),
        mass=100.0,
    )

    np.testing.assert_allclose(
        SpacecraftManeuverAccel(10.0, frame="rtn")(spacecraft),
        [0.0, 0.1, 0.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        SpacecraftManeuverAccel(10.0, frame="ntw")(spacecraft),
        [0.0, 0.1, 0.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        SpacecraftManeuverAccel(10.0, frame="body")(spacecraft),
        [0.0, 0.1, 0.0],
        atol=1e-12,
    )
    assert SpacecraftManeuverAccel(10.0, frame="rtn", isp=200.0).mass_flow_rate(spacecraft) == pytest.approx(
        10.0 / (200.0 * STANDARD_GRAVITY)
    )


def test_spacecraft_maneuver_accel_propagates_variable_finite_burn(gps_epoch):
    spacecraft = Spacecraft(
        r=[0.0, 0.0, 0.0],
        v=[0.0, 0.0, 0.0],
        t=gps_epoch,
        inertia=np.eye(3),
        mass=100.0,
    )
    # thrust_profile_constant gates on the absolute epoch, so its window moves
    # with the grid.
    burn = SpacecraftManeuverAccel(
        thrust_profile_constant(2.0, start=gps_epoch + 0.0, stop=gps_epoch + 10.0),
        frame="gcrf",
        direction=[1.0, 0.0, 0.0],
    )

    traj = spacecraft.propagate(
        times=[gps_epoch + 0.0, gps_epoch + 10.0], mu=0.0, acceleration=burn
    )

    np.testing.assert_allclose(traj.v[-1], [0.2, 0.0, 0.0], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(traj.r[-1], [1.0, 0.0, 0.0], rtol=1e-10, atol=1e-12)

    burn_with_isp = SpacecraftManeuverAccel(
        2.0,
        frame="gcrf",
        direction=[1.0, 0.0, 0.0],
        isp=200.0,
        start=gps_epoch + 0.0,
        stop=gps_epoch + 10.0,
    )
    mass_traj = spacecraft.propagate(
        times=[gps_epoch + 0.0, gps_epoch + 10.0], mu=0.0, acceleration=burn_with_isp
    )

    assert mass_traj.mass is not None
    assert mass_traj.mass[-1] == pytest.approx(100.0 - 2.0 * 10.0 / (200.0 * STANDARD_GRAVITY))


def test_spacecraft_propagate_tracks_thruster_mass_depletion():
    thrust = 1.0
    isp = 100.0
    burn_time = 10.0
    body = SpacecraftBody.box(name="bus", mass=10.0, size=(1.0, 1.0, 1.0)).with_thrusters(
        Thruster(thrust=thrust, direction_body=[1.0, 0.0, 0.0], isp=isp),
        append=False,
    )
    spacecraft = Spacecraft(
        r=[0.0, 0.0, 0.0],
        v=[0.0, 0.0, 0.0],
        q=[1.0, 0.0, 0.0, 0.0],
        omega=[0.0, 0.0, 0.0],
        body=body,
    )

    trajectory = spacecraft.propagate(
        times=[0.0, burn_time],
        mu=0.0,
        models=[SpacecraftThrusterAccel()],
        rtol=1e-10,
        atol=1e-12,
    )

    mass_flow_rate = thrust / (isp * STANDARD_GRAVITY)
    final_mass = spacecraft.mass - mass_flow_rate * burn_time
    expected_delta_v = isp * STANDARD_GRAVITY * np.log(spacecraft.mass / final_mass)

    np.testing.assert_allclose(trajectory.mass, [spacecraft.mass, final_mass], rtol=1e-12)
    np.testing.assert_allclose(trajectory.v[-1], [expected_delta_v, 0.0, 0.0], rtol=1e-9, atol=1e-12)
    assert trajectory.spacecraft().mass == pytest.approx(final_mass)


def test_propagate_6dof_tracks_explicit_mass_flow_rate():
    trajectory = propagate_6dof(
        r0=[0.0, 0.0, 0.0],
        v0=[0.0, 0.0, 0.0],
        times=[0.0, 2.0],
        inertia=np.eye(3),
        mu=0.0,
        mass0=5.0,
        mass_flow_rate=lambda t, r, v, q, omega: 0.25,
    )

    np.testing.assert_allclose(trajectory.mass, [5.0, 4.5])


def test_spacecraft_propagate_updates_body_mass_properties_during_burn():
    seen = []

    class Recorder:
        spacecraft_acceleration_model = True

        def __call__(self, *, spacecraft, t, r, v, q, omega):
            seen.append(
                (
                    float(spacecraft.mass),
                    float(spacecraft.body.current_mass),
                    float(spacecraft.body.current_center_of_mass[1]),
                    float(spacecraft.inertia[0, 0]),
                )
            )
            return np.zeros(3)

    body = SpacecraftBody.box(name="bus", mass=10.0, size=(1.0, 1.0, 1.0)).with_tanks(
        Tank(propellant_mass=10.0, dry_mass=0.0, position_body=[0.0, 2.0, 0.0])
    )
    spacecraft = Spacecraft(r=[0, 0, 0], v=[0, 0, 0], body=body)

    trajectory = spacecraft.propagate(
        times=[0.0, 1.0],
        mu=0.0,
        models=[Recorder()],
        mass_flow_rate=lambda t, r, v, q, omega: 5.0,
        max_step=0.1,
    )

    assert trajectory.mass[-1] == pytest.approx(15.0)
    assert min(item[0] for item in seen) < spacecraft.mass
    assert min(item[1] for item in seen) < body.current_mass
    assert min(item[2] for item in seen) < body.current_center_of_mass[1]
    assert min(item[3] for item in seen) < body.current_inertia[0, 0]
    sampled = trajectory.spacecraft(body=body)
    assert sampled.body.propellant_mass == pytest.approx(5.0)
    assert sampled.body.current_mass == pytest.approx(sampled.mass)


def test_propellant_empty_event_stops_at_body_dry_mass():
    body = SpacecraftBody.box(name="bus", mass=10.0, size=(1.0, 1.0, 1.0)).with_tanks(
        Tank(propellant_mass=5.0, dry_mass=1.0)
    )
    spacecraft = Spacecraft(r=[0, 0, 0], v=[0, 0, 0], body=body)

    trajectory = spacecraft.propagate(
        times=[0.0, 10.0],
        mu=0.0,
        mass_flow_rate=lambda t, r, v, q, omega: 2.0,
        stop_at_dry_mass=True,
    )

    assert trajectory.t[-1] == pytest.approx(2.5)
    assert trajectory.mass[-1] == pytest.approx(body.dry_mass_total)
    assert trajectory.t_events[0][0] == pytest.approx(2.5)
    unchecked = spacecraft.propagate(
        times=[0.0, 3.0],
        mu=0.0,
        mass_flow_rate=lambda t, r, v, q, omega: 2.0,
    )
    assert unchecked.t[-1] == pytest.approx(3.0)
    assert unchecked.mass[-1] == pytest.approx(body.dry_mass_total)
    assert np.all(unchecked.mass >= body.dry_mass_total)
    assert propellant_empty_event(body)(0.0, np.r_[np.zeros(13), body.dry_mass_total]) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="SpacecraftBody"):
        propellant_empty_event(object())


def test_spacecraft_propagate_coasts_without_propulsive_acceleration_after_depletion():
    body = (
        SpacecraftBody.box(name="bus", mass=10.0, size=(1.0, 1.0, 1.0))
        .with_tanks(Tank(propellant_mass=5.0, dry_mass=1.0))
        .with_thrusters(
            Thruster(thrust=1.0, direction_body=[1.0, 0.0, 0.0], isp=0.1),
            append=False,
        )
    )
    spacecraft = Spacecraft(r=[0, 0, 0], v=[0, 0, 0], body=body)

    trajectory = spacecraft.propagate(
        times=np.linspace(0.0, 10.0, 11),
        mu=0.0,
        acceleration=SpacecraftManeuverAccel(
            1.0,
            frame="gcrf",
            direction=[1.0, 0.0, 0.0],
            isp=0.1,
        ),
        dense_output=True,
    )

    assert trajectory.t[-1] == pytest.approx(10.0)
    assert np.all(trajectory.mass >= body.dry_mass_total)
    depleted = np.flatnonzero(np.isclose(trajectory.mass, body.dry_mass_total))
    assert depleted.size
    np.testing.assert_allclose(trajectory.v[depleted[0]:, 0], trajectory.v[depleted[0], 0])
    assert trajectory.solution(10.0)[13] == pytest.approx(body.dry_mass_total)


def test_rhs_central_gravity_matches_newtonian_acceleration():
    r = np.array([7_000_000.0, 1_000_000.0, 0.0])
    v = np.array([100.0, 7_400.0, 10.0])
    y = np.concatenate([r, v, [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    dy = sixdof_rhs(0.0, y, inertia=np.eye(3))

    np.testing.assert_allclose(dy[0:3], v)
    np.testing.assert_allclose(dy[3:6], -EARTH_MU * r / np.linalg.norm(r) ** 3)
    np.testing.assert_allclose(dy[10:13], 0.0)


def test_body_acceleration_rotates_through_current_attitude():
    q_z90 = normalize_quaternion([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    times = np.array([0.0, 10.0, 20.0])

    traj = propagate_6dof(
        r0=[0.0, 0.0, 0.0],
        v0=[0.0, 0.0, 0.0],
        q0=q_z90,
        times=times,
        mu=0.0,
        inertia=np.eye(3),
        body_acceleration=lambda t, r, v, q, omega: [2.0e-6, 0.0, 0.0],
    )

    np.testing.assert_allclose(traj.r[:, 0], 0.0, atol=1e-14)
    np.testing.assert_allclose(traj.v[:, 0], 0.0, atol=1e-14)
    np.testing.assert_allclose(traj.v[:, 1], 2.0e-6 * times, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(
        traj.r[:, 1],
        0.5 * 2.0e-6 * times**2,
        rtol=1e-10,
        atol=1e-14,
    )


def test_ntw_acceleration_uses_ssapy_component_order():
    y = np.concatenate(
        [
            [7_000_000.0, 0.0, 0.0],
            [0.0, 7_500.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    dy = sixdof_rhs(
        0.0,
        y,
        mu=0.0,
        inertia=np.eye(3),
        ntw_acceleration=lambda t, r, v, q, omega: [1.0e-6, 2.0e-6, 3.0e-6],
    )

    np.testing.assert_allclose(dy[3:6], [1.0e-6, 2.0e-6, 3.0e-6])


def test_constant_torque_changes_principal_axis_spin():
    times = np.array([0.0, 2.0, 4.0])

    traj = propagate_6dof(
        r0=[0.0, 0.0, 0.0],
        v0=[0.0, 0.0, 0.0],
        times=times,
        mu=0.0,
        inertia=np.diag([2.0, 3.0, 4.0]),
        torque=lambda t, r, v, q, omega: [2.0, 0.0, 0.0],
    )

    np.testing.assert_allclose(traj.omega[:, 0], times, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(traj.omega[:, 1:], 0.0, atol=1e-12)


def test_propagate_6dof_supports_terminal_events_and_dense_output(gps_epoch):
    def reaches_half_meter(_t, y):
        return y[0] - 0.5

    reaches_half_meter.terminal = True
    reaches_half_meter.direction = 1

    traj = propagate_6dof(
        r0=[0.0, 0.0, 0.0],
        v0=[1.0, 0.0, 0.0],
        t0=gps_epoch,
        times=gps_epoch + np.array([0.0, 0.25, 0.75, 1.0]),
        inertia=np.eye(3),
        mu=0.0,
        events=reaches_half_meter,
        dense_output=True,
    )

    assert traj.status == 1
    assert traj.t_events is not None
    assert traj.y_events is not None
    assert traj.t[-1] == pytest.approx(gps_epoch + 0.5, rel=0.0, abs=1.0e-6)
    assert traj.r[-1, 0] == pytest.approx(0.5)
    assert traj.t_events[0][0] == pytest.approx(gps_epoch + 0.5, rel=0.0, abs=1.0e-6)
    assert traj.y_events[0][0, 0] == pytest.approx(0.5)
    assert traj.solution is not None
    # Dense output is queried in absolute time.
    assert traj.solution(gps_epoch + 0.25)[0] == pytest.approx(0.25)


def test_physical_event_helpers_stop_radius_altitude_and_mass_crossings(gps_epoch):
    radius_event = radius_crossing_event(0.5, direction=1)
    radius_traj = propagate_6dof(
        r0=[0.0, 0.0, 0.0],
        v0=[1.0, 0.0, 0.0],
        t0=gps_epoch,
        times=gps_epoch + np.array([0.0, 0.25, 0.75, 1.0]),
        inertia=np.eye(3),
        mu=0.0,
        events=radius_event,
    )
    assert radius_traj.status == 1
    assert radius_traj.t[-1] == pytest.approx(gps_epoch + 0.5, rel=0.0, abs=1.0e-6)
    assert radius_traj.t_events[0][0] == pytest.approx(
        gps_epoch + 0.5, rel=0.0, abs=1.0e-6
    )

    altitude_event = altitude_crossing_event(1.0, earth_radius=10.0, direction=1)
    assert altitude_event(0.0, np.r_[11.0, 0.0, 0.0, np.zeros(10)]) == pytest.approx(0.0)

    mass_traj = propagate_6dof(
        r0=[0.0, 0.0, 0.0],
        v0=[0.0, 0.0, 0.0],
        times=[0.0, 0.75, 1.5],
        inertia=np.eye(3),
        mu=0.0,
        mass0=10.0,
        mass_flow_rate=lambda t, r, v, q, omega: 2.0,
        events=mass_floor_event(8.0),
    )
    assert mass_traj.status == 1
    assert mass_traj.t[-1] == pytest.approx(1.0)
    assert mass_traj.t_events[0][0] == pytest.approx(1.0)
    assert mass_traj.y_events[0][0, 13] == pytest.approx(8.0)
