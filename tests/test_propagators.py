
import numpy as np

from ssatk.accelerations_6dof import (
    SpacecraftManeuverAccel,
    SpacecraftReactionWheelTorque,
)
from ssatk.constants import EARTH_MU
from ssatk.propagators_6dof import (
    Spacecraft,
    propagate_spacecraft_segments,
)
from ssatk.propagators_orbit import (
    propagate_orbit_state,
    propagate_orbit_state_with_stm,
)
from ssatk.propagators_orbit.high_accuracy import _kepler_jacobian
from ssatk.satellites import SpacecraftBody, reaction_wheel_triplet


def test_high_accuracy_orbit_propagator_returns_near_circular_state_after_period():
    radius = 7_000_000.0
    speed = np.sqrt(EARTH_MU / radius)
    period = 2.0 * np.pi * np.sqrt(radius**3 / EARTH_MU)

    trajectory = propagate_orbit_state(
        r0=[radius, 0.0, 0.0],
        v0=[0.0, speed, 0.0],
        times=np.linspace(0.0, period, 16),
    )

    assert trajectory.r.shape == trajectory.v.shape == (16, 3)
    assert trajectory.nfev > 0
    np.testing.assert_allclose(trajectory.r[-1], trajectory.r[0], atol=30.0, rtol=0)
    np.testing.assert_allclose(trajectory.v[-1], trajectory.v[0], atol=0.05, rtol=0)
    phase = 2 * np.pi * np.linspace(0, 1, 16)
    expected_r = radius * np.column_stack((np.cos(phase), np.sin(phase), np.zeros(16)))
    expected_v = speed * np.column_stack((-np.sin(phase), np.cos(phase), np.zeros(16)))
    np.testing.assert_allclose(trajectory.r, expected_r, atol=30.0, rtol=0)
    np.testing.assert_allclose(trajectory.v, expected_v, atol=0.05, rtol=0)


def test_orbit_stm_matches_finite_difference_of_propagated_state():
    radius = 7_000_000.0
    speed = np.sqrt(EARTH_MU / radius)
    times = np.linspace(0.0, 900.0, 5)
    state = propagate_orbit_state_with_stm(
        r0=[radius, 0.0, 0.0],
        v0=[0.0, speed, 0.0],
        times=times,
    )
    delta = np.array([0.2, -0.1, 0.05, 1.0e-4, -2.0e-4, 3.0e-4])
    plus = propagate_orbit_state(
        r0=np.array([radius, 0.0, 0.0]) + delta[:3],
        v0=np.array([0.0, speed, 0.0]) + delta[3:],
        times=times,
    )
    minus = propagate_orbit_state(
        r0=np.array([radius, 0.0, 0.0]) - delta[:3],
        v0=np.array([0.0, speed, 0.0]) - delta[3:],
        times=times,
    )
    predicted = np.einsum("tij,j->ti", state.stm, delta)
    actual = 0.5 * np.column_stack((plus.r - minus.r, plus.v - minus.v))
    np.testing.assert_allclose(actual, predicted, rtol=3.0e-4, atol=2.0e-7)


def test_kepler_stm_jacobian_matches_central_gravity_derivative():
    r = np.array([7_000_000.0, -1_200_000.0, 800_000.0])
    radius = np.linalg.norm(r)
    expected = np.zeros((6, 6))
    expected[:3, 3:] = np.eye(3)
    expected[3:, :3] = -EARTH_MU * (
        np.eye(3) / radius**3 - 3.0 * np.outer(r, r) / radius**5
    )

    np.testing.assert_allclose(_kepler_jacobian(r, EARTH_MU), expected)


def test_high_accuracy_spacecraft_segments_chain_state_and_mass(gps_epoch):
    import ssatk

    assert ssatk.propagate_spacecraft_segments is propagate_spacecraft_segments
    spacecraft = Spacecraft(
        r=[0.0, 0.0, 0.0],
        v=[0.0, 0.0, 0.0],
        t=gps_epoch,
        inertia=np.eye(3),
        mass=100.0,
    )
    # The burn window is an absolute-epoch quantity, like the segment times.
    burn = SpacecraftManeuverAccel(
        1.0,
        frame="gcrf",
        direction=[1.0, 0.0, 0.0],
        isp=100.0,
        start=gps_epoch + 1.0,
        stop=gps_epoch + 2.0,
    )

    trajectory = propagate_spacecraft_segments(
        spacecraft,
        [
            {"times": [gps_epoch + 0.0, gps_epoch + 1.0], "mu": 0.0},
            {"times": [gps_epoch + 1.0, gps_epoch + 2.0], "models": [burn], "mu": 0.0},
        ],
    )

    np.testing.assert_allclose(trajectory.t, gps_epoch + np.array([0.0, 1.0, 2.0]))
    assert trajectory.mass is not None
    # SI standard gravity is defined as 9.80665 m/s^2. A one-second 1 N burn
    # consumes F/(Isp*g0) kg; the rocket equation supplies the velocity oracle.
    exhaust_speed = 100.0 * 9.80665
    final_mass = 100.0 - 1.0 / exhaust_speed
    np.testing.assert_allclose(trajectory.mass, [100.0, 100.0, final_mass], atol=1e-9, rtol=0)
    np.testing.assert_allclose(trajectory.v[-1],
                               [exhaust_speed * np.log(100.0 / final_mass), 0, 0], atol=1e-8, rtol=0)
    assert trajectory.nfev > 0


def test_high_accuracy_spacecraft_segments_preserve_wheel_momentum(gps_epoch):
    body = SpacecraftBody.cubesat(1, mass=10.0).with_reaction_wheels(
        *reaction_wheel_triplet(max_torque=0.1)
    )
    spacecraft = Spacecraft(
        r=[0.0, 0.0, 0.0],
        v=[0.0, 0.0, 0.0],
        t=gps_epoch,
        body=body,
    )

    trajectory = propagate_spacecraft_segments(
        spacecraft,
        [
            {"times": [gps_epoch + 0.0, gps_epoch + 1.0], "mu": 0.0},
            {
                "times": [gps_epoch + 1.0, gps_epoch + 2.0],
                "models": [SpacecraftReactionWheelTorque([0.0, 0.0, 0.03])],
                "mu": 0.0,
            },
        ],
    )

    assert trajectory.wheel_momentum.shape == (3, 3)
    np.testing.assert_allclose(trajectory.wheel_momentum[0], [0.0, 0.0, 0.0])
    # The internal actuator transfers equal and opposite angular momentum.
    np.testing.assert_allclose(trajectory.wheel_momentum[-1], [0, 0, -.03], atol=1e-10, rtol=0)
    # A 0.1 m cubic bus of mass 10 kg has Izz=m*a^2/6.
    np.testing.assert_allclose(trajectory.omega[-1], [0, 0, .03 / (10 * .1**2 / 6)],
                               atol=1e-9, rtol=0)
