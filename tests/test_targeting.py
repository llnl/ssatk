import numpy as np

import ssatk


def test_solve_6dof_target_reaches_terminal_velocity():
    spacecraft = ssatk.Spacecraft(
        r=[0.0, 0.0, 0.0],
        v=[0.0, 0.0, 0.0],
        inertia=np.eye(3),
    )

    result = ssatk.solve_6dof_target(
        spacecraft,
        times=np.linspace(0.0, 10.0, 11),
        target_v=[2.0, 0.0, 0.0],
        control_scale=[1.0, 1.0, 1.0],
        propagation_kwargs={"mu": 0.0},
    )

    assert result.success
    assert result.nfev <= 20
    np.testing.assert_allclose(result.control, [0.2, 0.0, 0.0], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(result.trajectory.v[-1], [2.0, 0.0, 0.0], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(result.residual, 0.0, atol=1e-12)


def test_solve_6dof_multi_segment_target_reaches_terminal_state_with_constraints():
    spacecraft = ssatk.Spacecraft(
        r=[0.0, 0.0, 0.0],
        v=[0.0, 0.0, 0.0],
        inertia=np.eye(3),
    )

    result = ssatk.solve_6dof_multi_segment_target(
        spacecraft,
        segments=[
            {"times": [0.0, 5.0], "mu": 0.0},
            {"times": [5.0, 10.0], "mu": 0.0},
        ],
        target_v=[3.0, 0.0, 0.0],
        control_scale=[1.0, 1.0, 1.0],
        bounds=([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]),
        constraints=lambda trajectory: [max(0.0, np.max(np.abs(trajectory.r[:, 1])) - 1e-10)],
    )

    assert result.success
    np.testing.assert_allclose(result.trajectory.v[-1], [3.0, 0.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(result.controls[:, 0].sum(), 0.6, atol=1e-10)
    assert np.all(np.abs(result.controls) <= 1.0)


def test_multiple_shooting_targets_independent_internal_node():
    spacecraft = ssatk.Spacecraft(
        r=[0.0, 0.0, 0.0],
        v=[0.0, 0.0, 0.0],
        inertia=np.eye(3),
    )

    result = ssatk.solve_6dof_multiple_shooting_target(
        spacecraft,
        segments=[
            {"times": [0.0, 5.0], "mu": 0.0},
            {"times": [5.0, 10.0], "mu": 0.0},
        ],
        terminal_residual=lambda final: (final.v - [2.0, 0.0, 0.0]),
        control_scale=[1.0, 1.0, 1.0],
        position_scale=1.0,
    )

    assert result.success
    assert len(result.node_states) == len(result.segment_trajectories) == 2
    np.testing.assert_allclose(result.trajectory.v[-1], [2.0, 0.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(
        result.segment_trajectories[0].r[-1], result.node_states[1].r, atol=1e-7
    )
    np.testing.assert_allclose(
        result.segment_trajectories[0].v[-1], result.node_states[1].v, atol=1e-8
    )


def test_multiple_shooting_preserves_pre_impulse_continuity_and_state_jump():
    spacecraft = ssatk.Spacecraft(
        r=[0.0, 0.0, 0.0],
        v=[0.0, 0.0, 0.0],
        inertia=np.eye(3),
    )

    result = ssatk.solve_6dof_multiple_shooting_target(
        spacecraft,
        segments=[
            {"times": [0.0, 5.0], "mu": 0.0},
            {
                "times": [5.0, 10.0],
                "mu": 0.0,
                "impulses": ssatk.ImpulseManeuver(
                    dv=[1.0, 0.0, 0.0],
                    q_reset=[0.0, 1.0, 0.0, 0.0],
                    omega_reset=[0.1, 0.0, 0.0],
                ),
            },
        ],
        target_v=[1.0, 0.0, 0.0],
        control_scale=[1.0, 1.0, 1.0],
        bounds=([-1.0e-12, -1.0e-12, -1.0e-12], [1.0e-12, 1.0e-12, 1.0e-12]),
        position_scale=1.0,
    )

    assert result.success
    first, second = result.segment_trajectories
    np.testing.assert_allclose(first.v[-1], result.node_states[1].v, atol=1e-12)
    np.testing.assert_allclose(second.v[0], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(second.q[0], [0.0, 1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(second.omega[0], [0.1, 0.0, 0.0], atol=1e-12)
    assert np.count_nonzero(np.isclose(result.trajectory.t, 5.0)) == 2


def test_multiple_shooting_continues_mass_and_wheel_state():
    body = (
        ssatk.SpacecraftBody(name="test", mass=8.0, inertia=np.eye(3))
        .with_tanks(ssatk.Tank(propellant_mass=2.0, name="tank"))
        .with_reaction_wheels(
            ssatk.ReactionWheel(
                axis_body=[1.0, 0.0, 0.0], max_torque=0.1,
                momentum_capacity=0.5, wheel_inertia=1.0,
            )
        )
    )
    spacecraft = ssatk.Spacecraft(
        r=[0.0, 0.0, 0.0],
        v=[0.0, 0.0, 0.0],
        body=body,
    )
    segment = {
        "mu": 0.0,
        "mass_flow_rate": lambda t, r, v, q, omega: 0.1,
        "wheel_torque": lambda t, r, v, q, omega: [0.05],
    }

    result = ssatk.solve_6dof_multiple_shooting_target(
        spacecraft,
        segments=[dict(segment, times=[0.0, 5.0]), dict(segment, times=[5.0, 10.0])],
        target_v=[0.0, 0.0, 0.0],
        control_scale=[1.0, 1.0, 1.0],
    )

    assert result.success
    first = result.segment_trajectories[0]
    assert first.mass is not None and first.wheel_momentum is not None
    np.testing.assert_allclose(first.mass[-1], result.node_states[1].mass, atol=1e-10)
    np.testing.assert_allclose(
        first.wheel_momentum[-1], result.node_states[1].wheel_momentum, atol=1e-10
    )
    assert abs(result.node_states[1].wheel_momentum[0]) <= 0.5
