import numpy as np

import ssatk
from ssatk.accelerations_6dof import (
    SpacecraftAccelConstBody,
    SpacecraftAccelConstNTW,
    constant_body_torque,
)
from ssatk.coordinates.attitude import normalize_quaternion, rotate_vector
from ssatk.propagators_6dof import (
    attitude_error_stm,
    propagate_6dof,
    propagate_6dof_covariance,
    propagate_6dof_variational,
)
from ssatk.propagators_6dof.sixdof import sixdof_rhs
from ssatk.propagators_6dof.variational import (
    _body_acceleration_jacobian,
    _free_rigid_body_jacobian,
    _gravity_gradient_jacobian,
    _model_state_jacobian,
)


def test_variational_stm_matches_terminal_finite_difference_response():
    assert ssatk.propagate_6dof_variational is propagate_6dof_variational
    kwargs = {
        "times": np.array([0.0, 0.2]),
        "r0": np.array([7_000_000.0, 0.0, 0.0]),
        "v0": np.array([0.0, 7_500.0, 0.0]),
        "q0": np.array([1.0, 0.0, 0.0, 0.0]),
        "omega0": np.array([0.0, 0.0, 0.01]),
        "inertia": np.diag([2.0, 3.0, 4.0]),
        "rtol": 1e-10,
        "atol": 1e-12,
    }
    result = propagate_6dof_variational(**kwargs, jacobian_step=1e-6)
    delta = np.zeros(13)
    delta[0] = 0.1
    delta[10] = 1e-7
    plus = propagate_6dof(**{**kwargs, "r0": kwargs["r0"] + delta[:3], "omega0": kwargs["omega0"] + delta[10:13]})
    minus = propagate_6dof(**{**kwargs, "r0": kwargs["r0"] - delta[:3], "omega0": kwargs["omega0"] - delta[10:13]})
    finite_difference = np.concatenate((
        plus.r[-1] - minus.r[-1],
        plus.v[-1] - minus.v[-1],
        plus.q[-1] - minus.q[-1],
        plus.omega[-1] - minus.omega[-1],
    )) / 2.0
    predicted = result.stm[-1] @ delta

    assert result.stm.shape == (2, 13, 13)
    np.testing.assert_allclose(predicted, finite_difference, rtol=2e-4, atol=1e-12)


def test_variational_constant_ntw_acceleration_uses_analytic_state_jacobian():
    model = SpacecraftAccelConstNTW([1.0e-4, 2.0e-4, -0.5e-4])
    calls = [0]
    state_jacobian = model.state_jacobian

    def counted_state_jacobian(**kwargs):
        calls[0] += 1
        return state_jacobian(**kwargs)

    model.state_jacobian = counted_state_jacobian
    kwargs = {
        "times": np.array([0.0, 0.2]),
        "r0": np.array([7_000_000.0, 0.0, 0.0]),
        "v0": np.array([0.0, 7_500.0, 0.0]),
        "q0": np.array([0.9, 0.1, -0.2, 0.3]),
        "omega0": np.array([0.0, 0.0, 0.01]),
        "inertia": np.diag([2.0, 3.0, 4.0]),
        "acceleration": model,
        "rtol": 1e-10,
        "atol": 1e-12,
    }
    result = propagate_6dof_variational(**kwargs)
    assert calls[0] > 0

    delta = np.zeros(13)
    delta[0] = 0.1
    delta[3] = 1.0e-4
    plus = propagate_6dof(
        **{**kwargs, "r0": kwargs["r0"] + delta[:3], "v0": kwargs["v0"] + delta[3:6]}
    )
    minus = propagate_6dof(
        **{**kwargs, "r0": kwargs["r0"] - delta[:3], "v0": kwargs["v0"] - delta[3:6]}
    )
    finite_difference = np.concatenate((
        plus.r[-1] - minus.r[-1],
        plus.v[-1] - minus.v[-1],
        plus.q[-1] - minus.q[-1],
        plus.omega[-1] - minus.omega[-1],
    )) / 2.0

    np.testing.assert_allclose(result.stm[-1] @ delta, finite_difference, rtol=2e-4, atol=1e-11)


def test_variational_constant_body_torque_uses_analytic_state_jacobian():
    model = constant_body_torque([0.0, 0.0, 1.0e-6])
    calls = [0]
    state_jacobian = model.state_jacobian

    def counted_state_jacobian(**kwargs):
        calls[0] += 1
        return state_jacobian(**kwargs)

    model.state_jacobian = counted_state_jacobian
    kwargs = {
        "times": np.array([0.0, 0.2]),
        "r0": np.array([7_000_000.0, 0.0, 0.0]),
        "v0": np.array([0.0, 7_500.0, 0.0]),
        "q0": np.array([1.0, 0.0, 0.0, 0.0]),
        "omega0": np.array([0.0, 0.0, 0.01]),
        "inertia": np.diag([2.0, 3.0, 4.0]),
        "torque": model,
        "rtol": 1e-10,
        "atol": 1e-12,
    }
    result = propagate_6dof_variational(**kwargs)
    assert calls[0] > 0

    delta = np.zeros(13)
    delta[10] = 1.0e-7
    plus = propagate_6dof(**{**kwargs, "omega0": kwargs["omega0"] + delta[10:13]})
    minus = propagate_6dof(**{**kwargs, "omega0": kwargs["omega0"] - delta[10:13]})
    finite_difference = np.concatenate((
        plus.r[-1] - minus.r[-1],
        plus.v[-1] - minus.v[-1],
        plus.q[-1] - minus.q[-1],
        plus.omega[-1] - minus.omega[-1],
    )) / 2.0

    np.testing.assert_allclose(result.stm[-1] @ delta, finite_difference, rtol=2e-4, atol=1e-11)


def test_constant_body_acceleration_state_jacobian_matches_finite_difference():
    model = SpacecraftAccelConstBody([1.0e-3, -2.0e-3, 0.5e-3])
    state = np.array([
        7.0e6, -1.2e6, 0.8e6, 1.2e3, 7.3e3, -0.5e3,
        0.9, 0.1, -0.2, 0.3, 0.01, -0.02, 0.03,
    ])
    jacobian = _model_state_jacobian(model, 0.0, state)
    finite_difference = np.empty((3, 13))

    def acceleration(y):
        return model(0.0, y[:3], y[3:6], normalize_quaternion(y[6:10]), y[10:13])

    for column in range(state.size):
        step = 1.0e-7 * max(1.0, abs(state[column]))
        delta = np.zeros(state.size)
        delta[column] = step
        finite_difference[:, column] = (
            acceleration(state + delta) - acceleration(state - delta)
        ) / (2.0 * step)

    np.testing.assert_allclose(jacobian, finite_difference, rtol=1e-6, atol=1e-12)


def test_attitude_error_stm_has_nonsingular_three_parameter_attitude_state():
    result = propagate_6dof_variational(
        times=[0.0, 0.01],
        inertia=np.diag([2.0, 3.0, 4.0]),
        r0=[7.0e6, 0.0, 0.0],
        v0=[0.0, 7_500.0, 0.0],
        q0=[0.9, 0.1, -0.2, 0.3],
        omega0=[0.01, -0.02, 0.03],
    )
    error_stm = attitude_error_stm(result)

    assert result.attitude_error_stm is not None
    assert error_stm.shape == (2, 12, 12)
    np.testing.assert_allclose(error_stm[0], np.eye(12), atol=1e-12)


def test_covariance_transform_uses_stm_and_process_noise():
    result = propagate_6dof_variational(
        times=[0.0, 0.01],
        inertia=np.eye(3),
        r0=[7e6, 0, 0],
        v0=[0, 7500, 0],
        rtol=1e-6,
        atol=1e-9,
    )
    covariance0 = np.eye(13)
    process_noise = np.broadcast_to(np.eye(13) * 1e-6, result.stm.shape).copy()
    covariance = propagate_6dof_covariance(result, covariance0, process_noise)
    expected = result.stm[-1] @ covariance0 @ result.stm[-1].T + process_noise[-1]

    assert ssatk.propagate_6dof_covariance is propagate_6dof_covariance
    np.testing.assert_allclose(covariance[-1], expected, atol=1e-12)
    np.testing.assert_allclose(covariance, np.swapaxes(covariance, -1, -2), atol=1e-12)

    local_stm = result.attitude_error_stm
    local_noise = np.broadcast_to(np.eye(12) * 1e-6, local_stm.shape).copy()
    local_covariance = propagate_6dof_covariance(
        result, np.eye(12), local_noise, coordinates="attitude_error"
    )
    local_expected = local_stm[-1] @ local_stm[-1].T + local_noise[-1]
    np.testing.assert_allclose(local_covariance[-1], local_expected, atol=1e-12)
    np.testing.assert_allclose(
        local_covariance,
        propagate_6dof_covariance(result, np.eye(12), local_noise),
        atol=1e-12,
    )


def test_free_rigid_body_jacobian_matches_rhs_finite_difference():
    inertia = np.diag([2.0, 3.0, 4.0])
    state = np.array([
        7.0e6, -1.2e6, 0.8e6, 1.2e3, 7.3e3, -0.5e3,
        0.9, 0.1, -0.2, 0.3, 0.01, -0.02, 0.03,
    ])
    jacobian = _free_rigid_body_jacobian(state, mu=ssatk.EARTH_MU, inertia=inertia)
    finite_difference = np.empty_like(jacobian)
    for column in range(state.size):
        step = 1.0e-7 * max(1.0, abs(state[column]))
        delta = np.zeros(state.size)
        delta[column] = step
        plus = sixdof_rhs(0.0, state + delta, inertia=inertia)
        minus = sixdof_rhs(0.0, state - delta, inertia=inertia)
        finite_difference[:, column] = (plus - minus) / (2.0 * step)

    np.testing.assert_allclose(jacobian, finite_difference, rtol=1.0e-6, atol=1.0e-10)


def test_free_rigid_body_jacobian_matches_passive_wheel_coupling():
    inertia = np.diag([2.0, 3.0, 4.0])
    wheel_axes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    state = np.array([
        7.0e6, -1.2e6, 0.8e6, 1.2e3, 7.3e3, -0.5e3,
        0.9, 0.1, -0.2, 0.3, 0.01, -0.02, 0.03, 0.4, -0.5,
    ])
    jacobian = _free_rigid_body_jacobian(
        state, mu=ssatk.EARTH_MU, inertia=inertia, wheel_axes=wheel_axes
    )
    finite_difference = np.empty_like(jacobian)
    for column in range(state.size):
        step = 1.0e-7 * max(1.0, abs(state[column]))
        delta = np.zeros(state.size)
        delta[column] = step
        plus = sixdof_rhs(0.0, state + delta, inertia=inertia, wheel_axes_body=wheel_axes)
        minus = sixdof_rhs(0.0, state - delta, inertia=inertia, wheel_axes_body=wheel_axes)
        finite_difference[:, column] = (plus - minus) / (2.0 * step)

    np.testing.assert_allclose(jacobian, finite_difference, rtol=1.0e-6, atol=1.0e-10)


def test_gravity_gradient_jacobian_matches_torque_finite_difference():
    inertia = np.diag([2.0, 3.0, 4.0])
    position = np.array([7.0e6, -1.2e6, 0.8e6])
    quaternion = np.array([0.9, 0.1, -0.2, 0.3])
    torque_dr, torque_dq = _gravity_gradient_jacobian(
        position, quaternion, inertia, ssatk.EARTH_MU
    )

    from ssatk.propagators_6dof.sixdof import _gravity_gradient_torque_prepared

    finite_dr = np.empty((3, 3))
    finite_dq = np.empty((3, 4))
    for column in range(3):
        step = 1.0e-6 * max(1.0, abs(position[column]))
        delta = np.zeros(3)
        delta[column] = step
        finite_dr[:, column] = (
            _gravity_gradient_torque_prepared(
                position + delta, quaternion, inertia, ssatk.EARTH_MU
            )
            - _gravity_gradient_torque_prepared(
                position - delta, quaternion, inertia, ssatk.EARTH_MU
            )
        ) / (2.0 * step)
    for column in range(4):
        step = 1.0e-7
        delta = np.zeros(4)
        delta[column] = step
        finite_dq[:, column] = (
            _gravity_gradient_torque_prepared(
                position, quaternion + delta, inertia, ssatk.EARTH_MU
            )
            - _gravity_gradient_torque_prepared(
                position, quaternion - delta, inertia, ssatk.EARTH_MU
            )
        ) / (2.0 * step)

    np.testing.assert_allclose(torque_dr, finite_dr, rtol=1.0e-6, atol=1.0e-18)
    np.testing.assert_allclose(torque_dq, finite_dq, rtol=1.0e-6, atol=1.0e-12)


def test_variational_jacobian_includes_analytic_body_force_attitude_partial():
    body_acceleration = lambda t, r, v, q, omega: [0.0, 1.0e-3, 0.0]
    body_acceleration.attitude_jacobian = lambda q: np.zeros((3, 4))
    state = np.array([
        7.0e6, -1.2e6, 0.8e6, 1.2e3, 7.3e3, -0.5e3,
        0.9, 0.1, -0.2, 0.3, 0.01, -0.02, 0.03,
    ])
    jacobian = _free_rigid_body_jacobian(
        state, mu=ssatk.EARTH_MU, inertia=np.diag([2.0, 3.0, 4.0])
    )
    jacobian[3:6, 6:10] = _body_acceleration_jacobian(
        body_acceleration, 0.0, state, q_raw=state[6:10]
    )
    finite = np.empty((3, 4))
    for column in range(4):
        step = 1.0e-7
        delta = np.zeros(4)
        delta[column] = step
        plus = rotate_vector(state[6:10] + delta, body_acceleration(0.0, state[:3], state[3:6], state[6:10] + delta, state[10:13]))
        minus = rotate_vector(state[6:10] - delta, body_acceleration(0.0, state[:3], state[3:6], state[6:10] - delta, state[10:13]))
        finite[:, column] = (plus - minus) / (2.0 * step)

    np.testing.assert_allclose(jacobian[3:6, 6:10], finite, rtol=1.0e-6, atol=1.0e-12)


def test_body_force_attitude_jacobian_chains_quaternion_normalization():
    def body_acceleration(t, r, v, q, omega):
        return 1.0e-3 * np.array([q[0], 2.0 * q[1], -q[2]])

    body_acceleration.attitude_jacobian = lambda q: 1.0e-3 * np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
    ])
    state = np.array([
        7.0e6, -1.2e6, 0.8e6, 1.2e3, 7.3e3, -0.5e3,
        0.9, 0.1, -0.2, 0.3, 0.01, -0.02, 0.03,
    ])
    jacobian = _body_acceleration_jacobian(
        body_acceleration, 0.0, state, q_raw=state[6:10]
    )
    finite = np.empty((3, 4))
    for column in range(4):
        step = 1.0e-7
        delta = np.zeros(4)
        delta[column] = step
        plus_q = normalize_quaternion(state[6:10] + delta)
        minus_q = normalize_quaternion(state[6:10] - delta)
        plus = rotate_vector(plus_q, body_acceleration(0.0, state[:3], state[3:6], plus_q, state[10:13]))
        minus = rotate_vector(minus_q, body_acceleration(0.0, state[:3], state[3:6], minus_q, state[10:13]))
        finite[:, column] = (plus - minus) / (2.0 * step)

    np.testing.assert_allclose(jacobian, finite, rtol=1.0e-6, atol=1.0e-12)
