from dataclasses import replace

import numpy as np

import ssatk
from ssatk.propagators_6dof import (
    FlexibleMode,
    HingedAppendage,
    SloshMode,
    attitude_error_stm,
    propagate_6dof_covariance,
    propagate_6dof_extended,
    propagate_6dof_extended_variational,
)


def test_extended_modes_propagate_and_couple_to_rigid_body():
    result = propagate_6dof_extended(
        times=np.linspace(0, 0.2, 5), inertia=np.eye(3), mu=0,
        r0=[7e6, 0, 0], v0=[0, 7500, 0], q0=[1, 0, 0, 0],
        hinge=HingedAppendage([0, 0, 1], 2, stiffness=4, angle0=0.1),
        flexible=FlexibleMode([1, 0, 0], 1, 3, displacement0=0.1),
        slosh=SloshMode([0, 1, 0], 1, 2, lever_arm_body=[1, 0, 0], displacement0=0.1),
        bus_mass=10,
    )
    assert result.trajectory.r.shape == (5, 3)
    assert result.hinge.shape == result.flexible.shape == result.slosh.shape == (5, 2)
    assert not np.allclose(result.trajectory.omega[0], result.trajectory.omega[-1])
    assert np.linalg.norm(result.trajectory.v[0] - result.trajectory.v[-1]) > 1e-6
    np.testing.assert_allclose(np.linalg.norm(result.trajectory.q, axis=1), 1.0, atol=1e-12)


def test_slosh_force_follows_attitude():
    result = propagate_6dof_extended(
        times=[0, 1e-3], inertia=np.eye(3), mu=0,
        r0=[1, 0, 0], v0=[0, 0, 0], q0=[np.sqrt(0.5), 0, 0, np.sqrt(0.5)],
        slosh=SloshMode([1, 0, 0], 1, 1, displacement0=1), bus_mass=1,
    )

    np.testing.assert_allclose(result.trajectory.v[-1], [0, 1e-3, 0], atol=1e-9)


def test_hinge_cubic_stiffness_matches_initial_torque_response():
    kwargs = {
        "times": np.linspace(0, 1e-4, 3), "inertia": np.eye(3), "mu": 0,
        "r0": [1, 0, 0], "v0": [0, 0, 0], "q0": [1, 0, 0, 0],
    }
    nonlinear = propagate_6dof_extended(
        hinge=HingedAppendage([0, 0, 1], 1, angle0=0.5, cubic_stiffness=8), **kwargs)

    np.testing.assert_allclose(nonlinear.hinge[-1, 1], -1e-4, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(nonlinear.trajectory.omega[-1, 2], 1e-4, rtol=1e-6, atol=1e-12)


def test_extended_variational_stm_covers_modes_and_covariance():
    mode = HingedAppendage([0.0, 0.0, 1.0], 1.0, stiffness=2.0, angle0=0.1)
    flexible = FlexibleMode([1.0, 0.0, 0.0], 1.0, 3.0, displacement0=0.02)
    slosh = SloshMode([0.0, 1.0, 0.0], 1.0, 2.0, displacement0=0.03)
    kwargs = {
        "times": [0.0, 0.02],
        "inertia": np.eye(3),
        "mu": 0.0,
        "r0": [1.0, 0.0, 0.0],
        "v0": [0.0, 0.0, 0.0],
        "q0": [1.0, 0.0, 0.0, 0.0],
        "mass0": 10.0,
        "hinge": mode,
        "flexible": flexible,
        "slosh": slosh,
        "bus_mass": 10.0,
        "rtol": 1e-10,
        "atol": 1e-12,
        "jacobian_step": 1e-6,
    }
    result = propagate_6dof_extended_variational(**kwargs)

    assert ssatk.propagate_6dof_extended_variational is propagate_6dof_extended_variational
    assert result.stm.shape == (2, 20, 20)
    np.testing.assert_allclose(result.stm[0], np.eye(20), atol=1e-12)
    np.testing.assert_allclose(attitude_error_stm(result)[0], np.eye(19), atol=1e-12)

    perturbation = 1e-6
    plus = propagate_6dof_extended(**{**kwargs, "hinge": replace(mode, angle0=mode.angle0 + perturbation)})
    minus = propagate_6dof_extended(**{**kwargs, "hinge": replace(mode, angle0=mode.angle0 - perturbation)})
    plus_state = np.concatenate((plus.trajectory.r[-1], plus.trajectory.v[-1], plus.trajectory.q[-1], plus.trajectory.omega[-1], [plus.trajectory.mass[-1]], plus.hinge[-1], plus.flexible[-1], plus.slosh[-1]))
    minus_state = np.concatenate((minus.trajectory.r[-1], minus.trajectory.v[-1], minus.trajectory.q[-1], minus.trajectory.omega[-1], [minus.trajectory.mass[-1]], minus.hinge[-1], minus.flexible[-1], minus.slosh[-1]))
    finite_difference = (plus_state - minus_state) / (2.0 * perturbation)
    np.testing.assert_allclose(result.stm[-1, :, 14], finite_difference, rtol=2e-4, atol=1e-10)

    covariance = propagate_6dof_covariance(result, np.eye(20))
    assert covariance.shape == (2, 20, 20)
    np.testing.assert_allclose(covariance, np.swapaxes(covariance, -1, -2), atol=1e-12)
