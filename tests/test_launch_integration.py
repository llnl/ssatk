"""State-unit tolerances and raw quaternion diagnostics against independent references."""

import numpy as np
import pytest

from ssatk.propagators_6dof import propagate_6dof


@pytest.mark.parametrize("method", ["DOP853", "RK45", "Radau"])
def test_spin_and_step_diagnostics_against_closed_form(method):
    """R1/R4: isotropic free spin/linear motion, 1e-8 m and 1e-9 quaternion.

    The internal mesh is invariant to requested output spacing. Raw norm
    drift stays below 1e-9; the internal step bound is 1/256 s plus roundoff.
    """
    epoch = 1.4e9
    omega = np.array([1., 2., -1.])
    rate = np.linalg.norm(omega)
    trajectories = []
    for grid in ([epoch, epoch+8], epoch+np.linspace(0, 8, 41)):
        trajectory = propagate_6dof(times=grid, r0=[7e6, 0, 0], v0=[1, 2, 3], t0=epoch,
            omega0=omega, inertia=2*np.eye(3), mu=0, dense_output=True, method=method,
            # Binary increments isolate integration error from position summation roundoff.
            rtol=1e-11, atol=[1e-7]*3+[1e-9]*3+[1e-12]*4+[1e-11]*3,
            max_step=1/256, first_step=1/256)
        elapsed = trajectory.t-epoch
        expected_q = np.column_stack((np.cos(rate*elapsed/2),
                                      np.sin(rate*elapsed[:, None]/2)*omega/rate))
        np.testing.assert_allclose(trajectory.q, expected_q, rtol=0, atol=1e-9)
        np.testing.assert_allclose(trajectory.r, [7e6, 0, 0]+elapsed[:, None]*[1, 2, 3], rtol=0, atol=1e-8)
        np.testing.assert_allclose(trajectory.omega, np.broadcast_to(omega, trajectory.omega.shape), rtol=0, atol=1e-12)
        diagnostics, = trajectory.integration_diagnostics
        assert diagnostics["max_endpoint_quaternion_norm_error"] < 1e-9
        assert diagnostics["max_output_quaternion_correction"] < 1e-9
        assert 0 < diagnostics["min_step_s"] <= diagnostics["max_step_s"] <= 1/256+1e-12
        trajectories.append(trajectory)
    first, second = [t.integration_diagnostics[0] for t in trajectories]
    assert first["accepted_steps"] == second["accepted_steps"]
    assert first["max_endpoint_quaternion_norm_error"] == second["max_endpoint_quaternion_norm_error"]


def test_staging_and_pitch_refinement_against_rocket_equation():
    """R1: two-burn velocity within 1e-6 m/s; R4: mass/norm/refinement budgets.

    The study includes delayed sampled control and phase resets, not only
    smooth orbit propagation. Its additional reference assertions retain
    the existing independently derived inventory and actuator checks.
    """
    from demos.launch.demo_launch_integration import run_reference

    study = run_reference()
    exhaust = 300*9.80665
    expected = exhaust*(np.log(85/37)+np.log(15/(15-10000/exhaust)))
    for run in study["cases"][0]["runs"]:
        actual = run["record"]["truth"][-1]["velocity_gcrf_m_s"][-1, 0]
        assert abs(actual-expected) < 1e-6
