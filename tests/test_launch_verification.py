"""Independent references for the launch verification report's numerical case."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from demos.launch.demo_launch_verification import mechanics_reference


def test_mechanics_report_against_constant_acceleration_and_rotation_exponential():
    """R1/R2: quadratic motion within 1 um / 1 nm/s and SciPy rotation within 10 nrad."""
    record = mechanics_reference()
    inputs = record["inputs"]
    times = inputs["time_s"][:, None]
    expected_r = inputs["r0_m"] + times*(inputs["v0_m_s"] + times*inputs["acceleration_m_s2"]/2)
    expected_v = inputs["v0_m_s"] + times*inputs["acceleration_m_s2"]
    reference = Rotation.from_rotvec(times*inputs["omega0_rad_s"])
    for run in record["runs"]:
        np.testing.assert_allclose(run["r_m"], expected_r, atol=1e-6, rtol=0)
        np.testing.assert_allclose(run["v_m_s"], expected_v, atol=1e-9, rtol=0)
        rotations = Rotation.from_quat(run["q"][:, [1, 2, 3, 0]])
        np.testing.assert_allclose((reference.inv()*rotations).magnitude(), 0., atol=1e-8, rtol=0)
    assert all(metric["status"] == "PASS" for metric in record["metrics"])


def test_finite_burn_against_gmat_chemical_thruster(tmp_path):
    """R1/R2: GMAT within 2 cm / 20 um/s / 10 ug and cutoff within 1 us."""
    from demos.benchmarks.demo_gmat_benchmark import run_finite_burn

    result = run_finite_burn(output_dir=tmp_path)
    if result["skipped"]:
        pytest.skip(result["reason"])
    assert result["max_position_error_m"] <= 2e-2
    assert result["max_velocity_error_m_s"] <= 2e-5
    assert result["max_mass_error_kg"] <= 1e-8
    assert result["rocket_equation_mass_error_kg"] <= 1e-8
    assert result["event_time_error_s"] <= 1e-6
