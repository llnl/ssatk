import numpy as np
import pytest

from demos.launch import demo_packaged_motor_burn


def test_packaged_motor_burn_matches_mass_balance_and_rocket_equation():
    """Analytic mass balance (kg) and Tsiolkovsky speed (m/s), relative error <= 1e-6.

    The adaptive integration resolves the piecewise-linear thrust at half its
    smallest sample interval; transverse speed is zero within 1e-12 m/s.
    These numerical allowances do not describe catalog or flight uncertainty.
    """
    result = demo_packaged_motor_burn.main()
    motor = result["metadata"]["motor"]
    initial_mass = motor["totalWeightG"] / 1000.0
    propellant_mass = motor["propWeightG"] / 1000.0
    dry_mass = initial_mass - propellant_mass
    exhaust_velocity = result["sampled_impulse_n_s"] / propellant_mass
    reference_delta_v = exhaust_velocity * np.log(initial_mass / dry_mass)

    assert result["final_mass_kg"] == pytest.approx(dry_mass, rel=1e-6, abs=0)
    assert result["delta_v_m_s"] == pytest.approx(reference_delta_v, rel=1e-6, abs=0)
    assert np.all(np.diff(result["mass_kg"]) <= 0.0)
    np.testing.assert_allclose(
        np.asarray(result["velocity_m_s"])[:, 1:], 0.0, atol=1e-12, rtol=0,
    )
