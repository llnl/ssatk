import csv
from io import StringIO

import numpy as np
import pytest
from scipy.integrate import trapezoid


def test_packaged_motor_burn_matches_mass_balance_and_rocket_equation():
    """R1 (rocket equation) and R4 (propellant mass balance), relative error
    <= 1e-6; R1 (independent impulse) against the catalog, relative error
    <= 1%.

    The impulse used for the rocket-equation reference is integrated here by
    this test, from the raw packaged CSV samples, independently of the
    demo's own ``sampled_impulse_n_s`` -- reusing that value would only check
    the demo against itself. It is also checked against the sidecar
    catalog's ``totImpulseNs``, which the C5 RASP curve used here meets
    within roughly 0.01%; 1% leaves comfortable room for a packaged curve
    that is close to, but not exactly, its catalog impulse.

    The adaptive integration resolves the piecewise-linear thrust at half its
    smallest sample interval; transverse speed is zero within 1e-12 m/s.
    These numerical allowances do not describe catalog or flight uncertainty.
    """
    pytest.importorskip("ssa_data_propulsion", reason="Optional propulsion extra is required for this demo.")
    from ssatk.data import read_data_text

    from demos.launch import demo_packaged_motor_burn

    result = demo_packaged_motor_burn.main()
    assert not result.get("skipped")
    motor = result["metadata"]["motor"]
    initial_mass = motor["totalWeightG"] / 1000.0
    propellant_mass = motor["propWeightG"] / 1000.0
    dry_mass = initial_mass - propellant_mass

    curve_path = (
        f"{demo_packaged_motor_burn.THRUSTCURVE_ORG_PD_THRUST_CURVE_DIR}"
        f"/{demo_packaged_motor_burn.MOTOR_CURVE}.csv"
    )
    rows = list(csv.DictReader(StringIO(read_data_text(curve_path))))
    times_s = np.array([float(row["time_s"]) for row in rows])
    thrusts_n = np.array([float(row["thrust_n"]) for row in rows])
    independent_impulse_n_s = float(trapezoid(thrusts_n, times_s))

    assert independent_impulse_n_s == pytest.approx(motor["totImpulseNs"], rel=0.01, abs=0)

    exhaust_velocity = independent_impulse_n_s / propellant_mass
    reference_delta_v = exhaust_velocity * np.log(initial_mass / dry_mass)

    assert result["final_mass_kg"] == pytest.approx(dry_mass, rel=1e-6, abs=0)
    assert result["delta_v_m_s"] == pytest.approx(reference_delta_v, rel=1e-6, abs=0)
    mass_kg = np.asarray(result["mass_kg"])
    assert np.all(np.diff(mass_kg) <= 0.0)
    assert mass_kg[0] - mass_kg[-1] == pytest.approx(propellant_mass, rel=1e-6, abs=0)
    np.testing.assert_allclose(
        np.asarray(result["velocity_m_s"])[:, 1:], 0.0, atol=1e-12, rtol=0,
    )
