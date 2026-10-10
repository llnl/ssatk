"""Optional Orekit 10.3.1 external parity for EIGEN-6S's fixed-axis J2 term."""

import numpy as np
import pytest

from demos.benchmarks.demo_orekit_benchmark import _eigen_6s_file, _run_orekit_j2
from ssatk.accelerations_6dof import SpacecraftAccelJ2
from ssatk.propagators_6dof import propagate_6dof_high_accuracy


def test_orekit_eigen_6s_degree_2_order_0_matches_ssatk_j2():
    """Orekit 10.3.1 Holmes-Featherstone / EIGEN-6S, GCRF z-axis, 63° LEO, 5400 s."""
    coefficient_file = _eigen_6s_file()
    if coefficient_file is None:
        pytest.skip("EIGEN-6S is unavailable; set OREKIT_EIGEN_6S or OREKIT_DATA_DIR")
    result = _run_orekit_j2(coefficient_file=coefficient_file, duration=5_400.0, step=60.0, allow_install=False)
    if result is None:
        pytest.skip("Orekit 10.3.1, Java, Maven, or EIGEN-6S is unavailable")
    orekit, constants = result
    mu, radius, c20 = constants["mu"], constants["ae"], constants["c20"]
    j2 = -np.sqrt(5.0) * c20
    ssatk = propagate_6dof_high_accuracy(
        r0=orekit[0, 1:4],
        v0=orekit[0, 4:7],
        times=orekit[:, 0],
        inertia=np.eye(3),
        mu=mu,
        acceleration=SpacecraftAccelJ2(mu=mu, radius=radius, j2=j2),
        max_step=60.0,
    )
    position_error = np.linalg.norm(ssatk.r - orekit[:, 1:4], axis=1)
    velocity_error = np.linalg.norm(ssatk.v - orekit[:, 4:7], axis=1)

    assert mu == pytest.approx(3.986004415e14, rel=0.0, abs=0.1)
    assert radius == pytest.approx(6_378_136.460, rel=0.0, abs=1e-9)
    assert c20 == pytest.approx(-4.84165299820e-4, rel=0.0, abs=1e-16)
    assert np.max(position_error) < 0.02
    assert np.max(velocity_error) < 2e-5
