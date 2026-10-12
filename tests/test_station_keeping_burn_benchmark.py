from demos.benchmarks.demo_station_keeping_burn import run
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from ssatk.constants import EARTH_MU


def test_station_keeping_burn_benchmark(tmp_path):
    result = run(output_dir=tmp_path)
    assert result["burn_count"] == 35
    # R2: independent Cartesian integration, with tangential drag and explicit
    # velocity jumps. Evaluate a from orbital energy instead of SSATK elements.
    def reference(controlled, tolerance):
        y = np.array([7e6, 0, 0, 0, np.sqrt(EARTH_MU/7e6), 0])
        def rhs(t, y):
            r, v = y[:3], y[3:]
            return np.r_[v, -EARTH_MU*r/np.linalg.norm(r)**3 - 1e-5*v/np.linalg.norm(v)]
        for index in range(36):
            if controlled and index:
                y[3:] += 1e-3*y[3:]/np.linalg.norm(y[3:])
            arc = solve_ivp(rhs, (0, 100), y, method="Radau", rtol=tolerance, atol=tolerance)
            assert arc.success
            y = arc.y[:, -1]
        return 1/(2/np.linalg.norm(y[:3])-np.dot(y[3:], y[3:])/EARTH_MU) - 7e6

    for controlled, key in ((False, "uncontrolled"), (True, "controlled")):
        expected = reference(controlled, 1e-12)
        assert reference(controlled, 1e-11) == pytest.approx(expected, abs=1e-3)
        assert result[f"{key}_semimajor_axis_change_m"] == pytest.approx(expected, abs=.01)
    assert (tmp_path / "station_keeping_burn_benchmark.json").is_file()
