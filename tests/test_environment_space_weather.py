
import pytest
from astropy.time import Time

from ssatk.environment import SpaceEnvironment


def test_nrlmsise00_density_uses_packaged_eop_and_space_weather():
    pytest.importorskip("pymsis")
    environment = SpaceEnvironment(atmosphere_density_model="nrlmsise00")
    density = environment.density(
        400_000.0,
        Time("2024-01-15T12:00:00", scale="utc").gps,
        [6_778_137.0, 0.0, 0.0],
        [0.0, 7_668.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        None,
    )

    assert density == pytest.approx(6.060943413821462e-12, rel=2.0e-10)
