
import csv
from datetime import timedelta

import numpy as np
import pytest
from astropy.coordinates import EarthLocation
import astropy.units as u
from astropy.time import Time

from ssatk.environment import SpaceEnvironment
from ssatk.coordinates.earth_fixed import gcrf_to_itrf_eop
from ssatk.data import data_path
from ssatk.environment_space_weather import load_packaged_space_weather


@pytest.mark.parametrize("version", [0, 2.1])
def test_nrlmsise00_density_uses_packaged_eop_and_space_weather(version):
    """R2: direct pymsis call with independently parsed weather and WGS84 LLH.

    This checks wrapper inputs/units, not the empirical model's accuracy against
    measured atmospheric density. Earth rotation is separately checked against
    ERFA in test_environment_eop.py; its output is a shared input here.
    """
    msis = pytest.importorskip("pymsis").msis
    environment = SpaceEnvironment(atmosphere_density_model="nrlmsise00", nrlmsise_version=version)
    epoch = Time("2024-01-15T12:00:00", scale="utc")
    density = environment.density(
        400_000.0,
        epoch.gps,
        [6_778_137.0, 0.0, 0.0],
        [0.0, 7_668.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        None,
    )

    with data_path("environment/space_weather/SW-All.csv") as path:
        with open(path, newline="") as stream:
            rows = {row["DATE"]: row for row in csv.DictReader(stream)}
    moment = epoch.to_datetime()
    current = rows[moment.date().isoformat()]
    previous = rows[(moment - timedelta(days=1)).date().isoformat()]
    history = []
    for offset in range(20):
        sample = moment - timedelta(hours=3*offset)
        history.append(float(rows[sample.date().isoformat()][f"AP{sample.hour//3+1}"]))
    aps = [float(current["AP_AVG"]), *history[:4], np.mean(history[4:12]), np.mean(history[12:20])]
    f107, f107a = float(previous["F10.7_ADJ"]), float(current["F10.7_ADJ_CENTER81"])
    actual_weather = load_packaged_space_weather().msis_inputs(epoch)
    np.testing.assert_allclose(actual_weather[:2], [f107, f107a], atol=1e-12, rtol=0)
    np.testing.assert_allclose(actual_weather[2], aps, atol=1e-12, rtol=0)
    xyz = gcrf_to_itrf_eop([[6_778_137.0, 0, 0]], [epoch.gps])[0]
    location = EarthLocation.from_geocentric(*xyz, unit=u.m)
    expected = msis.run([moment], [location.lon.deg], [location.lat.deg],
                        [location.height.to_value(u.km)], [f107], [f107a], [aps], version=version)
    # pymsis returns single precision; allow 2 ppm for geodetic conversion.
    assert density == pytest.approx(float(expected.reshape(-1, 11)[0, 0]), rel=2e-6, abs=0)
