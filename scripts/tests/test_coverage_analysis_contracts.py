"""Coverage analysis wiring and rendering contracts, separate from core physics."""

import numpy as np
from ssapy import Orbit

from ssatk.plots import coverage_analysis as analysis


def test_analysis_uses_site_sun_elevation_and_spacecraft_phase_without_writing(
    tmp_path, monkeypatch
):
    """R1/C: collinear Sun-satellite-site gives 180° phase and zenith Sun.

    Prescribed positions share one Cartesian frame; propagation and rotation
    are replaced only to isolate the analysis wiring. This is not ephemeris
    validation. Angles allow 1e-6 degrees; duration allows 1e-9 seconds.
    """
    site = analysis.site_ecef(45, 0)
    up = np.array([2**-0.5, 0, 2**-0.5])
    position = site + 500000 * up
    sun = position + 149597870700 * up
    orbit = Orbit.fromKeplerianElements(
        a=7e6, e=0, i=0, pa=0, raan=0, trueAnomaly=0, t=0
    )

    def states(orbit, times, **kwargs):
        return np.tile(position, (len(times), 1)), np.zeros((len(times), 3))

    monkeypatch.setattr(analysis.compute, "rv", states)
    monkeypatch.setattr(
        analysis,
        "propagate",
        lambda *a, **kw: (*states(orbit, [0, 60, 120]), np.array([0.0, 60, 120])),
    )
    monkeypatch.setattr(analysis, "gcrf_to_itrf", lambda r, t: r)
    monkeypatch.setattr(
        analysis, "compute_sun_positions", lambda t: np.tile(sun, (len(t), 1))
    )
    monkeypatch.chdir(tmp_path)
    result = analysis.analyse_satellite(
        "synthetic",
        orbit,
        "normal",
        45,
        0,
        0,
        n_lat=2,
        n_lon=2,
        coverage_days=120 / 86400,
        make_figures=False,
    )
    np.testing.assert_allclose(result["phase_angle"], 180, atol=1e-6, rtol=0)
    np.testing.assert_allclose(result["elevation_sun"], 90, atol=1e-6, rtol=0)
    assert abs(result["vis_min"] * 60 - 120) < 1e-9
    assert not result["observable"].any()
    assert not list(tmp_path.iterdir())
