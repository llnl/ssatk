"""Building a plot must not change astropy's process-wide IERS behaviour."""

import pytest
from astropy.time import Time

iers = pytest.importorskip("astropy.utils.iers")
pytest.importorskip("matplotlib")


def test_base_plot_leaves_ut1_available_after_the_iers_b_span():
    # R6 + R2: BasePlot3D's background IERS pre-load used to set
    # conf.auto_download = False and conf.auto_max_age = None for the whole
    # process. On astropy < 8 (all Python 3.10 installs) IERS_Auto then serves
    # IERS-B, so UT1 at any date past its end (2026-08-28 in
    # astropy-iers-data 0.2026.10.5) raised IERSRangeError afterwards: 23
    # tests failed on Python 3.10. Reference: astropy's own UT1-UTC for a date
    # 7 days past the IERS-B span, taken before the plot exists; it must be
    # unchanged to 1 ns afterwards.
    from ssatk.plots.base_plot import BasePlot3D
    from ssatk.plots.orbit_state import OrbitalState

    t = Time(float(iers.IERS_B.open()["MJD"][-1].value) + 7.0, format="mjd", scale="utc")
    try:
        reference_s = float(iers.IERS_Auto.open().ut1_utc(t).to_value("s"))
    except iers.IERSRangeError:
        pytest.skip("installed IERS-A predictions do not reach 7 days past IERS-B")
    config_before = (iers.conf.auto_download, iers.conf.auto_max_age)

    plot = BasePlot3D(OrbitalState())
    plot._iers_thread.join(timeout=60)

    assert (iers.conf.auto_download, iers.conf.auto_max_age) == config_before
    iers.IERS_Auto.iers_table = None  # force the re-open a later conversion would do
    after_s = float(iers.IERS_Auto.open().ut1_utc(t).to_value("s"))
    assert after_s == pytest.approx(reference_s, abs=1e-9)
