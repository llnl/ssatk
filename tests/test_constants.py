import numpy as np
import pytest

from ssatk import constants


def test_unit_conversions_match_their_si_and_iau_definitions():
    # R3: IAU 2012 B2 fixes 1 au = 149,597,870,700 m; IAU 2015 B2 fixes
    # 1 pc = (648000 / pi) au; the Julian year is 365.25 d of 86400 s; a radian
    # is 180/pi deg. Exact to 1e-12 relative.
    assert constants.au_to_m == 149_597_870_700
    assert constants.pc_to_au == pytest.approx(648000.0 / np.pi, rel=1e-12)
    assert constants.pc_to_m == pytest.approx(3.0856775814913673e16, rel=1e-12)
    assert constants.rad_to_deg == pytest.approx(57.29577951308232, rel=1e-12)
    assert constants.rad_to_arcsecond == pytest.approx(206264.80624709636, rel=1e-12)
    assert constants.year_to_second == 365.25 * 86400
    assert constants.year_to_day == 365.25
    assert constants.year_to_month == 12
    assert constants.year_to_week == pytest.approx(365.25 / 7, rel=1e-12)


def test_au_per_year_and_rebound_velocity_units():
    # R3: 1 au per Julian year is 4740.4705 m/s. REBOUND with G = 1 in au, Msun
    # uses the Gaussian time unit yr2pi = 1/k day, k = 0.01720209895 (IAU 1976),
    # so its velocity unit is 29784.69 m/s. 1e-9 relative, plus REBOUND's own
    # yr2pi as an independent check.
    rebound = pytest.importorskip("rebound")
    assert constants.aupyr_to_mps == pytest.approx(4740.470463533348, rel=1e-9)
    assert constants.v_rebound_to_si == pytest.approx(149_597_870_700 * 0.01720209895 / 86400, rel=1e-9)
    assert constants.v_rebound_to_si == pytest.approx(
        constants.au_to_m / rebound.units.times_SI["yr2pi"], rel=1e-9
    )


def test_physical_constants_match_codata_2018():
    # R3: CODATA 2018 / 2019 SI: c and k_B exact, G = 6.67430e-11 m^3 kg^-1 s^-2,
    # standard gravity 9.80665 m/s^2 (CGPM 1901). Exact to 1e-12 relative.
    assert constants.c == 299_792_458
    assert constants.kb == pytest.approx(1.380649e-23, rel=1e-12, abs=0)
    assert constants.G == pytest.approx(6.67430e-11, rel=1e-12, abs=0)
    assert constants.STANDARD_GRAVITY == 9.80665


@pytest.mark.parametrize(
    "name, semi_major_axis_au",
    [
        # JPL "Approximate Positions of the Planets" (Standish), Table 1,
        # J2000 mean elements valid 1800-2050.
        ("MERCURY_a", 0.38709927),
        ("VENUS_a", 0.72333566),
        ("EARTH_a", 1.00000261),
        ("MARS_a", 1.52371034),
        ("JUPITER_a", 5.20288700),
        ("SATURN_a", 9.53667594),
        ("URANUS_a", 19.18916464),
        ("NEPTUNE_a", 30.06992276),
    ],
)
def test_planet_semi_major_axes_match_jpl_mean_elements(name, semi_major_axis_au):
    # R3: four-figure values, held to 5e-4 relative (Saturn's rounded 9.5388
    # differs by 2.3e-4; the former Mars value 1.5273 differed by 2.4e-3).
    assert getattr(constants, name) == pytest.approx(semi_major_axis_au, rel=5e-4)
