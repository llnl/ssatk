"""Lambertian photometry against blackbody integrals and closed-form reflection."""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import GCRS, get_sun
from astropy.time import Time
from scipy.integrate import quad

import ssatk.compute.lambertian_magnitude as photometry


def _planck(lam, temperature):
    h, c, k = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    return 2 * h * c**2 / lam**5 / np.expm1(h * c / (lam * k * temperature))


def test_blackbody_band_fractions():
    # R1: integrating B_lambda over 0.01 um - 1 mm recovers sigma T^4 / pi
    # (fraction 1 to 1e-4). R2: in-band fractions match scipy quad of the
    # Planck function for V, J and LWIR to 1e-6 relative.
    assert photometry._planck_band_fraction(5772.0, 1e-8, 1e-3) == pytest.approx(1.0, abs=1e-4)
    sigma = 5.670374419e-8
    for band, temperature in (("V", 5772.0), ("J", 5772.0), ("LWIR", 300.0)):
        lo, hi = photometry.BANDS[band]
        expected = quad(_planck, lo, hi, args=(temperature,), epsrel=1e-10)[0] / (sigma * temperature**4 / np.pi)
        assert photometry._planck_band_fraction(temperature, lo, hi) == pytest.approx(expected, rel=1e-6)


def test_sun_v_magnitude_from_the_solar_model():
    # R3: the Sun's apparent V magnitude is -26.76 (Willmer 2018, ApJS 236, 47).
    # A 5772 K blackbody scaled to 1361 W/m^2 in the top-hat V band (500-600 nm)
    # gives -26.70 AB; within 0.1 mag.
    lo, hi = photometry.BANDS["V"]
    flux = photometry.SOLAR_CONST * photometry._planck_band_fraction(photometry.T_SUN, lo, hi)
    assert photometry._ab_mag(flux, lo, hi) == pytest.approx(-26.76, abs=0.1)


def test_direct_sunlight_reflection_matches_the_lambert_sphere_formula():
    # R1: F = S (1 au / d_sun)^2 A (R/d)^2 p(alpha) times the band fraction, with
    # p(alpha) = (2 / 3 pi)(sin a + (pi - a) cos a). Observer a raw GCRS vector
    # 1000 km from the object; Earthshine and moonshine off. 1e-6 relative
    # (the module and the test take the Sun from astropy get_sun).
    time = Time("2026-10-08T00:00:00", scale="utc")
    r_obj = np.array([0.0, 42164e3, 0.0])
    r_obs = r_obj + np.array([1000e3, 0.0, 0.0])
    out = photometry.lambertian_reflection(
        r_obj, r_obs, radius_m=2.0, albedo=0.25, time=time, band="V",
        include_earthshine=False, include_moonshine=False,
    )
    alpha = np.radians(out["angles_deg"]["phase_sun_obj_obs_deg"])
    visibility = out["angles_deg"]["sun_visibility"]
    phase = (2.0 / (3.0 * np.pi)) * (np.sin(alpha) + (np.pi - alpha) * np.cos(alpha))
    lo, hi = photometry.BANDS["V"]
    sun_vec = get_sun(time).transform_to(GCRS(obstime=time)).cartesian.xyz.to_value(u.m)
    d_sun = np.linalg.norm(sun_vec - r_obj)
    expected = (visibility * photometry.SOLAR_CONST * (photometry.AU_M / d_sun) ** 2 * 0.25
                * (2.0 / 1000e3) ** 2 * phase * photometry._planck_band_fraction(photometry.T_SUN, lo, hi))
    assert out["irradiance_inband_W_m2"]["sun"] == pytest.approx(expected, rel=1e-6)
    assert visibility == 1.0
