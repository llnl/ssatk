"""
Demo + pytest benchmarks for `lambertian_sphere_brightness`
(the model now living in ssatk.compute).

Two modes
---------
* pytest:   `pytest demo_lambertian_brightness.py`
            Numeric benchmarks against known reference quantities.
            NO graphics are produced.
* demo:     `python demo_lambertian_brightness.py`
            Same benchmarks AND benchmark figures, saved via
            `ssatk.plots.figsave` into figures/.

The model is imported from ssatk.compute when available, falling
back to the standalone lambertian_sphere_brightness.py next to this file.

Benchmark references (all independent of the model code)
---------------------------------------------------------
* Analytic Lambertian-sphere photometry (McCue et al. 1971; the same
  closed form the retired ssatk calc_M_v sphere term used),
  re-implemented inline in this file:
      F/F_sun = A * (R/d)^2 * (2/3pi) [sin a + (pi - a) cos a]
      m_V     = M_sun + 5 log10(r_sun_sat / 10 pc) - 2.5 log10(F/F_sun)
* Kasten & Young (1989) airmass reference values.
* Measured in-band solar irradiance (500-600 nm, ~165 W/m^2).
* Blackbody band-fraction tables (290 K in 8-14 um ~ 37%).
* Stefan-Boltzmann closed form for the thermal term.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import ssapy


# ----------------------------------------------------------------------
# Import the model: prefer ssatk.compute, fall back to local file
# ----------------------------------------------------------------------
def _import_model():
    """Locate `lambertian_sphere_brightness` and return its *defining*
    module (via inspect.getmodule), so private helpers and constants are
    available even when only the public functions are re-exported from
    ssatk.compute's __init__."""
    import inspect
    for name in (
        "ssatk.compute.lambertian_sphere_brightness",
        "ssatk.compute",
        "lambertian_sphere_brightness",
    ):
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        fn = getattr(mod, "lambertian_sphere_brightness", None)
        if fn is not None:
            defining = inspect.getmodule(fn)
            return defining if defining is not None else mod
    raise ImportError(
        "lambertian_sphere_brightness not found in ssatk.compute "
        "or as a local module."
    )


MODEL = _import_model()
lambertian_sphere_brightness = MODEL.lambertian_sphere_brightness
lambertian_reflection        = MODEL.lambertian_reflection
thermal_emission             = MODEL.thermal_emission
lambert_sphere_phase         = MODEL.lambert_sphere_phase
airmass_kasten_young         = MODEL.airmass_kasten_young
sun_visibility_factor        = MODEL.sun_visibility_factor
_planck_band_fraction        = MODEL._planck_band_fraction
R_EARTH      = MODEL.R_EARTH
SOLAR_CONST  = MODEL.SOLAR_CONST
SIGMA_SB     = MODEL.SIGMA_SB
DEFAULT_TIME = MODEL.DEFAULT_TIME

T0 = DEFAULT_TIME
AU_M = 149_597_870_700.0
TEN_PC_M = 3.0856775814913673e17
GEOCENTER = np.array([1e-6, 0.0, 0.0])     # 1 um from origin: "geocentric"
                                           # observer with safe vector math


# ----------------------------------------------------------------------
# Independent analytic reference (known quantity), written from scratch
# ----------------------------------------------------------------------
def _analytic_lambert_sun(r_sat, r_obs, r_sun, radius, albedo,
                          sun_abs_mag=4.80):
    """Closed-form Lambertian-sphere sun reflection (McCue 1971 form).

    Returns (fractional flux F/F_sun_at_object, apparent V magnitude)."""
    d = np.linalg.norm(r_sat - r_obs)
    v1 = (r_sun - r_sat) / np.linalg.norm(r_sun - r_sat)
    v2 = (r_obs - r_sat) / np.linalg.norm(r_obs - r_sat)
    a = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
    frac = (albedo * radius**2 / d**2
            * (2.0 / (3.0 * np.pi))
            * (np.sin(a) + (np.pi - a) * np.cos(a)))
    r_sun_sat = np.linalg.norm(r_sat - r_sun)
    m_sun_at_obj = sun_abs_mag + 5.0 * np.log10(r_sun_sat / TEN_PC_M)
    m = m_sun_at_obj - 2.5 * np.log10(frac) if frac > 0 else np.inf
    return frac, m


# ----------------------------------------------------------------------
# Shared benchmark geometry: sweep the phase angle at fixed range
# ----------------------------------------------------------------------
def _phase_sweep(n=61, d=1000e3 + R_EARTH, time=T0):
    """Object positions sweeping Sun-object-observer phase angle, observer
    at the geocenter.  Returns (positions list, r_sun)."""
    from ssapy.utils import sunPos
    r_sun = np.asarray(sunPos(time), dtype=float).ravel()
    s = r_sun / np.linalg.norm(r_sun)
    p = np.cross(s, [0.0, 0.0, 1.0])
    p /= np.linalg.norm(p)
    thetas = np.linspace(0.0, np.pi, n)
    return [d * (np.cos(th) * s + np.sin(th) * p) for th in thetas], r_sun


def _ours_vs_analytic(n=61):
    """Run the model and the analytic reference over the sweep."""
    pos, r_sun = _phase_sweep(n)
    out = {
        "ours_frac": [],
        "ours_mag": [],
        "ref_frac": [],
        "ref_mag": [],
        "phase_deg": [],
        "sun_vis": [],
    }
    for r_sat in pos:
        o = lambertian_reflection(
            r_sat, observer=GEOCENTER, radius_m=1.0, albedo=0.2,
            time=T0, band="V", k_extinction=0.0,
            include_earthshine=False, include_moonshine=False,
        )
        F_sun_at_obj = SOLAR_CONST * (
            AU_M / np.linalg.norm(r_sun - r_sat)) ** 2
        out["ours_frac"].append(
            o["irradiance_bolometric_W_m2"]["sun"] / F_sun_at_obj)
        out["ours_mag"].append(o["ab_mag_exoatmospheric"])
        out["phase_deg"].append(o["angles_deg"]["phase_sun_obj_obs_deg"])
        out["sun_vis"].append(o["angles_deg"]["sun_visibility"])

        frac, mag = _analytic_lambert_sun(
            r_sat, GEOCENTER, r_sun, radius=1.0, albedo=0.2)
        out["ref_frac"].append(frac)
        out["ref_mag"].append(mag)
    return {k: np.asarray(v) for k, v in out.items()}


# ======================================================================
# pytest benchmarks (no graphics)
# ======================================================================
def test_phase_function_endpoints():
    assert np.isclose(lambert_sphere_phase(0.0), 2.0 / 3.0, rtol=1e-12)
    assert np.isclose(lambert_sphere_phase(np.pi), 0.0, atol=1e-12)


def test_sun_term_matches_analytic_lambertian():
    """Model fractional sun flux == independent closed form, to machine
    precision, wherever the object is sunlit."""
    b = _ours_vs_analytic()
    lit = b["sun_vis"] >= 1.0
    assert lit.sum() > 30
    np.testing.assert_allclose(
        b["ours_frac"][lit], b["ref_frac"][lit], rtol=1e-10, atol=1e-25)


def test_magnitude_offset_is_small_and_constant():
    """AB (blackbody-sun) vs analytic Vega-ish V (M_sun = 4.80): constant
    zero-point offset, |offset| < 0.15 mag, scatter < 0.02 mag."""
    b = _ours_vs_analytic()
    lit = (b["sun_vis"] >= 1.0) & (b["phase_deg"] < 179.0)
    dmag = b["ours_mag"][lit] - b["ref_mag"][lit]
    assert np.std(dmag) < 0.02
    assert abs(np.mean(dmag)) < 0.15


def test_earth_shadow_zeroes_sun_component():
    """Anti-sun LEO point lies in the umbra: direct-sun flux must vanish."""
    pos, r_sun = _phase_sweep(n=3)
    r_sat = pos[-1]                          # anti-sun direction
    assert sun_visibility_factor(r_sat, r_sun) == 0.0
    out = lambertian_reflection(
        r_sat, observer=GEOCENTER, k_extinction=0.0,
        include_earthshine=False, include_moonshine=False, time=T0)
    assert out["irradiance_bolometric_W_m2"]["sun"] == 0.0


def test_airmass_kasten_young_reference():
    """KY1989 reference: X(0)=1, X(60 deg)~1.994, X(90 deg)~38."""
    assert np.isclose(airmass_kasten_young(0.0), 1.0, atol=2e-3)
    assert np.isclose(airmass_kasten_young(60.0), 1.994, atol=0.01)
    assert 35.0 < airmass_kasten_young(90.0) < 42.0


def test_solar_inband_flux_500_600nm():
    """Blackbody-sun model vs measured ~165 W/m^2 in 500-600 nm."""
    F = SOLAR_CONST * _planck_band_fraction(5772.0, 500e-9, 600e-9)
    assert abs(F - 165.0) / 165.0 < 0.10


def test_thermal_band_fraction_290K_LWIR():
    """~37% of 290 K blackbody power lies in 8-14 um (radiation tables)."""
    f = _planck_band_fraction(290.0, 8e-6, 14e-6)
    assert 0.33 < f < 0.41


def test_thermal_matches_stefan_boltzmann():
    """Bolometric thermal term == eps * sigma * T^4 * (R/d)^2 exactly."""
    pos, _ = _phase_sweep(n=2)
    r_sat = pos[0]
    T, R, A = 290.0, 1.0, 0.3
    e = thermal_emission(r_sat, observer=GEOCENTER, temperature_K=T,
                         radius_m=R, albedo=A, time=T0, k_extinction=0.0)
    d = e["range_m"]
    expected = (1 - A) * SIGMA_SB * T**4 * R**2 / d**2
    assert np.isclose(e["irradiance_bolometric_W_m2"]["thermal"],
                      expected, rtol=1e-12)


def test_reflection_plus_emission_equals_combined():
    pos, _ = _phase_sweep(n=5)
    for r_sat in pos:
        c = lambertian_sphere_brightness(
            r_sat,
            observer=GEOCENTER,
            temperature_K=290.0,
            radius_m=1.0,
            albedo=0.3,
            time=T0,
            band="SWIR",
            k_extinction=0.0,
        )
        r = lambertian_reflection(
            r_sat,
            observer=GEOCENTER,
            radius_m=1.0,
            albedo=0.3,
            time=T0,
            band="SWIR",
            k_extinction=0.0,
        )
        e = thermal_emission(
            r_sat,
            observer=GEOCENTER,
            temperature_K=290.0,
            radius_m=1.0,
            albedo=0.3,
            time=T0,
            band="SWIR",
            k_extinction=0.0,
        )
        assert np.isclose(
            r["irradiance_inband_total_W_m2"] + e["irradiance_inband_total_W_m2"],
            c["irradiance_inband_total_W_m2"], rtol=1e-12)


# ======================================================================
# Demo mode: figures (only when run as a script, never under pytest)
# ======================================================================
def _demo_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ssatk.plots import figsave

    os.makedirs("tests", exist_ok=True)
    b = _ours_vs_analytic(n=181)
    lit = b["sun_vis"] >= 1.0
    shadow = ~lit

    # ---- Figure 1: phase-curve benchmark vs analytic reference --------
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(b["phase_deg"], b["ref_mag"], "k-", lw=3, alpha=0.35,
             label="analytic Lambertian sphere (McCue 1971, M_sun=4.80)")
    ax1.plot(b["phase_deg"][lit], b["ours_mag"][lit], "C0.", ms=4,
             label="ssatk lambertian_sphere_brightness (sunlit)")
    if shadow.any():
        ax1.axvspan(b["phase_deg"][shadow].min(),
                    b["phase_deg"][shadow].max(), color="0.85",
                    label="Earth shadow (model only)")
    ax1.invert_yaxis()
    ax1.set_ylabel("apparent magnitude")
    ax1.set_title("Sun-reflection phase curve: 1 m sphere, albedo 0.2,\n"
                  "geocentric observer, LEO range")
    ax1.legend(loc="lower left", fontsize=9)
    good = lit & (b["phase_deg"] < 179.0)
    dmag = b["ours_mag"] - b["ref_mag"]
    ax2.plot(b["phase_deg"][good], dmag[good], "C3.", ms=4)
    ax2.axhline(np.mean(dmag[good]), color="C3", ls="--", lw=1,
                label=f"zero-point offset = {np.mean(dmag[good]):+.3f} mag "
                      f"(AB/blackbody-sun vs Vega V)")
    ax2.set_xlabel("Sun-object-observer phase angle [deg]")
    ax2.set_ylabel("model - analytic [mag]")
    ax2.legend(fontsize=9)
    fig.tight_layout()
    figsave(fig, "demo_phase_curve_benchmark.jpg")
    plt.close(fig)

    # ---- Figure 2: fractional-flux agreement (machine precision) ------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ratio = np.where(b["ref_frac"] > 0,
                     b["ours_frac"] / np.maximum(b["ref_frac"], 1e-300), np.nan)
    ax.semilogy(b["phase_deg"][good], np.abs(ratio[good] - 1.0) + 1e-17, "C0.")
    ax.set_xlabel("phase angle [deg]")
    ax.set_ylabel("|model / analytic  -  1|")
    ax.set_title("Fractional sun-reflection flux vs closed-form reference\n"
                 "(identical Lambertian-sphere model -> machine precision)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    figsave(fig, "demo_fractional_flux_benchmark.jpg")
    plt.close(fig)

    # ---- Figure 3: airmass benchmark + multi-band magnitudes ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    z = np.linspace(0, 89.9, 300)
    ax1.semilogy(z, airmass_kasten_young(z), "C0-", label="Kasten-Young 1989")
    ax1.semilogy(z[z < 80], 1 / np.cos(np.radians(z[z < 80])), "k--",
                 lw=1, label="sec(z) (plane-parallel)")
    for zr, Xr in [(0, 1.0), (60, 1.994), (85, 10.32)]:
        ax1.plot(zr, Xr, "rs")
    ax1.set_xlabel("zenith angle [deg]"); ax1.set_ylabel("airmass")
    ax1.set_title("Airmass model vs reference values (red squares)")
    ax1.legend(); ax1.grid(alpha=0.3)

    site = ssapy.EarthObserver(lon=-156.26, lat=20.71, elevation=3055.0)
    r_site, _ = site.getRV(T0)
    r_site = np.asarray(r_site, dtype=float).ravel()
    obj = r_site * (np.linalg.norm(r_site) + 800e3) / np.linalg.norm(r_site)
    bands = ["B", "V", "green", "red", "I", "SWIR", "SWIR2", "MWIR", "LWIR"]
    mags = [lambertian_sphere_brightness(
                obj, lon=-156.26, lat=20.71, elevation=3055.0,
                temperature_K=290.0, radius_m=1.0, albedo=0.3,
                time=T0, band=bb, k_extinction=0.0)["ab_mag_exoatmospheric"]
            for bb in bands]
    ax2.bar(bands, mags, color="C1")
    ax2.invert_yaxis()
    ax2.set_ylabel("exoatmospheric AB magnitude")
    ax2.set_title("1 m sphere, 290 K, albedo 0.3, 800 km overhead:\n"
                  "reflection-dominated VIS -> thermal-dominated LWIR")
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    figsave(fig, "demo_airmass_and_bands.jpg")
    plt.close(fig)

    print("Saved via figsave: figures/demo_phase_curve_benchmark.jpg")
    print("Saved via figsave: figures/demo_fractional_flux_benchmark.jpg")
    print("Saved via figsave: figures/demo_airmass_and_bands.jpg")


if __name__ == "__main__":
    print(f"Model imported from: {MODEL.__name__}")
    for fn in [test_phase_function_endpoints,
               test_sun_term_matches_analytic_lambertian,
               test_magnitude_offset_is_small_and_constant,
               test_earth_shadow_zeroes_sun_component,
               test_airmass_kasten_young_reference,
               test_solar_inband_flux_500_600nm,
               test_thermal_band_fraction_290K_LWIR,
               test_thermal_matches_stefan_boltzmann,
               test_reflection_plus_emission_equals_combined]:
        fn()
        print(f"PASS  {fn.__name__}")
    _demo_figures()
