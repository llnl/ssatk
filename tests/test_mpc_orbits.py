import numpy as np
import pytest
from astropy import units as u

from ssatk.io.mpc import (
    MPC_GM_SUN,
    mpc_epoch_to_time,
    mpc_heliocentric_states,
    parse_mpcorb,
    select_mpc_object,
    unpack_mpc_date,
    unpack_mpc_designation,
)

# (433) Eros as published in the MPC NEA.txt export on 2026-11-05.
EROS_MPCORB = (
    "00433   10.39  0.15 K2669  62.51145  178.91814  304.26797   10.82855  0.2228780  0.55970463"
    "   1.4582437  0 E2026-TG8 18065  59 1893-2026 0.61 M-v 3Ek MPCORBFIT  1804    (433) Eros"
    "               20260919"
)

# JPL Horizons, 433 Eros (solution JPL#659), heliocentric (500@10), ecliptic
# and mean equinox of J2000.0, at JD 2461200.5 TDB, units km and km/s.
HORIZONS_EROS_GM_KM3_S2 = 1.3271244004127939e11
HORIZONS_EROS_ELEMENTS = {
    "semimajor_axis_au": 2.181501549919854e08 / 149597870.7,
    "eccentricity": 2.228779626572815e-01,
    "inclination_deg": 1.082854410400055e01,
    "ascending_node_deg": 3.042679713340772e02,
    "arg_perihelion_deg": 1.789181319120550e02,
    "mean_anomaly_deg": 6.251145486133948e01,
    "epoch_jd_tt": 2461200.5,
}
HORIZONS_EROS_ECLIPTIC_KM = np.array(
    [
        [-1.733830319358056e08, -1.032308946749103e08, -3.852532749381296e07],
        [8.353170444746741e00, -2.474053445733561e01, -1.344165154492732e00],
    ]
)
HORIZONS_EROS_ICRF_KM = np.array(
    [
        [-1.733830319358056e08, -7.938799891374873e07, -7.640918859900680e07],
        [8.353170444746741e00, -2.216431837835302e01, -1.107446685038000e01],
    ]
)


def test_packed_forms_match_mpc_published_examples():
    # Examples from https://minorplanetcenter.net/iau/info/PackedDes.html and
    # https://minorplanetcenter.net/iau/info/PackedDates.html.
    designations = {
        "J95X00A": "1995 XA",
        "J95X01L": "1995 XL1",
        "J95F13B": "1995 FB13",
        "J98SA8Q": "1998 SQ108",
        "J98SC7V": "1998 SV127",
        "J98SG2S": "1998 SS162",
        "K99AJ3Z": "2099 AZ193",
        "K08Aa0A": "2008 AA360",
        "K07Tf8A": "2007 TA418",
        "PLS2040": "2040 P-L",
        "T1S3138": "3138 T-1",
        "03202": "3202",
        "A0345": "100345",
        "a0017": "360017",
        "K3289": "203289",
        "~0000": "620000",
        "~000z": "620061",
        "~AZaz": "3140113",
        "~zzzz": "15396335",
    }
    for packed, readable in designations.items():
        assert unpack_mpc_designation(packed) == readable

    dates = {
        "J9611": (1996, 1, 1.0),
        "J961A": (1996, 1, 10.0),
        "J969U": (1996, 9, 30.0),
        "J96A1": (1996, 10, 1.0),
        "K01AM": (2001, 10, 22.0),
    }
    for packed, expected in dates.items():
        assert unpack_mpc_date(packed) == expected
    year, month, day = unpack_mpc_date("K01AM138303")
    assert (year, month) == (2001, 10)
    assert day == pytest.approx(22.138303, abs=1e-12)
    # 2026 June 9.0 TT is JD 2461200.5 (Horizons epoch above).
    assert mpc_epoch_to_time("K2669").tt.jd == pytest.approx(2461200.5, abs=1e-9)


def test_element_conversion_reproduces_horizons_vectors():
    # Same solution on both sides: Horizons osculating elements, with the
    # Horizons solar GM, must reproduce the Horizons state vectors.
    mu = HORIZONS_EROS_GM_KM3_S2 * 1e9
    r, v = mpc_heliocentric_states(HORIZONS_EROS_ELEMENTS, mu=mu)
    np.testing.assert_allclose(r, HORIZONS_EROS_ECLIPTIC_KM[0] * 1e3, rtol=0, atol=1.0)
    np.testing.assert_allclose(v, HORIZONS_EROS_ECLIPTIC_KM[1] * 1e3, rtol=0, atol=1e-6)

    # SSATK's J2000 obliquity constant (23.439291 deg) is rounded from the
    # IAU 1976 value 84381.448 arcsec (23.4392911 deg) that Horizons uses; the
    # 1.1e-7 deg difference moves this 1.4 au position by 214 m (measured).
    r, v = mpc_heliocentric_states(HORIZONS_EROS_ELEMENTS, frame="icrf", mu=mu)
    np.testing.assert_allclose(r, HORIZONS_EROS_ICRF_KM[0] * 1e3, rtol=0, atol=300.0)
    np.testing.assert_allclose(v, HORIZONS_EROS_ICRF_KM[1] * 1e3, rtol=0, atol=1e-4)


def test_mpc_record_agrees_with_independent_jpl_solution():
    orbits = parse_mpcorb([EROS_MPCORB])
    eros = select_mpc_object(orbits, "Eros")
    assert eros["number"] == 433
    assert eros["orbit_type"] == "Amor" and eros["is_neo"] and not eros["is_pha"]
    assert eros["epoch_jd_tt"] == pytest.approx(2461200.5, abs=1e-9)

    # MPC and JPL fit the same observations independently; their Eros states
    # differed by 30.8 km and 2.1 mm/s at this epoch (measured).
    r, v = mpc_heliocentric_states(eros)
    assert np.linalg.norm(r - HORIZONS_EROS_ECLIPTIC_KM[0] * 1e3) < 50e3
    assert np.linalg.norm(v - HORIZONS_EROS_ECLIPTIC_KM[1] * 1e3) < 5e-3

    # Two-body propagation over exactly one Kepler period returns to the
    # epoch state.
    period = 2 * np.pi * np.sqrt((eros["semimajor_axis_au"] * 149597870700.0) ** 3 / MPC_GM_SUN)
    later = mpc_epoch_to_time(eros["epoch_packed"]) + period * u.s
    r_later, v_later = mpc_heliocentric_states(eros, time=later)
    np.testing.assert_allclose(r_later, r, rtol=0, atol=1.0)
    np.testing.assert_allclose(v_later, v, rtol=0, atol=1e-6)
