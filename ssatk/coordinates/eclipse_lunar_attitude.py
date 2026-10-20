"""Physical lunar attitude for eclipse rendering.

Strict LLNL modes use SSAPy's DE440 principal-axis lunar orientation.  The
reference backend uses the NAIF/IAU 2009 text-PCK series with an explicit
UTC-to-TDB time path.  Every public matrix maps Moon body-fixed vectors into
GCRF; attitude rotates texture and body axes but never deforms the solid Moon.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

_POLE_RA = (269.9949, 0.0031, 0.0)
_POLE_DEC = (66.5392, 0.0130, 0.0)
_PM = (38.3213, 13.17635815, -1.4e-12)
_NUT_RA = np.asarray([
    -3.8787, -0.1204, 0.0700, -0.0172, 0.0, 0.0072, 0.0,
    0.0, 0.0, -0.0052, 0.0, 0.0, 0.0043,
], dtype=float)
_NUT_DEC = np.asarray([
    1.5419, 0.0239, -0.0278, 0.0068, 0.0, -0.0029, 0.0009,
    0.0, 0.0, 0.0008, 0.0, 0.0, -0.0009,
], dtype=float)
_NUT_PM = np.asarray([
    3.5610, 0.1208, -0.0642, 0.0158, 0.0252, -0.0066, -0.0047,
    -0.0046, 0.0028, 0.0052, 0.0040, 0.0019, -0.0044,
], dtype=float)
_NUT_ANGLES = np.asarray([
    [125.045, -1935.5364525],
    [250.089, -3871.0729050],
    [260.008, 475263.3328725],
    [176.625, 487269.6299850],
    [357.529, 35999.0509575],
    [311.589, 964468.4993100],
    [134.963, 477198.8693250],
    [276.617, 12006.3007650],
    [34.226, 63863.5132425],
    [15.134, -5806.6093575],
    [119.743, 131.8406400],
    [239.961, 6003.1503825],
    [25.053, 473327.7964200],
], dtype=float)


@dataclass(frozen=True)
class LunarAttitude:
    jd_utc: float
    body_to_gcrf: np.ndarray
    source: str
    time_argument: str
    pole_ra_deg: float | None = None
    pole_dec_deg: float | None = None
    prime_meridian_deg: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "jd_utc": float(self.jd_utc),
            "body_to_gcrf": np.asarray(self.body_to_gcrf, dtype=float).tolist(),
            "source": self.source,
            "time_argument": self.time_argument,
            "pole_ra_deg": self.pole_ra_deg,
            "pole_dec_deg": self.pole_dec_deg,
            "prime_meridian_deg": self.prime_meridian_deg,
        }


def _r1_passive(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])


def _r3_passive(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def _jd_tdb(jd_utc: float) -> tuple[float, str]:
    try:
        from astropy.time import Time
        return float(Time(float(jd_utc), format="jd", scale="utc").tdb.jd), "Astropy UTC-to-TDB"
    except Exception:
        # 69.184 s = TT-UTC for both validated events; periodic TDB-TT is
        # included to millisecond scale.  The fallback is deterministic and is
        # never relabelled as the binary-PCK SSAPy solution.
        tt = float(jd_utc) + 69.184 / 86400.0
        T = (tt - 2451545.0) / 36525.0
        g = math.radians((357.5277233 + 35999.05034 * T) % 360.0)
        tdb_minus_tt_s = 0.001657 * math.sin(g) + 0.000022 * math.sin(2.0 * g)
        return tt + tdb_minus_tt_s / 86400.0, "deterministic TT/TDB approximation"


def _validate_rotation(matrix: np.ndarray) -> None:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("lunar attitude must be a 3x3 matrix")
    orth = float(np.max(np.abs(matrix.T @ matrix - np.eye(3))))
    det = float(np.linalg.det(matrix))
    if not np.all(np.isfinite(matrix)) or orth > 1e-8 or abs(det - 1.0) > 1e-8:
        raise ValueError(f"lunar attitude is not a proper rotation (orth={orth}, det={det})")


def iau_moon_attitude(jd_utc: float) -> LunarAttitude:
    tdb, time_source = _jd_tdb(float(jd_utc))
    d = tdb - 2451545.0
    T = d / 36525.0
    angles = np.radians((_NUT_ANGLES[:, 0] + _NUT_ANGLES[:, 1] * T) % 360.0)
    ra = _POLE_RA[0] + _POLE_RA[1] * T + _POLE_RA[2] * T * T + float(np.dot(_NUT_RA, np.sin(angles)))
    dec = _POLE_DEC[0] + _POLE_DEC[1] * T + _POLE_DEC[2] * T * T + float(np.dot(_NUT_DEC, np.cos(angles)))
    w = _PM[0] + _PM[1] * d + _PM[2] * d * d + float(np.dot(_NUT_PM, np.sin(angles)))
    w %= 360.0
    inertial_to_body = (
        _r3_passive(math.radians(w))
        @ _r1_passive(math.radians(90.0 - dec))
        @ _r3_passive(math.radians(ra + 90.0))
    )
    body_to_gcrf = inertial_to_body.T
    _validate_rotation(body_to_gcrf)
    return LunarAttitude(
        jd_utc=float(jd_utc),
        body_to_gcrf=body_to_gcrf,
        source="NAIF pck00011 / IAU 2009 lunar orientation",
        time_argument=time_source,
        pole_ra_deg=float(ra % 360.0),
        pole_dec_deg=float(dec),
        prime_meridian_deg=float(w),
    )


def moon_attitude_for_event(event, jd_utc: float) -> LunarAttitude:
    selected = str(event.metadata.get("state_source", "reference"))
    if selected == "ssapy":
        from ssatk.compute.eclipse_runtime import ssapy_body_to_gcrf_matrices

        matrix = np.asarray(ssapy_body_to_gcrf_matrices(float(jd_utc), "moon"), dtype=float)
        _validate_rotation(matrix)
        return LunarAttitude(
            jd_utc=float(jd_utc),
            body_to_gcrf=matrix,
            source="SSAPy DE440 Moon principal-axis orientation",
            time_argument="UTC JD -> Astropy GPS seconds -> SSAPy MoonOrientation",
        )
    return iau_moon_attitude(float(jd_utc))


