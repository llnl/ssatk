"""SSAPy ephemerides, body orientation and SSATK frames for eclipse states.

Every function takes UTC Julian Dates and returns kilometres. SSAPy bodies
take GPS seconds and return metres; the conversion happens here once.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np


class BackendUnavailableError(RuntimeError):
    """Raised when a requested state backend cannot be used."""


@dataclass(frozen=True)
class BackendSelection:
    requested: str
    selected: str
    ephemeris_backend: str
    frame_backend: str = "SSATK GCRF<->ITRF (SSAPy groundTrack)"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_EPHEMERIS = {
    "ssapy": "SSAPy DE430 (ssa-data-lunar)",
    "reference": "NASA/GSFC Besselian elements and published circumstances",
}


def resolve_backend(mode: str = "ssapy") -> BackendSelection:
    """Return the state backend for ``mode`` (``ssapy`` or ``reference``)."""
    selected = str(mode).strip().lower()
    if selected not in _EPHEMERIS:
        raise ValueError("backend must be 'ssapy' or 'reference'")
    return BackendSelection(selected, selected, _EPHEMERIS[selected])


def jd_utc_to_gps_seconds(jd_utc) -> np.ndarray:
    """UTC Julian Dates to GPS seconds, including leap seconds (Astropy)."""
    from astropy.time import Time

    return np.asarray(Time(np.asarray(jd_utc, dtype=float), format="jd", scale="utc").gps, dtype=float)


@lru_cache(maxsize=None)
def _body(name: str):
    # One kernel-backed body per process: animations evaluate many frames.
    from ssapy import get_body

    return get_body(name)


def _rows(value, count: int) -> np.ndarray:
    arr = np.squeeze(np.asarray(value, dtype=float))
    if arr.shape == (3, count) and count != 3:
        return arr.T
    return arr.reshape(count, 3)


_LIGHT_SPEED_KM_S = 299_792.458


def _ssapy_position_km(name: str, gps: np.ndarray) -> np.ndarray:
    return _rows(_body(name).position(gps), len(gps)) / 1_000.0


def ssapy_ephemeris_positions_km(jd_utc) -> tuple[np.ndarray, np.ndarray]:
    """Return light-time-corrected geocentric GCRF ``(moon_km, sun_km)``, each ``(N, 3)``.

    An eclipse happens where the light is, so each body is evaluated when the
    light reaching Earth left it. Geometric Sun positions lead the apparent Sun
    by about 20 arcsec (499 s of light time), which makes predicted contacts
    roughly 40 s late. NASA/GSFC Besselian elements use apparent positions.
    """
    jd = np.asarray(jd_utc, dtype=float).reshape(-1)
    gps = jd_utc_to_gps_seconds(jd)
    result = []
    for name in ("moon", "sun"):
        position = _ssapy_position_km(name, gps)
        for _ in range(2):
            position = _ssapy_position_km(name, gps - np.linalg.norm(position, axis=1) / _LIGHT_SPEED_KM_S)
        result.append(position)
    moon, sun = result
    if not (np.all(np.isfinite(moon)) and np.all(np.isfinite(sun))):
        raise BackendUnavailableError("SSAPy returned non-finite Sun or Moon positions")
    return moon, sun


def ssapy_body_to_gcrf_matrices(jd_utc, body: str = "moon") -> np.ndarray:
    """Body-fixed-to-GCRF rotation(s) for the Earth or the Moon.

    SSAPy's ``Body.orientation`` returns GCRF-to-body; the Moon uses the DE440
    binary PCK. The renderers need the transpose.
    """
    key = str(body).strip().lower()
    if key not in {"earth", "moon"}:
        raise ValueError("body must be 'earth' or 'moon'")
    jd = np.asarray(jd_utc, dtype=float)
    gps = jd_utc_to_gps_seconds(jd.reshape(-1))
    orientation = _body(key).orientation
    matrices = np.stack([np.asarray(orientation(float(t)), dtype=float).reshape(3, 3).T for t in gps])
    return matrices[0] if jd.ndim == 0 else matrices.reshape(jd.shape + (3, 3))


@lru_cache(maxsize=8192)
def _itrf_to_gcrf_matrix(jd_utc: float) -> np.ndarray:
    # GCRF and ITRF share an origin, so the SSATK transform is a rotation.
    # Transforming the basis once per epoch lets renderers convert thousands
    # of points and directions without a groundTrack call for each.
    from ssatk.coordinates import itrf_to_gcrf

    basis_m = np.eye(3) * 1.0e7
    gps = jd_utc_to_gps_seconds(np.full(3, jd_utc))
    return (np.asarray(itrf_to_gcrf(basis_m, gps), dtype=float) / 1.0e7).T


def _rotate(points_km, jd_utc, *, inverse: bool) -> np.ndarray:
    points = np.asarray(points_km, dtype=float)
    values = points.reshape(-1, 3)
    jd = np.broadcast_to(np.asarray(jd_utc, dtype=float).reshape(-1), (len(values),))
    result = np.empty_like(values)
    for value in np.unique(jd):
        rows = jd == value
        matrix = _itrf_to_gcrf_matrix(float(value))
        result[rows] = values[rows] @ (matrix if inverse else matrix.T)
    return result[0] if points.ndim == 1 else result


def gcrf_to_itrf_km(points_gcrf_km, jd_utc) -> np.ndarray:
    return _rotate(points_gcrf_km, jd_utc, inverse=True)


def itrf_to_gcrf_km(points_itrf_km, jd_utc) -> np.ndarray:
    return _rotate(points_itrf_km, jd_utc, inverse=False)


def true_of_date_to_gcrf_matrix(jd_utc: float) -> np.ndarray:
    """Rotation from the true equator and equinox of date to GCRF (IAU 2006/2000A).

    NASA's published apparent right ascensions and declinations are referred
    to the true equator of date; GCRF is fixed near J2000. Since J2000 the two
    have drifted apart by about 0.34 deg.
    """
    import erfa
    from astropy.time import Time

    tt = Time(float(jd_utc), format="jd", scale="utc").tt
    return np.asarray(erfa.pnm06a(tt.jd1, tt.jd2), dtype=float).T
