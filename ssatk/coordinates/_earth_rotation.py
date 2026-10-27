"""The one GCRS<->ITRS rotation SSATK uses.

Every public Earth-fixed/inertial conversion in :mod:`ssatk.coordinates`
(``gcrf_to_itrf``, ``itrf_to_gcrf``, ``gcrf_to_itrf_eop``, ``itrf_to_gcrf_eop``,
``gcrf_to_itrf_astropy``, ``eci_to_ecf_matrix``, ``llh_to_gcrf``,
``gcrf_to_llh``, and the ``yastropy`` wrappers of the same names) builds its
rotation matrix here instead of maintaining its own chain. SSATK used to carry
two: IAU 1976/1980 precession-nutation with GAST (``erfa.pnm80`` +
``erfa.gst94``, the chain ``ssapy.groundTrack`` also uses) in
``earth_fixed.py``/``frames.py``, and IAU 2006/2000A through Astropy's
GCRS/ITRS transform in ``geodetic.py``/``gcrf_to_itrf_astropy``. The two
disagreed by 0.65-1.34 m at a fixed pad over 2020-2024 -- more than frame bias
alone explains -- and only one of them read SSATK's packaged EOP data. The
IAU 1976/1980 chain is deleted; this module is the IAU 2006/2000A survivor,
fed only by :mod:`ssatk.environment_eop`'s packaged IERS snapshot, so it needs
no network access and no astropy IERS configuration.

Model
-----
CIO-based IAU 2006 precession / IAU 2000A nutation: the CIP coordinates ``X``,
``Y`` and the CIO locator ``s`` come from ``erfa.xys06a``, with the IERS
celestial-pole offsets ``dX``, ``dY`` (from ``finals2000A.all``, when the
record has them) added to ``X``, ``Y`` before forming the celestial-to-
terrestrial matrix with ``erfa.c2txy`` (Earth Rotation Angle at UT1, the TIO
locator ``s'``, and polar motion from UT1-UTC and ``x_p``/``y_p`` in the same
EOP record). This is the same composition ``erfa.c2t06a`` performs, plus the
dX/dY correction that convenience function omits.
"""

from __future__ import annotations

import warnings

import erfa
import numpy as np
from astropy.time import Time

from ..environment_eop import EarthOrientationExtrapolationWarning
from ..time_functions import to_gps

__all__ = ["gcrs_to_itrs_matrix", "gcrs_to_itrs_matrix_and_rate"]

_ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)
_MAS_TO_RAD = _ARCSEC_TO_RAD / 1000.0

# Half-width (s) of the central difference used for dR/dt (see
# gcrs_to_itrs_matrix_and_rate). Measured against a surface point's closed-form
# speed (omega_E * rho; tests/test_earth_rotation_route.py), the achievable
# accuracy is *not* monotonic in the step: precession/nutation/polar motion
# change negligibly over a few seconds, so dR/dt is dominated by the Earth
# Rotation Angle's rate, but erfa.era00's own finite difference is limited by
# float64 precision in the UT1 Julian date -- smaller steps lose precision to
# cancellation (measured ~0.08 m/s surface-speed error at 1 ms), while larger
# steps pick up real O(step^2) curvature (~0.15 m/s at 600 s). 5-10 s is the
# measured sweet spot, with errors of a few 1e-5 m/s at a WGS84 surface point.
_RATE_STEP_S = 5.0


def _gps_array(t) -> np.ndarray:
    gps = to_gps(t)
    return np.atleast_1d(np.asarray(gps, dtype=float))


def _eop_table(eop):
    if eop is not None:
        return eop
    from ..environment_eop import load_packaged_eop

    return load_packaged_eop()


def _records(gps: np.ndarray, eop, allow_predicted: bool, *, extrapolate: bool = False):
    """EOP records for ``gps``, each from ``eop.at``.

    With ``extrapolate=False`` (the default; used by the strict functions),
    an epoch outside ``[eop.start_gps, eop.end_gps]`` raises ``ValueError``
    from ``eop.at`` unchanged. With ``extrapolate=True`` (used by SSATK's
    lenient public functions), such an epoch is clamped to the nearest table
    end instead, holding that record's UT1-UTC, polar motion and
    celestial-pole offsets constant, and a single
    :class:`~ssatk.environment_eop.EarthOrientationExtrapolationWarning`
    reports the actual epoch range that needed it. The (observed vs.
    predicted) ``allow_predicted`` check still applies to the clamped record.
    """

    if not extrapolate:
        return [eop.at(value, allow_predicted=allow_predicted) for value in gps]

    out_of_range = (gps < eop.start_gps) | (gps > eop.end_gps)
    if np.any(out_of_range):
        lo = float(gps[out_of_range].min())
        hi = float(gps[out_of_range].max())
        warnings.warn(
            f"{int(out_of_range.sum())} of {gps.size} epoch(s) (GPS seconds "
            f"{lo:.0f}..{hi:.0f}) fall outside the packaged EOP table "
            f"[{eop.start_gps:.0f}, {eop.end_gps:.0f}] s; holding the nearest "
            "table record's UT1-UTC, polar motion and dX/dY constant.",
            EarthOrientationExtrapolationWarning,
            stacklevel=4,
        )
    clamped = np.clip(gps, eop.start_gps, eop.end_gps)
    return [eop.at(value, allow_predicted=allow_predicted) for value in clamped]


def _cip_and_polar_terms(gps: np.ndarray, records):
    """Return ``(tta, ttb, uta, utb, x, y, xp, yp)`` for ``gps``/``records``.

    ``utb`` (the UT1 Julian date fraction) is the only output that a caller
    should vary to probe nearby epochs: precession-nutation (``x``, ``y``),
    polar motion (``xp``, ``yp``) and TT change negligibly over the few
    seconds :func:`gcrs_to_itrs_matrix_and_rate` probes, so it reuses these
    rather than re-querying the EOP table at the shifted epochs.
    """

    time = Time(gps, format="gps", scale="utc")
    tt_mjd = np.asarray(time.tt.mjd, dtype=float)
    utc_mjd = np.asarray(time.utc.mjd, dtype=float)

    n = len(records)
    tta = np.full(n, 2400000.5)
    ttb = tt_mjd
    uta = np.full(n, 2400000.5)
    utb = np.array([utc_mjd[i] + records[i].ut1_minus_utc_s / 86400.0 for i in range(n)])

    x0, y0, _s = erfa.xys06a(tta, ttb)
    dx = np.array([(record.dx_mas or 0.0) for record in records]) * _MAS_TO_RAD
    dy = np.array([(record.dy_mas or 0.0) for record in records]) * _MAS_TO_RAD
    x = x0 + dx
    y = y0 + dy

    xp = np.array([record.polar_motion_x_arcsec for record in records]) * _ARCSEC_TO_RAD
    yp = np.array([record.polar_motion_y_arcsec for record in records]) * _ARCSEC_TO_RAD

    return tta, ttb, uta, utb, x, y, xp, yp


def _rotation_matrices(gps: np.ndarray, records) -> np.ndarray:
    """Vectorized IAU 2006/2000A GCRS->ITRS matrices for ``gps`` / ``records``."""

    tta, ttb, uta, utb, x, y, xp, yp = _cip_and_polar_terms(gps, records)
    return erfa.c2txy(tta, ttb, uta, utb, x, y, xp, yp)


def gcrs_to_itrs_matrix(
    t, *, eop=None, allow_predicted: bool = False, extrapolate: bool = False
) -> np.ndarray:
    """GCRS->ITRS rotation matrix/matrices at ``t`` (``r_itrs = R @ r_gcrs``).

    Parameters
    ----------
    t : float, array-like, or astropy.time.Time
        GPS seconds (or Time), one epoch or one per output matrix.
    eop : ssatk.environment_eop.EarthOrientationTable, optional
        Defaults to :func:`ssatk.environment_eop.load_packaged_eop`.
    allow_predicted : bool, optional
        Forwarded to ``EarthOrientationTable.at``; default ``False`` rejects
        an epoch that falls on a predicted (not observed) EOP record.
    extrapolate : bool, optional
        Default ``False``: an epoch outside the EOP table's span raises
        ``ValueError`` (see below). ``True`` instead holds the nearest table
        record's values constant and warns once with
        :class:`~ssatk.environment_eop.EarthOrientationExtrapolationWarning`;
        SSATK's lenient public functions (``gcrf_to_itrf``, ``itrf_to_gcrf``,
        ``gcrf_to_itrf_astropy``, ``eci_to_ecf_matrix``, and the
        ``geodetic``/``yastropy`` conversions) pass ``True`` here.

    Returns
    -------
    numpy.ndarray
        ``(3, 3)`` for a scalar ``t``, ``(N, 3, 3)`` for an array.

    Raises
    ------
    ValueError
        If ``t`` lies outside the EOP table's span and ``extrapolate`` is
        ``False``, or (unless ``allow_predicted``) intersects a predicted
        record. Earth orientation is never silently assumed (e.g.
        UT1-UTC = 0) outside the table.
    """

    t_arr = np.asarray(to_gps(t), dtype=float)
    scalar = t_arr.ndim == 0
    gps = np.atleast_1d(t_arr)
    eop_table = _eop_table(eop)
    records = _records(gps, eop_table, allow_predicted, extrapolate=extrapolate)
    matrices = _rotation_matrices(gps, records)
    return matrices[0] if scalar else matrices


def gcrs_to_itrs_matrix_and_rate(
    t,
    *,
    eop=None,
    allow_predicted: bool = False,
    extrapolate: bool = False,
    step_s: float = _RATE_STEP_S,
):
    """Return ``(R, dR/dt)`` at ``t``.

    ``dR/dt`` is a central difference restricted to the Earth Rotation Angle
    term: precession-nutation, polar motion and TT are evaluated once at the
    requested epoch (via the EOP table, so this still rejects -- or, with
    ``extrapolate=True``, extrapolates -- an epoch the table can't support)
    and held fixed, while the UT1 Julian date that feeds ``erfa.era00`` inside
    the same ``erfa.c2txy`` composition is shifted by ``+-step_s`` to form the
    difference. See the module-level note on ``_RATE_STEP_S`` for the
    measured accuracy this buys over differencing the full table lookup (and
    the epoch-independent float64 floor either way), and
    :func:`gcrs_to_itrs_matrix` for ``extrapolate``. Use the result as
    ``v_itrs = R @ v_gcrs + (dR/dt) @ r_gcrs``.
    """

    t_arr = np.asarray(to_gps(t), dtype=float)
    scalar = t_arr.ndim == 0
    gps = np.atleast_1d(t_arr)
    eop_table = _eop_table(eop)
    records = _records(gps, eop_table, allow_predicted, extrapolate=extrapolate)

    tta, ttb, uta, utb, x, y, xp, yp = _cip_and_polar_terms(gps, records)
    step_days = step_s / 86400.0

    r = erfa.c2txy(tta, ttb, uta, utb, x, y, xp, yp)
    r_ahead = erfa.c2txy(tta, ttb, uta, utb + step_days, x, y, xp, yp)
    r_behind = erfa.c2txy(tta, ttb, uta, utb - step_days, x, y, xp, yp)
    rate = (r_ahead - r_behind) / (2.0 * step_s)

    if scalar:
        return r[0], rate[0]
    return r, rate
