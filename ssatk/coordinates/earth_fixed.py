import erfa
import numpy as np
from ssapy import groundTrack
from ssapy.utils import _gpsToTT, iers_interp

from ..time_functions import to_gps
from .velocity import v_from_r


def gcrf_to_itrf(r_gcrf, t, v=None, *, step_s=1.0):
    """
    Convert GCRF coordinates to ITRF coordinates.

    Parameters
    ----------
    r_gcrf : numpy.ndarray
        GCRF positions in meters, shape (n, 3).
    t : numpy.ndarray
        Times (GPS seconds or astropy Time), one per position.
    v : numpy.ndarray or True, optional
        GCRF velocities in m/s, shape (n, 3), to transform to Earth-fixed
        velocities. ``True`` instead differentiates the ITRF positions
        numerically, which needs a densely sampled trajectory.
    step_s : float, optional
        Half-width in seconds of the central difference used to differentiate
        the rotation when ``v`` is an array. Default 1 s.

    Returns
    -------
    numpy.ndarray or tuple
        ITRF positions in meters, or ``(position, velocity)`` when ``v`` is
        given. The velocity is Earth-relative: ``R v + (dR/dt) r``.
    """
    t = to_gps(t)
    x, y, z = groundTrack(r_gcrf, t, format="cartesian")
    pos = np.array([x, y, z]).T
    if v is None or v is False:
        return pos
    if v is True:
        return pos, v_from_r(pos, t)

    r_gcrf = np.asarray(r_gcrf, dtype=float)
    v_gcrf = np.asarray(v, dtype=float)
    if v_gcrf.shape != r_gcrf.shape:
        raise ValueError("v must have the same shape as r_gcrf.")
    t = np.asarray(t, dtype=float)
    ahead = np.array(groundTrack(r_gcrf + v_gcrf * step_s, t + step_s, format="cartesian")).T
    behind = np.array(groundTrack(r_gcrf - v_gcrf * step_s, t - step_s, format="cartesian")).T
    return pos, (ahead - behind) / (2.0 * step_s)


def gcrf_to_itrf_astropy(state_vectors, t):
    """Convert GCRF positions to geocentric ITRF using Astropy."""
    import astropy.units as u
    from astropy.coordinates import GCRS, ITRS, SkyCoord

    state_vectors = np.asarray(state_vectors, dtype=float)
    if state_vectors.ndim != 2 or state_vectors.shape[1] != 3:
        raise ValueError("state_vectors must have shape (N, 3)")

    sc = SkyCoord(
        x=state_vectors[:, 0] * u.m,
        y=state_vectors[:, 1] * u.m,
        z=state_vectors[:, 2] * u.m,
        representation_type="cartesian",
        frame=GCRS(obstime=t),
    )
    sc_itrs = sc.transform_to(ITRS(obstime=t))

    return np.array([
        sc_itrs.cartesian.x.to_value(u.m),
        sc_itrs.cartesian.y.to_value(u.m),
        sc_itrs.cartesian.z.to_value(u.m),
    ]).T


def gcrf_to_itrf_eop(state_vectors, t, *, eop=None, allow_predicted=False):
    """Convert GCRF positions to ITRF using an explicit EOP data source.

    The rotation follows the SSAPy ``EarthOrientation`` convention: IAU 1976
    precession/nutation, UT1-dependent Greenwich sidereal time, and IERS polar
    motion. ``eop`` may be an ``EarthOrientationTable`` or any callable with
    the same ``at(time, allow_predicted=...)`` interface.
    """

    rotation = _eop_rotation(state_vectors, t, eop=eop, allow_predicted=allow_predicted)
    return np.einsum("nij,nj->ni", rotation, _state_matrix(state_vectors, "state_vectors"))


def itrf_to_gcrf_eop(state_vectors, t, *, eop=None, allow_predicted=False):
    """Convert ITRF positions to GCRF using an explicit EOP data source."""

    rotation = _eop_rotation(state_vectors, t, eop=eop, allow_predicted=allow_predicted)
    return np.einsum("nji,nj->ni", rotation, _state_matrix(state_vectors, "state_vectors"))


def _eop_rotation(state_vectors, t, *, eop, allow_predicted):
    import erfa
    from astropy.time import Time

    state_vectors = _state_matrix(state_vectors, "state_vectors")
    gps = _time_array(t, len(state_vectors))
    if eop is None:
        from ..environment_eop import load_packaged_eop

        eop = load_packaged_eop()
    records = [eop.at(value, allow_predicted=allow_predicted) for value in gps]
    time = Time(gps, format="gps", scale="utc")
    pn = np.asarray([erfa.pnm80(2400000.5, value) for value in time.tt.mjd])
    gst = np.asarray(
        [
            erfa.gst94(2400000.5, utc_mjd + record.ut1_minus_utc_s / 86400.0)
            for utc_mjd, record in zip(time.utc.mjd, records)
        ]
    )
    cosine, sine = np.cos(gst), np.sin(gst)
    gst_matrix = np.zeros((len(records), 3, 3), dtype=float)
    gst_matrix[:, 0, 0] = cosine
    gst_matrix[:, 0, 1] = sine
    gst_matrix[:, 1, 0] = -sine
    gst_matrix[:, 1, 1] = cosine
    gst_matrix[:, 2, 2] = 1.0

    arcsec_to_rad = np.pi / (180.0 * 3600.0)
    polar = np.eye(3, dtype=float)[None, :, :].repeat(len(records), axis=0)
    polar_motion_x = np.asarray([record.polar_motion_x_arcsec for record in records]) * arcsec_to_rad
    polar_motion_y = np.asarray([record.polar_motion_y_arcsec for record in records]) * arcsec_to_rad
    polar[:, 0, 2] = polar_motion_x
    polar[:, 1, 2] = -polar_motion_y
    polar[:, 2, 0] = -polar_motion_x
    polar[:, 2, 1] = polar_motion_y
    return np.einsum("nij,njk->nik", polar, np.einsum("nij,njk->nik", gst_matrix, pn))


def _state_matrix(values, name):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3).")
    return values


def _time_array(values, count):
    from astropy.time import Time

    if isinstance(values, Time):
        gps = np.atleast_1d(np.asarray(values.gps, dtype=float))
    else:
        try:
            gps = np.atleast_1d(np.asarray(values, dtype=float))
        except (TypeError, ValueError):
            gps = np.atleast_1d(np.asarray(Time(values, scale="utc").gps, dtype=float))
    if gps.size == 1:
        gps = np.repeat(gps, count)
    if gps.ndim != 1 or gps.size != count:
        raise ValueError("t must be scalar or contain one time per state vector.")
    return gps


def itrf_to_gcrf(r_itrf: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Convert ITRF positions to GCRF, undoing the SSAPy ``groundTrack`` transform."""
    time = to_gps(time)

    r_itrf = np.asarray(r_itrf)
    time = np.asarray(time)
    if r_itrf.shape[0] != time.shape[0]:
        raise ValueError("Number of positions must match number of times.")
    if r_itrf.shape[-1] != 3 or time.ndim != 1:
        raise ValueError("r_itrf must be (n, 3) and time must be (n,)")

    n = len(time)
    mjd_tt = _gpsToTT(time)
    d_ut1_tt_mjd, pmx, pmy = iers_interp(time)
    pn = erfa.pnm80(2400000.5, mjd_tt)
    gst = erfa.gst94(2400000.5, mjd_tt + d_ut1_tt_mjd)
    cg, sg = np.cos(gst), np.sin(gst)

    gst_mat = np.zeros((n, 3, 3), dtype=float)
    gst_mat[:, 0, 0] = cg
    gst_mat[:, 0, 1] = sg
    gst_mat[:, 1, 0] = -sg
    gst_mat[:, 1, 1] = cg
    gst_mat[:, 2, 2] = 1.0

    polar = np.eye(3, dtype=float)[np.newaxis, :, :].repeat(n, axis=0)
    polar[:, 0, 2] = pmx
    polar[:, 1, 2] = -pmy
    polar[:, 2, 0] = -pmx
    polar[:, 2, 1] = pmy

    u = np.einsum("tij,tjk->tik", gst_mat, pn)
    transform = np.einsum("tij,tjk->tik", polar, u)
    return np.einsum("tij,tj->ti", transform.transpose(0, 2, 1), r_itrf)
