"""GCRF <-> ITRF conversions, all built on :mod:`ssatk.coordinates._earth_rotation`.

Every function here used to carry one of two independent Earth-orientation
chains: ``gcrf_to_itrf``/``itrf_to_gcrf`` rotated through SSAPy's
``groundTrack``/IAU 1976+1980 convention, while ``gcrf_to itrf_eop`` built its
own ``erfa.pnm80``/``erfa.gst94`` chain and ``gcrf_to_itrf_astropy`` deferred
to Astropy's IAU 2006/2000A GCRS/ITRS transform. All four now share the one
IAU 2006/2000A route in ``_earth_rotation``, fed by SSATK's packaged EOP data
instead of SSAPy's or Astropy's own IERS tables, so a given epoch always
produces the same rotation and none of them touch the network.

``allow_predicted`` default: ``gcrf_to_itrf_eop``/``itrf_to_gcrf_eop`` already
rejected a predicted EOP record by default before this change, so they keep
that default. ``gcrf_to_itrf``, ``itrf_to_gcrf``, ``gcrf_to_itrf_astropy`` and
``eci_to_ecf_matrix`` (frames.py) never had this distinction -- SSAPy's
``groundTrack``/``iers_interp`` and Astropy's own ITRS/GCRS transform always
just used whatever data they had -- and near-future epochs routinely fall on
the predicted side of SSATK's packaged snapshot, so making every ordinary call
raise by default would be a much larger behavior change than switching models.
They default ``allow_predicted=True``, preserving that prior behavior, while
still exposing the argument for a caller that wants the stricter check.
"""

import numpy as np

from ..time_functions import to_gps
from ._earth_rotation import gcrs_to_itrs_matrix, gcrs_to_itrs_matrix_and_rate
from .velocity import v_from_r


def gcrf_to_itrf(r_gcrf, t, v=None, *, step_s=1.0, allow_predicted=True):
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
        the GCRS->ITRS rotation when ``v`` is an array. Default 1 s.
    allow_predicted : bool, optional
        Forwarded to the packaged EOP table; default ``True`` (see the
        module docstring).

    Returns
    -------
    numpy.ndarray or tuple
        ITRF positions in meters, or ``(position, velocity)`` when ``v`` is
        given. The velocity is Earth-relative: ``R v + (dR/dt) r``.
    """
    gps = to_gps(t)
    r_gcrf = _state_matrix(r_gcrf, "r_gcrf")
    gps_arr = _time_array(gps, len(r_gcrf))

    if v is None or v is False:
        rotation = gcrs_to_itrs_matrix(gps_arr, allow_predicted=allow_predicted)
        return np.einsum("nij,nj->ni", rotation, r_gcrf)
    if v is True:
        rotation = gcrs_to_itrs_matrix(gps_arr, allow_predicted=allow_predicted)
        pos = np.einsum("nij,nj->ni", rotation, r_gcrf)
        return pos, v_from_r(pos, gps_arr)

    v_gcrf = np.asarray(v, dtype=float)
    if v_gcrf.shape != r_gcrf.shape:
        raise ValueError("v must have the same shape as r_gcrf.")
    rotation, rate = gcrs_to_itrs_matrix_and_rate(
        gps_arr, step_s=step_s, allow_predicted=allow_predicted
    )
    pos = np.einsum("nij,nj->ni", rotation, r_gcrf)
    vel = np.einsum("nij,nj->ni", rotation, v_gcrf) + np.einsum("nij,nj->ni", rate, r_gcrf)
    return pos, vel


def itrf_to_gcrf(r_itrf: np.ndarray, time: np.ndarray, *, allow_predicted: bool = True) -> np.ndarray:
    """Convert ITRF positions to GCRF.

    ``allow_predicted`` defaults ``True``; see the module docstring.
    """
    gps = to_gps(time)
    r_itrf = _state_matrix(r_itrf, "r_itrf")
    gps_arr = _time_array(gps, len(r_itrf))

    rotation = gcrs_to_itrs_matrix(gps_arr, allow_predicted=allow_predicted)
    return np.einsum("nji,nj->ni", rotation, r_itrf)


def gcrf_to_itrf_astropy(state_vectors, t, *, allow_predicted: bool = True):
    """Convert GCRF positions to geocentric ITRF positions (meters).

    Despite the name, this is the same IAU 2006/2000A route as every other
    function in this module -- kept for callers that already import it by
    this name. It used to call Astropy's GCRS->ITRS transform directly, which
    needs Astropy's own (network-backed) IERS table; it now shares SSATK's
    packaged-EOP route, so it no longer touches the network.
    ``allow_predicted`` defaults ``True``; see the module docstring.
    """
    state_vectors = _state_matrix(state_vectors, "state_vectors")
    gps = _time_array(to_gps(t), len(state_vectors))
    rotation = gcrs_to_itrs_matrix(gps, allow_predicted=allow_predicted)
    return np.einsum("nij,nj->ni", rotation, state_vectors)


def gcrf_to_itrf_eop(state_vectors, t, *, eop=None, allow_predicted=False):
    """Convert GCRF positions to ITRF using an explicit EOP data source.

    The rotation is IAU 2006/2000A (:mod:`ssatk.coordinates._earth_rotation`):
    CIO-based precession-nutation with the IERS celestial-pole offsets, Earth
    Rotation Angle at UT1, and polar motion. ``eop`` may be an
    ``EarthOrientationTable`` or any object with the same
    ``at(time, allow_predicted=...)`` interface; it defaults to SSATK's
    packaged snapshot.
    """
    state_vectors = _state_matrix(state_vectors, "state_vectors")
    gps = _time_array(t, len(state_vectors))
    rotation = gcrs_to_itrs_matrix(gps, eop=eop, allow_predicted=allow_predicted)
    return np.einsum("nij,nj->ni", rotation, state_vectors)


def itrf_to_gcrf_eop(state_vectors, t, *, eop=None, allow_predicted=False):
    """Convert ITRF positions to GCRF using an explicit EOP data source.

    See :func:`gcrf_to_itrf_eop` for the model and the ``eop``/
    ``allow_predicted`` arguments.
    """
    state_vectors = _state_matrix(state_vectors, "state_vectors")
    gps = _time_array(t, len(state_vectors))
    rotation = gcrs_to_itrs_matrix(gps, eop=eop, allow_predicted=allow_predicted)
    return np.einsum("nji,nj->ni", rotation, state_vectors)


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
