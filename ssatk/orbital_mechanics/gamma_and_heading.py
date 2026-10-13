"""Flight-path angle (gamma) and heading in the Earth-fixed frame."""

import numpy as np

from ..coordinates import gcrf_to_itrf, v_from_r

_Z_HAT = np.array([0.0, 0.0, 1.0])


def _gamma_and_heading_from_rv(r_itrf, v_itrf):
    """Flight-path angle and heading (deg) from Earth-fixed position and velocity.

    Gamma is the geocentric flight-path angle, positive when climbing:
    ``sin(gamma) = r_hat . v_hat``. Heading is the azimuth of the horizontal
    part of the velocity, clockwise from local north toward local east, in
    [0, 360). Local east is ``z_hat x r_hat`` and local north is
    ``r_hat x east``. Heading is NaN at a pole or without horizontal motion;
    both angles are NaN at zero speed. This is a geocentric horizon, not the
    WGS84 ellipsoid normal.
    """
    r_itrf = np.atleast_2d(np.asarray(r_itrf, dtype=float))
    v_itrf = np.atleast_2d(np.asarray(v_itrf, dtype=float))
    if r_itrf.ndim != 2 or r_itrf.shape[1] != 3 or v_itrf.shape != r_itrf.shape:
        raise ValueError("Positions and velocities must have matching (N, 3) shapes")
    radius = np.linalg.norm(r_itrf, axis=1)
    if (
        not np.all(np.isfinite(r_itrf))
        or not np.all(np.isfinite(v_itrf))
        or np.any(radius == 0)
    ):
        raise ValueError("States must be finite with nonzero geocentric positions")
    r_unit = r_itrf / radius[:, None]
    speed = np.linalg.norm(v_itrf, axis=1)
    radial = np.einsum("ij,ij->i", v_itrf, r_unit)
    horizontal = np.linalg.norm(v_itrf - radial[:, None] * r_unit, axis=1)
    # Only remove directionless components at floating-point roundoff scale.
    tolerance = 64 * np.finfo(float).eps
    has_horizontal = horizontal > tolerance * speed
    horizontal = np.where(has_horizontal, horizontal, 0.0)
    gamma = np.degrees(np.arctan2(radial, horizontal))
    gamma[speed == 0] = np.nan
    east = np.cross(_Z_HAT, r_unit)
    east_length = np.linalg.norm(east, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        east = east / east_length[:, None]
    north = np.cross(r_unit, east)
    heading = np.mod(
        np.degrees(
            np.arctan2(
                np.einsum("ij,ij->i", v_itrf, east),
                np.einsum("ij,ij->i", v_itrf, north),
            )
        ),
        360.0,
    )
    heading[heading == 360.0] = 0.0
    heading[~has_horizontal | (east_length <= tolerance)] = np.nan
    return gamma, heading


def calc_gamma(r, t, *, v=None):
    """
    Earth-relative flight-path angle for a GCRF trajectory.

    Parameters
    ----------
    r : numpy.ndarray
        GCRF positions in meters, shape (n, 3).
    t : numpy.ndarray or astropy.time.Time
        GPS seconds or astropy Time, one per position.
    v : numpy.ndarray, optional
        GCRF velocities in m/s. When supplied, the full state transformation
        includes Earth's rotation; otherwise ITRF positions are differentiated.

    Returns
    -------
    numpy.ndarray
        Gamma in degrees, positive when climbing: ``sin(gamma) = r_hat . v_hat``
        with ITRF position and velocity. If ``v`` is absent, the velocity is
        estimated from the ITRF positions. Zero speed gives NaN.

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    return calc_gamma_and_heading(r, t, v=v)[0]


def calc_heading_itrf(r_itrf, v_itrf):
    """
    Heading of an Earth-fixed velocity.

    Parameters
    ----------
    r_itrf : numpy.ndarray
        ITRF positions in meters, shape (n, 3).
    v_itrf : numpy.ndarray
        ITRF velocities in m/s, shape (n, 3).

    Returns
    -------
    numpy.ndarray
        Heading in degrees in [0, 360), clockwise from local north toward local
        east (north = 0, east = 90). East and north are the geocentric local
        directions ``z_hat x r_hat`` and ``r_hat x east``; heading is NaN
        directly over a pole or without horizontal motion.

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    return _gamma_and_heading_from_rv(r_itrf, v_itrf)[1]


def calc_gamma_and_heading(r, t, *, v=None):
    """
    Earth-relative flight-path angle and heading for a GCRF trajectory.

    Parameters
    ----------
    r : numpy.ndarray
        GCRF positions in meters, shape (n, 3).
    t : numpy.ndarray or astropy.time.Time
        GPS seconds or astropy Time, one per position.
    v : numpy.ndarray, optional
        GCRF velocities in m/s, shape (n, 3). Prefer supplied velocities for
        sparse trajectories. Without them, finite differences approximate the
        ITRF velocity and cannot recover motion unresolved by the samples.

    Returns
    -------
    tuple of numpy.ndarray
        ``(gamma, heading)`` in degrees, with the conventions of
        :func:`calc_gamma` and :func:`calc_heading_itrf`.

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    r_itrf, v_itrf = gcrf_to_itrf(r, t, v=True if v is None else v)
    return _gamma_and_heading_from_rv(r_itrf, v_itrf)


def calc_gamma_and_heading_itrf(r_itrf, t, *, v=None):
    """
    Flight-path angle and heading for an ITRF trajectory.

    Parameters
    ----------
    r_itrf : numpy.ndarray
        ITRF positions in meters, shape (n, 3).
    t : numpy.ndarray or astropy.time.Time
        GPS seconds or astropy Time, one per position.
    v : numpy.ndarray, optional
        ITRF velocities in m/s, shape (n, 3). If absent, differentiate positions.

    Returns
    -------
    tuple of numpy.ndarray
        ``(gamma, heading)`` in degrees, with the conventions of
        :func:`calc_gamma` and :func:`calc_heading_itrf`. If ``v`` is absent,
        the velocity is the finite-difference derivative of ``r_itrf``.

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    v_itrf = v_from_r(r_itrf, t) if v is None else v
    return _gamma_and_heading_from_rv(r_itrf, v_itrf)
