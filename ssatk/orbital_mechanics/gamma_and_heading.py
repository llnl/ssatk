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
    ``r_hat x east``, so heading is NaN directly over a pole.
    """
    r_itrf = np.atleast_2d(np.asarray(r_itrf, dtype=float))
    v_itrf = np.atleast_2d(np.asarray(v_itrf, dtype=float))
    r_unit = r_itrf / np.linalg.norm(r_itrf, axis=1, keepdims=True)
    v_unit = v_itrf / np.linalg.norm(v_itrf, axis=1, keepdims=True)

    gamma = np.degrees(np.arcsin(np.clip(np.einsum("ij,ij->i", r_unit, v_unit), -1.0, 1.0)))

    east = np.cross(_Z_HAT, r_unit)
    with np.errstate(invalid="ignore", divide="ignore"):
        east = east / np.linalg.norm(east, axis=1, keepdims=True)
    north = np.cross(r_unit, east)
    heading = np.degrees(np.arctan2(np.einsum("ij,ij->i", v_unit, east), np.einsum("ij,ij->i", v_unit, north)))
    return gamma, np.mod(heading, 360.0)


def calc_gamma(r, t):
    """
    Earth-relative flight-path angle for a GCRF trajectory.

    Parameters
    ----------
    r : numpy.ndarray
        GCRF positions in meters, shape (n, 3).
    t : numpy.ndarray or astropy.time.Time
        GPS seconds or astropy Time, one per position.

    Returns
    -------
    numpy.ndarray
        Gamma in degrees, positive when climbing: ``sin(gamma) = r_hat . v_hat``
        with ITRF position and velocity. The velocity is the finite-difference
        derivative of the ITRF positions, so it is Earth-relative.

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    r_itrf, v_itrf = gcrf_to_itrf(r, t, v=True)
    return _gamma_and_heading_from_rv(r_itrf, v_itrf)[0]


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
        directly over a pole.

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    return _gamma_and_heading_from_rv(r_itrf, v_itrf)[1]


def calc_gamma_and_heading(r, t):
    """
    Earth-relative flight-path angle and heading for a GCRF trajectory.

    Parameters
    ----------
    r : numpy.ndarray
        GCRF positions in meters, shape (n, 3).
    t : numpy.ndarray or astropy.time.Time
        GPS seconds or astropy Time, one per position.

    Returns
    -------
    tuple of numpy.ndarray
        ``(gamma, heading)`` in degrees, with the conventions of
        :func:`calc_gamma` and :func:`calc_heading_itrf`.

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    r_itrf, v_itrf = gcrf_to_itrf(r, t, v=True)
    return _gamma_and_heading_from_rv(r_itrf, v_itrf)


def calc_gamma_and_heading_itrf(r_itrf, t):
    """
    Flight-path angle and heading for an ITRF trajectory.

    Parameters
    ----------
    r_itrf : numpy.ndarray
        ITRF positions in meters, shape (n, 3).
    t : numpy.ndarray or astropy.time.Time
        GPS seconds or astropy Time, one per position.

    Returns
    -------
    tuple of numpy.ndarray
        ``(gamma, heading)`` in degrees, with the conventions of
        :func:`calc_gamma` and :func:`calc_heading_itrf`. The velocity is the
        finite-difference derivative of ``r_itrf``.

    Author
    ------
    Travis Yeager (yeager7@llnl.gov)
    """
    v_itrf = v_from_r(r_itrf, t)
    return _gamma_and_heading_from_rv(r_itrf, v_itrf)
