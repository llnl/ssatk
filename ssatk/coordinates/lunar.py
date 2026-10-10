import numpy as np
from ssapy import get_body
from ssapy.body import MoonPosition
from ssapy.utils import normed
from astropy.time import Time

from .velocity import v_from_r


def gcrf_to_lunar(r: np.ndarray, t: np.ndarray, v: np.ndarray = None) -> np.ndarray:
    """Convert GCRF position/velocity vectors to the rotating lunar frame."""

    class MoonRotator:
        def __init__(self):
            self.mpm = MoonPosition()

        def __call__(self, r: np.ndarray, t: np.ndarray) -> np.ndarray:
            if isinstance(t, Time):
                t = t.gps
            rmoon = self.mpm(t)
            vmoon = (self.mpm(t + 5.0) - self.mpm(t - 5.0)) / 10.0
            xhat = normed(rmoon.T).T
            vpar = np.einsum("ab,ab->b", xhat, vmoon) * xhat
            yhat = normed((vmoon - vpar).T).T
            zhat = np.cross(xhat, yhat, axisa=0, axisb=0).T
            rotation = np.empty((3, 3, len(t)))
            rotation[0] = xhat
            rotation[1] = yhat
            rotation[2] = zhat
            return np.einsum("abc,cb->ca", rotation, r)

    rotator = MoonRotator()
    if v is None:
        return rotator(r, t)
    r_lunar = rotator(r, t)
    return r_lunar, v_from_r(r_lunar, t)


def gcrf_to_lunar_fixed(r: np.ndarray, t: np.ndarray, v: np.ndarray = None) -> np.ndarray:
    """Convert GCRF vectors to the Moon-centred Earth-Moon rotating frame.

    +X points from Earth to the Moon (so Earth lies on -X), +Y along the
    Moon's velocity component perpendicular to X, and +Z along the orbit
    normal. This is the frame cislunar plots want; it is not the lunar
    body-fixed frame used by selenographic maps. For that, use
    :func:`gcrf_to_lunar_body`.
    """
    moon_body = get_body("moon")
    r_lunar = gcrf_to_lunar(r, t) - gcrf_to_lunar(moon_body.position(t).T, t)
    if v is None:
        return r_lunar
    return r_lunar, v_from_r(r_lunar, t)


def _gps_seconds(t):
    """Return GPS seconds as a 1-D float array from scalar, array, or Time input."""
    if isinstance(t, Time):
        return np.atleast_1d(t.gps).astype(float)
    values = np.atleast_1d(np.asarray(t, dtype=object)).reshape(-1)
    if values.size and isinstance(values[0], Time):
        return np.array([value.gps for value in values], dtype=float)
    return values.astype(float)


def lunar_body_rotation(t):
    """Return GCRF-to-lunar-body rotation matrices with shape ``(N, 3, 3)``.

    The body frame is SSAPy's ``MoonOrientation``, the DE440 lunar
    principal-axis (PA) frame: +X near the mean sub-Earth point (0 deg
    selenographic longitude), +Y toward 90 deg E, +Z along the spin axis.
    Lunar maps (LOLA, LROC) are published in the mean-Earth/polar-axis (ME)
    frame, which differs from PA by a small constant rotation (sub-kilometre at
    the surface; the angles are in the DE440 lunar frames kernel).

    This is not the frame of :func:`gcrf_to_lunar_fixed`, which is the
    Moon-centred Earth-Moon rotating frame (Earth on -X, +Z along the orbit
    normal).
    """
    # Keep the Body referenced while its kernels are in use: SSAPy closes a
    # Body's kernels when the Body is collected, so get_body("moon").orientation(t)
    # on a temporary fails. MoonOrientation evaluates one epoch per call.
    moon = get_body("moon")
    return np.stack([np.asarray(moon.orientation(ti), dtype=float) for ti in _gps_seconds(t)])


def gcrf_to_lunar_body(r: np.ndarray, t) -> np.ndarray:
    """Convert GCRF positions in metres to Moon-centred lunar body-fixed metres.

    ``r`` has shape ``(N, 3)`` or ``(3,)``; ``t`` is one GPS time or Astropy
    ``Time`` per row, or a single epoch applied to every row. See
    :func:`lunar_body_rotation` for the frame definition.
    """
    r = np.atleast_2d(np.asarray(r, dtype=float))
    if r.shape[-1] != 3:
        raise ValueError(f"r must contain 3-vectors; got shape {r.shape}")
    t_gps = _gps_seconds(t)
    if t_gps.size == 1 and len(r) > 1:
        t_gps = np.repeat(t_gps, len(r))
    if t_gps.size != len(r):
        raise ValueError("t must contain one time per position or a single epoch")
    moon = get_body("moon")
    r_moon = np.asarray(moon.position(t_gps), dtype=float).reshape(3, -1).T
    return np.einsum("nij,nj->ni", lunar_body_rotation(t_gps), r - r_moon)


def get_lunar_rv(t):
    """Return Moon GCRF position and velocity for scalar or vector time input."""
    if isinstance(t, Time):
        t = t.gps
    elif np.size(t) > 1 and isinstance(t[0], Time):
        t = np.array([ti.gps for ti in t], dtype=float)

    moon = get_body("moon")
    t = np.asarray(t, dtype=float)
    r = moon.position(t).T
    # A +/-1 s central difference at each epoch. Differencing the requested
    # epochs themselves gives chord velocities that depend on the sampling
    # (120 m/s wrong for daily samples).
    dt = 1.0
    v = (moon.position(t + dt).T - moon.position(t - dt).T) / (2.0 * dt)
    return np.atleast_2d(r), np.atleast_2d(v)
