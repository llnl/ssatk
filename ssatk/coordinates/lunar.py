import numpy as np
from ssapy import get_body
from astropy.time import Time


def _lunar_rotation(moon, t):
    """GCRF-to-Earth-Moon rotating axes, with shape (N, 3, 3)."""
    rmoon = np.asarray(moon.position(t), dtype=float).reshape(3, -1).T
    vmoon = (np.asarray(moon.position(t + 5.0)).reshape(3, -1).T
             - np.asarray(moon.position(t - 5.0)).reshape(3, -1).T) / 10.0
    xhat = rmoon / np.linalg.norm(rmoon, axis=1)[:, None]
    transverse = vmoon - np.sum(xhat * vmoon, axis=1)[:, None] * xhat
    yhat = transverse / np.linalg.norm(transverse, axis=1)[:, None]
    return np.stack((xhat, yhat, np.cross(xhat, yhat)), axis=1)


def _rotating_lunar_state(r, t, v, *, moon_centred, step_s):
    r = np.asarray(r, dtype=float)
    scalar = r.ndim == 1
    if r.ndim not in (1, 2) or r.shape[-1] != 3 or not np.all(np.isfinite(r)):
        raise ValueError("r must contain finite 3-vectors in metres.")
    r = np.atleast_2d(r)
    epochs = _gps_seconds(t)
    if epochs.size == 1:
        epochs = np.repeat(epochs, len(r))
    if epochs.size != len(r) or not np.all(np.isfinite(epochs)):
        raise ValueError("t must be finite with one epoch per position or a single epoch.")
    step_s = float(step_s)
    if not np.isfinite(step_s) or step_s <= 0.0:
        raise ValueError("step_s must be finite and positive.")
    moon = get_body("moon")
    rotation = _lunar_rotation(moon, epochs)
    relative = r
    if moon_centred:
        relative = r - np.asarray(moon.position(epochs)).reshape(3, -1).T
    position = np.einsum("nij,nj->ni", rotation, relative)
    if v is None:
        return position[0] if scalar else position
    v = np.asarray(v, dtype=float)
    if v.shape not in ((3,), r.shape) or not np.all(np.isfinite(v)):
        raise ValueError("v must contain finite 3-vectors in metres per second matching r.")
    v = np.broadcast_to(v, r.shape)
    if moon_centred:
        moon_velocity = (np.asarray(moon.position(epochs + step_s)).reshape(3, -1).T
                         - np.asarray(moon.position(epochs - step_s)).reshape(3, -1).T) / (2.0 * step_s)
        v = v - moon_velocity
    rotation_rate = (_lunar_rotation(moon, epochs + step_s)
                     - _lunar_rotation(moon, epochs - step_s)) / (2.0 * step_s)
    velocity = (np.einsum("nij,nj->ni", rotation, v)
                + np.einsum("nij,nj->ni", rotation_rate, relative))
    return (position[0], velocity[0]) if scalar else (position, velocity)


def gcrf_to_lunar(r: np.ndarray, t: np.ndarray, v: np.ndarray = None, *, step_s=1.0):
    """Convert GCRF states to the Earth-centred Earth-Moon rotating frame.

    Positions are metres, velocities metres per second, and epochs GPS seconds
    or Astropy Time. A supplied velocity is transformed as R v + Rdot r;
    Rdot is evaluated with a +/- ``step_s`` second central difference at each
    epoch, independently of the requested sampling. Inputs are (N, 3) or one
    (3,) state; a scalar epoch applies to every row.
    """
    return _rotating_lunar_state(r, t, v, moon_centred=False, step_s=step_s)


def gcrf_to_lunar_fixed(r: np.ndarray, t: np.ndarray, v: np.ndarray = None, *, step_s=1.0):
    """Convert GCRF vectors to the Moon-centred Earth-Moon rotating frame.

    +X points from Earth to the Moon (so Earth lies on -X), +Y along the
    Moon's velocity component perpendicular to X, and +Z along the orbit
    normal. This is the frame cislunar plots want; it is not the lunar
    body-fixed frame used by selenographic maps. For that, use
    :func:`gcrf_to_lunar_body`.

    Units and shapes follow :func:`gcrf_to_lunar`. A supplied velocity becomes
    R (v - v_moon) + Rdot (r - r_moon), with ephemeris derivatives evaluated
    locally at each epoch using +/- ``step_s`` seconds.
    """
    return _rotating_lunar_state(r, t, v, moon_centred=True, step_s=step_s)


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



def lunar_map_rotation(t):
    """GCRF -> lunar mean-Earth map axes, shape (N, 3, 3).

    Apply NAIF's DE440 PA-to-ME421-aligned constant rotation to SSAPy's
    dated DE440 principal axes. North-up LOLA/LROC maps use ME axes;
    :func:`lunar_body_rotation` continues to return PA axes for dynamics.
    Source: moon_de440_250416.tf, TKFRAME_31009_ANGLES (arcseconds).
    https://naif.jpl.nasa.gov/pub/naif/generic_kernels/fk/satellites/moon_de440_250416.tf
    """
    from scipy.spatial.transform import Rotation
    pa_to_map = Rotation.from_euler("zyx", np.array([67.8526, 78.6944, .2785])/3600,
                                   degrees=True).as_matrix()
    return pa_to_map @ lunar_body_rotation(t)


def gcrf_to_lunar_map(r, t):
    """Moon-centred ME map coordinates (metres) from geocentric GCRF metres."""
    from scipy.spatial.transform import Rotation
    pa_to_map = Rotation.from_euler("zyx", np.array([67.8526, 78.6944, .2785])/3600,
                                   degrees=True).as_matrix()
    return gcrf_to_lunar_body(r, t) @ pa_to_map.T


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
