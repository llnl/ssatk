import numpy as np

def equally_spaced_ta(
    n_samples,
    orbit=None,
    a=None,
    e=None,
    rp=None,
    ra=None,
    n_dense=20000,
    degrees=False,
):
    """
    Compute true anomalies (`ta`) for points approximately equally spaced
    by arc length around an elliptical orbit.

    Accepts either:
        - orbit : SSAPy Orbit object
        - a and e
        - rp and ra  (periapsis and apoapsis radii)

    Constraints:
        - n_samples must be even
        - if n_samples >= 2, 0 and pi are always included

    Parameters
    ----------
    n_samples : int
        Number of samples to return. Must be even.
    orbit : ssapy.Orbit, optional
        SSAPy Orbit object.
    a : float, optional
        Semi-major axis.
    e : float, optional
        Eccentricity.
    rp : float, optional
        Periapsis radius.
    ra : float, optional
        Apoapsis radius.
    n_dense : int, optional
        Number of dense samples used internally.
    degrees : bool, optional
        If True, return degrees. Otherwise radians.

    Returns
    -------
    ta : ndarray
        True anomalies, either in radians or degrees.
    """

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if n_samples % 2 != 0:
        raise ValueError("n_samples must be even.")

    # Resolve ellipse parameters
    if orbit is not None:
        # Assumes SSAPy Orbit exposes a and e as attributes
        a = float(np.atleast_1d(orbit.a)[0])
        e = float(np.atleast_1d(orbit.e)[0])
    elif a is not None and e is not None:
        a = float(a)
        e = float(e)
    elif rp is not None and ra is not None:
        rp = float(rp)
        ra = float(ra)
        if rp <= 0 or ra <= 0:
            raise ValueError("rp and ra must be positive.")
        a = 0.5 * (rp + ra)
        e = (ra - rp) / (ra + rp)
    else:
        raise ValueError(
            "Provide either orbit, or (a and e), or (rp and ra)."
        )

    if not (0 <= e < 1):
        raise ValueError("Only elliptical orbits with 0 <= e < 1 are supported.")
    if a <= 0:
        raise ValueError("Semi-major axis must be positive.")

    # Semi-minor axis
    b = a * np.sqrt(1 - e**2)

    # Dense eccentric-anomaly sampling over full ellipse
    E = np.linspace(0.0, 2.0 * np.pi, n_dense, endpoint=False)

    # Ellipse coordinates centered at ellipse center
    x = a * np.cos(E)
    y = b * np.sin(E)

    # Use the first half [0, pi] and mirror it to guarantee 0 and pi
    half_mask = E <= np.pi
    E_half = E[half_mask]
    x_half = a * np.cos(E_half)
    y_half = b * np.sin(E_half)

    # Cumulative arc length on half-ellipse via chord approximation
    dx_half = np.diff(x_half)
    dy_half = np.diff(y_half)
    ds_half = np.sqrt(dx_half**2 + dy_half**2)
    s_half = np.r_[0.0, np.cumsum(ds_half)]
    half_length = s_half[-1]

    # Number of samples on each half
    n_half = n_samples // 2

    # Include 0 on the first half; exclude pi so it appears only in mirrored half
    s_targets_half = np.linspace(0.0, half_length, n_half, endpoint=False)

    # Interpolate eccentric anomaly values for equal arc-length points
    E_half_equal = np.interp(s_targets_half, s_half, E_half)

    # Mirror to second half
    E_equal = np.r_[E_half_equal, E_half_equal + np.pi]
    E_equal = np.mod(E_equal, 2.0 * np.pi)

    # Convert eccentric anomaly -> true anomaly using SSAPy relation
    # orbit.py provides _ellipticalEccentricToTrueAnomaly(E, e)
    beta = e / (1 + np.sqrt((1 - e) * (1 + e)))
    ta = E_equal + 2 * np.arctan(beta * np.sin(E_equal) / (1 - beta * np.cos(E_equal)))
    ta = np.mod(ta, 2.0 * np.pi)

    # Sort results
    ta = np.sort(ta)

    # Force exact 0 and pi to avoid tiny numerical drift
    if n_samples >= 2:
        ta[0] = 0.0
        idx_pi = np.argmin(np.abs(ta - np.pi))
        ta[idx_pi] = np.pi

    if degrees:
        ta = np.degrees(ta)

    return ta