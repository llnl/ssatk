import numpy as np

def rescale_burn(a0, m0, t0, m=None, t=None, mode="constant_thrust"):
    """
    Rescale a burn given a reference case (a0, m0, t0).

    Parameters
    ----------
    a0 : float or array
        Reference acceleration (m/s^2)
    m0 : float or array
        Reference mass (kg)
    t0 : float or array
        Reference duration (s)
    m : float or array or None
        New mass (kg). If None, uses m0.
    t : float or array or None
        New duration (s). If None, uses t0.
    mode : str
        "constant_thrust"  -> thrust F is held constant; duration doesn't change accel.
        "constant_impulse" -> total impulse I = F*t is held constant; changing duration changes thrust.

    Returns
    -------
    a : float or array
        New acceleration (m/s^2)
    t : float or array
        New duration (s) (the input 't' if provided, else t0)
    dv : float or array
        Approx delta-v assuming mass stays constant during the burn: dv = a * t
    F : float or array
        Thrust used for the new case (N)
    I : float or array
        Total impulse used for the new case (N*s)
    """
    a0 = np.asarray(a0, dtype=float)
    m0 = np.asarray(m0, dtype=float)
    t0 = np.asarray(t0, dtype=float)

    m_new = m0 if m is None else np.asarray(m, dtype=float)
    t_new = t0 if t is None else np.asarray(t, dtype=float)

    # Reference thrust and impulse inferred from the reference case
    F0 = a0 * m0
    I0 = F0 * t0

    if mode == "constant_thrust":
        F = F0
        a = F / m_new
        I = F * t_new
    elif mode == "constant_impulse":
        I = I0
        F = I / t_new
        a = F / m_new
    else:
        raise ValueError("mode must be 'constant_thrust' or 'constant_impulse'")

    dv = a * t_new  # simple (no mass change during burn)
    return a, t_new, dv, F, I
