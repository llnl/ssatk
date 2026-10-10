from ..constants import EARTH_MU, EARTH_RADIUS
import numpy as np


def ellipse_from_rv(r, v, num=500, f_span=None, t0=None):
    """
    Sample `num` points on the conic (ellipse / parabola / hyperbola)
    defined by state vector (r, v).  For e >= 1 the arc is truncated so
    that the radius never exceeds 2*|r|.

    Returns a dict much like ellipse_arc.py (see docstring there).
    """
    # ---------- input -------------------------------------------------------
    r = np.asarray(r, float)
    v = np.asarray(v, float)
    if r.shape != (3,) or v.shape != (3,):
        raise ValueError("r and v must be length-3 vectors")

    mu = float(EARTH_MU)
    R_E = float(EARTH_RADIUS)
    r_mag = np.linalg.norm(r)
    r_limit = 2.0 * r_mag
    tol = 1.0e-10

    # ---------- invariants --------------------------------------------------
    h_vec = np.cross(r, v)
    h = np.linalg.norm(h_vec)
    if h == 0.0:
        raise ValueError("Angular-momentum vector is zero")

    e_vec = np.cross(v, h_vec) / mu - r / r_mag
    e = np.linalg.norm(e_vec)
    Energy = np.dot(v, v) / 2.0 - mu / r_mag
    a = -mu / (2.0 * Energy) if Energy != 0.0 else np.inf  # negative for hyperbola, inf for parabola
    p = h * h / mu

    # ---------- orientation angles -----------------------------------------
    i_rad = np.arccos(h_vec[2] / h)
    k_hat = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(k_hat, h_vec)
    n_norm = np.linalg.norm(n_vec)

    if n_norm != 0.0:
        raan = np.arccos(n_vec[0] / n_norm)
        if n_vec[1] < 0.0:
            raan = 2.0 * np.pi - raan
    else:
        raan = 0.0

    if n_norm != 0.0 and e > tol:
        pa = np.arccos(np.dot(n_vec, e_vec) / (n_norm * e))
        if e_vec[2] < 0.0:
            pa = 2.0 * np.pi - pa
    else:
        pa = 0.0

    if e > tol:
        ta0 = np.arccos(np.dot(e_vec, r) / (e * r_mag))
        if np.dot(r, v) < 0.0:
            ta0 = 2.0 * np.pi - ta0
    else:
        ta0 = 0.0

    # ---------- true‑anomaly grid ------------------------------------------
    if e < 1.0 - tol:                     # ellipse
        f = np.linspace(0.0, 2.0 * np.pi, max(3, num), endpoint=False)
    else:                                 # parabola / hyperbola
        cos_cap = (p / r_limit - 1.0) / e
        if e > 1.0 + tol:                 # hyperbola asymptote guard
            cos_cap = max(cos_cap, -1.0 / e + 1.0e-8)
        cos_cap = np.clip(cos_cap, -1.0, 1.0)
        f_cap = float(np.arccos(cos_cap))
        if e > 1.0 + tol:
            f_asym = np.arccos(-1.0 / e) - 1.0e-6
            f_cap = min(f_cap, f_asym)
        if f_span is not None:
            f_cap = min(f_cap, abs(float(f_span)))
        f = np.linspace(-f_cap, f_cap, max(3, num))

    # ---------- perifocal positions / velocities ---------------------------
    r_pf = p / (1.0 + e * np.cos(f))
    x_pf = r_pf * np.cos(f)
    y_pf = r_pf * np.sin(f)
    z_pf = np.zeros_like(f)

    fac = np.sqrt(mu / p)
    vx_pf = -fac * np.sin(f)
    vy_pf =  fac * (e + np.cos(f))
    vz_pf = np.zeros_like(f)

    # ---------- rotation to inertial ---------------------------------------
    def R3(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, -s, 0.0],
                         [s,  c, 0.0],
                         [0.0, 0.0, 1.0]])

    def R1(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[1.0, 0.0, 0.0],
                         [0.0,  c, -s],
                         [0.0,  s,  c]])

    # Perifocal -> inertial: R3(RAAN) R1(i) R3(pa), with R1/R3 active rotations.
    Q = R3(raan) @ R1(i_rad) @ R3(pa)

    pts  = (Q @ np.vstack((x_pf,  y_pf,  z_pf))).T
    vels = (Q @ np.vstack((vx_pf, vy_pf, vz_pf))).T

    # ---------- timing ------------------------------------------------------
    t_rel = np.zeros_like(f)
    period = None
    eta = None
    if e < 1.0 - tol:
        eta = np.sqrt(1.0 - e * e)
        E = 2.0 * np.arctan2(np.tan(f / 2.0), np.sqrt((1.0 + e) / (1.0 - e)))
        E = np.unwrap(E)
        M = E - e * np.sin(E)
        n_mean = np.sqrt(mu / (a ** 3))
        t_rel = (M - M[0]) / n_mean
        period = 2.0 * np.pi / n_mean
    elif abs(e - 1.0) <= tol:
        D = np.tan(f / 2.0)
        # Barker's equation: t = (1/2) sqrt(p^3 / mu) (D + D^3 / 3)
        t_rel = 0.5 * (p ** 1.5 / np.sqrt(mu)) * (D + D ** 3 / 3.0)
        t_rel -= t_rel[0]
    else:
        sinhH = np.sqrt((e - 1.0) / (e + 1.0)) * np.tan(f / 2.0)
        H = np.arcsinh(sinhH)
        M_h = e * np.sinh(H) - H
        n_h = np.sqrt(mu / (-a) ** 3)
        t_rel = (M_h - M_h[0]) / n_h

    # ---------- extras ------------------------------------------------------
    rp = a * (1.0 - e) if e < 1.0 - tol else p / 2.0
    ra = a * (1.0 + e) if e < 1.0 - tol else None
    mean_motion = None if period is None else 2.0 * np.pi / period

    h_hat = h_vec / h
    u_hat = e_vec / e if e > tol else pts[0] / np.linalg.norm(pts[0])
    v_hat = np.cross(h_hat, u_hat)
    rot_dir = 1 if h_hat[2] >= 0.0 else -1

    if not np.isfinite(a):
        F2 = np.full(3, np.nan)  # a parabola's second focus is at infinity
    else:
        F2 = 2.0 * a * e_vec if e > tol else np.zeros(3)

    result = {
        # trajectory
        "r": pts,
        "v": vels,
        "t_rel": t_rel,
        "t_abs": (np.asarray(t0) + t_rel) if t0 is not None else None,

        # elements
        "a": a, "e": e, "i": i_rad,
        "raan": raan, "pa": pa, "ta": ta0,
        "L": (raan + pa + ta0) % (2.0 * np.pi),

        # handy scalars
        "rp": rp, "ra": ra,
        "rp_alt": None if rp is None else rp - R_E,
        "ra_alt": None if ra is None else ra - R_E,
        "b": a * np.sqrt(1 - e * e) if e < 1.0 - tol else None,
        "p": p, "eta": eta, "period": period,
        "mean_motion": mean_motion,

        # invariants
        "h_vec": h_vec, "h": h, "Energy": Energy,
        "e_vec": e_vec, "n_vec": n_vec,

        # convenience
        "r0": pts[0], "v0": vels[0], "F2": F2,
        "plane_basis": (u_hat, v_hat, h_hat),
        "rot_dir": rot_dir,

        # constant
        "mu": mu,
    }

    return result
