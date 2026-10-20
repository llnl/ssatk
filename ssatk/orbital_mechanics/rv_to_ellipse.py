import numpy as np
from ..constants import EARTH_MU, EARTH_RADIUS

try:
    from astropy.time import Time
except ImportError:
    Time = None


def rv_to_ellipse(
    r,
    v,
    *,
    t0=None,                    # epoch echoed as first element of 't_abs'
    num: int | None = None,     # total # samples (≥1)
    mu: float = EARTH_MU,       # GM [m³ s⁻²]
    R_body: float = EARTH_RADIUS
):
    """
    Return the ellipse-arc dictionary used by ellipse_arc.py.
    When `num` ≥ 2, generate that many samples starting with (r0, v0):

        • Elliptic   – advance uniformly in true anomaly through 2π.
        • Non-elliptic – advance until ``r = 2 * norm(r0)``.

    The first sample is always the exact input state.
    """
    # ── convert inputs ───────────────────────────────────────────────
    r = np.asarray(r, float)
    v = np.asarray(v, float)
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)

    # ── orbital invariants ───────────────────────────────────────────
    h_vec = np.cross(r, v);  h = np.linalg.norm(h_vec)
    k_hat = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(k_hat, h_vec);  n = np.linalg.norm(n_vec)
    e_vec = (np.cross(v, h_vec) / mu) - r / r_mag
    e = np.linalg.norm(e_vec)
    Energy = v_mag**2 / 2.0 - mu / r_mag
    a = -mu / (2.0 * Energy) if Energy < 0 else np.inf
    p = h**2 / mu
    eta = np.sqrt(max(0.0, 1.0 - e**2))
    b = a * eta if Energy < 0 else np.nan

    # ── fundamental angles ──────────────────────────────────────────
    i   = np.arccos(h_vec[2] / h)
    raan = np.arccos(n_vec[0] / n) if n else 0.0
    if n and n_vec[1] < 0: raan = 2*np.pi - raan
    pa = np.arccos(np.clip(np.dot(n_vec, e_vec)/(n*e), -1, 1)) if n and e else 0.0
    if n and e and e_vec[2] < 0: pa = 2*np.pi - pa
    ta = np.arccos(np.clip(np.dot(e_vec, r)/(e*r_mag), -1, 1)) if e else 0.0
    if e and np.dot(r, v) < 0: ta = 2*np.pi - ta

    # mean longitude (elliptic only, needed for period timing)
    if Energy < 0:
        cosE = (e + np.cos(ta)) / (1 + e*np.cos(ta))
        sinE = (np.sin(ta) * eta) / (1 + e*np.cos(ta))
        E0 = np.arctan2(sinE, cosE) % (2*np.pi)
        M0 = E0 - e*np.sin(E0)
        L = (raan + pa + M0) % (2*np.pi)
    else:
        M0 = 0.0
        L = np.nan

    # ── geometric / timing scalars ───────────────────────────────────
    rp = a*(1-e) if Energy < 0 else p/(1+e)
    ra = a*(1+e) if Energy < 0 else np.inf
    rp_alt, ra_alt = rp - R_body, ra - R_body
    mean_motion = np.sqrt(mu/a**3) if Energy < 0 else np.nan
    period = 2*np.pi/mean_motion if Energy < 0 else np.nan

    # in-plane frame
    u_hat = r / r_mag
    w_hat = h_vec / h
    v_hat = np.cross(w_hat, u_hat);  v_hat /= np.linalg.norm(v_hat)
    plane_basis = (u_hat, v_hat, w_hat)
    F2 = 2*a*e_vec

    # ── sampling setup ───────────────────────────────────────────────
    total = max(1, num or 1)          # at least one sample
    elliptical = Energy < 0
    r_samples = np.empty((total, 3))
    v_samples = np.empty((total, 3))
    t_rel = np.zeros(total)

    # first entry is exactly the input state
    r_samples[0] = r
    v_samples[0] = v

    if total > 1:
        if elliptical:
            # uniform Δf through 0→2π, excluding the initial point
            f_steps = np.linspace(0, 2*np.pi, total, endpoint=False)[1:]
        else:
            # determine Δf that brings r to 2·|r0|
            cos_f_lim = (p/(2*r_mag) - 1)/e
            cos_f_lim = np.clip(cos_f_lim, -1.0, 1.0)
            f_lim = np.arccos(cos_f_lim)          # true anomaly where r = 2|r0|, outbound
            # ta is stored in [0, 2 pi); an inbound state has a negative true
            # anomaly. The steps are increments from ta, so they run to f_lim - ta.
            ta_signed = ta - 2*np.pi if ta > np.pi else ta
            f_steps = np.linspace(0, f_lim - ta_signed, total)[1:]
            ta = ta_signed

        # generate the remaining samples
        # u_hat points at r0, which sits at true anomaly ta, so a sample at
        # true anomaly ta + df lies at angle df from u_hat.
        for k, df in enumerate(f_steps, start=1):
            f = ta + df
            r_k = p / (1 + e*np.cos(f))
            r_hat = np.cos(df)*u_hat + np.sin(df)*v_hat
            t_hat = -np.sin(df)*u_hat + np.cos(df)*v_hat
            r_samples[k] = r_k * r_hat

            # velocity
            v_r = (mu/h)*e*np.sin(f)
            v_t = (mu/h)*(1 + e*np.cos(f))
            v_samples[k] = v_r*r_hat + v_t*t_hat

        # relative times from Kepler's equation
        t_rel[1:] = _time_since(ta, ta + f_steps, e=e, a=a, p=p, mu=mu)

    # ── absolute times array ─────────────────────────────────────────
    if t0 is not None:
        if Time is not None and isinstance(t0, Time):
            from astropy.time import TimeDelta
            t_abs = t0 + TimeDelta(t_rel, format="sec")
        else:
            t_abs = t0 + t_rel
    else:
        t_abs = None

    # ── assemble final dictionary ────────────────────────────────────
    return {
        "r": r_samples,
        "v": v_samples,
        "t_rel": t_rel,
        "t_abs": t_abs,

        "a": a, "e": e, "i": i, "raan": raan, "pa": pa, "ta": ta, "L": L,
        "rp": rp, "ra": ra, "rp_alt": rp_alt, "ra_alt": ra_alt,
        "b": b, "p": p, "mean_motion": mean_motion, "eta": eta, "period": period,
        "h_vec": h_vec, "h": h, "Energy": Energy, "e_vec": e_vec, "n_vec": n_vec,
        "r0": r, "v0": v, "F2": F2,
        "plane_basis": plane_basis, "rot_dir": 1,
        "mu": mu,
    }



def _time_since(f0, f, *, e, a, p, mu):
    """Time (s) to move from true anomaly f0 to each f (f >= f0, unwrapped) on a conic."""
    f = np.asarray(f, dtype=float)
    if e < 1.0:
        n = np.sqrt(mu / a**3)

        def mean_anomaly(nu):
            E = 2.0 * np.arctan(np.sqrt((1.0 - e) / (1.0 + e)) * np.tan(nu / 2.0))
            return E - e * np.sin(E)

        # Count whole revolutions separately so tan(nu/2) never crosses pi.
        def unwrapped(nu):
            turns = np.floor((nu + np.pi) / (2.0 * np.pi))
            return mean_anomaly(nu - 2.0 * np.pi * turns) + 2.0 * np.pi * turns

        return (unwrapped(f) - unwrapped(f0)) / n
    if e == 1.0:
        def barker(nu):
            D = np.tan(nu / 2.0)
            return np.sqrt(p**3 / mu) * 0.5 * (D + D**3 / 3.0)

        return barker(f) - barker(f0)

    n = np.sqrt(mu / (-a) ** 3) if np.isfinite(a) else np.sqrt(mu / (p / (e * e - 1.0)) ** 3)

    def hyperbolic_mean_anomaly(nu):
        H = 2.0 * np.arctanh(np.sqrt((e - 1.0) / (e + 1.0)) * np.tan(nu / 2.0))
        return e * np.sinh(H) - H

    return (hyperbolic_mean_anomaly(f) - hyperbolic_mean_anomaly(f0)) / n


# ── quick demo ───────────────────────────────────────────────────────
if __name__ == "__main__":
    r0 = [7000e3, 0.0, 0.0]
    v0 = [0.0, 7.5e3, 1.0e3]
    info = rv_to_ellipse(r0, v0, num=5)
    print("First r sample equals r0?", np.allclose(info["r"][0], r0))
    print("t_rel:", info["t_rel"])
