import numpy as np
from typing import Tuple, Optional

# Closed-form two-body relations; see e.g. Vallado, Fundamentals of
# Astrodynamics and Applications (4th ed.), chapters 1-2 and 6.


def escape_velocity(mu: float, r: float) -> float:
    """Escape velocity: v_e = sqrt(2*mu / r)"""
    return np.sqrt(2 * mu / r)


def circular_velocity(mu: float, r: float) -> float:
    """Circular orbital velocity: v = sqrt(mu / r)"""
    return np.sqrt(mu / r)


def vis_viva(mu: float, r: float, a: float) -> float:
    """Vis-viva equation: v^2 = mu * (2/r - 1/a)"""
    return np.sqrt(mu * (2 / r - 1 / a))


def specific_orbital_energy(mu: float, r: float, v: float) -> float:
    """Specific orbital energy: epsilon = v^2/2 - mu/r"""
    return v**2 / 2 - mu / r


def specific_angular_momentum(r_vec: np.ndarray, v_vec: np.ndarray) -> np.ndarray:
    """Specific angular momentum vector: h = r x v"""
    return np.cross(r_vec, v_vec)


def eccentricity_vector(r_vec: np.ndarray, v_vec: np.ndarray, mu: float) -> np.ndarray:
    """Eccentricity vector: e_vec = (v x h)/mu - r/norm(r)."""
    h = specific_angular_momentum(r_vec, v_vec)
    return np.cross(v_vec, h) / mu - r_vec / np.linalg.norm(r_vec)


def orbital_elements_from_state(r_vec: np.ndarray, v_vec: np.ndarray, mu: float):
    """Compute classical orbital elements from state vectors.
    Returns (a, e, i, RAAN, arg_periapsis, true_anomaly, mean_anomaly)
    a: semi-major axis (m)
    e: eccentricity (scalar)
    i: inclination (rad)
    RAAN: right ascension of ascending node (rad)
    arg_periapsis: argument of periapsis (rad)
    true_anomaly: (rad)
    mean_anomaly: (rad)
    """
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    energy = specific_orbital_energy(mu, r, v)
    if abs(energy) < 1e-12:
        a = np.inf
    else:
        a = -mu / (2 * energy)
    h_vec = specific_angular_momentum(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    # inclination
    i = np.arccos(h_vec[2] / h)
    # node line
    K = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(K, h_vec)
    n = np.linalg.norm(n_vec)
    # eccentricity
    e_vec = eccentricity_vector(r_vec, v_vec, mu)
    e = np.linalg.norm(e_vec)
    # RAAN
    RAAN = 0.0
    if n_vec[1] >= 0:
        RAAN = np.arccos(n_vec[0] / n) if n > 1e-12 else 0.0
    else:
        RAAN = 2 * np.pi - np.arccos(n_vec[0] / n) if n > 1e-12 else 0.0
    # argument of periapsis
    arg_periapsis = 0.0
    if n > 1e-12 and e > 1e-12:
        cos_argp = np.dot(n_vec, e_vec) / (n * e)
        cos_argp = np.clip(cos_argp, -1.0, 1.0)
        if e_vec[2] >= 0:
            arg_periapsis = np.arccos(cos_argp)
        else:
            arg_periapsis = 2 * np.pi - np.arccos(cos_argp)
    # true anomaly
    true_anomaly = 0.0
    if e > 1e-12:
        cos_nu = np.dot(e_vec, r_vec) / (e * r)
        cos_nu = np.clip(cos_nu, -1.0, 1.0)
        if np.dot(r_vec, v_vec) >= 0:
            true_anomaly = np.arccos(cos_nu)
        else:
            true_anomaly = 2 * np.pi - np.arccos(cos_nu)
    else:
        # circular: true anomaly from node line
        if n > 1e-12:
            cos_nu = np.dot(n_vec, r_vec) / (n * r)
            cos_nu = np.clip(cos_nu, -1.0, 1.0)
            if r_vec[2] >= 0:
                true_anomaly = np.arccos(cos_nu)
            else:
                true_anomaly = 2 * np.pi - np.arccos(cos_nu)
    # eccentric anomaly and mean anomaly
    if e < 1.0:
        # elliptical
        E = kepler_E_from_M_from_nu(true_anomaly, e)
        M = E - e * np.sin(E)
    else:
        # hyperbolic or parabolic: mean anomaly not defined simply
        M = None
    return a, e, i, RAAN, arg_periapsis, true_anomaly, M


def kepler_E_from_M(M: float, e: float, tol: float = 1e-8, max_iter: int = 100) -> float:
    """Solve Kepler's equation M = E - e*sin(E) for E (elliptical)"""
    E = M if e < 0.8 else np.pi
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        df = 1 - e * np.cos(E)
        dE = -f / df
        E += dE
        if np.abs(dE) < tol:
            break
    return E


def kepler_E_from_M_from_nu(nu: float, e: float) -> float:
    """Compute eccentric anomaly E from true anomaly nu for elliptical orbits"""
    return 2 * np.arctan2(np.sqrt(1 - e) * np.sin(nu / 2), np.sqrt(1 + e) * np.cos(nu / 2))


def orbital_period(a: float, mu: float) -> Optional[float]:
    """Orbital period for ellipse: T = 2*pi*sqrt(a^3/mu)"""
    if a > 0:
        return 2 * np.pi * np.sqrt(a**3 / mu)
    return None


def hohmann_transfer_delta_v(r1: float, r2: float, mu: float) -> Tuple[float, float, float]:
    """Return (dv1, dv2, total_dv) for Hohmann transfer between circular orbits r1 and r2"""
    a_t = (r1 + r2) / 2.0
    v1 = circular_velocity(mu, r1)
    v_transfer_perigee = vis_viva(mu, r1, a_t)
    dv1 = np.abs(v_transfer_perigee - v1)
    v2 = circular_velocity(mu, r2)
    v_transfer_apogee = vis_viva(mu, r2, a_t)
    dv2 = np.abs(v2 - v_transfer_apogee)
    return dv1, dv2, dv1 + dv2


def bi_elliptic_transfer_delta_v(r1: float, r2: float, rb: float, mu: float) -> Tuple[float, float, float, float]:
    """Return delta-v components for a bi-elliptic transfer via apoapsis ``rb``."""
    if r1 <= 0 or r2 <= 0 or rb <= 0:
        raise ValueError("r1, r2, and rb must be positive radii")
    if rb <= max(r1, r2):
        raise ValueError("rb must be larger than both circular-orbit radii")

    a1 = (r1 + rb) / 2.0
    a2 = (r2 + rb) / 2.0
    v1 = circular_velocity(mu, r1)
    v_transfer1 = vis_viva(mu, r1, a1)
    dv1 = np.abs(v_transfer1 - v1)
    v_transfer2 = vis_viva(mu, rb, a1)
    v_transfer3 = vis_viva(mu, rb, a2)
    dv2 = np.abs(v_transfer3 - v_transfer2)
    v2 = circular_velocity(mu, r2)
    v_transfer4 = vis_viva(mu, r2, a2)
    dv3 = np.abs(v2 - v_transfer4)
    return dv1, dv2, dv3, dv1 + dv2 + dv3


def plane_change_delta_v(v: float, i1: float, i2: float) -> float:
    """Delta-v for a plane change: ``2 * v * sin(abs(i2 - i1) / 2)``."""
    return 2 * v * np.abs(np.sin((i2 - i1) / 2.0))


def sphere_of_influence_radius(a: float, m_secondary: float, m_primary: float) -> float:
    """Sphere of influence radius: r_SOI = a*(m_secondary/m_primary)^(2/5)"""
    return a * (m_secondary / m_primary) ** (2.0 / 5.0)
