"""
core/frames.py
──────────────
Reference-frame transforms for trajectories and vectors.

Supported frames
----------------
ECI   Earth-Centred Inertial  (J2000 / GCRF) — the native SSAPy frame
ECF   Earth-Centred Fixed     (rotates with Earth at OMEGA_E)
LVLH  Local Vertical / Local Horizontal  (RSW: R along r, W = h-hat, S = W×R)
RTN   Radial–Transverse–Normal  (same as LVLH / RSW — alias)
NTW   Normal–Transverse–W       (N = T×W, T along v, W = orbit normal)

Usage
-----
from ssatk.coordinates.frames import FrameTransform, Frame

tf = FrameTransform(Frame.LVLH)
r_lvlh = tf.transform_points(r_eci, v_eci)

# Or transform an entire Trajectory
traj_lvlh = tf.transform_trajectory(traj, ref_state)
"""

from __future__ import annotations

from enum import Enum
import numpy as np

from ssatk.constants import WGS84_EARTH_OMEGA

# Sourced from ssatk.constants rather than a hardcoded literal --
# this was independently duplicated (at the same value, by coincidence)
# in orbit_state.py.
OMEGA_E = WGS84_EARTH_OMEGA   # rad/s Earth rotation rate

__all__ = [
    "Frame",
    "FrameTransform",
    "eci_to_ecf_matrix",
    "greenwich_azimuth_rad",
    "eci_to_lon_lat",
    "lvlh_axes",
    "lvlh_matrix",
    "ntw_axes",
    "ntw_matrix",
]


# ── Frame enum ───────────────────────────────────────────────────────────────
class Frame(str, Enum):
    ECI  = "ECI"
    ECF  = "ECF"
    LVLH = "LVLH"
    RTN  = "RTN"   # alias for LVLH
    NTW  = "NTW"

    @property
    def label(self) -> str:
        return {
            "ECI":  "Earth-Centred Inertial (J2000)",
            "ECF":  "Earth-Centred Fixed (rotating)",
            "LVLH": "Local Vertical / Local Horizontal",
            "RTN":  "Radial–Transverse–Normal",
            "NTW":  "Normal–Transverse–W (velocity-aligned)",
        }[self.value]


# ── helper: unit vector ───────────────────────────────────────────────────────
def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-15 else v


# ── rotation matrices ─────────────────────────────────────────────────────────
def eci_to_ecf_matrix(t_gps) -> np.ndarray:
    """
    GCRF -> ITRF rotation matrix at GPS time(s) ``t_gps``.

    Same chain as :func:`ssapy.compute.groundTrack` and
    ``ssapy.EarthObserver``: IAU 1976/1980 precession-nutation (``erfa.pnm80``
    at TT), Greenwich apparent sidereal time (``erfa.gst94`` at UT1 from the
    IERS table), then polar motion. Returns ``(3, 3)`` for a scalar time and
    ``(N, 3, 3)`` for an array, with ``r_itrf = M @ r_gcrf``.

    This previously rotated by Greenwich *mean* sidereal time alone. Applied to
    GCRF vectors that omits precession and nutation since J2000: on
    2026-10-09 it put an equatorial point 1235 arcsec (38.2 km) west and
    540 arcsec south of astropy's GCRS->ITRS result.
    """
    import erfa
    from ssapy.utils import iers_interp

    t = np.asarray(t_gps, dtype=float)
    scalar = t.ndim == 0
    t = np.atleast_1d(t)
    mjd_tt = 44244.0 + (t + 51.184) / 86400.0          # GPS -> TT, as MJD
    d_ut1_tt_mjd, pmx, pmy = iers_interp(t)
    pn = erfa.pnm80(2400000.5, mjd_tt)
    gst = erfa.gst94(2400000.5, mjd_tt + d_ut1_tt_mjd)

    cg, sg = np.cos(gst), np.sin(gst)
    r3 = np.zeros((t.size, 3, 3))
    r3[:, 0, 0] = cg
    r3[:, 0, 1] = sg
    r3[:, 1, 0] = -sg
    r3[:, 1, 1] = cg
    r3[:, 2, 2] = 1.0

    polar = np.broadcast_to(np.eye(3), (t.size, 3, 3)).copy()
    polar[:, 0, 2] = pmx
    polar[:, 1, 2] = -pmy
    polar[:, 2, 0] = -pmx
    polar[:, 2, 1] = pmy

    m = polar @ r3 @ pn
    return m[0] if scalar else m


def greenwich_azimuth_rad(t_gps):
    """Right ascension (rad) of the ITRF x-axis (Greenwich meridian) in GCRF.

    This is the single angle to turn an Earth texture about GCRF +z so that
    the prime meridian lands where :func:`eci_to_ecf_matrix` puts it. A pure
    z-rotation cannot also reproduce the tilt of the true pole from GCRF +z
    (0.15 deg in 2026), so latitudes on such a texture stay off by up to that
    amount; use :func:`eci_to_ecf_matrix` directly when that matters.
    """
    m = eci_to_ecf_matrix(t_gps)
    return np.arctan2(m[..., 0, 1], m[..., 0, 0])


def lvlh_matrix(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    3×3 matrix whose rows are [R_hat, S_hat, W_hat] in ECI.
    R = r/``|r|``  (radial),  W = h/``|h|``  (orbit normal),  S = W × R
    Transforms ECI → LVLH/RSW.
    """
    R_hat = _unit(r)
    W_hat = _unit(np.cross(r, v))
    S_hat = np.cross(W_hat, R_hat)
    return np.array([R_hat, S_hat, W_hat])


def ntw_matrix(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    3×3 matrix whose rows are [N_hat, T_hat, W_hat] in ECI.
    T = v/``|v|``  (tangential / along-track),
    W = h/``|h|``  (orbit normal),
    N = T × W  (in-plane, radial for circular prograde equatorial orbits).
    Transforms ECI → NTW.

    The component ordering intentionally matches SSAPy's AccelConstNTW and
    ssatk.coordinates.satellite_frames convention: [N, T, W].
    """
    T_hat = _unit(v)
    W_hat = _unit(np.cross(r, v))
    N_hat = np.cross(T_hat, W_hat)
    return np.array([N_hat, T_hat, W_hat])


def ntw_axes(r: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (T_hat, N_hat, W_hat) as separate ECI unit vectors.

    This helper returns named plotting axes rather than NTW component order.
    Use :func:`ntw_matrix` for canonical SSAPy [N, T, W] components.
    """
    T = _unit(v)
    W = _unit(np.cross(r, v))
    N = np.cross(T, W)
    return T, N, W


def lvlh_axes(r: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (R_hat, S_hat, W_hat) as separate ECI unit vectors."""
    R = _unit(r)
    W = _unit(np.cross(r, v))
    S = np.cross(W, R)
    return R, S, W


# ── FrameTransform ────────────────────────────────────────────────────────────
class FrameTransform:
    """
    Transform position arrays (and optionally velocity arrays) between frames.

    All inputs/outputs in km (positions) or km/s (velocities).

    Parameters
    ----------
    frame : Frame
        Target frame.
    t_gps : float | None
        GPS epoch for ECF rotation.  Required if frame == ECF.
    """

    def __init__(self, frame: Frame | str, t_gps: float | None = None):
        self.frame = Frame(frame)
        self.t_gps = t_gps

    # ── single-point transform ────────────────────────────────────────────────
    def transform_point(
        self,
        r_eci : np.ndarray,
        v_eci : np.ndarray | None = None,
        t_gps : float | None = None,
    ) -> np.ndarray:
        """
        Transform a single ECI position into the target frame.

        r_eci : (3,) km
        v_eci : (3,) km/s  — required for LVLH / RTN / NTW
        t_gps : GPS seconds — required for ECF (overrides self.t_gps)
        """
        r = np.asarray(r_eci, dtype=float)
        v = np.asarray(v_eci, dtype=float) if v_eci is not None else None

        if self.frame == Frame.ECI:
            return r.copy()

        if self.frame == Frame.ECF:
            tg = t_gps or self.t_gps or 0.0
            return eci_to_ecf_matrix(tg) @ r

        if self.frame in (Frame.LVLH, Frame.RTN):
            if v is None:
                raise ValueError("velocity required for LVLH/RTN transform")
            return lvlh_matrix(r, v) @ r

        if self.frame == Frame.NTW:
            if v is None:
                raise ValueError("velocity required for NTW transform")
            return ntw_matrix(r, v) @ r

        raise ValueError(f"Unknown frame: {self.frame}")

    def transform_vector(
        self,
        vec   : np.ndarray,
        r_eci : np.ndarray,
        v_eci : np.ndarray | None = None,
        t_gps : float | None = None,
    ) -> np.ndarray:
        """
        Rotate an arbitrary ECI vector into the target frame.
        (Same rotation as transform_point but without the implied meaning of position.)
        """
        return self.transform_point(vec, v_eci=v_eci, t_gps=t_gps) if self.frame == Frame.ECF \
               else self._rotation_matrix(r_eci, v_eci, t_gps) @ vec

    def _rotation_matrix(self, r, v, t_gps=None) -> np.ndarray:
        if self.frame == Frame.ECI:
            return np.eye(3)
        if self.frame == Frame.ECF:
            return eci_to_ecf_matrix(t_gps or self.t_gps or 0.0)
        if self.frame in (Frame.LVLH, Frame.RTN):
            return lvlh_matrix(r, v)
        if self.frame == Frame.NTW:
            return ntw_matrix(r, v)
        return np.eye(3)

    # ── trajectory transform ──────────────────────────────────────────────────
    def transform_trajectory(
        self,
        r_eci : np.ndarray,
        v_eci : np.ndarray,
        t_gps : np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Transform an (N, 3) ECI position array into the target frame.

        Parameters
        ----------
        r_eci : (N, 3) km
        v_eci : (N, 3) km/s
        t_gps : (N,) GPS seconds — required for ECF

        Returns
        -------
        (N, 3) positions in target frame, km
        """
        r = np.asarray(r_eci, dtype=float)
        v = np.asarray(v_eci, dtype=float)
        N = r.shape[0]
        out = np.empty_like(r)

        if self.frame == Frame.ECI:
            return r.copy()

        if self.frame == Frame.ECF:
            if t_gps is None:
                raise ValueError("t_gps required for ECF transform")
            for i in range(N):
                out[i] = eci_to_ecf_matrix(t_gps[i]) @ r[i]
            return out

        if self.frame in (Frame.LVLH, Frame.RTN):
            for i in range(N):
                M = lvlh_matrix(r[i], v[i])
                out[i] = M @ r[i]
            return out

        if self.frame == Frame.NTW:
            for i in range(N):
                M = ntw_matrix(r[i], v[i])
                out[i] = M @ r[i]
            return out

        raise ValueError(f"Unknown frame: {self.frame}")

    # ── convenience: relative trajectory (centred on first point) ─────────────
    def relative_trajectory(
        self,
        r_eci : np.ndarray,
        v_eci : np.ndarray,
        t_gps : np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Transform trajectory and subtract first point — useful for
        LVLH plots centred on the reference spacecraft.
        """
        out = self.transform_trajectory(r_eci, v_eci, t_gps)
        return out - out[0]


# ── Convenience functions ─────────────────────────────────────────────────────
def eci_to_lon_lat(r_eci_km: np.ndarray, t_gps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert (N,3) GCRF positions -> (lon_deg, geocentric lat_deg) sub-satellite point.
    Rotates with :func:`eci_to_ecf_matrix`, then uses spherical geometry.
    """
    r = np.atleast_2d(np.asarray(r_eci_km, dtype=float))
    t = np.broadcast_to(np.asarray(t_gps, dtype=float), (len(r),))
    r_ecf = np.einsum("nij,nj->ni", np.atleast_3d(eci_to_ecf_matrix(t)).reshape(len(r), 3, 3), r)
    lon = np.degrees(np.arctan2(r_ecf[:, 1], r_ecf[:, 0]))
    lat = np.degrees(np.arcsin(r_ecf[:, 2] / np.linalg.norm(r_ecf, axis=1)))
    return lon, lat
