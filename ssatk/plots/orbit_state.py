"""
core/orbit_state.py
───────────────────
Central OrbitalState object.  Everything in the toolkit starts here.

Usage
-----
::

    state = OrbitalState(a_km=6928, e=0.001, inc_deg=51.6)
    traj  = state.propagate(n_orbits=3, dt_s=60)
    print(state.regime, state.period_hr, state.j2_raan_drift_deg_day)

Force-model customisation
-------------------------
::

    cfg = PropagatorConfig(
        propagator  = "rk78",
        gravity     = "8x8",
        third_body  = "both",
        non_grav    = "drag",
    )
    state = OrbitalState(..., config=cfg)
    traj  = state.propagate(n_orbits=5, dt_s=30, callback=my_fn)
"""

from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, NamedTuple, Optional

import numpy as np

from ssatk.constants import EARTH_MU_KM3_S2, EARTH_RADIUS_KM, J2_wgs, LD_KM, WGS84_EARTH_OMEGA
from .plotutils import normalize_orbit_trajectory

# ── Optional SSAPy ────────────────────────────────────────────────────────────
try:
    import ssapy
    import ssapy.compute
    from astropy.time import Time
    _HAS_SSAPY = True
except ImportError:
    _HAS_SSAPY = False
    warnings.warn("SSAPy not found — analytic Keplerian fallback active.", stacklevel=2)

# ── Constants ─────────────────────────────────────────────────────────────────
# RE_KM, OMEGA_E, AU_KM, MOON_A_KM sourced from ssatk.constants /
# astropy (same pattern as sun_view.py) rather than hardcoded literals --
# these used to be independently duplicated across several files in this
# package (layers.py, frames.py, groundtrack_enhanced.py), all at the same
# value by coincidence rather than by having one shared source.
RE_KM      = EARTH_RADIUS_KM   # Earth equatorial radius (WGS84), km
MU         = EARTH_MU_KM3_S2   # GM Earth, km³ s⁻²
J2         = J2_wgs            # Earth J2
OMEGA_E    = WGS84_EARTH_OMEGA        # Earth rotation rate, rad s⁻¹
MOON_A_KM  = LD_KM


# ─── PropagatorConfig ────────────────────────────────────────────────────────
@dataclass
class PropagatorConfig:
    """
    All force-model and integrator knobs in one place.

    propagator : "keplerian" | "scipy" | "rk4" | "rk78"
    gravity    : "point_mass" | "j2" | "4x4" | "8x8"
    third_body : None | "moon" | "sun" | "both"
    non_grav   : None | "drag" | "srp" | "both"

    Drag parameters (only used when non_grav includes "drag"):
      cd, area_m2, mass_kg

    SRP parameters (only used when non_grav includes "srp"):
      cr, area_m2, mass_kg
    """
    propagator : str   = "keplerian"
    gravity    : str   = "j2"
    third_body : Optional[str] = None
    non_grav   : Optional[str] = None

    # Drag
    cd      : float = 2.2
    area_m2 : float = 10.0
    mass_kg : float = 500.0

    # SRP
    cr      : float = 1.3

    def label(self) -> str:
        parts = [self.propagator.upper(), self.gravity]
        if self.third_body:
            parts.append(self.third_body)
        if self.non_grav:
            parts.append(self.non_grav)
        return " + ".join(parts)


# ─── Trajectory ──────────────────────────────────────────────────────────────
class Trajectory(NamedTuple):
    """Propagated trajectory arrays — all in ECI, km / km·s⁻¹."""
    r   : np.ndarray  # (N, 3) position, km
    v   : np.ndarray  # (N, 3) velocity, km/s
    t   : np.ndarray  # (N,)   GPS seconds
    ok  : bool = True
    msg : str  = ""


# ─── Preset library ──────────────────────────────────────────────────────────
# NOTE on the two cislunar-scale entries below ("Cislunar Test Orbit" and
# "Cislunar Transfer Test Orbit"): these are coarse two-body Keplerian
# stand-ins used only to validate that a plot/propagator can handle
# cislunar-scale distances. They are NOT real mission trajectories or
# real libration-point orbits:
#
#   - A genuine Earth-Moon L1 halo orbit is a solution to the Circular
#     Restricted Three-Body Problem (CR3BP), which accounts for the
#     Earth's and Moon's gravity simultaneously. It fundamentally cannot
#     be represented by classical Keplerian elements (a, e, i, RAAN,
#     argp, nu) at all — there is no such thing as "a halo orbit's
#     semi-major axis." Propagating one of these elements sets with a
#     two-body propagator and viewing the result in a frame that rotates
#     with the Moon produces a spiral/loop artifact that has no
#     resemblance to a real halo orbit's shape.
#
#   - A real Artemis-style mission (Earth orbit -> translunar injection
#     -> lunar flyby/orbit -> return) changes orbital regime partway
#     through and likewise cannot be represented by one set of classical
#     elements. See ssatk/plots/artemis_profile_demo.py for a
#     stitched multi-phase approximation, or
#     ssatk/plots/artemis1_horizons_plot.py for REAL Artemis I
#     trajectory data pulled from JPL Horizons.
PRESETS: dict[str, dict] = {
    "ISS (LEO)":            dict(a_km=RE_KM+408,   e=3e-4, inc_deg=51.6, raan_deg=0,   argp_deg=0,   nu_deg=0),
    "Starlink (LEO)":       dict(a_km=RE_KM+550,   e=1e-4, inc_deg=53.0, raan_deg=0,   argp_deg=0,   nu_deg=0),
    "Sun-Sync (SSO)":       dict(a_km=RE_KM+700,   e=1e-3, inc_deg=98.2, raan_deg=0,   argp_deg=0,   nu_deg=0),
    "GPS (MEO)":            dict(a_km=26_560,       e=1e-2, inc_deg=55.0, raan_deg=0,   argp_deg=0,   nu_deg=0),
    "GEO":                  dict(a_km=42_164,       e=1e-4, inc_deg=0.05, raan_deg=0,   argp_deg=0,   nu_deg=0),
    "Molniya (HEO)":        dict(a_km=26_560,       e=0.72, inc_deg=63.4, raan_deg=0,   argp_deg=270, nu_deg=0),
    "Tundra":               dict(a_km=42_164,       e=0.26, inc_deg=63.4, raan_deg=0,   argp_deg=270, nu_deg=0),
    "GTO":                  dict(a_km=(RE_KM+250+42_164)/2, e=0.73, inc_deg=27.0, raan_deg=0, argp_deg=180, nu_deg=0),
    "Cislunar Test Orbit":           dict(a_km=MOON_A_KM*0.85, e=0.03, inc_deg=5.1, raan_deg=0, argp_deg=0, nu_deg=0),
    "Cislunar Transfer Test Orbit":  dict(a_km=RE_KM+400_000, e=0.97, inc_deg=90,  raan_deg=0,  argp_deg=180, nu_deg=0),
}

# ─── OrbitalState ────────────────────────────────────────────────────────────
class OrbitalState:
    """
    Keplerian elements + propagation configuration + derived quantities.

    Parameters
    ----------
    a_km, e, inc_deg, raan_deg, argp_deg, nu_deg : float
        Classical orbital elements.
    epoch : datetime | str
        UTC epoch.  Defaults to now.
    config : PropagatorConfig
        Force-model configuration.  Defaults to Keplerian.

    Properties (computed on demand, cached)
    ----------------------------------------
    period_s, period_min, period_hr
    h_a_km, h_p_km  (apoapsis / periapsis altitude)
    v_a, v_p        (km/s at apo/peri)
    regime          "LEO" | "MEO" | "GEO" | "HEO" | "CISLUNAR"
    j2_raan_drift_deg_day   secular nodal regression
    j2_argp_drift_deg_day   secular apsidal advance

    Propagation, conversion, and cloning methods are documented below.
    """

    # ── construction ─────────────────────────────────────────────────────────
    def __init__(
        self,
        a_km     : float = RE_KM + 550,
        e        : float = 0.001,
        inc_deg  : float = 51.6,
        raan_deg : float = 0.0,
        argp_deg : float = 0.0,
        nu_deg   : float = 0.0,
        epoch    : datetime | str | None = None,
        config   : PropagatorConfig | None = None,
        name     : str = "Orbit",
    ):
        self.a_km     = float(a_km)
        self.e        = float(e)
        self.inc_deg  = float(inc_deg)
        self.raan_deg = float(raan_deg)
        self.argp_deg = float(argp_deg)
        self.nu_deg   = float(nu_deg)
        self.name     = name
        self.config   = config or PropagatorConfig()

        if epoch is None:
            self.epoch = datetime.now(timezone.utc)
        elif isinstance(epoch, str):
            self.epoch = datetime.fromisoformat(epoch).replace(tzinfo=timezone.utc)
        else:
            self.epoch = epoch

        self._cache: dict = {}

    # ── cache invalidation ────────────────────────────────────────────────────
    def _invalidate(self):
        self._cache.clear()

    def _c(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    # ── element setters (invalidate cache) ───────────────────────────────────
    def set_elements(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, float(v))
        self._invalidate()

    # ── basic derived quantities ──────────────────────────────────────────────
    @property
    def r_a(self): return self.a_km * (1 + self.e)   # apoapsis radius km
    @property
    def r_p(self): return self.a_km * (1 - self.e)   # periapsis radius km
    @property
    def h_a_km(self): return self.r_a - RE_KM
    @property
    def h_p_km(self): return self.r_p - RE_KM

    @property
    def period_s(self):
        return self._c("T", lambda: 2*np.pi*np.sqrt(self.a_km**3 / MU))
    @property
    def period_min(self): return self.period_s / 60
    @property
    def period_hr(self):  return self.period_s / 3600

    @property
    def v_p(self):
        return np.sqrt(MU * (2/self.r_p - 1/self.a_km)) if self.r_p > 0 else 0.0
    @property
    def v_a(self):
        return np.sqrt(MU * (2/self.r_a - 1/self.a_km)) if self.r_a > 0 else 0.0
    @property
    def v_circ(self):
        return np.sqrt(MU / self.a_km)

    @property
    def specific_energy(self): return -MU / (2 * self.a_km)
    @property
    def specific_angular_momentum(self):
        p = self.a_km * (1 - self.e**2)
        return np.sqrt(MU * p)

    @property
    def regime(self) -> str:
        ha = self.h_a_km
        if ha < 2_000:               return "LEO"
        if ha < 35_000:              return "MEO"
        if abs(ha - 35_786) < 750:   return "GEO"
        if self.a_km > MOON_A_KM*0.4: return "CISLUNAR"
        return "HEO"

    # ── J2 secular drift rates ────────────────────────────────────────────────
    @property
    def j2_raan_drift_deg_day(self) -> float:
        """Secular nodal regression due to J2, deg/day."""
        n   = 2*np.pi / self.period_s
        p   = self.a_km * (1 - self.e**2)
        cos_i = np.cos(np.radians(self.inc_deg))
        rate_rad_s = -1.5 * n * J2 * (RE_KM/p)**2 * cos_i
        return np.degrees(rate_rad_s) * 86_400

    @property
    def j2_argp_drift_deg_day(self) -> float:
        """Secular apsidal advance due to J2, deg/day."""
        n   = 2*np.pi / self.period_s
        p   = self.a_km * (1 - self.e**2)
        sin2_i = np.sin(np.radians(self.inc_deg))**2
        rate_rad_s = 0.75 * n * J2 * (RE_KM/p)**2 * (5*sin2_i - 4)
        return np.degrees(rate_rad_s) * 86_400

    # ── warnings ─────────────────────────────────────────────────────────────
    @property
    def warnings(self) -> list[str]:
        w = []
        if self.h_p_km < 0:      w.append("✖ Periapsis below Earth surface")
        elif self.h_p_km < 150:  w.append("⚠ Periapsis <150 km — re-entry likely")
        if self.e >= 1.0:        w.append("✖ e ≥ 1.0 — hyperbolic trajectory")
        if self.e < 0:           w.append("✖ Negative eccentricity")
        if self.inc_deg < 0 or self.inc_deg > 180:
            w.append("⚠ Inclination out of [0, 180]°")
        return w

    # ── elements → ECI state vector ──────────────────────────────────────────
    def to_rv(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (r_km, v_km_s) ECI at true anomaly nu."""
        e   = self.e
        a   = self.a_km
        inc = np.radians(self.inc_deg)
        W   = np.radians(self.raan_deg)
        w   = np.radians(self.argp_deg)
        nu  = np.radians(self.nu_deg)

        p   = a * (1 - e**2)
        r   = p / (1 + e * np.cos(nu))

        # perifocal frame
        r_pf = r * np.array([np.cos(nu), np.sin(nu), 0.0])
        v_pf = np.sqrt(MU/p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

        # rotation matrix perifocal → ECI
        cW, sW = np.cos(W), np.sin(W)
        ci, si = np.cos(inc), np.sin(inc)
        cw, sw = np.cos(w), np.sin(w)

        R = np.array([
            [cW*cw - sW*sw*ci, -cW*sw - sW*cw*ci,  sW*si],
            [sW*cw + cW*sw*ci, -sW*sw + cW*cw*ci, -cW*si],
            [sw*si,             cw*si,              ci   ],
        ])
        return R @ r_pf, R @ v_pf

    # ── osculating ellipse (analytic, instant) ────────────────────────────────
    def osculating_ellipse(self, n_pts: int = 360) -> np.ndarray:
        """
        Return (n_pts, 3) array tracing the osculating ellipse in ECI.
        Fast — no propagator.
        """
        nus = np.linspace(0, 2*np.pi, n_pts)
        e   = self.e
        a   = self.a_km
        inc = np.radians(self.inc_deg)
        W   = np.radians(self.raan_deg)
        w   = np.radians(self.argp_deg)
        p   = a * (1 - e**2)

        cW, sW = np.cos(W), np.sin(W)
        ci, si = np.cos(inc), np.sin(inc)
        cw, sw = np.cos(w), np.sin(w)

        R = np.array([
            [cW*cw - sW*sw*ci, -cW*sw - sW*cw*ci,  sW*si],
            [sW*cw + cW*sw*ci, -sW*sw + cW*cw*ci, -cW*si],
            [sw*si,             cw*si,              ci   ],
        ])
        pts = []
        for nu in nus:
            r_mag = p / (1 + e*np.cos(nu))
            r_pf  = r_mag * np.array([np.cos(nu), np.sin(nu), 0.0])
            pts.append(R @ r_pf)
        return np.array(pts)

    # ── GPS epoch helper ─────────────────────────────────────────────────────
    def _epoch_gps(self) -> float:
        """Convert self.epoch to GPS seconds (SSAPy native time)."""
        if _HAS_SSAPY:
            try:
                from astropy.time import Time as AstroTime
                t = AstroTime(self.epoch)
                return t.gps
            except Exception:
                pass
        # fallback: approximate GPS = unix + 18 leap-seconds offset to J2000
        gps_epoch = datetime(1980, 1, 6, tzinfo=timezone.utc)
        return (self.epoch - gps_epoch).total_seconds()

    # ── SSAPy orbit object ───────────────────────────────────────────────────
    def to_ssapy(self):
        """Return ssapy.Orbit at self.epoch.  Raises if SSAPy unavailable."""
        if not _HAS_SSAPY:
            raise RuntimeError("SSAPy not installed")
        r, v = self.to_rv()
        t0 = self._epoch_gps()
        return ssapy.Orbit(r=r * 1e3, v=v * 1e3, t=t0)  # SSAPy uses metres

    # ── build SSAPy propagator + accelerations ───────────────────────────────
    def _build_propagator(self):
        """Return (propagator_obj, accel_list) for use with ssapy.rv."""
        cfg = self.config
        accels = []

        # Use the correct SSAPy API throughout — verified against SSAPy docs:
        # AccelKepler(mu), AccelHarmonic(body, n, m), AccelThirdBody(get_body(...))
        # AccelJ2(), AccelKepler() (no args), AccelHarmonic(n,m) (no body) do NOT exist.
        from ssapy.gravity import AccelHarmonic, AccelThirdBody
        from ssapy.body import get_body as _get_body
        from ssapy.accel import AccelKepler
        earth = _get_body("earth")

        # ── gravity ──────────────────────────────────────────────────────
        if cfg.gravity == "point_mass":
            accels.append(AccelKepler(earth.mu))
        elif cfg.gravity == "j2":
            accels.append(AccelKepler(earth.mu))
            accels.append(AccelHarmonic(earth, 2, 0))   # J2 zonal term
        elif cfg.gravity == "4x4":
            accels.append(AccelKepler(earth.mu))
            accels.append(AccelHarmonic(earth, 4, 4))
        elif cfg.gravity == "8x8":
            accels.append(AccelKepler(earth.mu))
            accels.append(AccelHarmonic(earth, 8, 8))

        # ── third body ───────────────────────────────────────────────────
        # MUST use get_body("moon"/"sun") — these carry the real positional
        # ephemeris. A bare Body(mu=...) has no position data, defaulting
        # to (0,0,0) which produces a huge spurious acceleration at t=t0.
        if cfg.third_body in ("moon", "both"):
            accels.append(AccelThirdBody(_get_body("moon")))
        if cfg.third_body in ("sun", "both"):
            accels.append(AccelThirdBody(_get_body("sun")))

        # ── non-gravitational ─────────────────────────────────────────────
        if cfg.non_grav in ("drag", "both"):
            accels.append(ssapy.AccelDrag(
                cd=cfg.cd, area=cfg.area_m2, mass=cfg.mass_kg,
            ))
        if cfg.non_grav in ("srp", "both"):
            accels.append(ssapy.AccelSRP(
                cr=cfg.cr, area=cfg.area_m2, mass=cfg.mass_kg,
            ))

        # ── integrator ───────────────────────────────────────────────────
        # SSAPy's propagator expects a single Accel object, not a bare list.
        # Wrap multiple accelerations in AccelSum first.
        from ssapy.accel import AccelSum
        accel_obj = AccelSum(accels) if len(accels) > 1 else (accels[0] if accels else None)

        if cfg.propagator == "keplerian" or not accels:
            return ssapy.propagator.KeplerianPropagator(), []
        elif cfg.propagator == "scipy":
            return ssapy.propagator.SciPyPropagator(accel_obj), accels
        elif cfg.propagator == "rk4":
            dt = min(60.0, self.period_s / 360)
            return ssapy.propagator.RK4Propagator(accel_obj, h=dt), accels
        else:  # rk78
            return ssapy.propagator.RK78Propagator(accel_obj), accels

    # ── analytic Keplerian propagation (no SSAPy) ────────────────────────────
    def _propagate_analytic(self, times_s: np.ndarray) -> Trajectory:
        """Pure-numpy two-body propagation using universal variables."""
        r0, v0 = self.to_rv()
        r0 *= 1e3; v0 *= 1e3  # work in metres
        mu = MU * 1e9

        rs, vs = [], []
        for dt in times_s:
            # Kepler's equation via eccentric anomaly
            n   = np.sqrt(mu / (self.a_km * 1e3)**3)
            M0  = self._nu_to_M(np.radians(self.nu_deg))
            M   = M0 + n * dt
            E   = self._solve_kepler(M, self.e)
            nu  = 2 * np.arctan2(
                np.sqrt(1+self.e)*np.sin(E/2),
                np.sqrt(1-self.e)*np.cos(E/2),
            )
            # rebuild state at new nu
            old_nu = self.nu_deg
            self.nu_deg = np.degrees(nu)
            r, v = self.to_rv()
            rs.append(r); vs.append(v)
            self.nu_deg = old_nu

        return Trajectory(
            r=np.array(rs), v=np.array(vs),
            t=self._epoch_gps() + times_s,
        )

    @staticmethod
    def _nu_to_M(nu: float, e: float = 0.0) -> float:
        E = 2*np.arctan2(np.sqrt(1-e)*np.sin(nu/2), np.sqrt(1+e)*np.cos(nu/2))
        return E - e*np.sin(E)

    @staticmethod
    def _solve_kepler(M: float, e: float, tol: float = 1e-10) -> float:
        E = M if e < 0.8 else np.pi
        for _ in range(50):
            dE = (M - E + e*np.sin(E)) / (1 - e*np.cos(E))
            E += dE
            if abs(dE) < tol:
                break
        return E

    # ── propagate ────────────────────────────────────────────────────────────
    def propagate(
        self,
        n_orbits    : float = 3.0,
        dt_s        : float = 60.0,
        callback    : Callable[[Trajectory], None] | None = None,
        stop_event  : threading.Event | None = None,
    ) -> Trajectory:
        """
        Propagate the orbit.

        Parameters
        ----------
        n_orbits   : number of orbital periods to propagate
        dt_s       : time step, seconds
        callback   : called with completed Trajectory when done
        stop_event : threading.Event — set to abort early

        Returns
        -------
        Trajectory  (or partial trajectory if stopped early)
        """
        invalid_messages = [
            message
            for message in self.warnings
            if (
                "e ≥ 1.0" in message
                or "Negative eccentricity" in message
                or "Inclination out of [0, 180]" in message
            )
        ]
        if invalid_messages:
            t = Trajectory(r=np.zeros((1, 3)), v=np.zeros((1, 3)),
                           t=np.array([self._epoch_gps()]),
                           ok=False, msg="; ".join(invalid_messages))
            if callback:
                callback(t)
            return t

        # Guard: periapsis below surface → analytic only with warning
        if self.h_p_km < 0:
            t = Trajectory(r=np.zeros((1, 3)), v=np.zeros((1, 3)),
                           t=np.array([self._epoch_gps()]),
                           ok=False, msg="Periapsis below Earth surface")
            if callback:
                callback(t)
            return t

        duration_s = n_orbits * self.period_s
        times = np.arange(0, duration_s + dt_s, dt_s)
        t0    = self._epoch_gps()

        # ── SSAPy path ───────────────────────────────────────────────────
        if _HAS_SSAPY and self.config.propagator != "keplerian":
            try:
                orbit = self.to_ssapy()
                prop, accels = self._build_propagator()
                t_arr  = t0 + times
                r_m, v_m = ssapy.rv(orbit, t_arr, propagator=prop)
                # SSAPy returns metres — convert to km
                r_km = r_m / 1e3
                v_km = v_m / 1e3
                traj = Trajectory(r=r_km, v=v_km, t=t_arr)
                if callback:
                    callback(traj)
                return traj
            except Exception as ex:
                warnings.warn(f"SSAPy propagation failed: {ex} — using analytic fallback")

        # ── analytic fallback ─────────────────────────────────────────────
        traj = self._propagate_analytic(times)
        if callback:
            callback(traj)
        return traj

    def propagate_async(
        self,
        n_orbits   : float = 3.0,
        dt_s       : float = 60.0,
        on_done    : Callable[[Trajectory], None] | None = None,
        on_error   : Callable[[Exception], None] | None = None,
    ) -> tuple[threading.Thread, threading.Event]:
        """
        Run propagate() on a background thread.

        Returns (thread, stop_event).
        Set stop_event to interrupt propagation.
        """
        stop = threading.Event()

        def _worker():
            try:
                traj = self.propagate(n_orbits=n_orbits, dt_s=dt_s,
                                      stop_event=stop)
                if on_done and not stop.is_set():
                    on_done(traj)
            except Exception as ex:
                if on_error:
                    on_error(ex)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        return t, stop

    # ── alternate constructors ────────────────────────────────────────────────
    @classmethod
    def from_preset(cls, name: str, **kwargs) -> "OrbitalState":
        if name in ("Cislunar Test Orbit", "Cislunar Transfer Test Orbit"):
            warnings.warn(
                f"Preset '{name}' is a two-body Keplerian "
                f"approximation, not a real CR3BP libration-point orbit or "
                f"real mission trajectory. See the PRESETS comment in this "
                f"module for details, or use "
                f"ssatk/plots/artemis1_horizons_plot.py for real "
                f"Artemis I data.",
                stacklevel=2,
            )

        if name not in PRESETS:
            raise KeyError(f"Unknown preset '{name}'. Available: {list(PRESETS)}")
        return cls(**{**PRESETS[name], **kwargs})

    @classmethod
    def from_tle(cls, tle_text: str, **kwargs) -> "OrbitalState":
        """Parse a 2- or 3-line TLE into an OrbitalState (mean → osculating approx)."""
        lines = [l.strip() for l in tle_text.strip().splitlines() if l.strip()]
        if len(lines) == 3:
            _, l1, l2 = lines
        elif len(lines) == 2:
            l1, l2 = lines
        else:
            raise ValueError("Expected 2 or 3 TLE lines")

        e_raw  = float("0." + l2[26:33].strip())
        n      = float(l2[52:63])          # rev/day
        a_km   = (MU / (n * 2*np.pi/86_400)**2) ** (1/3)
        inc    = float(l2[8:16])
        raan   = float(l2[17:25])
        argp   = float(l2[34:42])
        M_deg  = float(l2[43:51])
        # M → nu (low-e approximation)
        M = np.radians(M_deg)
        E = cls._solve_kepler(M, e_raw)
        nu = np.degrees(2*np.arctan2(np.sqrt(1+e_raw)*np.sin(E/2),
                                      np.sqrt(1-e_raw)*np.cos(E/2)))
        return cls(a_km=a_km, e=e_raw, inc_deg=inc, raan_deg=raan,
                   argp_deg=argp, nu_deg=nu % 360, **kwargs)

    @classmethod
    def from_rv(cls, r_km: np.ndarray, v_km_s: np.ndarray,
                epoch=None, **kwargs) -> "OrbitalState":
        """Convert ECI state vector to OrbitalState via orbital elements."""
        r = np.asarray(r_km, dtype=float)
        v = np.asarray(v_km_s, dtype=float)
        r_mag = np.linalg.norm(r)
        v_mag = np.linalg.norm(v)

        h_vec = np.cross(r, v)
        h_mag = np.linalg.norm(h_vec)
        n_vec = np.cross([0, 0, 1], h_vec)
        n_mag = np.linalg.norm(n_vec)

        e_vec = ((v_mag**2 - MU/r_mag)*r - np.dot(r, v)*v) / MU
        e     = np.linalg.norm(e_vec)

        energy = v_mag**2/2 - MU/r_mag
        a_km   = -MU / (2*energy)

        inc  = np.degrees(np.arccos(h_vec[2] / h_mag))
        raan = np.degrees(np.arccos(n_vec[0] / n_mag)) if n_mag > 1e-10 else 0.0
        if n_vec[1] < 0: raan = 360 - raan

        argp = np.degrees(np.arccos(np.dot(n_vec, e_vec) / (n_mag * e))) if (n_mag*e > 1e-10) else 0.0
        if e_vec[2] < 0: argp = 360 - argp

        nu = np.degrees(np.arccos(np.dot(e_vec, r) / (e * r_mag))) if e > 1e-10 else 0.0
        if np.dot(r, v) < 0: nu = 360 - nu

        return cls(a_km=a_km, e=e, inc_deg=inc, raan_deg=raan,
                   argp_deg=argp, nu_deg=nu, epoch=epoch, **kwargs)

    @classmethod
    def from_ssapy(cls, orbit, t=None, **kwargs) -> "OrbitalState":
        """Convert an SSAPy ``Orbit`` or vector-valued ``Orbit.at`` result.

        SSAPy stores positions in metres and velocities in metres/second;
        this class stores kilometres and kilometres/second.  If the supplied
        orbit contains a propagated time series, the first sample defines the
        osculating state used by the scene-oriented plotting classes.
        """
        r_km, v_km_s, t_time = normalize_orbit_trajectory(
            orbit=orbit,
            t=t,
            require_velocity=True,
            r_units="m",
            v_units="m/s",
        )
        epoch = kwargs.pop("epoch", None)
        if epoch is None and len(t_time):
            epoch = t_time[0].utc.to_datetime(timezone=timezone.utc)
        return cls.from_rv(r_km[0], v_km_s[0], epoch=epoch, **kwargs)

    def clone(self, **overrides) -> "OrbitalState":
        kw = dict(a_km=self.a_km, e=self.e, inc_deg=self.inc_deg,
                  raan_deg=self.raan_deg, argp_deg=self.argp_deg,
                  nu_deg=self.nu_deg, epoch=self.epoch,
                  config=self.config, name=self.name)
        kw.update(overrides)
        return OrbitalState(**kw)

    # ── repr ──────────────────────────────────────────────────────────────────
    def __repr__(self):
        return (f"OrbitalState('{self.name}' | {self.regime} | "
                f"a={self.a_km:.1f}km e={self.e:.4f} i={self.inc_deg:.2f}° | "
                f"T={self.period_hr:.3f}h)")
