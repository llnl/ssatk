"""
core/satellite.py
─────────────────
3D satellite model, manoeuvre events, attitude visualisation.

Usage
-----
::

    from ssatk.plots.satellite import Satellite3D, BurnEvent
    import numpy as np

    sat = Satellite3D(model_path="models/cubesat.obj", mass_kg=500)
    sat.show_ntw   = True          # draw N, T, W axes
    sat.show_burns = True

    # Add an impulsive burn (delta-v in NTW frame)
    sat.add_burn(BurnEvent(
        epoch_offset_s = 1800,          # seconds from propagation start
        dv_ntw_km_s    = np.array([0, 0.02, 0]),  # along T (prograde), [N,T,W]
        mode           = "impulsive",
    ))

    # Get NTW unit vectors at a given state
    r = np.array([6928, 0, 0])  # km
    v = np.array([0, 7.612, 0])  # km/s
    T, N, W = sat.ntw_vectors(r, v)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from ssatk.constants import G0

from ssatk.coordinates.frames import ntw_axes, lvlh_axes


# ─── BurnEvent ────────────────────────────────────────────────────────────────
@dataclass
class BurnEvent:
    """
    Single manoeuvre event.

    Parameters
    ----------
    epoch_offset_s : float
        Seconds from propagation start when burn occurs.
    dv_ntw_km_s : array (3,)
        Delta-V in the canonical SSAPy NTW component order [N, T, W],
        where T is along-track and W is orbit-normal. Units are km/s.
    mode : "impulsive" | "finite"
        Impulsive: instantaneous Δv applied.
        Finite: constant-thrust burn arc computed with fixed-step RK4.
    label : str
        Human-readable label shown on plots.

    Finite-burn parameters (only used when mode="finite")
    ---------------
    thrust_N : float   Engine thrust, Newtons
    isp_s    : float   Specific impulse, seconds
    mass_kg  : float   Spacecraft wet mass at burn start, kg
    """
    epoch_offset_s : float
    dv_ntw_km_s    : np.ndarray
    mode           : Literal["impulsive", "finite"] = "impulsive"
    label          : str = ""

    # finite-burn parameters
    thrust_N : float = 50.0
    isp_s    : float = 300.0
    mass_kg  : float = 500.0

    def __post_init__(self):
        self.dv_ntw_km_s = np.asarray(self.dv_ntw_km_s, dtype=float)
        if not self.label:
            dv = np.linalg.norm(self.dv_ntw_km_s)
            self.label = f"Δv {dv*1000:.1f} m/s @ t+{self.epoch_offset_s:.0f}s"

    @property
    def dv_mag_m_s(self) -> float:
        return float(np.linalg.norm(self.dv_ntw_km_s)) * 1000

    @property
    def dv_mag_km_s(self) -> float:
        return float(np.linalg.norm(self.dv_ntw_km_s))

    def dv_eci(self, r_km: np.ndarray, v_km_s: np.ndarray) -> np.ndarray:
        """Rotate canonical [N, T, W] Δv into ECI using the current state."""
        T, N, W = ntw_axes(r_km, v_km_s)
        dv_n, dv_t, dv_w = self.dv_ntw_km_s
        return dv_t*T + dv_n*N + dv_w*W

    def burn_duration_s(self) -> float:
        """Estimated finite-burn duration using rocket equation."""
        if self.mode == "impulsive":
            return 0.0
        ve = self.isp_s * G0 / 1000  # km/s
        mass_final = self.mass_kg * np.exp(-self.dv_mag_km_s / ve)
        dm = self.mass_kg - mass_final
        thrust_km_s2 = self.thrust_N / (self.mass_kg * 1000)  # km/s²
        return float(self.dv_mag_km_s / thrust_km_s2) if thrust_km_s2 > 0 else 0.0


# ─── Satellite3D ─────────────────────────────────────────────────────────────
class Satellite3D:
    """
    3D satellite model + attitude + manoeuvre visualisation.

    Attributes
    ----------
    model_path : str | None
        Path to a Wavefront .obj file.  None → render as a simple box.
    mass_kg    : float
        Dry mass (used for finite burns when no per-burn mass is set).
    show_ntw   : bool
        Draw T (green), N (red), W (blue) unit vectors on the satellite.
    show_lvlh  : bool
        Draw R (white), S (yellow), W (blue) LVLH axes.
    show_burns : bool
        Draw Δv arrow for each BurnEvent.
    ntw_scale  : float
        Length of the NTW/LVLH axis arrows, km.
    """

    # ── construction ─────────────────────────────────────────────────────────
    def __init__(
        self,
        model_path : str | Path | None = None,
        mass_kg    : float = 500.0,
        name       : str = "Satellite",
    ):
        self.model_path = Path(model_path) if model_path else None
        self.mass_kg    = mass_kg
        self.name       = name

        self.show_ntw   = True
        self.show_lvlh  = False
        self.show_burns = True
        self.ntw_scale  = 500.0    # km  (scales with scene)

        self.burns: list[BurnEvent] = []

        # cached OBJ geometry
        self._verts: np.ndarray | None = None  # (V, 3)
        self._faces: list[list[int]] = []

    # ── manoeuvre management ──────────────────────────────────────────────────
    def add_burn(self, burn: BurnEvent):
        """Append a BurnEvent, sorted by epoch_offset_s."""
        self.burns.append(burn)
        self.burns.sort(key=lambda b: b.epoch_offset_s)

    def remove_burn(self, index: int):
        self.burns.pop(index)

    def total_delta_v_m_s(self) -> float:
        return sum(b.dv_mag_m_s for b in self.burns)

    # ── NTW / LVLH axis vectors in ECI ───────────────────────────────────────
    def ntw_vectors(
        self,
        r_km    : np.ndarray,
        v_km_s  : np.ndarray,
        scale   : float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (T, N, W) unit vectors (scaled) in ECI for visualisation.
        T = along-track (green), N = cross-track (red), W = orbit-normal (blue).
        """
        s = scale or self.ntw_scale
        T, N, W = ntw_axes(r_km, v_km_s)
        return T*s, N*s, W*s

    def lvlh_vectors(
        self,
        r_km    : np.ndarray,
        v_km_s  : np.ndarray,
        scale   : float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (R, S, W) unit vectors (scaled) in ECI for visualisation."""
        s = scale or self.ntw_scale
        R, S, W = lvlh_axes(r_km, v_km_s)
        return R*s, S*s, W*s

    # ── burn Δv ECI vector at event time ─────────────────────────────────────
    def burn_vector_eci(self, burn: BurnEvent, r_km, v_km_s) -> np.ndarray:
        """Return scaled ECI Δv arrow for plotting."""
        dv_eci = burn.dv_eci(r_km, v_km_s)
        mag    = np.linalg.norm(dv_eci)
        if mag < 1e-15:
            return dv_eci
        # scale arrow so it's visible at orbit scale
        vis_scale = self.ntw_scale * (mag / 0.1)   # 0.1 km/s → 1× ntw_scale
        return dv_eci / mag * vis_scale

    # ── apply burns to OrbitalState, return list of post-burn states ──────────
    def apply_burns_to_trajectory(
        self,
        traj,               # Trajectory namedtuple
        orbit_state,        # OrbitalState
    ) -> list:             # list of (OrbitalState, burn_point_index)
        """
        For each BurnEvent find the closest trajectory point, apply Δv,
        and return a new OrbitalState.  Finite burns use RK4 on the burn arc.

        Returns
        -------
        list of (new_OrbitalState, traj_index, burn)
        """
        from .orbit_state import OrbitalState
        results = []
        for burn in self.burns:
            # Find trajectory index closest to burn epoch
            t0_gps = traj.t[0]
            target_gps = t0_gps + burn.epoch_offset_s
            idx = int(np.argmin(np.abs(traj.t - target_gps)))
            idx = min(idx, len(traj.r) - 1)

            r = traj.r[idx].copy()   # km
            v = traj.v[idx].copy()   # km/s
            dv_eci = burn.dv_eci(r, v)

            if burn.mode == "impulsive":
                v_new = v + dv_eci
            else:
                v_new = self._finite_burn_rk4(r, v, burn)

            new_state = OrbitalState.from_rv(
                r_km=r,
                v_km_s=v_new,
                epoch=orbit_state.epoch,
                config=orbit_state.config,
                name=f"{orbit_state.name}+{burn.label}",
            )
            results.append((new_state, idx, burn))
        return results

    def _finite_burn_rk4(
        self,
        r0_km  : np.ndarray,
        v0_km_s: np.ndarray,
        burn   : BurnEvent,
        n_steps: int = 100,
    ) -> np.ndarray:
        """
        Integrate only the burn arc with fixed-step RK4.
        Returns post-burn velocity in km/s.
        """
        from .orbit_state import MU
        g0_km_s2 = G0 / 1000  # km/s²
        ve = burn.isp_s * g0_km_s2  # km/s exhaust velocity

        # Burn direction: NTW Δv unit vector
        dv_dir_eci = burn.dv_eci(r0_km, v0_km_s)
        mag = np.linalg.norm(dv_dir_eci)
        if mag < 1e-15:
            return v0_km_s.copy()
        thrust_dir = dv_dir_eci / mag  # unit vector in ECI

        thrust_km_s2 = burn.thrust_N / (burn.mass_kg * 1000)  # km/s²

        duration = max(burn.burn_duration_s(), 1.0)
        dt = duration / n_steps

        r, v, m = r0_km.copy(), v0_km_s.copy(), burn.mass_kg

        def _deriv(r, v, m):
            r_mag = np.linalg.norm(r)
            a_grav   = -MU / r_mag**3 * r
            mdot     = -burn.thrust_N / (ve * 1000)          # kg/s
            a_thrust = thrust_dir * burn.thrust_N / (m * 1000)  # km/s²
            return v.copy(), a_grav + a_thrust, mdot

        for _ in range(n_steps):
            dr1, dv1, dm1 = _deriv(r, v, m)
            dr2, dv2, dm2 = _deriv(r + 0.5*dt*dr1, v + 0.5*dt*dv1, m + 0.5*dt*dm1)
            dr3, dv3, dm3 = _deriv(r + 0.5*dt*dr2, v + 0.5*dt*dv2, m + 0.5*dt*dm2)
            dr4, dv4, dm4 = _deriv(r + dt*dr3, v + dt*dv3, m + dt*dm3)
            r += dt/6*(dr1 + 2*dr2 + 2*dr3 + dr4)
            v += dt/6*(dv1 + 2*dv2 + 2*dv3 + dv4)
            m += dt/6*(dm1 + 2*dm2 + 2*dm3 + dm4)
            m  = max(m, 1.0)  # dry mass floor

        return v

    # ── OBJ loader ────────────────────────────────────────────────────────────
    def load_obj(self) -> bool:
        """
        Parse a Wavefront .obj file.
        Returns True on success.

        Sets self._verts (V, 3) and self._faces (list of index lists).
        Vertices are normalised so the model fits in a unit sphere.
        """
        if self.model_path is None or not self.model_path.exists():
            return False
        verts, faces = [], []
        try:
            with open(self.model_path) as fh:
                for line in fh:
                    line = line.strip()
                    if line.startswith("v "):
                        parts = line.split()
                        verts.append([float(parts[1]),
                                      float(parts[2]),
                                      float(parts[3])])
                    elif line.startswith("f "):
                        parts = line.split()[1:]
                        # handle v, v/vt, v/vt/vn formats
                        idx = [int(p.split("/")[0]) - 1 for p in parts]
                        faces.append(idx)
        except Exception as ex:
            print(f"[Satellite3D] OBJ load error: {ex}")
            return False

        if not verts:
            return False
        v = np.array(verts, dtype=float)
        # centre and normalise to unit sphere
        v -= v.mean(axis=0)
        r_max = np.linalg.norm(v, axis=1).max()
        if r_max > 1e-10:
            v /= r_max
        self._verts = v
        self._faces = faces
        return True

    def model_vertices_eci(
        self,
        r_km   : np.ndarray,
        v_km_s : np.ndarray,
        scale_km: float = 50.0,
    ) -> np.ndarray:
        """
        Return OBJ vertices transformed into ECI at the satellite's current position.

        The model is oriented so +T is along-track, +N is cross-track, +W is orbit-normal.
        scale_km controls the rendered size.
        """
        if self._verts is None:
            # fallback: simple box
            self._verts = _unit_box_verts()
            self._faces = _unit_box_faces()

        T, N, W = ntw_axes(r_km, v_km_s)
        # rotation: model X→T, Y→N, Z→W
        R = np.column_stack([T, N, W])   # (3,3) columns are frame axes
        verts_eci = (self._verts * scale_km) @ R.T + r_km
        return verts_eci

    @property
    def faces(self) -> list[list[int]]:
        if not self._faces:
            self.load_obj()
        return self._faces


# ── fallback geometry: unit box ───────────────────────────────────────────────
def _unit_box_verts() -> np.ndarray:
    s = 0.5
    return np.array([
        [-s, -s, -s], [ s, -s, -s], [ s,  s, -s], [-s,  s, -s],
        [-s, -s,  s], [ s, -s,  s], [ s,  s,  s], [-s,  s,  s],
    ], dtype=float)


def _unit_box_faces() -> list[list[int]]:
    return [
        [0,1,2,3], [4,5,6,7],  # bottom / top
        [0,1,5,4], [2,3,7,6],  # front / back
        [0,3,7,4], [1,2,6,5],  # left / right
    ]
