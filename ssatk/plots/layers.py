"""
core/layers.py
──────────────
Pluggable layer system.  Each layer knows how to add itself to both
a matplotlib Axes3D and a Plotly figure.

Available layers
----------------
  "stars"       StarfieldLayer     — real catalog from bright_stars.csv
  "earth"       EarthLayer         — textured sphere, matplotlib + Plotly
  "moon"        MoonLayer          — textured Moon sphere
  "sun"         OrbitSunLayer           — Sun position marker + direction arrow
  "groundtrack" GroundTrackLayer   — sub-satellite lat/lon projected on sphere
  "eclipse"     EclipseLayer       — eclipse fraction + solar β angle readout
  "terminator"  TerminatorLayer    — day/night boundary circle on Earth
  "van_allen"   VanAllenLayer      — inner + outer belt tori
  "magfield"    MagfieldLayer      — IGRF 2025 field lines (ppigrf + RK4)
  "lagrange"    LagrangeLayer      — L1–L5 markers (cislunar regime)
  "burns"       BurnLayer          — Δv arrows from Satellite3D.burns
  "ntw"         NTWLayer           — T, N, W axis vectors on satellite

Usage
-----
layer = EarthLayer(texture_path="path/to/earth.png")
layer.add_to_plotly(fig, orbit_state, traj)
layer.add_to_mpl(ax, orbit_state, traj)
"""

from __future__ import annotations

import abc
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from ssatk.constants import EARTH_RADIUS_KM, LD_KM, MOON_RADIUS_KM
from ssatk.plots.scene_primitives import earth_rotation_deg_from_time

# ── optional heavy imports ────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    import ppigrf
    _HAS_PPIGRF = True
except ImportError:
    _HAS_PPIGRF = False

# ── constants ─────────────────────────────────────────────────────────────────
# RE_KM / MOON_A_KM / MOON_R_KM sourced from ssatk.constants rather than
# hardcoded literals -- these were independently duplicated (at the same
# value, by coincidence) across this file, orbit_state.py, and
# groundtrack_enhanced.py.
RE_KM      = EARTH_RADIUS_KM
MOON_A_KM  = LD_KM
MOON_R_KM  = MOON_RADIUS_KM

_SPECTRAL_COLOR = {
    "O": "#9bb0ff", "B": "#aabfff", "A": "#cad7ff",
    "F": "#f8f7ff", "G": "#fff4e8", "K": "#ffd2a1", "M": "#ffcc6f",
}


def _precession_matrix(epoch_jd: float) -> np.ndarray:
    """IAU 1976 precession from J2000.0 to given Julian date.

    Identical algorithm to starfield.py so both matplotlib and Plotly
    paths produce the same star positions.  Shift: ~50 arcsec/yr
    (~3 matplotlib px at 2022, ~3.5 px at 2026).
    """
    T = (epoch_jd - 2_451_545.0) / 36_525.0
    zeta  = ((2306.2181 + 1.39656*T - 0.000139*T**2)*T
             + (0.30188 - 0.000344*T)*T**2 + 0.017998*T**3)
    z     = ((2306.2181 + 1.39656*T - 0.000139*T**2)*T
             + (1.09468 + 0.000066*T)*T**2 + 0.018203*T**3)
    theta = ((2004.3109 - 0.85330*T - 0.000217*T**2)*T
             - (0.42665 + 0.000217*T)*T**2 - 0.041775*T**3)
    zeta_r, z_r, th_r = (np.radians(v / 3600.0) for v in (zeta, z, theta))
    Rz_zeta = np.array([[ np.cos(zeta_r), np.sin(zeta_r), 0.0],
                         [-np.sin(zeta_r), np.cos(zeta_r), 0.0],
                         [            0.0,             0.0, 1.0]])
    Ry_th   = np.array([[ np.cos(th_r), 0.0, -np.sin(th_r)],
                         [          0.0, 1.0,           0.0],
                         [ np.sin(th_r), 0.0,  np.cos(th_r)]])
    Rz_z    = np.array([[ np.cos(z_r), np.sin(z_r), 0.0],
                         [-np.sin(z_r), np.cos(z_r), 0.0],
                         [         0.0,          0.0, 1.0]])
    return Rz_z @ Ry_th @ Rz_zeta


# ── Sensor FOV geometry helpers ───────────────────────────────────────────────

def _orthonormal_basis(axis: np.ndarray):
    """Return two unit vectors (u, v) perpendicular to *axis* and each other."""
    axis = axis / np.linalg.norm(axis)
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, helper); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    return u, v


def _cone_mesh3d(apex, direction, half_angle_deg, length_km,
                 n_sides=64, color="#00CED1", opacity=0.35, name="Sensor FOV",
                 sun_hat=None):
    """Build a closed Plotly Mesh3d cone.
    If *sun_hat* (GCRF unit vector toward Sun) is provided, vertices are
    coloured by solar illumination (lit side bright, shadow side dark).
    """
    apex = np.asarray(apex, dtype=float)
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    import math as _math
    radius = length_km * _math.tan(_math.radians(half_angle_deg))
    base_centre = apex + direction * length_km
    u, v = _orthonormal_basis(direction)
    angles = np.linspace(0., 2. * _math.pi, n_sides, endpoint=False)
    rim = (base_centre[None, :]
           + radius * (np.cos(angles)[:, None] * u[None, :]
                       + np.sin(angles)[:, None] * v[None, :]))
    vx = np.empty(n_sides + 2); vy = np.empty(n_sides + 2); vz = np.empty(n_sides + 2)
    vx[0], vy[0], vz[0] = apex
    vx[1:n_sides+1] = rim[:, 0]; vy[1:n_sides+1] = rim[:, 1]; vz[1:n_sides+1] = rim[:, 2]
    ci = n_sides + 1; vx[ci], vy[ci], vz[ci] = base_centre
    ti, tj, tk = [], [], []
    for s in range(n_sides):
        c = s + 1; nx = (s + 1) % n_sides + 1
        ti += [0, ci]; tj += [c, c]; tk += [nx, nx]
    if not _HAS_PLOTLY:
        return None

    if sun_hat is not None:
        # Per-vertex illumination intensity ∈ [0,1]
        # Side-face normals: outward perpendicular to cone surface
        sun_hat = np.asarray(sun_hat, dtype=float)
        sun_hat /= np.linalg.norm(sun_hat)
        cos_ha = _math.cos(_math.radians(half_angle_deg))
        sin_ha = _math.sin(_math.radians(half_angle_deg))
        ambient = 0.25
        intensity = np.empty(n_sides + 2)
        for s in range(n_sides):
            radial = np.cos(angles[s]) * u + np.sin(angles[s]) * v
            # Outward normal to cone lateral surface
            normal = cos_ha * radial - sin_ha * direction
            normal /= np.linalg.norm(normal)
            intensity[s + 1] = ambient + (1 - ambient) * max(0.0, float(np.dot(normal, sun_hat)))
        # Apex: average of rim intensities (geometric mean is 0.5 for symmetric cone)
        intensity[0] = float(np.mean(intensity[1:n_sides+1]))
        # Base cap: normal = +direction
        intensity[ci] = ambient + (1 - ambient) * max(0.0, float(np.dot(direction, sun_hat)))

        # Parse hex colour for sunlit endpoint
        _hx = color.lstrip("#")
        _r, _g, _b = int(_hx[0:2],16)/255, int(_hx[2:4],16)/255, int(_hx[4:6],16)/255
        _lit  = f"rgb({int(_r*255)},{int(_g*255)},{int(_b*255)})"
        _shad = "rgb(15,20,60)"
        return go.Mesh3d(
            x=vx, y=vy, z=vz, i=ti, j=tj, k=tk,
            intensity=intensity,
            intensitymode="vertex",
            colorscale=[[0, _shad], [0.3, _shad], [1.0, _lit]],
            cmin=0, cmax=1,
            showscale=False,
            opacity=opacity, name=name, showlegend=True,
            flatshading=False,
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0),
        )

    return go.Mesh3d(
        x=vx, y=vy, z=vz, i=ti, j=tj, k=tk,
        color=color, opacity=opacity, name=name, showlegend=True,
        flatshading=False,
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.3, roughness=0.5),
    )


def _footprint_on_earth(apex_km, direction, half_angle_deg, n_pts=300):
    """Intersect the FOV cone rim with Earth's sphere (R=6 371 km).

    Returns (xs, ys, zs) arrays in km for the footprint circle, or None
    if the cone edge does not reach Earth (e.g. anti-nadir pointing).
    """
    import math as _m
    R_E  = 6_371.0
    apex = np.asarray(apex_km, dtype=float)
    d    = np.asarray(direction, dtype=float); d /= np.linalg.norm(d)
    u, v = _orthonormal_basis(d)
    sin_ha = _m.sin(_m.radians(half_angle_deg))
    cos_ha = _m.cos(_m.radians(half_angle_deg))
    phi = np.linspace(0.0, 2.0 * _m.pi, n_pts, endpoint=False)
    xs, ys, zs = [], [], []
    for p in phi:
        ray = sin_ha * (_m.cos(p) * u + _m.sin(p) * v) + cos_ha * d
        ray /= np.linalg.norm(ray)
        A = 1.0
        B = 2.0 * float(np.dot(apex, ray))
        C = float(np.dot(apex, apex)) - R_E**2
        disc = B*B - 4*A*C
        if disc < 0:
            return None          # rim misses Earth
        t1 = (-B - _m.sqrt(disc)) / 2.0
        t2 = (-B + _m.sqrt(disc)) / 2.0
        t  = t1 if t1 > 1e-3 else (t2 if t2 > 1e-3 else None)
        if t is None:
            return None
        pt = apex + t * ray
        xs.append(pt[0]); ys.append(pt[1]); zs.append(pt[2])
    return np.array(xs), np.array(ys), np.array(zs)


def _footprint_illumination_traces(xs, ys, zs, sun_hat, cone_color, name="Footprint"):
    """Split a footprint circle into sunlit and shadow arcs.

    Returns up to two go.Scatter3d traces:
      • sunlit arc  — bright cone colour
      • shadow arc  — dark blue-grey
    """
    if not _HAS_PLOTLY:
        return []
    sun_hat  = np.asarray(sun_hat, dtype=float); sun_hat /= np.linalg.norm(sun_hat)
    pts      = np.stack([xs, ys, zs], axis=1)       # (N, 3)
    norms    = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    dot      = norms @ sun_hat                       # (N,) — positive = sunlit
    lit_mask = dot > 0

    traces = []
    for mask, col, label, w in [
        (lit_mask,  cone_color, f"{name} (day)",    3),
        (~lit_mask, "#1a2a4a",  f"{name} (night)",  2),
    ]:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        # Keep contiguous segments (wrap-around handled by None gaps)
        seg_x, seg_y, seg_z = [], [], []
        for k in range(len(idx)):
            i = idx[k]
            seg_x.append(xs[i]); seg_y.append(ys[i]); seg_z.append(zs[i])
            # Insert gap if next index is not consecutive
            if k < len(idx)-1 and idx[k+1] != i+1:
                seg_x.append(None); seg_y.append(None); seg_z.append(None)
        traces.append(go.Scatter3d(
            x=seg_x, y=seg_y, z=seg_z,
            mode="lines",
            line=dict(color=col, width=w),
            name=label, showlegend=True,
        ))
    return traces


def _boresight_line3d(apex, direction, length_km,
                      color="#FFFFFF", name="Boresight", dash="dot"):
    """Dashed boresight line along the cone axis."""
    tip = np.asarray(apex) + np.asarray(direction) * length_km
    if not _HAS_PLOTLY:
        return None
    return go.Scatter3d(
        x=[apex[0], tip[0]], y=[apex[1], tip[1]], z=[apex[2], tip[2]],
        mode="lines",
        line=dict(color=color, width=2, dash=dash),
        name=name, showlegend=False,
    )


# ─── BaseLayer ────────────────────────────────────────────────────────────────
class BaseLayer(abc.ABC):
    """
    Abstract base.  Concrete layers implement add_to_mpl and add_to_plotly.

    Parameters
    ----------
    key     : str   unique identifier used in BasePlot3D.layers dict
    name    : str   human-readable label
    enabled : bool  master on/off toggle
    """

    def __init__(self, key: str, name: str, enabled: bool = True):
        self.key     = key
        self.name    = name
        self.enabled = enabled
        self._artists_mpl: list = []   # mpl artist handles for removal

    @abc.abstractmethod
    def add_to_mpl(self, ax, orbit_state, traj=None, satellite=None, **kw):
        """Draw onto a matplotlib Axes3D.  Returns list of artists."""

    @abc.abstractmethod
    def add_to_plotly(self, fig, orbit_state, traj=None, satellite=None, **kw):
        """Add traces/shapes to a plotly go.Figure."""

    def remove_from_mpl(self):
        for a in self._artists_mpl:
            try:
                a.remove()
            except Exception:
                pass
        self._artists_mpl.clear()

    def __repr__(self):
        state = "on" if self.enabled else "off"
        return f"<{self.__class__.__name__} key='{self.key}' [{state}]>"


# ─── StarfieldLayer ───────────────────────────────────────────────────────────
class StarfieldLayer(BaseLayer):
    """Real star catalog (bright_stars.csv) with spectral colours + magnitude sizing."""

    def __init__(self, catalog_path: str | Path, sky_radius_factor: float = 15.0,
                 epoch_jd: float | None = None, **kw):
        super().__init__("stars", "Starfield", **kw)
        self.catalog_path      = Path(catalog_path)
        self.sky_radius_factor = sky_radius_factor
        self.epoch_jd          = epoch_jd   # Julian date for IAU 1976 precession; None = J2000
        self._stars: Optional[np.ndarray] = None   # has-loaded flag
        self._loaded_epoch_jd: object = object()   # sentinel — forces first _load()

    def _load(self, scene_radius_km: float):
        # Invalidate cache when epoch changes (e.g. user picked a new date in GUI)
        if self._stars is not None and self._loaded_epoch_jd == self.epoch_jd:
            return

        if not self.catalog_path.exists():
            warnings.warn(f"Star catalog not found: {self.catalog_path}")
            self._xyz    = np.empty((0, 3))
            self._sizes  = np.empty(0)
            self._colors = []
            self._stars  = self._xyz           # mark as loaded
            self._loaded_epoch_jd = self.epoch_jd
            return
        try:
            import csv
            ras, decs, mags, sptypes = [], [], [], []
            with open(self.catalog_path, newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        ras.append(float(row.get("ra", row.get("RA", 0))))
                        decs.append(float(row.get("dec", row.get("Dec", 0))))
                        mags.append(float(row.get("mag", row.get("Vmag", 5))))
                        sptypes.append(str(row.get("sptype", row.get("SpType", "G")))[:1])
                    except (ValueError, KeyError):
                        continue
        except Exception as ex:
            warnings.warn(f"Star catalog read error: {ex}")
            self._xyz    = np.empty((0, 3))
            self._sizes  = np.empty(0)
            self._colors = []
            self._stars  = self._xyz
            self._loaded_epoch_jd = self.epoch_jd
            return

        # RA in HYG is in hours (0–24) — convert to radians via degrees (×15)
        # BUG FIX: earlier code omitted the ×15, compressing RA by 15×
        ra_rad  = np.radians(np.array(ras) * 15.0)
        dec_rad = np.radians(np.array(decs))

        # Unit vectors on J2000.0 celestial sphere (GCRF-aligned)
        cx = np.cos(dec_rad) * np.cos(ra_rad)
        cy = np.cos(dec_rad) * np.sin(ra_rad)
        cz = np.sin(dec_rad)

        # Apply IAU 1976 precession if epoch_jd is set.
        # This shifts J2000 catalog positions to the mean equator-of-date
        # so stars land in the correct GCRF directions for the mission date.
        # Shift magnitude: ~50 arcsec/yr (~3 px at 2022, ~3.5 px at 2026).
        if self.epoch_jd is not None:
            P = _precession_matrix(self.epoch_jd)
            vecs = P @ np.stack([cx, cy, cz])   # (3, N)
            cx, cy, cz = vecs[0], vecs[1], vecs[2]

        R = scene_radius_km * self.sky_radius_factor
        self._xyz    = np.column_stack([cx * R, cy * R, cz * R])
        self._sizes  = np.clip(5 - np.array(mags), 0.5, 6.0)
        self._colors = [_SPECTRAL_COLOR.get(s, "#ffffff") for s in sptypes]
        self._stars  = self._xyz           # mark as loaded
        self._loaded_epoch_jd = self.epoch_jd

    def add_to_mpl(self, ax, orbit_state, traj=None, satellite=None, scene_radius_km=None, **kw):
        if not _HAS_MPL:
            return []
        r = scene_radius_km or orbit_state.r_a * 1.1
        self._load(r)
        if self._xyz.shape[0] == 0:
            return []
        a = ax.scatter(
            self._xyz[:, 0], self._xyz[:, 1], self._xyz[:, 2],
            s=self._sizes, c=self._colors, alpha=0.7,
            zorder=0, depthshade=False,
        )
        self._artists_mpl = [a]
        return [a]

    def add_to_plotly(self, fig, orbit_state, traj=None, satellite=None, scene_radius_km=None, **kw):
        if not _HAS_PLOTLY:
            return
        r = scene_radius_km or orbit_state.r_a * 1.1
        self._load(r)
        if self._xyz.shape[0] == 0:
            return
        fig.add_trace(go.Scatter3d(
            x=self._xyz[:, 0], y=self._xyz[:, 1], z=self._xyz[:, 2],
            mode="markers",
            marker=dict(size=self._sizes*0.5, color=self._colors, opacity=0.7),
            name="Stars", hoverinfo="skip",
            showlegend=False,
        ))


# ─── EarthLayer ───────────────────────────────────────────────────────────────
class EarthLayer(BaseLayer):
    """Textured Earth sphere.  Falls back to solid blue if texture missing."""

    def __init__(self, texture_path: str | Path | None = None,
                 n_lat: int = 72, n_lon: int = 144, radius_scale: float = 1.0,
                 time=None, rotation_deg=None, **kw):
        super().__init__("earth", "Earth", **kw)
        self.texture_path = Path(texture_path) if texture_path else None
        self.n_lat = n_lat
        self.n_lon = n_lon
        self.radius_scale = radius_scale   # visual size multiplier; doesn't move position
        self.rotation_deg = earth_rotation_deg_from_time(time) if rotation_deg is None else float(rotation_deg)
        self._tex: Optional[np.ndarray] = None  # (H, W, 3) uint8

    def _load_texture(self):
        if self._tex is not None or not _HAS_PIL:
            return
        if self.texture_path and self.texture_path.exists():
            try:
                img = Image.open(self.texture_path).convert("RGB")
                img = img.resize((self.n_lon, self.n_lat), Image.LANCZOS)
                self._tex = np.array(img)
            except Exception as ex:
                warnings.warn(f"Earth texture load error: {ex}")

    def _sphere_xyz(self, R=RE_KM):
        # Texture row 0 is the north pole for standard equirectangular Earth
        # maps, so build the geometry north-to-south to keep Matplotlib
        # facecolors geodetically aligned.
        lat = np.linspace(np.pi/2, -np.pi/2, self.n_lat)
        lon = np.linspace(-np.pi,   np.pi,   self.n_lon) + np.radians(self.rotation_deg)
        Lon, Lat = np.meshgrid(lon, lat)
        X = R * np.cos(Lat) * np.cos(Lon)
        Y = R * np.cos(Lat) * np.sin(Lon)
        Z = R * np.sin(Lat)
        return X, Y, Z

    def add_to_mpl(self, ax, orbit_state, traj=None, satellite=None, **kw):
        if not _HAS_MPL:
            return []
        self._load_texture()
        X, Y, Z = self._sphere_xyz()
        if self._tex is not None:
            fcolors = self._tex / 255.0
            surf = ax.plot_surface(X, Y, Z, facecolors=fcolors,
                                   rstride=1, cstride=1, shade=False,
                                   zorder=1, alpha=1.0)
        else:
            surf = ax.plot_surface(X, Y, Z, color="#1a4f8a",
                                   alpha=0.85, zorder=1)
        self._artists_mpl = [surf]
        return [surf]

    def add_to_plotly(self, fig, orbit_state, traj=None, satellite=None, **kw):
        if not _HAS_PLOTLY:
            return
        self._load_texture()
        # Latitude 90→−90 (north first) so texture row 0 (North Pole in any
        # standard equirectangular map) maps to +Z, not -Z.
        lat = np.linspace( 90, -90,  self.n_lat)
        lon = np.linspace(-180, 180, self.n_lon) + float(self.rotation_deg)
        Lon, Lat = np.meshgrid(lon, lat)
        _R = RE_KM * self.radius_scale
        X = _R * np.cos(np.radians(Lat)) * np.cos(np.radians(Lon))
        Y = _R * np.cos(np.radians(Lat)) * np.sin(np.radians(Lon))
        Z = _R * np.sin(np.radians(Lat))

        if self._tex is not None:
            # flatten texture into per-vertex colours for go.Mesh3d
            verts_x = X.ravel(); verts_y = Y.ravel(); verts_z = Z.ravel()
            H, W = self.n_lat, self.n_lon
            i_idx, j_idx, k_idx = [], [], []
            cols = []
            for r in range(H - 1):
                for c in range(W - 1):
                    v0 = r*W + c; v1 = v0+1; v2 = (r+1)*W + c; v3 = v2+1
                    # SSAPy's bundled earth.png and standard Blue Marble
                    # equirectangular maps already run left-to-right over
                    # [-180°, +180°], matching the mesh longitude grid.
                    pixel = self._tex[r, c]
                    col = f"rgb({pixel[0]},{pixel[1]},{pixel[2]})"
                    for tri in [(v0,v1,v2),(v1,v3,v2)]:
                        i_idx.append(tri[0]); j_idx.append(tri[1]); k_idx.append(tri[2])
                        cols.append(col)
            fig.add_trace(go.Mesh3d(
                x=verts_x, y=verts_y, z=verts_z,
                i=i_idx, j=j_idx, k=k_idx,
                facecolor=cols,
                name="Earth", hoverinfo="skip", showlegend=False,
            ))
        else:
            fig.add_trace(go.Surface(
                x=X, y=Y, z=Z,
                colorscale=[[0,"#1a4f8a"],[1,"#1a4f8a"]],
                showscale=False, name="Earth", opacity=0.9,
            ))


# ─── MoonLayer ────────────────────────────────────────────────────────────────
class MoonLayer(BaseLayer):
    """Textured Moon sphere positioned at current lunar position."""

    def __init__(self, texture_path: str | Path | None = None,
                 n_pts: int = 60, radius_scale: float = 1.0, **kw):
        super().__init__("moon", "Moon", **kw)
        self.texture_path = Path(texture_path) if texture_path else None
        self.n_pts = n_pts
        self.radius_scale = radius_scale   # visual size multiplier; doesn't move position
        self._tex: Optional[np.ndarray] = None

    def _load_texture(self):
        if self._tex is not None or not _HAS_PIL:
            return
        if self.texture_path and self.texture_path.exists():
            try:
                img = Image.open(self.texture_path).convert("RGB")
                img = img.resize((self.n_pts, self.n_pts//2), Image.LANCZOS)
                self._tex = np.array(img)
            except Exception:
                pass

    def _moon_position_km(self, t_gps: float) -> np.ndarray:
        """Approximate Moon ECI position (low-precision)."""
        try:
            import ssapy.compute
            return ssapy.compute.moonPos(t_gps) / 1e3  # m → km
        except Exception:
            pass
        # fallback: simple circular approximation
        T  = t_gps / 86_400.0 / 27.321582  # fraction of lunar month
        th = 2*np.pi * T
        return MOON_A_KM * np.array([np.cos(th), np.sin(th)*np.cos(np.radians(5.1)),
                                      np.sin(th)*np.sin(np.radians(5.1))])

    def add_to_mpl(self, ax, orbit_state, traj=None, satellite=None, **kw):
        if not _HAS_MPL:
            return []
        self._load_texture()
        moon_pos = self._moon_position_km(orbit_state._epoch_gps())
        n = self.n_pts
        u = np.linspace(0, 2*np.pi, n)
        v = np.linspace(0,   np.pi, n)
        X = moon_pos[0] + MOON_R_KM * np.outer(np.cos(u), np.sin(v))
        Y = moon_pos[1] + MOON_R_KM * np.outer(np.sin(u), np.sin(v))
        Z = moon_pos[2] + MOON_R_KM * np.outer(np.ones(n), np.cos(v))
        if self._tex is not None:
            fcolors = np.zeros((*X.shape, 3))
            H, W = self._tex.shape[:2]
            for i in range(n):
                for j in range(n):
                    ti = int(i / n * H) % H
                    tj = int(j / n * W) % W
                    fcolors[j, i] = self._tex[ti, tj] / 255.0
            surf = ax.plot_surface(X, Y, Z, facecolors=fcolors, shade=False,
                                   rstride=1, cstride=1, zorder=3)
        else:
            surf = ax.plot_surface(X, Y, Z, color="#888888", alpha=0.9, zorder=3)
        self._artists_mpl = [surf]
        return [surf]

    def add_to_plotly(self, fig, orbit_state, traj=None, satellite=None, **kw):
        if not _HAS_PLOTLY:
            return
        try:
            moon_pos = self._moon_position_km(orbit_state._epoch_gps())
        except Exception as ex:
            print(f"[MoonLayer] position lookup failed ({ex}); using fixed fallback.")
            moon_pos = np.array([MOON_A_KM, 0.0, 0.0])
        n = self.n_pts
        u = np.linspace(0, 2*np.pi, n)
        v = np.linspace(0,   np.pi, n)
        _R = MOON_R_KM * self.radius_scale
        X = moon_pos[0] + _R * np.outer(np.cos(u), np.sin(v))
        Y = moon_pos[1] + _R * np.outer(np.sin(u), np.sin(v))
        Z = moon_pos[2] + _R * np.outer(np.ones(n), np.cos(v))
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            colorscale=[[0,"#888888"],[1,"#cccccc"]],
            showscale=False, opacity=0.95,
            name="Moon", hoverinfo="skip",
        ))
        # Marker fallback ensures the Moon is findable even at scales where the
        # sphere itself (radius ~1737 km) is too small to see against a scene
        # sized to lunar distance.
        fig.add_trace(go.Scatter3d(
            x=[moon_pos[0]], y=[moon_pos[1]], z=[moon_pos[2]],
            mode="markers+text", marker=dict(size=6, color="#cccccc"),
            text=["Moon"], textfont=dict(color="#cccccc", size=10),
            name="Moon marker", hoverinfo="skip", showlegend=False,
        ))


# ─── OrbitSunLayer ─────────────────────────────────────────────────────────────────
class OrbitSunLayer(BaseLayer):
    """Sun position marker + direction arrow + optional light-cone."""

    def __init__(self, arrow_scale_km: float = RE_KM * 5, **kw):
        super().__init__("sun", "Sun", **kw)
        self.arrow_scale_km = arrow_scale_km

    def _sun_pos(self, t_gps: float, scene_r: float) -> np.ndarray:
        try:
            import ssapy.compute
            s = ssapy.compute.sunPos(t_gps)
            if s.ndim == 2:
                s = s[:, 0]
            # s/|s| is a dimensionless unit vector — scale it directly by the
            # desired scene distance (km). The old code additionally divided
            # by 1e3, which placed the sun ~1000x closer than intended
            # (effectively right on top of Earth).
            return s / np.linalg.norm(s) * scene_r * 1.5
        except Exception:
            pass
        # fallback: simplified ecliptic
        jd = 2_444_244.5 + t_gps / 86_400
        n_days = jd - 2_451_545.0
        L = np.radians((280.460 + 0.9856474*n_days) % 360)
        g = np.radians((357.528 + 0.9856003*n_days) % 360)
        lam = L + np.radians(1.915*np.sin(g) + 0.020*np.sin(2*g))
        return np.array([np.cos(lam), np.sin(lam), 0]) * scene_r * 1.3

    def add_to_mpl(self, ax, orbit_state, traj=None, satellite=None,
                   scene_radius_km=None, **kw):
        if not _HAS_MPL:
            return []
        r = scene_radius_km or orbit_state.r_a
        t = orbit_state._epoch_gps()
        pos = self._sun_pos(t, r)
        a = ax.scatter(*pos, s=300, c="#FFD700", zorder=5, depthshade=False)
        origin = np.zeros(3)
        arrow = ax.quiver(*origin, *pos/np.linalg.norm(pos)*r*0.5,
                          color="#FFD700", alpha=0.4, linewidth=1)
        self._artists_mpl = [a, arrow]
        return [a, arrow]

    def add_to_plotly(self, fig, orbit_state, traj=None, satellite=None,
                      scene_radius_km=None, **kw):
        if not _HAS_PLOTLY:
            return
        r = scene_radius_km or max(orbit_state.r_a, RE_KM * 3)
        try:
            t = orbit_state._epoch_gps()
            pos = self._sun_pos(t, r)
        except Exception as ex:
            print(f"[OrbitSunLayer] position lookup failed ({ex}); using fixed fallback.")
            pos = np.array([r * 1.3, 0.0, 0.0])
        fig.add_trace(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            mode="markers+text",
            marker=dict(size=14, color="#FFD700", symbol="circle"),
            text=["☀"], textfont=dict(color="#FFD700", size=16),
            name="Sun", hovertemplate="Sun<extra></extra>",
        ))
        # Direction line from Earth toward the Sun — capped to a sane
        # fraction of the scene regardless of how far `pos` actually is.
        # (Previously a go.Cone glyph, which could render as an oversized
        # blob dominating the whole scene's autoscale — a thin dashed line,
        # consistent with how the dipole axis / L-point vectors are drawn
        # elsewhere in this file, can't do that.)
        try:
            u_hat = pos / np.linalg.norm(pos)
        except Exception:
            u_hat = np.array([1.0, 0.0, 0.0])
        arrow_len = min(self.arrow_scale_km, r * 0.4)
        tip = u_hat * arrow_len
        fig.add_trace(go.Scatter3d(
            x=[0, tip[0]], y=[0, tip[1]], z=[0, tip[2]],
            mode="lines",
            line=dict(color="#FFD700", width=3, dash="dash"),
            name="Sun direction", hoverinfo="skip", showlegend=False,
        ))


# ─── GroundTrackLayer ─────────────────────────────────────────────────────────
class GroundTrackLayer(BaseLayer):
    """Sub-satellite ground track projected on the Earth sphere."""

    def __init__(self, color: str = "#00FF9C", linewidth: float = 1.5,
                 n_pts: int = 300, **kw):
        super().__init__("groundtrack", "Ground Track", **kw)
        self.color     = color
        self.linewidth = linewidth
        self.n_pts     = n_pts

    def _compute(self, traj, orbit_state) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (x, y, z) on Earth surface along ground track."""
        from ssatk.coordinates.frames import eci_to_lon_lat
        if traj is None:
            traj = orbit_state.propagate(n_orbits=1, dt_s=60)
        # sub-sample
        step = max(1, len(traj.r) // self.n_pts)
        r = traj.r[::step]
        t = traj.t[::step]
        lons, lats = eci_to_lon_lat(r, t)
        lons_r = np.radians(lons)
        lats_r = np.radians(lats)
        x = RE_KM * np.cos(lats_r) * np.cos(lons_r)
        y = RE_KM * np.cos(lats_r) * np.sin(lons_r)
        z = RE_KM * np.sin(lats_r)
        return x, y, z

    def add_to_mpl(self, ax, orbit_state, traj=None, satellite=None, **kw):
        if not _HAS_MPL:
            return []
        x, y, z = self._compute(traj, orbit_state)
        line, = ax.plot(x, y, z, color=self.color,
                        linewidth=self.linewidth, zorder=2, alpha=0.8)
        self._artists_mpl = [line]
        return [line]

    def add_to_plotly(self, fig, orbit_state, traj=None, satellite=None, **kw):
        if not _HAS_PLOTLY:
            return
        x, y, z = self._compute(traj, orbit_state)
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="lines",
            line=dict(color=self.color, width=2),
            name="Ground Track",
            hoverinfo="skip",
        ))


# ─── TerminatorLayer ─────────────────────────────────────────────────────────
class TerminatorLayer(BaseLayer):
    """Day/night terminator circle on Earth's surface."""

    def __init__(self, color: str = "#FFB830", alpha: float = 0.5, **kw):
        super().__init__("terminator", "Day/Night Terminator", **kw)
        self.color = color
        self.alpha = alpha

    def _terminator_xyz(self, t_gps: float) -> tuple:
        from .layers import OrbitSunLayer
        dummy = OrbitSunLayer()
        sun = dummy._sun_pos(t_gps, RE_KM * 200)
        sun_hat = sun / np.linalg.norm(sun)
        # Terminator is a great circle perpendicular to the sun direction
        # Build an orthonormal basis
        if abs(sun_hat[0]) < 0.9:
            ref = np.array([1, 0, 0])
        else:
            ref = np.array([0, 1, 0])
        e1 = np.cross(sun_hat, ref); e1 /= np.linalg.norm(e1)
        e2 = np.cross(sun_hat, e1); e2 /= np.linalg.norm(e2)
        theta = np.linspace(0, 2*np.pi, 360)
        pts = RE_KM * (np.outer(np.cos(theta), e1) + np.outer(np.sin(theta), e2))
        return pts[:, 0], pts[:, 1], pts[:, 2]

    def add_to_mpl(self, ax, orbit_state, traj=None, satellite=None, **kw):
        if not _HAS_MPL:
            return []
        x, y, z = self._terminator_xyz(orbit_state._epoch_gps())
        line, = ax.plot(x, y, z, color=self.color,
                        linewidth=1.5, alpha=self.alpha, zorder=2, linestyle="--")
        self._artists_mpl = [line]
        return [line]

    def add_to_plotly(self, fig, orbit_state, traj=None, satellite=None, **kw):
        if not _HAS_PLOTLY:
            return
        x, y, z = self._terminator_xyz(orbit_state._epoch_gps())
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="lines",
            line=dict(color=self.color, width=2, dash="dash"),
            name="Terminator", hoverinfo="skip",
        ))


# ─── EclipseLayer ─────────────────────────────────────────────────────────────
class EclipseLayer(BaseLayer):
    """Mark eclipse entry/exit on the orbit track."""

    def __init__(self, color_shadow: str = "#333366", **kw):
        super().__init__("eclipse", "Eclipse", **kw)
        self.color_shadow = color_shadow

    def _eclipse_mask(self, traj, orbit_state) -> np.ndarray:
        """Return boolean mask — True where spacecraft is in Earth's shadow."""
        try:
            import ssapy.compute
            t_arr = traj.t
            sun   = ssapy.compute.sunPos(t_arr)  # (3, N) metres
            if sun.ndim == 1:
                sun = sun[:, np.newaxis] * np.ones((1, len(t_arr)))
            r_m   = traj.r.T * 1e3                # (3, N) metres
            # cylinder shadow model
            shadow = ssapy.compute.earthShadowCoords(r_m, sun)
            in_shadow = (shadow[0] < 0) & (shadow[1] < RE_KM*1e3)
            return in_shadow
        except Exception:
            # geometric fallback: cylindrical Earth shadow
            t_arr = traj.t
            # sun direction via OrbitSunLayer helper
            sl = OrbitSunLayer()
            masks = []
            for i, t in enumerate(t_arr):
                sun_hat = sl._sun_pos(t, 1.0); sun_hat /= np.linalg.norm(sun_hat)
                r = traj.r[i]
                # project satellite onto anti-sun direction
                proj = -np.dot(r, sun_hat)
                perp = np.linalg.norm(r - proj*(-sun_hat))
                masks.append(proj > 0 and perp < RE_KM)
            return np.array(masks)

    def add_to_mpl(self, ax, orbit_state, traj=None, satellite=None, **kw):
        if not _HAS_MPL or traj is None:
            return []
        mask = self._eclipse_mask(traj, orbit_state)
        r_shadow = traj.r[mask]
        if len(r_shadow) == 0:
            return []
        a = ax.scatter(r_shadow[:, 0], r_shadow[:, 1], r_shadow[:, 2],
                       s=4, c=self.color_shadow, alpha=0.6, zorder=3, depthshade=False)
        self._artists_mpl = [a]
        return [a]

    def add_to_plotly(self, fig, orbit_state, traj=None, satellite=None, **kw):
        if not _HAS_PLOTLY:
            return
        if traj is None:
            print("[EclipseLayer] no propagated trajectory available — "
                  "run ▶ Full propagation (not Instant preview) to see eclipse markers.")
            return
        try:
            mask = self._eclipse_mask(traj, orbit_state)
        except Exception as ex:
            print(f"[EclipseLayer] eclipse computation failed: {ex}")
            return
        r_s = traj.r[mask]
        if len(r_s) == 0:
            print("[EclipseLayer] no shadow crossing found in the propagated window "
                  "— try increasing Orbits, or the orbit plane may rarely cross Earth's shadow.")
            return
        fig.add_trace(go.Scatter3d(
            x=r_s[:, 0], y=r_s[:, 1], z=r_s[:, 2],
            mode="markers",
            marker=dict(size=3, color=self.color_shadow, opacity=0.8,
                        line=dict(color="#FFFFFF", width=0.5)),
            name="Eclipse", hoverinfo="skip",
        ))


# ─── VanAllenLayer ───────────────────────────────────────────────────────────
class VanAllenLayer(BaseLayer):
    """Inner (1–2 RE) and outer (3–6 RE) Van Allen belt tori."""

    INNER = dict(r_min=1.0, r_max=2.0, color_plotly="#FFB830", color_mpl="gold",      alpha=0.3)
    OUTER = dict(r_min=3.0, r_max=6.0, color_plotly="#7B2FFF", color_mpl="mediumpurple", alpha=0.2)

    def __init__(self, show_inner=True, show_outer=True, n_pts=40, **kw):
        super().__init__("van_allen", "Van Allen Belts", **kw)
        self.show_inner = show_inner
        self.show_outer = show_outer
        self.n_pts = n_pts

    def _torus_mesh(self, R: float, r: float, n: int):
        """(x,y,z) for a torus with major radius R, tube radius r."""
        u = np.linspace(0, 2*np.pi, n)
        v = np.linspace(0, 2*np.pi, n)
        U, V = np.meshgrid(u, v)
        X = (R + r*np.cos(V)) * np.cos(U)
        Y = (R + r*np.cos(V)) * np.sin(U)
        Z = r * np.sin(V)
        return X*RE_KM, Y*RE_KM, Z*RE_KM

    def _add_belt(self, spec: dict, ax_or_fig, mode: str):
        R_mid = (spec["r_min"] + spec["r_max"]) / 2
        r_tube = (spec["r_max"] - spec["r_min"]) / 2
        X, Y, Z = self._torus_mesh(R_mid, r_tube, self.n_pts)
        if mode == "mpl" and _HAS_MPL:
            ax_or_fig.plot_surface(X, Y, Z, color=spec["color_mpl"],
                                   alpha=spec["alpha"], zorder=1)
        elif mode == "plotly" and _HAS_PLOTLY:
            vx=X.ravel(); vy=Y.ravel(); vz=Z.ravel()
            n=self.n_pts
            i_idx, j_idx, k_idx = [], [], []
            for r in range(n-1):
                for c in range(n-1):
                    v0=r*n+c; v1=v0+1; v2=(r+1)*n+c; v3=v2+1
                    i_idx+=[v0,v1]; j_idx+=[v1,v3]; k_idx+=[v2,v2]
            ax_or_fig.add_trace(go.Mesh3d(
                x=vx, y=vy, z=vz,
                i=i_idx, j=j_idx, k=k_idx,
                color=spec["color_plotly"], opacity=spec["alpha"],
                name=f"Van Allen {'Inner' if spec is self.INNER else 'Outer'}",
                hoverinfo="skip", showlegend=True,
            ))

    def add_to_mpl(self, ax, orbit_state, traj=None, satellite=None, **kw):
        if self.show_inner: self._add_belt(self.INNER, ax, "mpl")
        if self.show_outer: self._add_belt(self.OUTER, ax, "mpl")
        return []

    def add_to_plotly(self, fig, orbit_state, traj=None, satellite=None, **kw):
        if self.show_inner: self._add_belt(self.INNER, fig, "plotly")
        if self.show_outer: self._add_belt(self.OUTER, fig, "plotly")


# ─── MagfieldLayer ───────────────────────────────────────────────────────────
class MagfieldLayer(BaseLayer):
    """IGRF 2025 field lines traced with ppigrf + vectorised RK4."""

    SEED_LATS = [20, 30, 40, 55, 65, 75]
    MAX_R_RE  = 15.0

    def __init__(self, seed_lats=None, max_r_re=None, **kw):
        super().__init__("magfield", "Magnetic Field Lines", **kw)
        self.seed_lats = seed_lats or self.SEED_LATS
        self.max_r_re  = max_r_re  or self.MAX_R_RE

    def _dipole_axis(self) -> np.ndarray:
        """IGRF 2025 north magnetic pole unit vector."""
        lon = np.radians(136.0); lat = np.radians(85.5)
        return np.array([np.cos(lat)*np.cos(lon),
                         np.cos(lat)*np.sin(lon),
                         np.sin(lat)])

    def _trace_lines(self):
        """Return list of (N,3) field line arrays in km."""
        if not _HAS_PPIGRF:
            warnings.warn("ppigrf not installed — MagfieldLayer disabled")
            return []
        import datetime
        lines = []
        pole = self._dipole_axis()
        ref  = np.array([1,0,0]) if abs(pole[0]) < 0.9 else np.array([0,1,0])
        e1 = np.cross(pole, ref); e1 /= np.linalg.norm(e1)
        e2 = np.cross(pole, e1)

        date = datetime.datetime.now()
        max_r = self.max_r_re * RE_KM

        def B_ECI(r_km):
            r_m = np.linalg.norm(r_km)
            if r_m < RE_KM * 0.99:
                return None
            lat = np.degrees(np.arcsin(r_km[2] / r_m))
            lon = np.degrees(np.arctan2(r_km[1], r_km[0]))
            alt_km = r_m - RE_KM
            try:
                bn, be, bd = ppigrf.igrf(lon, lat, alt_km, date)
                # NED → ECI
                lat_r = np.radians(lat); lon_r = np.radians(lon)
                en = np.array([-np.sin(lat_r)*np.cos(lon_r),
                               -np.sin(lat_r)*np.sin(lon_r),
                                np.cos(lat_r)])
                ee = np.array([-np.sin(lon_r), np.cos(lon_r), 0])
                ed = np.array([-np.cos(lat_r)*np.cos(lon_r),
                               -np.cos(lat_r)*np.sin(lon_r),
                               -np.sin(lat_r)])
                B = float(bn)*en + float(be)*ee - float(bd)*ed
                mag = np.linalg.norm(B)
                return B / mag if mag > 1e-15 else None
            except Exception:
                return None

        for sign in [1, -1]:
            for lat_deg in self.seed_lats:
                theta = np.radians(90 - lat_deg)
                for phi in np.linspace(0, 2*np.pi, 6, endpoint=False):
                    seed = RE_KM * 1.02 * (
                        np.sin(theta)*np.cos(phi)*e1 +
                        np.sin(theta)*np.sin(phi)*e2 +
                        np.cos(theta)*pole
                    )
                    pts = [seed.copy()]
                    r   = seed.copy()
                    h   = 50.0 * sign  # km step
                    for _ in range(4000):
                        if np.linalg.norm(r) > max_r:
                            break
                        b = B_ECI(r)
                        if b is None:
                            break
                        k1 = h * b
                        b2 = B_ECI(r + 0.5*k1)
                        if b2 is None: break
                        k2 = h * b2
                        b3 = B_ECI(r + 0.5*k2)
                        if b3 is None: break
                        k3 = h * b3
                        b4 = B_ECI(r + k3)
                        if b4 is None: break
                        k4 = h * b4
                        r  = r + (k1 + 2*k2 + 2*k3 + k4) / 6
                        pts.append(r.copy())
                    if len(pts) > 2:
                        lines.append(np.array(pts))
        return lines

    def add_to_mpl(self, ax, orbit_state, traj=None, satellite=None, **kw):
        if not _HAS_MPL:
            return []
        lines = self._trace_lines()
        artists = []
        for line in lines:
            a, = ax.plot(line[:, 0], line[:, 1], line[:, 2],
                         color="#00BFFF", linewidth=0.7, alpha=0.6, zorder=1)
            artists.append(a)
        self._artists_mpl = artists
        return artists

    def add_to_plotly(self, fig, orbit_state, traj=None, satellite=None, **kw):
        if not _HAS_PLOTLY:
            return
        lines = self._trace_lines()
        for i, line in enumerate(lines):
            fig.add_trace(go.Scatter3d(
                x=line[:, 0], y=line[:, 1], z=line[:, 2],
                mode="lines",
                line=dict(color="#00BFFF", width=1),
                name="Field lines" if i == 0 else None,
                showlegend=(i == 0),
                hoverinfo="skip",
            ))


# ─── LagrangeLayer ───────────────────────────────────────────────────────────
class LagrangeLayer(BaseLayer):
    """Earth–Moon L1–L5 Lagrange point markers."""

    def __init__(self, **kw):
        super().__init__("lagrange", "Lagrange Points", **kw)

    def _points(self, moon_pos_km: np.ndarray):
        """Return dict of point label → ECI position."""
        m_hat = moon_pos_km / np.linalg.norm(moon_pos_km)
        perp  = np.cross(m_hat, [0, 0, 1]); perp /= np.linalg.norm(perp)
        d     = np.linalg.norm(moon_pos_km)
        # approximate distances (mass-ratio for Earth-Moon)
        mu_ratio = 0.01215
        r_L1 = d * (1 - (mu_ratio/3)**(1/3))
        r_L2 = d * (1 + (mu_ratio/3)**(1/3))
        r_L3 = -d * (1 + 5/12 * mu_ratio)
        return {
            "L1": r_L1 * m_hat,
            "L2": r_L2 * m_hat,
            "L3": r_L3 * m_hat,
            "L4": d * (m_hat*0.5 + perp*np.sqrt(3)/2),
            "L5": d * (m_hat*0.5 - perp*np.sqrt(3)/2),
        }

    def add_to_mpl(self, ax, orbit_state, traj=None, satellite=None, **kw):
        if not _HAS_MPL:
            return []
        ml = MoonLayer()
        moon_pos = ml._moon_position_km(orbit_state._epoch_gps())
        pts = self._points(moon_pos)
        artists = []
        colors = {"L1":"#ff6b6b","L2":"#ffd93d","L3":"#6bcb77","L4":"#4d96ff","L5":"#c77dff"}
        for label, pos in pts.items():
            a = ax.scatter(*pos, s=60, c=colors[label], zorder=5, depthshade=False)
            t = ax.text(*pos, f" {label}", color=colors[label], fontsize=7, zorder=5)
            artists.extend([a, t])
        self._artists_mpl = artists
        return artists

    def add_to_plotly(self, fig, orbit_state, traj=None, satellite=None, **kw):
        if not _HAS_PLOTLY:
            return
        try:
            ml = MoonLayer()
            moon_pos = ml._moon_position_km(orbit_state._epoch_gps())
            pts = self._points(moon_pos)
        except Exception as ex:
            print(f"[LagrangeLayer] computation failed: {ex}")
            return
        colors = {"L1":"#ff6b6b","L2":"#ffd93d","L3":"#6bcb77","L4":"#4d96ff","L5":"#c77dff"}
        for label, pos in pts.items():
            fig.add_trace(go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                mode="markers+text",
                marker=dict(size=8, color=colors[label], line=dict(color="#FFFFFF", width=1)),
                text=[label],
                textfont=dict(color=colors[label], size=12),
                name=label,
            ))


# ─── NTWLayer ────────────────────────────────────────────────────────────────
class NTWLayer(BaseLayer):
    """Draw T, N, W axis vectors on the satellite position."""

    COLORS = {"T": "#00FF9C", "N": "#FF4D6D", "W": "#4D96FF"}

    def __init__(self, satellite=None, **kw):
        super().__init__("ntw", "NTW Frame Vectors", **kw)
        self._sat = satellite

    def add_to_mpl(self, ax, orbit_state, traj=None, satellite=None, **kw):
        if not _HAS_MPL:
            return []
        sat = satellite or self._sat
        if sat is None or not sat.show_ntw:
            return []
        r, v = orbit_state.to_rv()
        T, N, W = sat.ntw_vectors(r, v)
        artists = []
        for label, vec in [("T", T), ("N", N), ("W", W)]:
            a = ax.quiver(*r, *vec, color=self.COLORS[label],
                          linewidth=2, arrow_length_ratio=0.2, zorder=5)
            artists.append(a)
        self._artists_mpl = artists
        return artists

    def add_to_plotly(self, fig, orbit_state, traj=None, satellite=None, **kw):
        if not _HAS_PLOTLY:
            return
        sat = satellite or self._sat
        if sat is None or not sat.show_ntw:
            return
        r, v = orbit_state.to_rv()
        T, N, W = sat.ntw_vectors(r, v)
        for label, vec in [("T", T), ("N", N), ("W", W)]:
            tip = r + vec
            fig.add_trace(go.Scatter3d(
                x=[r[0], tip[0]], y=[r[1], tip[1]], z=[r[2], tip[2]],
                mode="lines",
                line=dict(color=self.COLORS[label], width=4),
                name=label,
            ))


# ─── BurnLayer ───────────────────────────────────────────────────────────────
class BurnLayer(BaseLayer):
    """Draw Δv arrows for each BurnEvent in the Satellite3D."""

    def __init__(self, satellite=None, **kw):
        super().__init__("burns", "Burn Events", **kw)
        self._sat = satellite

    def add_to_mpl(self, ax, orbit_state, traj=None, satellite=None, **kw):
        if not _HAS_MPL or traj is None:
            return []
        sat = satellite or self._sat
        if sat is None or not sat.burns:
            return []
        artists = []
        results = sat.apply_burns_to_trajectory(traj, orbit_state)
        for new_state, idx, burn in results:
            r = traj.r[idx]
            v = traj.v[idx]
            dv_eci = sat.burn_vector_eci(burn, r, v)
            a = ax.quiver(*r, *dv_eci, color="#FFB830",
                          linewidth=2.5, arrow_length_ratio=0.3, zorder=6)
            dot = ax.scatter(*r, s=80, c="#FFB830", zorder=7, depthshade=False)
            artists.extend([a, dot])
        self._artists_mpl = artists
        return artists

    def add_to_plotly(self, fig, orbit_state, traj=None, satellite=None, **kw):
        if not _HAS_PLOTLY or traj is None:
            return
        sat = satellite or self._sat
        if sat is None or not sat.burns:
            return
        results = sat.apply_burns_to_trajectory(traj, orbit_state)
        for new_state, idx, burn in results:
            r   = traj.r[idx]
            v   = traj.v[idx]
            dv  = sat.burn_vector_eci(burn, r, v)
            tip = r + dv
            fig.add_trace(go.Scatter3d(
                x=[r[0], tip[0]], y=[r[1], tip[1]], z=[r[2], tip[2]],
                mode="lines+markers",
                line=dict(color="#FFB830", width=4),
                marker=dict(size=[5, 0], color="#FFB830"),
                name=burn.label,
                hovertemplate=f"{burn.label}<br>Δv={burn.dv_mag_m_s:.1f} m/s<extra></extra>",
            ))


# ─── SensorFOVLayer ──────────────────────────────────────────────────────────
class SensorFOVLayer(BaseLayer):
    """
    Sensor Field-of-View cone on a Plotly 3D scene.

    The cone apex is placed at the satellite position at *time_index* along a
    propagated orbit.  Pointing modes:

      "nadir"         — toward Earth centre  (Earth-observing sensor)
      "anti-nadir"    — away from Earth       (star tracker / deep-space comm)
      "velocity"      — ram direction         (forward-facing sensor)
      "anti-velocity" — anti-ram
      "custom"        — fixed GCRF unit vector you supply

    Usage
    -----
    ::

        fov = SensorFOVLayer(
            r_gcrf_km=r_arr / 1e3,    # (N,3) metres → km
            v_gcrf_kms=v_arr / 1e3,   # (N,3) m/s    → km/s  (needed for velocity modes)
            time_index=42,
            half_angle_deg=15.0,
            cone_length_km=8_000.0,
            pointing_mode="nadir",
        )
        fov.add_to_plotly(fig, orbit_state)
    """

    POINTING_MODES = ("nadir", "anti-nadir", "velocity", "anti-velocity", "custom")

    def __init__(
        self,
        r_gcrf_km,
        v_gcrf_kms=None,
        time_index: int = 0,
        half_angle_deg: float = 15.0,
        cone_length_km: float = 8_000.0,
        pointing_mode: str = "nadir",
        custom_direction=(1.0, 0.0, 0.0),
        color: str = "#00CED1",
        opacity: float = 0.35,
        n_sides: int = 64,
        show_boresight: bool = True,
        boresight_color: str = "#FFFFFF",
        sun_direction_gcrf=None,
        show_sun_shading: bool = True,
        show_footprint: bool = True,
        **kw,
    ):
        super().__init__("sensor_fov", "Sensor FOV", **kw)
        self.r_gcrf_km        = np.asarray(r_gcrf_km,  dtype=float)
        self.v_gcrf_kms       = np.asarray(v_gcrf_kms, dtype=float) if v_gcrf_kms is not None else None
        self.time_index       = int(time_index)
        self.half_angle_deg   = float(half_angle_deg)
        self.cone_length_km   = float(cone_length_km)
        self.pointing_mode    = pointing_mode
        self.custom_direction = np.asarray(custom_direction, dtype=float)
        self.color            = color
        self.opacity          = opacity
        self.n_sides          = int(n_sides)
        self.show_boresight   = show_boresight
        self.boresight_color  = boresight_color
        self.sun_direction_gcrf = (np.asarray(sun_direction_gcrf, dtype=float)
                                   if sun_direction_gcrf is not None else None)
        self.show_sun_shading = show_sun_shading
        self.show_footprint   = show_footprint

        if pointing_mode not in self.POINTING_MODES:
            raise ValueError(f"pointing_mode must be one of {self.POINTING_MODES}, got {pointing_mode!r}")
        if pointing_mode in ("velocity", "anti-velocity") and v_gcrf_kms is None:
            raise ValueError("v_gcrf_kms required for velocity / anti-velocity pointing modes.")

    @property
    def apex_km(self) -> np.ndarray:
        return self.r_gcrf_km[self.time_index]

    @property
    def boresight_direction(self) -> np.ndarray:
        r = self.r_gcrf_km[self.time_index]
        m = self.pointing_mode
        if   m == "nadir":         d = -r
        elif m == "anti-nadir":    d =  r
        elif m == "velocity":      d =  self.v_gcrf_kms[self.time_index]
        elif m == "anti-velocity": d = -self.v_gcrf_kms[self.time_index]
        else:                      d =  self.custom_direction.copy()
        norm = np.linalg.norm(d)
        if norm < 1e-12:
            raise ValueError("Boresight direction vector has zero length.")
        return d / norm

    def add_to_mpl(self, ax, orbit_state, traj=None, satellite=None, **kw):
        # Matplotlib cone rendering not implemented; silently skip.
        return []

    def build_traces(self) -> list:
        """Return all Plotly traces for the current time_index.

        Always returns exactly 5 traces in a fixed order so Plotly animation
        frames can reliably reference them by index:

        * Satellite marker (Scatter3d)
        * FOV cone (Mesh3d)
        * Boresight line (Scatter3d)
        * Footprint day arc (Scatter3d)
        * Footprint night arc (Scatter3d)

        Missing / disabled items are replaced by empty placeholder traces.
        """
        if not _HAS_PLOTLY:
            return []

        apex      = self.apex_km
        direction = self.boresight_direction
        sun_hat   = self.sun_direction_gcrf if self.show_sun_shading else None

        _empty_scatter = go.Scatter3d(x=[], y=[], z=[], mode="lines",
                                       showlegend=False, hoverinfo="skip")
        _empty_mesh    = go.Mesh3d(x=[], y=[], z=[], i=[], j=[], k=[],
                                    showlegend=False, hoverinfo="skip")

        # [0] Satellite marker
        pos = apex
        sat = go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            mode="markers",
            marker=dict(size=7, color="#FFD700", symbol="circle",
                        line=dict(color="#FFFFFF", width=1)),
            name="Satellite", showlegend=True,
        )

        # [1] FOV cone
        cone = _cone_mesh3d(apex, direction, self.half_angle_deg, self.cone_length_km,
                            self.n_sides, self.color, self.opacity, self.name,
                            sun_hat=sun_hat)
        cone = cone if cone is not None else _empty_mesh

        # [2] Boresight line
        if self.show_boresight:
            bs = _boresight_line3d(apex, direction, self.cone_length_km,
                                   self.boresight_color)
            bs = bs if bs is not None else _empty_scatter
        else:
            bs = _empty_scatter

        # [3] + [4] Ground footprint day / night arcs
        fp_day   = _empty_scatter
        fp_night = _empty_scatter
        if self.show_footprint and sun_hat is not None:
            fp = _footprint_on_earth(apex, direction, self.half_angle_deg)
            if fp is not None:
                xs, ys, zs = fp
                arcs = _footprint_illumination_traces(xs, ys, zs, sun_hat,
                                                      self.color, "Footprint")
                if len(arcs) >= 1: fp_day   = arcs[0]
                if len(arcs) >= 2: fp_night = arcs[1]

        return [sat, cone, bs, fp_day, fp_night]

    def add_to_plotly(self, fig, orbit_state, traj=None, satellite=None, **kw):
        if not _HAS_PLOTLY:
            return
        for t in self.build_traces():
            fig.add_trace(t)


# ── Factory ───────────────────────────────────────────────────────────────────
_LAYER_REGISTRY: dict[str, type] = {
    "stars":       StarfieldLayer,
    "earth":       EarthLayer,
    "moon":        MoonLayer,
    "sun":         OrbitSunLayer,
    "groundtrack": GroundTrackLayer,
    "terminator":  TerminatorLayer,
    "eclipse":     EclipseLayer,
    "van_allen":   VanAllenLayer,
    "magfield":    MagfieldLayer,
    "lagrange":    LagrangeLayer,
    "ntw":         NTWLayer,
    "burns":       BurnLayer,
    "sensor_fov":  SensorFOVLayer,
}


def create_layer(key: str, **kwargs) -> BaseLayer:
    """
    Factory function.  Creates a layer by its key string.

    Example
    -------
    layer = create_layer("earth", texture_path="/path/to/earth.png")
    """
    if key not in _LAYER_REGISTRY:
        raise KeyError(f"Unknown layer '{key}'. Available: {list(_LAYER_REGISTRY)}")
    return _LAYER_REGISTRY[key](**kwargs)


def available_layers() -> list[str]:
    return list(_LAYER_REGISTRY)
