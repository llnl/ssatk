"""
core/base_plot.py
─────────────────
BasePlot3D   — matplotlib render engine (static exports, animations)
PlotlyScene  — Plotly render engine (interactive, used by Streamlit GUI)

Both share the same layer system and OrbitalState API.

Usage — matplotlib
------------------
from ssatk.plots import OrbitalState, BasePlot3D
from ssatk.plots.layers import EarthLayer, StarfieldLayer, GroundTrackLayer

state = OrbitalState(a_km=6928, e=0.001, inc_deg=51.6)
plot  = BasePlot3D(state, figsize=(12, 9))
plot.add_layer(StarfieldLayer("/path/bright_stars.csv"))
plot.add_layer(EarthLayer("/path/earth.png"))
plot.add_layer(GroundTrackLayer())
plot.render()
plot.save("output/orbit.png")

Usage — Plotly (Streamlit)
--------------------------
from core import OrbitalState, PlotlyScene

state = OrbitalState(a_km=6928, e=0.001, inc_deg=51.6)
scene = PlotlyScene(state)
scene.add_layer("earth")
scene.add_layer("stars", catalog_path="/path/bright_stars.csv")
scene.add_layer("groundtrack")
fig = scene.build()
st.plotly_chart(fig, use_container_width=True)
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# ── optional imports ──────────────────────────────────────────────────────────
try:
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

from .layers import BaseLayer, create_layer
from .orbit_state import OrbitalState, Trajectory
from ssatk.coordinates.frames import Frame, FrameTransform

# ── palette ───────────────────────────────────────────────────────────────────
_ORBIT_COLOR   = "#00FF9C"
_SAT_COLOR     = "#FFB830"
_BG_COLOR      = "#09090F"
_GRID_COLOR    = "#1a2a3a"
_TEXT_COLOR    = "#C8D8E8"


def _coerce_orbital_state(state):
    if isinstance(state, OrbitalState):
        return state
    if all(hasattr(state, attr) for attr in ("r", "v", "t")):
        return OrbitalState.from_ssapy(state)
    raise TypeError("state must be an OrbitalState or SSAPy Orbit-like object")


# ══════════════════════════════════════════════════════════════════════════════
# BasePlot3D  (matplotlib)
# ══════════════════════════════════════════════════════════════════════════════
class BasePlot3D:
    """
    Matplotlib-based 3D plot engine with pluggable layers.

    Attributes
    ----------
    state     : OrbitalState
    frame     : Frame        reference frame for display
    dark      : bool         dark background
    layers    : dict         key → BaseLayer
    satellite : Satellite3D | None
    """

    def __init__(
        self,
        state    : OrbitalState,
        figsize  : tuple = (12, 9),
        dark     : bool  = True,
        frame    : Frame | str = Frame.ECI,
        satellite = None,
    ):
        if not _HAS_MPL:
            raise ImportError("matplotlib is required for BasePlot3D")
        self.state     = _coerce_orbital_state(state)
        self.dark      = dark
        self.frame     = Frame(frame)
        self.satellite = satellite
        self.layers: dict[str, BaseLayer] = {}

        self._traj: Optional[Trajectory] = None
        self._traj_lock = threading.Lock()

        self._fidelity = "fast"
        self._iers_thread: Optional[threading.Thread] = None
        self._on_fidelity_change: Optional[Callable] = None

        # matplotlib objects
        self._fig: Optional[plt.Figure] = None
        self._ax:  Optional[Axes3D]     = None
        self._orbit_line  = None
        self._sat_dot     = None
        self._anim: Optional[FuncAnimation] = None

        self._figsize = figsize
        self._setup_figure()

        # start background IERS upgrade
        self._start_iers_thread()

    # ── figure init ───────────────────────────────────────────────────────────
    def _setup_figure(self):
        if self.dark:
            plt.style.use("dark_background")
        self._fig = plt.figure(figsize=self._figsize, facecolor=_BG_COLOR)
        self._ax  = self._fig.add_subplot(111, projection="3d")
        self._style_axes()

    def _style_axes(self):
        ax = self._ax
        ax.set_facecolor(_BG_COLOR)
        ax.grid(True, color=_GRID_COLOR, linestyle=":", linewidth=0.5)
        ax.xaxis.pane.fill = False; ax.xaxis.pane.set_edgecolor(_GRID_COLOR)
        ax.yaxis.pane.fill = False; ax.yaxis.pane.set_edgecolor(_GRID_COLOR)
        ax.zaxis.pane.fill = False; ax.zaxis.pane.set_edgecolor(_GRID_COLOR)
        ax.tick_params(colors=_TEXT_COLOR, labelsize=7)
        ax.set_xlabel("X (km)", color=_TEXT_COLOR, fontsize=8, labelpad=4)
        ax.set_ylabel("Y (km)", color=_TEXT_COLOR, fontsize=8, labelpad=4)
        ax.set_zlabel("Z (km)", color=_TEXT_COLOR, fontsize=8, labelpad=4)

    # ── layer management ──────────────────────────────────────────────────────
    def add_layer(self, layer: BaseLayer | str, **kwargs) -> "BasePlot3D":
        if isinstance(layer, str):
            layer = create_layer(layer, **kwargs)
        self.layers[layer.key] = layer
        return self

    def remove_layer(self, key: str) -> "BasePlot3D":
        if key in self.layers:
            self.layers[key].remove_from_mpl()
            del self.layers[key]
        return self

    def toggle_layer(self, key: str, enabled: bool | None = None) -> "BasePlot3D":
        if key in self.layers:
            l = self.layers[key]
            l.enabled = not l.enabled if enabled is None else enabled
        return self

    # ── frame control ─────────────────────────────────────────────────────────
    def set_frame(self, frame: Frame | str) -> "BasePlot3D":
        self.frame = Frame(frame)
        return self

    # ── IERS background upgrade ───────────────────────────────────────────────
    @property
    def fidelity(self) -> str:
        """Return the current Earth-orientation fidelity state."""
        return self._fidelity

    def _start_iers_thread(self):
        def _worker():
            try:
                import warnings
                from astropy.utils import iers
                self._fidelity = "loading"
                if self._on_fidelity_change:
                    self._on_fidelity_change("loading")
                # Pre-load the bundled IERS-B table without touching astropy's
                # process-wide IERS configuration. This used to set
                # conf.auto_download = False and conf.auto_max_age = None for
                # the whole process; on astropy < 8 (the only versions for
                # Python 3.10) IERS_Auto then serves IERS-B, which ends weeks
                # before today, so every later UT1 conversion anywhere in the
                # process raised IERSRangeError.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    iers.IERS_B.open()
                self._fidelity = "high"
                if self._on_fidelity_change:
                    self._on_fidelity_change("high")
            except Exception:
                self._fidelity = "fast"
                if self._on_fidelity_change:
                    self._on_fidelity_change("fast")

        self._iers_thread = threading.Thread(target=_worker, daemon=True)
        self._iers_thread.start()

    def on_fidelity_change(self, callback: Callable[[str], None]) -> "BasePlot3D":
        """Register a callback (fidelity_str) → None called when IERS status changes."""
        self._on_fidelity_change = callback
        return self

    # ── compute / cache trajectory ────────────────────────────────────────────
    def _get_trajectory(self, n_orbits: float = 3.0, dt_s: float = 60.0) -> Trajectory:
        with self._traj_lock:
            if self._traj is None:
                self._traj = self.state.propagate(n_orbits=n_orbits, dt_s=dt_s)
        return self._traj

    def invalidate_trajectory(self):
        with self._traj_lock:
            self._traj = None

    # ── frame transform of trajectory ─────────────────────────────────────────
    def _transform_traj(self, traj: Trajectory) -> np.ndarray:
        tf = FrameTransform(self.frame)
        return tf.transform_trajectory(traj.r, traj.v, traj.t)

    # ── osculating ellipse (instant, no propagation) ─────────────────────────
    def draw_osculating(self, color: str = _ORBIT_COLOR, linewidth: float = 1.5):
        """Draw the analytic osculating ellipse.  Returns the line artist."""
        pts = self.state.osculating_ellipse(n_pts=360)
        tf  = FrameTransform(self.frame)
        r0, v0 = self.state.to_rv()
        # For LVLH/NTW we transform each point; ECI/ECF need different approach
        pts_t = tf.transform_trajectory(
            pts,
            np.tile(v0, (len(pts), 1)),   # velocity constant (approx)
            np.full(len(pts), self.state._epoch_gps()),
        )
        if self._orbit_line:
            try:
                self._orbit_line.remove()
            except Exception:
                pass
        line, = self._ax.plot(
            pts_t[:, 0], pts_t[:, 1], pts_t[:, 2],
            color=color, linewidth=linewidth, zorder=2,
        )
        self._orbit_line = line
        # set axes limits
        r_max = np.max(np.linalg.norm(pts, axis=1)) * 1.15
        self._ax.set_xlim(-r_max, r_max)
        self._ax.set_ylim(-r_max, r_max)
        self._ax.set_zlim(-r_max, r_max)
        self._ax.set_box_aspect((1, 1, 1))   # Matplotlib's default is 4:4:3
        return line

    # ── full render ───────────────────────────────────────────────────────────
    def render(
        self,
        n_orbits   : float = 3.0,
        dt_s       : float = 60.0,
        orbit_color: str   = _ORBIT_COLOR,
        sat_color  : str   = _SAT_COLOR,
        title      : str | None = None,
    ) -> plt.Figure:
        """
        Full render: propagate, draw all layers, draw orbit track + satellite dot.
        Returns the matplotlib Figure.
        """
        ax = self._ax
        ax.cla()
        self._style_axes()

        traj = self._get_trajectory(n_orbits, dt_s)
        r_t  = self._transform_traj(traj)

        # scene radius for layer scaling
        r_max = np.max(np.linalg.norm(traj.r, axis=1)) * 1.15
        ax.set_xlim(-r_max, r_max)
        ax.set_ylim(-r_max, r_max)
        ax.set_zlim(-r_max, r_max)
        ax.set_box_aspect((1, 1, 1))   # Matplotlib's default is 4:4:3

        # ── draw layers (back to front) ───────────────────────────────────
        layer_order = [
            "stars", "van_allen", "magfield",
            "earth", "moon", "sun", "terminator",
            "groundtrack", "eclipse",
            "lagrange", "burns", "ntw",
        ]
        drawn = set()
        for key in layer_order:
            if key in self.layers and self.layers[key].enabled:
                self.layers[key].add_to_mpl(
                    ax, self.state, traj=traj,
                    satellite=self.satellite,
                    scene_radius_km=r_max,
                )
                drawn.add(key)
        # any extra layers not in the ordered list
        for key, layer in self.layers.items():
            if key not in drawn and layer.enabled:
                layer.add_to_mpl(ax, self.state, traj=traj,
                                  satellite=self.satellite,
                                  scene_radius_km=r_max)

        # ── orbit track ───────────────────────────────────────────────────
        ax.plot(r_t[:, 0], r_t[:, 1], r_t[:, 2],
                color=orbit_color, linewidth=1.5, zorder=3)

        # ── satellite dot at current position ─────────────────────────────
        r0, _ = self.state.to_rv()
        self._sat_dot = ax.scatter(*r0, s=40, c=sat_color,
                                   zorder=4, depthshade=False)

        # ── satellite 3D model + NTW ──────────────────────────────────────
        if self.satellite is not None:
            _, v0 = self.state.to_rv()
            if self.satellite.show_ntw:
                T, N, W = self.satellite.ntw_vectors(r0, v0)
                for vec, col in [(T, "#00FF9C"), (N, "#FF4D6D"), (W, "#4D96FF")]:
                    ax.quiver(*r0, *vec, color=col, linewidth=2,
                              arrow_length_ratio=0.25, zorder=5)

        # ── title + fidelity badge ────────────────────────────────────────
        badge = {"fast": "⚡ Fast", "loading": "⟳ IERS…", "high": "✓ Hi-fi"}
        t_str = title or f"{self.state.name}  [{badge.get(self._fidelity, '')}]"
        ax.set_title(t_str, color=_TEXT_COLOR, fontsize=10, pad=8)

        # ── frame label ───────────────────────────────────────────────────
        if self.frame != Frame.ECI:
            ax.text2D(0.02, 0.97, f"Frame: {self.frame.value}",
                      transform=ax.transAxes, color="#FFB830", fontsize=8)

        self._fig.tight_layout()
        return self._fig

    # ── animation ─────────────────────────────────────────────────────────────
    def animate(
        self,
        n_orbits    : float = 3.0,
        dt_s        : float = 60.0,
        interval_ms : int   = 40,
        trail_pts   : int   = 60,
    ) -> FuncAnimation:
        """
        Return a FuncAnimation of the satellite moving along its orbit.
        Renders fast — only updates the dot and trail, not the whole scene.
        """
        # first render the static background
        self.render(n_orbits=n_orbits, dt_s=dt_s)
        traj  = self._get_trajectory()
        r_t   = self._transform_traj(traj)
        N     = len(r_t)

        # dynamic artists
        trail_line, = self._ax.plot([], [], [], color=_ORBIT_COLOR,
                                    linewidth=1.5, alpha=0.4, zorder=3)
        dot = self._ax.scatter([], [], [], s=60, c=_SAT_COLOR,
                               zorder=5, depthshade=False)
        # remove old static dot
        if self._sat_dot:
            try: self._sat_dot.remove()
            except Exception: pass

        def _init():
            trail_line.set_data_3d([], [], [])
            return trail_line, dot

        def _update(frame):
            idx   = frame % N
            start = max(0, idx - trail_pts)
            seg   = r_t[start:idx+1]
            trail_line.set_data_3d(seg[:, 0], seg[:, 1], seg[:, 2])
            pos = r_t[idx]
            dot._offsets3d = (pos[0:1], pos[1:2], pos[2:3])
            return trail_line, dot

        self._anim = FuncAnimation(
            self._fig, _update, frames=N, init_func=_init,
            interval=interval_ms, blit=True,
        )
        return self._anim

    # ── save ─────────────────────────────────────────────────────────────────
    def save(self, path: str | Path, dpi: int = 150, **kwargs):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fig.savefig(str(path), dpi=dpi, facecolor=_BG_COLOR,
                          bbox_inches="tight", **kwargs)
        return path

    def save_animation(self, path: str | Path, fps: int = 25, **kwargs):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._anim is None:
            raise RuntimeError("call animate() first")
        self._anim.save(str(path), fps=fps, **kwargs)
        return path

    @property
    def fig(self) -> plt.Figure:
        return self._fig

    @property
    def ax(self) -> Axes3D:
        return self._ax


# ══════════════════════════════════════════════════════════════════════════════
# PlotlyScene  (interactive / Streamlit)
# ══════════════════════════════════════════════════════════════════════════════
class PlotlyScene:
    """
    Plotly-based 3D scene — used by the Streamlit GUI for live interactive preview.

    All layers that implement add_to_plotly() work here.
    The build() method returns a go.Figure ready for st.plotly_chart().

    Usage
    -----
    scene = PlotlyScene(state)
    scene.add_layer("earth", texture_path="...")
    scene.add_layer("stars", catalog_path="...")
    scene.add_layer("groundtrack")
    fig = scene.build(n_orbits=3, dt_s=60)
    """

    def __init__(
        self,
        state     : OrbitalState,
        frame     : Frame | str = Frame.ECI,
        dark      : bool = True,
        satellite = None,
    ):
        if not _HAS_PLOTLY:
            raise ImportError("plotly is required for PlotlyScene")
        self.state     = _coerce_orbital_state(state)
        self.frame     = Frame(frame)
        self.dark      = dark
        self.satellite = satellite
        self.layers: dict[str, BaseLayer] = {}

        self._traj: Optional[Trajectory] = None
        self._traj_lock = threading.Lock()

        # IERS
        self._fidelity = "fast"
        self._iers_thread: Optional[threading.Thread] = None
        self._on_fidelity_change: Optional[Callable] = None
        self._start_iers_thread()

    # ── layer management ──────────────────────────────────────────────────────
    def add_layer(self, layer: BaseLayer | str, **kwargs) -> "PlotlyScene":
        if isinstance(layer, str):
            layer = create_layer(layer, **kwargs)
        self.layers[layer.key] = layer
        return self

    def remove_layer(self, key: str) -> "PlotlyScene":
        self.layers.pop(key, None)
        return self

    def toggle_layer(self, key: str, enabled: bool | None = None) -> "PlotlyScene":
        if key in self.layers:
            l = self.layers[key]
            l.enabled = not l.enabled if enabled is None else enabled
        return self

    def set_frame(self, frame: Frame | str) -> "PlotlyScene":
        self.frame = Frame(frame)
        return self

    # ── IERS ─────────────────────────────────────────────────────────────────
    @property
    def fidelity(self) -> str:
        return self._fidelity

    def _start_iers_thread(self):
        def _worker():
            try:
                import warnings
                from astropy.utils import iers
                self._fidelity = "loading"
                if self._on_fidelity_change:
                    self._on_fidelity_change("loading")
                # Pre-load the bundled IERS-B table without touching astropy's
                # process-wide IERS configuration. This used to set
                # conf.auto_download = False and conf.auto_max_age = None for
                # the whole process; on astropy < 8 (the only versions for
                # Python 3.10) IERS_Auto then serves IERS-B, which ends weeks
                # before today, so every later UT1 conversion anywhere in the
                # process raised IERSRangeError.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    iers.IERS_B.open()
                self._fidelity = "high"
                if self._on_fidelity_change:
                    self._on_fidelity_change("high")
            except Exception:
                self._fidelity = "fast"

        self._iers_thread = threading.Thread(target=_worker, daemon=True)
        self._iers_thread.start()

    def on_fidelity_change(self, callback: Callable) -> "PlotlyScene":
        self._on_fidelity_change = callback
        return self

    # ── trajectory ────────────────────────────────────────────────────────────
    def _get_trajectory(self, n_orbits=3.0, dt_s=60.0) -> Trajectory:
        with self._traj_lock:
            if self._traj is None:
                self._traj = self.state.propagate(n_orbits=n_orbits, dt_s=dt_s)
        return self._traj

    def invalidate_trajectory(self):
        with self._traj_lock:
            self._traj = None

    def _transform_traj(self, traj: Trajectory) -> np.ndarray:
        tf = FrameTransform(self.frame)
        return tf.transform_trajectory(traj.r, traj.v, traj.t)

    # ── build figure ──────────────────────────────────────────────────────────
    def build(
        self,
        n_orbits    : float = 3.0,
        dt_s        : float = 60.0,
        orbit_color : str   = _ORBIT_COLOR,
        sat_color   : str   = _SAT_COLOR,
        show_osculating: bool = False,
        height_px   : int   = 620,
    ) -> go.Figure:
        """
        Build and return the Plotly figure.

        Parameters
        ----------
        n_orbits      : orbits to propagate
        dt_s          : time step seconds
        orbit_color   : CSS hex for orbit line
        sat_color     : CSS hex for satellite marker
        show_osculating : overlay the analytic osculating ellipse
        height_px     : figure height
        """
        fig = go.Figure()

        traj  = self._get_trajectory(n_orbits, dt_s)
        r_t   = self._transform_traj(traj)
        r_max = float(np.max(np.linalg.norm(traj.r, axis=1))) * 1.15

        # ── layers (back to front) ────────────────────────────────────────
        layer_order = [
            "stars", "van_allen", "magfield",
            "earth", "moon", "sun", "terminator",
            "groundtrack", "eclipse",
            "lagrange", "burns", "ntw",
        ]
        drawn = set()
        for key in layer_order:
            if key in self.layers and self.layers[key].enabled:
                self.layers[key].add_to_plotly(
                    fig, self.state, traj=traj,
                    satellite=self.satellite,
                    scene_radius_km=r_max,
                )
                drawn.add(key)
        for key, layer in self.layers.items():
            if key not in drawn and layer.enabled:
                layer.add_to_plotly(fig, self.state, traj=traj,
                                     satellite=self.satellite,
                                     scene_radius_km=r_max)

        # ── propagated orbit track ────────────────────────────────────────
        fig.add_trace(go.Scatter3d(
            x=r_t[:, 0], y=r_t[:, 1], z=r_t[:, 2],
            mode="lines",
            line=dict(color=orbit_color, width=2),
            name=self.state.name,
            hovertemplate="X=%{x:.0f} km<br>Y=%{y:.0f} km<br>Z=%{z:.0f} km<extra></extra>",
        ))

        # ── osculating ellipse overlay ────────────────────────────────────
        if show_osculating:
            ell = self.state.osculating_ellipse(n_pts=360)
            fig.add_trace(go.Scatter3d(
                x=ell[:, 0], y=ell[:, 1], z=ell[:, 2],
                mode="lines",
                line=dict(color=orbit_color, width=1.5, dash="dot"),
                name="Osculating ellipse", opacity=0.5,
            ))

        # ── satellite dot ─────────────────────────────────────────────────
        r0, v0 = self.state.to_rv()
        fig.add_trace(go.Scatter3d(
            x=[r0[0]], y=[r0[1]], z=[r0[2]],
            mode="markers",
            marker=dict(size=5, color=sat_color, symbol="circle"),
            name="Satellite",
            hovertemplate=(
                f"{self.state.name}<br>"
                f"r={np.linalg.norm(r0):.1f} km  v={np.linalg.norm(v0):.3f} km/s"
                "<extra></extra>"
            ),
        ))

        # ── NTW vectors ───────────────────────────────────────────────────
        if self.satellite and self.satellite.show_ntw:
            T, N, W = self.satellite.ntw_vectors(r0, v0)
            for label, vec, col in [("T", T, "#00FF9C"), ("N", N, "#FF4D6D"), ("W", W, "#4D96FF")]:
                tip = r0 + vec
                fig.add_trace(go.Scatter3d(
                    x=[r0[0], tip[0]], y=[r0[1], tip[1]], z=[r0[2], tip[2]],
                    mode="lines",
                    line=dict(color=col, width=4),
                    name=label,
                ))

        # ── layout ────────────────────────────────────────────────────────
        bg = "#09090F" if self.dark else "#ffffff"
        fig.update_layout(
            height=height_px,
            paper_bgcolor=bg,
            plot_bgcolor=bg,
            scene=dict(
                xaxis=dict(title="X (km)", showgrid=True, gridcolor=_GRID_COLOR,
                           backgroundcolor=bg, color=_TEXT_COLOR, range=[-r_max, r_max]),
                yaxis=dict(title="Y (km)", showgrid=True, gridcolor=_GRID_COLOR,
                           backgroundcolor=bg, color=_TEXT_COLOR, range=[-r_max, r_max]),
                zaxis=dict(title="Z (km)", showgrid=True, gridcolor=_GRID_COLOR,
                           backgroundcolor=bg, color=_TEXT_COLOR, range=[-r_max, r_max]),
                bgcolor=bg,
                camera=dict(eye=dict(x=1.4, y=1.4, z=0.8)),
                aspectmode="cube",
            ),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=_TEXT_COLOR, size=11)),
            margin=dict(l=0, r=0, t=30, b=0),
            title=dict(
                text=(f"{self.state.name}  |  {self.state.regime}  |  "
                      f"T={self.state.period_hr:.3f} h  |  "
                      f"{'⚡' if self._fidelity=='fast' else '⟳' if self._fidelity=='loading' else '✓'}  "
                      f"{self.frame.value}"),
                font=dict(color=_TEXT_COLOR, size=12, family="JetBrains Mono, monospace"),
                x=0.02,
            ),
        )
        return fig

    # ── quick osculating-only figure (for live slider updates) ───────────────
    def build_fast(self, orbit_color: str = _ORBIT_COLOR) -> go.Figure:
        """
        Instant figure with only the osculating ellipse + Earth.
        No propagation.  Used for live parameter slider feedback.
        """
        fig = go.Figure()
        r_max = self.state.r_a * 1.15

        # Earth
        if "earth" in self.layers and self.layers["earth"].enabled:
            self.layers["earth"].add_to_plotly(fig, self.state)

        # Osculating ellipse
        ell = self.state.osculating_ellipse(n_pts=360)
        fig.add_trace(go.Scatter3d(
            x=ell[:, 0], y=ell[:, 1], z=ell[:, 2],
            mode="lines",
            line=dict(color=orbit_color, width=2),
            name="Orbit (osculating)",
        ))

        # Satellite
        r0, v0 = self.state.to_rv()
        fig.add_trace(go.Scatter3d(
            x=[r0[0]], y=[r0[1]], z=[r0[2]],
            mode="markers",
            marker=dict(size=5, color=_SAT_COLOR),
            name="Satellite",
        ))

        bg = "#09090F"
        fig.update_layout(
            height=480,
            paper_bgcolor=bg, plot_bgcolor=bg,
            scene=dict(
                xaxis=dict(range=[-r_max, r_max], gridcolor=_GRID_COLOR,
                           backgroundcolor=bg, color=_TEXT_COLOR),
                yaxis=dict(range=[-r_max, r_max], gridcolor=_GRID_COLOR,
                           backgroundcolor=bg, color=_TEXT_COLOR),
                zaxis=dict(range=[-r_max, r_max], gridcolor=_GRID_COLOR,
                           backgroundcolor=bg, color=_TEXT_COLOR),
                bgcolor=bg, aspectmode="cube",
                camera=dict(eye=dict(x=1.4, y=1.4, z=0.8)),
            ),
            margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=_TEXT_COLOR)),
            title=dict(
                text=f"{self.state.name}  |  {self.state.regime}  "
                     f"|  T={self.state.period_hr:.3f} h",
                font=dict(color=_TEXT_COLOR, size=11,
                          family="JetBrains Mono, monospace"),
            ),
        )
        return fig

    # ── multiple orbit comparison ─────────────────────────────────────────────
    @staticmethod
    def compare(
        states      : list[OrbitalState],
        n_orbits    : float = 3.0,
        dt_s        : float = 60.0,
        dark        : bool  = True,
        height_px   : int   = 700,
        **layer_kwargs,
    ) -> go.Figure:
        """
        Build a Plotly figure showing multiple orbits on one scene.
        Each state gets a different colour.  Shared Earth + stars underneath.

        Parameters
        ----------
        states     : list of OrbitalState
        n_orbits   : orbits to propagate per state
        dt_s       : time step
        layer_kwargs : passed to EarthLayer / StarfieldLayer

        Returns
        -------
        go.Figure
        """
        COLORS = ["#00FF9C", "#FFB830", "#7B2FFF", "#FF4D6D",
                  "#4D96FF", "#C8D8E8", "#ffd93d", "#6bcb77"]
        bg = "#09090F" if dark else "#ffffff"
        fig = go.Figure()

        # shared Earth
        from .layers import EarthLayer
        el = EarthLayer(**{k: v for k, v in layer_kwargs.items() if k in ("texture_path",)})
        el.add_to_plotly(fig, states[0])

        r_maxes = []
        for i, state in enumerate(states):
            col  = COLORS[i % len(COLORS)]
            traj = state.propagate(n_orbits=n_orbits, dt_s=dt_s)
            r_maxes.append(np.max(np.linalg.norm(traj.r, axis=1)))
            fig.add_trace(go.Scatter3d(
                x=traj.r[:, 0], y=traj.r[:, 1], z=traj.r[:, 2],
                mode="lines",
                line=dict(color=col, width=2),
                name=state.name,
            ))
            r0, _ = state.to_rv()
            fig.add_trace(go.Scatter3d(
                x=[r0[0]], y=[r0[1]], z=[r0[2]],
                mode="markers",
                marker=dict(size=5, color=col),
                showlegend=False, name=state.name,
            ))

        r_max = max(r_maxes) * 1.15 if r_maxes else 10_000
        fig.update_layout(
            height=height_px,
            paper_bgcolor=bg, plot_bgcolor=bg,
            scene=dict(
                xaxis=dict(range=[-r_max, r_max], gridcolor=_GRID_COLOR,
                           backgroundcolor=bg, color=_TEXT_COLOR),
                yaxis=dict(range=[-r_max, r_max], gridcolor=_GRID_COLOR,
                           backgroundcolor=bg, color=_TEXT_COLOR),
                zaxis=dict(range=[-r_max, r_max], gridcolor=_GRID_COLOR,
                           backgroundcolor=bg, color=_TEXT_COLOR),
                bgcolor=bg, aspectmode="cube",
            ),
            legend=dict(bgcolor="rgba(0,0,0,0)",
                        font=dict(color=_TEXT_COLOR, size=11)),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        return fig
