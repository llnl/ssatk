# `sun_view.py` — Sun / Earth / Moon 3D rendering

Plotly-based Sun model, day/night-shaded Earth and Moon, and a background
starfield, for Space Situational Awareness Toolkit (SSATK)'s 3D scenes.

## Public API

```python
from ssatk.plots.sun_view import (
    sun_position_eci,     # configured Astropy ephemeris direction + distance
    SunLayer,             # glowing sun sphere + corona
    EarthShadingLayer,    # textured, day/night-shaded Earth
    MoonShadingLayer,     # textured, day/night-shaded Moon
    starfield_trace,      # real HYG catalog stars (synthetic fallback)
)

sun_hat, dist_au = sun_position_eci(t)      # t: astropy Time or JD float
sun_pos = sun_hat * VISUAL_DIST_KM_LEO

fig.add_trace(starfield_trace(sky_radius=VISUAL_DIST_KM_LEO * 15))
fig.add_traces(SunLayer(sun_pos).build_traces())
fig.add_traces(EarthShadingLayer(sun_pos).build_traces())
```

There is no `Scene`/`add_layer` registry — traces are built directly and
added to a plotly `Figure` with `fig.add_traces(...)`.

## Constants: real vs. deliberately not-to-scale

This module mixes two kinds of constants, and the distinction matters when
reading or modifying it.

### Real physical constants (sourced, never hardcoded)

These come from `ssatk.constants` (or `astropy` where the toolkit
has no equivalent) and carry their true physical values. They should stay
sourced from those authorities, never re-hardcoded as literals:

| Name        | Source                                   | Meaning                          |
|-------------|------------------------------------------|----------------------------------|
| `R_EARTH_KM`| `ssatk.constants.EARTH_RADIUS`   | Earth equatorial radius (WGS84)  |
| `R_MOON_KM` | `ssatk.constants.MOON_RADIUS`    | Moon radius                      |
| `R_SUN_KM`  | `ssatk.constants.SUN_RADIUS`     | Sun radius                       |
| `AU_KM`     | `astropy.units.au`                       | 1 astronomical unit in km        |

`sun_position_eci()` likewise uses the **real** JPL (DE-series) solar
ephemeris via `astropy.get_body("sun", t)` — the same real-API approach as
`ssatk/accelerations_orbit/accel_sun.py` — not a hand-rolled analytic
series.

### Deliberately artistic constants (NOT physical scale)

These exist purely so the scene is legible. They are **intentionally not**
real distances/sizes, because a true-scale rendering is impossible to view:
the real Sun is ~150 million km away and ~695,000 km across, which next to a
6,378 km Earth would render as either an invisible speck or a scene so vast
that Earth and Moon vanish. Do **not** "fix" these by substituting real
constants — that would break the visualization, not improve it.

| Name                   | Value       | Why it's artistic                                   |
|------------------------|-------------|-----------------------------------------------------|
| `VISUAL_DIST_KM_LEO`   | 80,000 km   | Where the Sun is *placed* in the scene (real ~1 AU) |
| `VISUAL_DIST_KM_CISLUNAR` | 600,000 km | Same, for cislunar-scale scenes                   |
| `VISUAL_SUN_RADIUS_KM` | 5,500 km    | Rendered Sun sphere size (real `R_SUN_KM` ≈ 695,700)|

The real `R_SUN_KM` is still imported and available (e.g. for physically
correct angular-size math elsewhere), but the *rendered* sun sphere
deliberately uses `VISUAL_SUN_RADIUS_KM` instead.

**Earth and the Moon, by contrast, are drawn at true scale** — their real
radii (`R_EARTH_KM`, `R_MOON_KM`) and the Moon's real orbital distance
(`ssatk.constants.LD`) are small enough to render honestly. Only the
Sun's size and placement are schematic.

## Rendering notes

- **Day/night shading** is baked into solid per-vertex `go.Mesh3d`
  `vertexcolor` (real texture color × a diffuse lit-factor, `ambient=0.12`
  on the night side up to `1.0` facing the Sun). It is **not** done with
  per-vertex transparency — Plotly's 3D surface/mesh traces only support a
  single opacity per trace, so an earlier alpha-overlay approach rendered as
  a flat opaque blob and was replaced.
- **Textures** (`earth.png`, `moon.png`) are located via
  `ssapy.utils.find_file`, at native resolution (capped at 1024 px), with a
  plain color-gradient fallback if unavailable.
- **`aspectmode="data"`** is required on the scene, or Plotly stretches the
  independent axes and renders every sphere as an ellipsoid.
- The scene's **axis range** is set large enough to contain the starfield
  (so stars aren't clipped at the axis boundary — Plotly *does* clip 3D
  traces to the range), while the initial **camera zoom** is set separately
  via `camera.eye` to open framed on the near-field Sun/Earth.
