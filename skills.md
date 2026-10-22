# Using SSATK

How to use the Space Situational Awareness Toolkit (SSATK) from code. Rules for
changing this repository are in [AGENTS.md](AGENTS.md); the full narrative
guide is `docs/usage.rst` and the API reference is `docs/api.rst`.

## Package identity

| Item | Value |
| --- | --- |
| PyPI distribution | `ssatk` |
| Import name | `ssatk` |
| Repository | https://github.com/LLNL/ssatk |
| Core dependency | `llnl-ssapy`, imported as `ssapy` |
| Required data | `ssa-data-core`, `ssa-data-gravity`, `ssa-data-lunar`, `ssa-data-lunar-gravity` |
| Python | 3.10 or newer |

```bash
python3 -m pip install ssatk                    # library and required data
python3 -m pip install "ssatk[propulsion,video,static]"   # optional extras as needed
```

Optional extras: `propulsion` and `benchmarks` (data), `static` (Plotly image
export), `pdf`, `video` (MP4 output), `browser`, `notebook`, `monitoring`,
`geomagnetics`, `atmosphere`, `validation`, and `all-data`.

## Conventions

- **Units:** states are SI in memory, metres and metres per second. CCSDS files
  are read and written in km and km/s and converted at the boundary.
- **Time:** GPS seconds as floats, or Astropy `Time` objects. Add offsets with
  units: `Time(...) + seconds * u.s`.
- **Frames:** inertial states are GCRF unless a function says otherwise.
- **Namespace:** `import ssatk` is the entry point. Constants are in
  `ssatk.constants` and at top level (`ssatk.EARTH_MU`, `ssatk.constants.RGEO`).
  SSAPy objects such as `ssatk.Orbit`, `ssatk.rv`, `ssatk.groundTrack` and
  `ssatk.AccelKepler` load on first use. Where SSATK and SSAPy share a name,
  the SSATK one wins; reach SSAPy itself as `ssatk.ssapy`. Wildcard imports are
  not supported; import modules or names explicitly.
- **Output:** plotting helpers and `ssatk.plots.figsave` write relative paths
  under `~/ssatk_output/figures` (override the root with `$SSATK_OUTPUT_DIR`);
  absolute paths are used as given. Caches go to `$SSATK_CACHE` when set.

## Spacecraft geometry

Use `ssatk.geometry.SpacecraftMesh` as the canonical geometry type; the
`ssatk.plots.spacecraft_view.SpacecraftMesh` name is a compatibility alias.
Vertices are body-frame metres. Mesh construction preserves the supplied
origin and scale and does not normalize them.

```python
from ssatk.geometry import SpacecraftMesh, convert_mesh

mesh = SpacecraftMesh.from_file("vehicle.obj")
mesh = SpacecraftMesh.from_file("vehicle.step", units="mm")
mesh.export("vehicle.obj")
convert_mesh("vehicle.stl", "vehicle.ply", units="mm")
```

`from_file(path, scale=None, units=None, linear_deflection=.1,
angular_deflection=.5)` reads OBJ directly. `units` accepts exactly one of
`m`, `cm`, `mm`, `in`, or `ft`, and cannot be combined with `scale`. STL,
PLY, OFF, GLTF, GLB, and 3MF require the optional `ssatk[mesh]` extra
(`trimesh`). STEP, STP, IGES, and IGS require optional `ssatk[cad]`
(`cadquery-ocp`); OpenCascade reads CAD in internal millimetres, so the
default CAD conversion is `.001` to SI metres.
Neither loader promises SolidWorks, Inventor, CATIA, or FBX inputs. Export
those formats externally to neutral CAD or mesh. `export()` writes OBJ or
trimesh mesh formats, not CAD. RGB can be retained when supplied by a backend;
textures and PBR materials are not preserved.

`mesh.to_dict()` and `SpacecraftMesh.from_dict()` use
`ssatk.spacecraft_mesh.v1`, with `units="m"` and `frame="body"`. For dynamics,
`SpacecraftBody.from_file(path, mass=..., inertia=..., **...)` keeps physical
properties as explicit caller inputs; omitted inertia uses a uniform
bounding-box estimate. Geometry import does not infer mass, inertia, material,
optical coefficients, or self-occlusion physics.

`spacecraft_view` accepts one shared `mesh=` or `body=`. `satellite_viewer`
accepts one shared model through `mesh=`/`body=` or one model per orbit through
`meshes=`; these choices are exclusive. Its default `mesh_scale="glyph"`
exaggerates display size, while `"physical"` uses metre dimensions. WebGL
presets come from the same Python presets as `spacecraft_view`; LOD glyphs are
display-only.

## Module map

| Module | Use it for |
| --- | --- |
| `ssatk.orbital_mechanics` | Keplerian elements, transfers (`transfer_hohmann`, `transfer_bielliptic`, `transfer_ssapy`, `transfer_optimal`), burns, orbit fits, constellations |
| `ssatk.propagators_orbit` | Translational propagation (`propagate_orbit_state`, `propagate_orbit_state_with_stm`) |
| `ssatk.propagators_6dof` | 6-DoF spacecraft propagation and targeting (`solve_6dof_target`, `solve_6dof_multi_segment_target`, `ImpulseManeuver`) |
| `ssatk.accelerations_orbit`, `ssatk.accelerations_6dof` | Acceleration, force, torque and thrust models |
| `ssatk.coordinates` | GCRF, ITRF, geodetic, lunar, sky and satellite frames (`coordinates.satellite_frames`) |
| `ssatk.ssa` | Conjunction screening, closest approach, encounter frames, collision probability |
| `ssatk.navigation` | Batch orbit fits, EKFs, ground-station and optical sensors, star-tracker attitude |
| `ssatk.compute` | Sampling, coverage statistics, photometry, FFTs, Lyapunov exponents |
| `ssatk.environment_eop`, `ssatk.environment_space_weather` | Packaged Earth-orientation and space-weather tables |
| `ssatk.io` | CCSDS CDM, OMM and OEM; reference-case export and comparison; `ssatk_save` and `ssatk_read`; HDF5, CSV, JSON |
| `ssatk.plots` | `orbit_plot` and its views, globes, ground tracks, dashboards, animations, `figsave` |
| `ssatk.data` | Packaged data access (`data_resource`, `data_path`, `read_data_text`, `read_data_binary`) |
| `ssatk.launch`, `ssatk.engines`, `ssatk.satellites`, `ssatk.operations` | Launch sites and vehicles, propulsion catalogs, spacecraft presets, mission timelines |

## Common tasks

Orbit from elements, sampled states:

```python
import ssatk

orbit = ssatk.Orbit.fromKeplerianElements(
    a=ssatk.constants.RGEO, e=0.0, i=0.0, pa=0.0, raan=0.0, trueAnomaly=0.0, t=0.0
)
r, v = ssatk.rv(orbit, time=[0.0, 60.0])   # (2, 3) arrays in m and m/s
```

High-accuracy propagation (SciPy `DOP853` by default):

```python
import numpy as np
from ssatk.constants import EARTH_MU
from ssatk.propagators_orbit import propagate_orbit_state

radius = 7_000_000.0
traj = propagate_orbit_state(
    r0=[radius, 0.0, 0.0],
    v0=[0.0, np.sqrt(EARTH_MU / radius), 0.0],
    times=np.linspace(0.0, 3600.0, 121),
)
```

Satellite operation frames (matrices map frame components to GCRF):

```python
from ssatk.coordinates.satellite_frames import transform_from_gcrf, transform_to_gcrf

dv_gcrf = transform_to_gcrf([0.0, 1.0, 0.0], "ntw", r=r[0], v=v[0])
dv_ntw = transform_from_gcrf(dv_gcrf, "ntw", r=r[0], v=v[0])
```

Collision probability for a 2-D encounter-plane mean and covariance:

```python
from ssatk.ssa import probability_of_collision

pc = probability_of_collision([10.0, 0.0], [[100.0, 0.0], [0.0, 100.0]], 5.0)
```

Saving and reading data products (relative paths go under the output root):

```python
from ssatk.io import ssatk_read, ssatk_save

path = ssatk_save({"r": r, "v": v}, "examples/state.h5")
state = ssatk_read(path)
```

Plotting:

```python
from ssatk.plots import orbit_plot

orbit_plot(r, view=("xy", "xz", "3d"), frame="itrf", save="examples/orbit_views.png")
```

For transfer design with several boundary conditions or route choices, pass
`transfer_optimal` a structured `problem={"boundary": ..., "objective": ...,
"constraints": ..., "route": ..., "solver": ...}` rather than a long flat
argument list; its result diagnostics report the schema
`ssatk.transfer_problem.v1`. `transfer_bielliptic` is the analytic
three-impulse coplanar circular case only and does not solve phasing.

## Data

Read packaged data through `ssatk.data`, never through paths relative to the
repository or the installed package:

```python
from ssatk.data import data_resource, read_data_text

eop_file = data_resource("environment/eop/finals2000A.all")
```

Optional demo inputs come from `ssatk.io.demo_data.ensure_demo_data_file`,
which looks in the output root and nearby `ssatk_output` folders, downloads from
a known public source when allowed, and otherwise returns `None` with a warning
so demos can skip offline. Do not add data files to this repository.

## Command-line tools

| Command | Purpose |
| --- | --- |
| `ssatk-demo-gallery` | Build the demo gallery (`--output`, `--timeout`, `--strict`) |
| `ssatk-benchmark` | Timing and analytical-validation benchmarks |
| `ssatk-magnetosphere` | Render the 3-D geomagnetic field and radiation belts |
| `ssatk-bake-moon` | Bake WebGL Moon normal and horizon textures into the cache |

## Where to look next

- `demos/<category>/` for runnable, current examples of each workflow.
- `docs/usage.rst` for CCSDS messages, satellite frames, STMs, 6-DoF targeting
  and plot saving in detail.
- `tests/` for reference values and tolerances that pin each model's accuracy.
