# Space Situational Awareness Toolkit (SSATK)

The **Space Situational Awareness Toolkit (SSATK)** — `pip install ssatk`,
`import ssatk` — is a collection of higher-level, analysis-ready extensions for the
[SSAPy](https://github.com/llnl/SSAPy) orbital-modeling ecosystem. Where SSAPy
provides the core high-fidelity propagation and modeling engine, the Toolkit
adds astrodynamics utilities, orbital-transfer design, coordinate/time
conversions, brightness and observables modeling, launch and propulsion
helpers, orbit and 6-DoF propagators, rich plotting, and data I/O to support day-to-day research and
engineering workflows.

SSAPy itself is a fast, flexible, high-fidelity orbital modeling and analysis
tool for orbits spanning from low-Earth orbit into the cislunar regime, with
configurable force models (Earth and lunar gravity, radiation pressure, drag,
planetary perturbations, maneuvers), multiple propagators, orbit determination,
Monte Carlo / uncertainty-quantification workflows, and ground/space observer
models. See the SSAPy repository for full details:
<https://github.com/llnl/SSAPy>.

## Why SSATK?

SSATK is the analyst-facing layer around SSAPy: it keeps common orbital
mechanics, plotting, transfer-design, data I/O, and early 6-DoF spacecraft
workflows in one import path while preserving SSAPy as the core propagation
engine. The benchmarking review compares SSATK against adjacent astrodynamics,
mission-design, and 6-DoF tools:
[`docs/benchmarking_ssatk.rst`](docs/benchmarking_ssatk.rst). The 6-DoF design
study explains why SSATK is adding a lightweight spacecraft body/component
layer instead of trying to replace established tools such as Basilisk, Tudat,
GMAT, Orekit, STK, or FreeFlyer:
[`docs/design/6dof_architecture.rst`](docs/design/6dof_architecture.rst).

---

## Features

- **Orbital mechanics & astrodynamics** — Keplerian utilities, ellipse fitting,
  r/v conversions, Lagrange points, and synthetic orbit populations.
- **Orbital transfers** — a full transfer-design suite: Hohmann, coplanar,
  Lambert, inclination-change, and rendezvous transfers; continuous-thrust
  transfers; shooter and optimal-transfer methods; and burn-to-delta-v
  conversions with finite-burn modeling.
- **Coordinate transforms & time conversions** — GCRF-to-ITRF, GCRF-to-NTW,
  GCRF-to-LLH/lon-lat, GCRF-to-lunar, J2000-to-GCRF, Cartesian/spherical/
  cylindrical, equatorial/ecliptic, sky-angle helpers, and satellite operation
  frames (NTW, RTN, LVLH, VNB, body, topocentric, and line-of-sight), using a
  right-handed NTW (N = T x W) convention consistent with SSAPy.
- **Observables & brightness modeling** — Lambertian magnitude / brightness,
  including object thermal emission, Earth-shadow effects, and ground
  reflectance.
- **Plotting & visualization** — orbit, ground-track, cislunar (2-D and 3-D),
  and transfer plots; interactive dashboards; and animated GIF/video output.
- **Propagators** — adaptive DOP853 translational propagation, fixed-step RK4/leapfrog helpers, and 6-DoF propagation.
- **6-DoF dynamics** — coupled translational and rigid-body attitude
  propagation with quaternion attitude states, optional user acceleration and
  torque models, gravity-gradient torque, fixed-facet drag/SRP, thrusters,
  articulated facet transforms, magnetic torquers, reaction wheels, dynamic
  tank mass properties, dry-mass stopping events, and a basic quaternion PD
  controller.
- **Launch & propulsion** — launch-pad definitions, gravity-turn ascent,
  engine catalogs, thrust profiles, and fuel/burn utilities.
- **Data I/O** — CCSDS CDM KVN 1.0 and OMM XML 2.0 interoperability, HDF5
  helpers (including dictionary/HDF5 conversion with array handling and
  selective key loading), plus CSV, JSON, XML, pickle, and TLE/3LE parsing.
  OMM XML support covers the `meanElements` and `tleParameters` data-block
  subset; structured covariance and other structured data blocks are rejected
  explicitly.
- **SSAPy wrappers & HPC helpers** — convenience wrappers around SSAPy orbits,
  propagators, and satellite keyword arguments, plus utilities for HPC
  workflows.
- **Demo gallery** — a runnable gallery of worked examples with inline output.

---

## Installation

SSATK is a standard Python package, published on PyPI as `ssatk`.

> **Renamed in 1.1.0.** SSATK was published as `ssapy-toolkit` and imported as
> `ssapy_toolkit` through 1.0.7. Existing installs can run
> `python -m pip install -U ssapy-toolkit`: that release is a thin package that
> pulls in `ssatk` and keeps `import ssapy_toolkit` working (it forwards to
> `ssatk` and emits a `FutureWarning`). New installs should use
> `python -m pip install ssatk`. The `ssapy-*` command names remain as aliases
> of the `ssatk-*` commands.

```
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

This installs the package in editable mode along with development dependencies
(testing, linting, docs tools, JavaScript validation helpers, etc.). Runtime
plotting dependencies support HTML, image, and GIF output through Plotly,
Matplotlib, Pillow, imageio, and split SSATK data assets. Install
`ssatk[static]` for Plotly static-image export through Kaleido,
`ssatk[pdf]` for appending pages to existing PDF plots,
`ssatk[notebook]` for IPython display and ipyvolume Earth/Moon meshes,
`ssatk[video]` for OpenCV MP4 output and a bundled FFmpeg fallback, and
`ssatk[browser]` for Selenium browser capture. Node.js 20+ is used only to validate the
self-contained JavaScript viewer sources; GitHub Actions installs it with
`actions/setup-node`, and local developers can use system Node.js or `nodeenv`.
Install `ssatk[monitoring]` to enable the optional current-process RSS
memory helper.

SSATK builds on SSAPy; see the
[SSAPy](https://github.com/llnl/SSAPy) repository for its installation details.

---

## Usage

The Toolkit is intended to be the main user-facing entry point. Shared SSAPy
constants are available through the Toolkit, so users do not need to know
whether a constant originates in base SSAPy or Toolkit-specific helpers:

```
import ssatk

print(ssatk.EARTH_MU)
print(ssatk.constants.RGEO)

orbit = ssatk.Orbit.fromKeplerianElements(
    a=ssatk.constants.RGEO,
    e=0.0,
    i=0.0,
    pa=0.0,
    raan=0.0,
    trueAnomaly=0.0,
    t=0.0,
)
r, v = ssatk.rv(orbit, time=[0.0, 60.0])
```

Base SSAPy core objects such as `Orbit`, `rv`, `groundTrack`, and `AccelKepler`
are lazily available through `ssatk`. Toolkit duplicate helpers take
precedence at the top level, so names such as `ssatk.norm`,
`ssatk.deg0to360`, and `ssatk.period` resolve to Toolkit
implementations. Toolkit submodules also win on collisions such as
`ssatk.io` and `ssatk.utils`; the base package remains available
as `ssatk.ssapy` when direct SSAPy module access is needed.
Earth/Moon helpers formerly provided by `ssapy.plotUtils` are available as
`ssatk.draw_earth`, `draw_moon`, `load_earth_file`, and
`load_moon_file`.

CCSDS Conjunction Data Message (CDM) KVN 1.0 files use SI values in memory:

```python
from ssatk.io.ccsds_cdm import read_cdm, write_cdm

cdm = read_cdm("conjunction.cdm")
object1_gcrf = cdm.object1.state_gcrf()
write_cdm(cdm, "canonical.cdm")
```

The reader accepts calendar or ordinal Coordinated Universal Time (UTC), all
three CDM 1.0 state frames (`GCRF`, `EME2000`, and `ITRF`), and the mandatory
RTN covariance plus complete optional drag, solar-radiation-pressure, and
thrust rows. Later alternate-covariance extensions are rejected explicitly.

For workflow functions, import the specific Toolkit module you need:

```
from ssatk.orbital_mechanics import keplerian
from ssatk.orbital_mechanics import transfer_hohmann
from ssatk.orbital_mechanics import transfer_bielliptic
from ssatk.coordinates import gcrf_to_itrf
from ssatk.plots import orbit_plot
```

For high-accuracy translational propagation, use the adaptive DOP853 wrapper in
`propagators_orbit` instead of the older fixed-step helpers:

```
import numpy as np
from ssatk.constants import EARTH_MU
from ssatk.propagators_orbit import propagate_orbit_state

radius = 7_000_000.0
speed = np.sqrt(EARTH_MU / radius)
traj = propagate_orbit_state(
    r0=[radius, 0.0, 0.0],
    v0=[0.0, speed, 0.0],
    times=np.linspace(0.0, 3600.0, 121),
)
```

For rigid-body spacecraft dynamics, use `Spacecraft` when you want an
`Orbit`-like object with attitude, angular rate, inertia, and mass attached.
Use `Spacecraft.from_orbit(orbit, ...)` to attach 6-DoF state to an SSAPy
`Orbit`, `spacecraft.to_orbit()` to return the translational state to SSAPy,
and `propagate_6dof` directly for lower-level numerical propagation.

```
import numpy as np
import ssatk
from ssatk.accelerations_6dof import SpacecraftAccelJ2, constant_body_thrust
from ssatk.plots import orbit_plot

sat = ssatk.Spacecraft(
    r=[7_000_000.0, 0.0, 0.0],
    v=[0.0, 7_500.0, 0.0],
    q=[1.0, 0.0, 0.0, 0.0],       # [w, x, y, z], body to inertial
    omega=[0.0, 0.0, 0.001],      # body-frame rad/s
    inertia=np.diag([10.0, 12.0, 8.0]),
    mass=100.0,
)

traj = sat.propagate(
    times=np.linspace(0.0, 600.0, 61),
    acceleration=SpacecraftAccelJ2(),
    body_acceleration=constant_body_thrust([0.0, 0.01, 0.0], sat.mass),
    gravity_gradient=True,
)

orbit_plot(traj.r, traj.t, view="3d")
```

Reusable 6-DoF acceleration models live in ``ssatk.accelerations_6dof`` and
include SSAPy-like classes for Kepler gravity, J2, third-body gravity,
cannonball drag, cannonball solar radiation pressure, constant inertial/NTW/body
accelerations, summed acceleration/torque models, and attitude-dependent
flat-plate drag/SRP models, facet drag/SRP models, thruster torque, and
magnetic-dipole and gravity-gradient torque. Reaction wheels and
``SpacecraftAttitudePD`` provide a small actuator/control layer for attitude
studies; reaction-wheel momentum is propagated as an optional state when the
body defines wheels, and configured ``momentum_capacity`` values prevent
commands from driving wheels beyond their stored angular-momentum limits.
``SpaceEnvironment`` supplies epoch-aware Sun/Moon ephemerides, atmosphere
density and velocity, magnetic field, and disk-overlap Earth/Moon eclipse
fractions for environment-backed force models. Use ``third_bodies=True`` for
Moon/Sun perturbations, ``third_bodies="planets"`` for Mercury through Neptune
except Earth, or ``third_bodies="all"`` for a full Solar-System perturbation
set. Optional gravity-gradient torque supports central Earth or Earth/Moon/Sun
models. Common force-stack presets are available through
``SpaceEnvironment.force_models(preset="leo"|"earth_orbit"|"cislunar"|"all")``
and individual options can still override each preset. The default atmosphere
velocity is rigid Earth co-rotation; pass ``atmosphere_velocity_model=...`` for
wind or corotation overrides. The default magnetic field is a dependency-light
centered Earth dipole; use ``magnetic_field_model="igrf"`` for optional
``ppigrf``-backed IGRF field synthesis or pass a callable for mission-specific
models.
Thrusters report positive propellant mass flow from thrust and specific impulse;
``propagate_6dof`` can propagate mass when a mass-flow model is supplied.
For bodies with tanks, propagated mass updates tank propellant proportionally so
center of mass and inertia evolve during finite burns. Use
`propellant_empty_event` or `mass_floor_event` to stop burns at dry mass.

Finite maneuver accelerations use ``SpacecraftManeuverAccel``. Use
``frame="rtn"``/``"lvlh"``/``"ric"`` for common radial-transverse-normal
operations, ``frame="vnb"`` for velocity-normal-binormal commands,
``frame="body"`` for body-mounted thrust, or ``frame="ntw"`` for exact SSAPy
``[N, T, W]`` convention. Thrust can be constant, trapezoidal, smoothstep,
exponential, pulsed, callable, or loaded from CSV with ``ThrustCurve``; citable
engine data belongs in the appropriate split SSATK data repository rather than this source repository and can be
loaded with ``load_thrust_curve_data(...)`` once packaged.

Preset spacecraft bodies live in `ssatk.satellites`. Use
`satellite_design(...)` to start from a common bus, override dimensions or
mass, then add components, tanks, facets, or thrusters as needed:

```
from ssatk.accelerations_6dof import (
    SpacecraftFacetDrag,
    SpacecraftFacetSolRad,
    SpacecraftManeuverAccel,
    SpacecraftThrusterAccel,
)

body = ssatk.satellite_design(
    "earth_observation",
    mass=500.0,
    solar_array_area=10.0,
).with_thrusters(
    ssatk.Thruster(thrust=0.2, direction_body=[1, 0, 0], position_body=[0, 0.5, 0]),
).with_components(
    ssatk.Component(mass=25.0, position_body=[0.0, 0.0, 0.7], name="payload"),
).with_magnetic_dipoles(
    ssatk.MagneticDipole(moment_body=[0.2, 0.0, 0.0], name="x_magnetorquer"),
).with_reaction_wheels(
    *ssatk.reaction_wheel_triplet(max_torque=0.02),
)

sat = ssatk.Spacecraft(r=[7e6, 0, 0], v=[0, 7500, 0], body=body)
q_target = ssatk.attitude_quaternion_from_frame("nadir_velocity", r=sat.r, v=sat.v)
burn = SpacecraftManeuverAccel(
    ssatk.thrust_profile_trapezoid(0.2, start=120.0, burn_time=60.0, rise_time=5.0),
    frame="rtn",
    direction=[0, 1, 0],
    isp=220.0,
)
traj = sat.propagate(
    times=np.linspace(0.0, 600.0, 61),
    models=[
        SpacecraftFacetDrag(density=1e-12),
        SpacecraftFacetSolRad([ssatk.AU, 0, 0]),
        burn,
        ssatk.SpacecraftMagneticTorque([0, 2e-5, 0]),
        ssatk.SpacecraftReactionWheelTorque([0, 0, 0.01]),
        ssatk.SpacecraftAttitudePD(q_target=q_target, kp=0.05, kd=0.2, max_torque=0.02),
        SpacecraftThrusterAccel(),
    ],
)
```

`transfer_bielliptic` computes the analytic three-impulse, two-half-ellipse
transfer between coplanar circular orbits through an intermediate apoapsis
radius. It is useful for quick radius-to-radius trade studies; use
`transfer_ssapy` or `transfer_optimal` when fixed epochs, target phasing,
perturbed propagation, or non-circular boundary states matter.
Transfer entry points accept either SSAPy `Orbit` objects (`orbit1`/`orbit2`,
`initial`/`target`) or raw inertial state vectors (`r1, v1, r2, v2`). For
`transfer_optimal`, set `departure_mode="now"` or `leave_now=True` to depart
from the supplied state; leave the default `departure_mode="optimize"` to search
for the best departure phase/time.
Set `stage_mode="immediate"` or `stage_mode="timed"` to explicitly search
staged transfers through candidate staging orbits; `stage_mode="best"` compares
the direct and staged routes. Timed staging allows each post-stage leg to wait
for an appropriate phase instead of leaving the staging orbit immediately. The
default `n_stage_stops=1` searches one intermediate staging orbit; increase
`n_stage_stops` and set `stage_beam_width` for bounded multi-stop searches.
For larger design trades, `transfer_optimal` also accepts a structured
`problem={...}` schema that groups boundary conditions, objective, constraints,
route, and solver controls in one call:

```
from ssatk.orbital_mechanics import transfer_optimal

result = transfer_optimal(
    problem={
        "boundary": {
            "initial": orbit1,              # or r1/v1/r2/v2 at top level
            "target": orbit2,
            "departure_mode": "leave now", # or "leave whenever"
            # inject: free-phase/first burn only; intercept: target position only;
            # rendezvous: target position + velocity; insertion: free-phase velocity match
            "arrival_mode": "rendezvous",
        },
        "objective": {"minimize": "delta_v", "delta_v_mode": "total"},
        "constraints": {
            "tof_range": (1800.0, 86400.0),
            "dv_budget": None,
            "perigee_altitude_min": 100e3,
            "max_burns": 4,
        },
        "route": {
            "mode": "multi_stage",         # direct, immediate, multi_stage, best
            "timing": "optimized",         # immediate or optimized/timed
            "n_stage_stops": 1,
            "stage_candidates": {"radii": [20_000e3, 40_000e3]},
        },
        "solver": {"n_grid": (8, 8), "polish": False, "refine": False},
    },
)
```

The result diagnostics include `problem_schema="ssatk.transfer_problem.v1"`
when the structured interface is used.

`orbit_plot` is the main entry point for in-space trajectory plots. It uses a
four-panel orbit view by default, and also accepts compact selectors for common
slices and cislunar views:

```
orbit_plot(r, t, frame="gcrf")                         # xy, xz, yz, and 3-D
orbit_plot(r, t, view="xy", frame="itrf")             # one 2-D slice
orbit_plot(r, t, view=("xy", "xz", "3d"))            # custom panels
orbit_plot(r, t, view="lunar_yz")                     # lunar-fixed YZ slice
orbit_plot(r, t, view="lunar_xy", coordinate="gcrf")  # override coordinates
orbit_plot(r, t, view="ground track")                 # wide ground-track map
orbit_plot(r, t, view=("groundtrack", "globe"))       # map + 3-D globe
orbit_plot(r, t, view="dashboard")                    # map, globe, and slices
orbit_plot(r, t, view="cislunar_3d")                  # lunar-fixed 3-D view
orbit_plot(r, t, view="cislunar_xy")                  # GCRF + lunar XY views
orbit_plot(r, t, view="cislunar_dashboard")           # cislunar dashboard
orbit_plot(r, t, view="xy", save="quicklooks/orbit.mp4")  # animated MP4
orbit_plot(r, t, view="xy", save="quicklooks/orbit.gif")  # animated GIF
```

All plotting helpers accept `save`, `savefig`, `save_fig`, `save_figure`,
`savepath`, and `save_path` as equivalent save-path keywords. Relative names
are saved under `~/ssatk_output/figures`; absolute paths are used exactly as
provided. Set `SSATK_OUTPUT_DIR` to choose a different output root explicitly.
Use `ssatk_path` and `ssatk_fig` for direct path and figure-save helpers.

For general data products, `ssatk_save` and `ssatk_read` choose the storage
format from the file extension. Bare and relative data filenames are rooted
under `~/ssatk_output`; bare and relative figure filenames are rooted under
`~/ssatk_output/figures`; absolute paths are honored.

```
ssatk.ssatk_save({"r": r, "v": v, "t": t}, "runs/orbit.h5")
state = ssatk.ssatk_read("runs/orbit.h5")

ssatk.ssatk_save(r, "arrays/state.npy")
ssatk.ssatk_save({"r": r, "v": v}, "arrays/state.npz")
ssatk.ssatk_save(table, "tables/summary.csv")
ssatk.ssatk_save(fig, "quicklooks/orbit.png")
```

For keyed HDF5 or NPZ outputs, pass `key=`. Non-mapping objects default to
`"data"`; dictionaries use their own keys; nested dictionaries become nested
HDF5 groups or slash-delimited NPZ members.

More detailed examples can be found in the categorized `demos/` directory. The
demo gallery runner searches those subfolders recursively. To render the full
demo gallery as a visualization document:

```
ssatk-demo-gallery
```

The command can be run from any directory after installation. It writes the
HTML report to `~/ssatk_output/documents/index.html` by default and prints
the exact output path when it finishes. Use `--open` to open the report in a
browser, `--output PATH` to choose a different output directory, or
`--demos-dir PATH` to run demos from a source checkout explicitly. The default
does not fall back to the clone directory; set `SSATK_OUTPUT_DIR` if you want
a non-home output root.

---

## Development

To run the test suite:

```
pytest tests
```

The current CI lint gate checks fatal Ruff errors in changed Python files; it is
not a full formatting or style pass:

```bash
set -euo pipefail
base_ref="$(git merge-base origin/main HEAD)"
mapfile -t python_files < <(
  git diff --name-only --diff-filter=ACMR "$base_ref" HEAD -- '*.py'
)
if ((${#python_files[@]})); then
  ruff check --select E9,F63,F7,F82 "${python_files[@]}"
fi
```

Optional local repo mapping with Graphify:

```
pipx install graphifyy  # or: python -m pip install graphifyy
bash scripts/install_graphify_hook.sh
```

The installer writes a local `.git/hooks/post-commit` hook. After each commit,
the hook runs `graphify . --update --wiki` in the background when the
`graphify` CLI is available and writes ignored output under `graphify-out/`.
Disable it for one commit with `SSATK_GRAPHIFY_HOOK=0 git commit ...`, or force
foreground execution with `SSATK_GRAPHIFY_FOREGROUND=1 git commit ...`.

---

## Documentation

Project documentation is built with Sphinx and hosted on Read the Docs.
Once configured, the latest documentation will be available at:

<https://ssapy-toolkit.readthedocs.io>

To build the docs locally (after installing dev dependencies):

```
cd docs
make html
```

The built HTML files will be in `docs/_build/html/`.

---

## Contributing

Contributions are welcome via pull request against the `main` branch. Work that
primarily concerns the core propagation/modeling engine should target the
[SSAPy](https://github.com/llnl/SSAPy) repository instead.

---

## License

SSATK is distributed under the terms of the BSD 3-Clause license. All
new contributions must be made under the same license. See the
[LICENSE](LICENSE) file for details.

SPDX-License-Identifier: BSD-3-Clause

LLNL-CODE-2015996
