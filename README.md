# Space Situational Awareness Toolkit (SSATK)

SSATK is a Python toolkit for space situational awareness (SSA) and
astrodynamics. One package, `import ssatk`, covers the analysis chain from
state vectors to figures:

- propagating orbits and six-degree-of-freedom (6-DoF) spacecraft,
- designing impulsive and continuous-thrust transfers,
- converting between inertial, Earth-fixed, lunar and satellite frames,
- modelling the space environment (Sun and Moon, eclipses, atmosphere,
  geomagnetic field, radiation belts),
- screening conjunctions and fitting orbits to measurements,
- reading and writing standard orbit-data formats, and
- plotting orbits, ground tracks, eclipses, sensor coverage and the
  magnetosphere, including an interactive WebGL satellite viewer.

```bash
python -m pip install ssatk
```

---

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [What SSATK covers](#what-ssatk-covers)
- [Validation](#validation)
- [Demos](#demos)
- [Documentation](#documentation)
- [Development](#development)
- [Dependencies](#dependencies)
- [Citing SSATK](#citing-ssatk)
- [License](#license)

---

## Installation

SSATK supports Python 3.10–3.13.

```bash
python -m pip install ssatk
```

The install pulls four data packages automatically, about 210 MB in total:
`ssatk-data-core`, `ssatk-data-gravity`, `ssatk-data-lunar` and
`ssatk-data-lunar-gravity`. They hold the IERS Earth-orientation and space-weather
snapshots, gravity models, planetary and lunar ephemerides, star catalogue, and
Earth and Moon textures. SSATK reads them from the installed packages; only
explicit download helpers (TLE catalogue updates, OMNI solar-wind records and
optional demo datasets) contact the network.

One dependency compiles a C++ extension during installation (about one
minute), so a C++ compiler must be available.

Optional extras:

| Extra | Adds |
|---|---|
| `ssatk[geomagnetics]` | IGRF field synthesis and T89/T96 magnetosphere models (`ppigrf`, `geopack`) |
| `ssatk[atmosphere]` | NRLMSISE-00 atmospheric density (`pymsis`) |
| `ssatk[propulsion]` | Certified rocket-motor thrust curves (`ssatk-data-propulsion`) |
| `ssatk[benchmarks]` | Benchmark mission datasets (`ssatk-data-benchmarks`) |
| `ssatk[all-data]` | All six `ssatk-data-*` packages |
| `ssatk[static]` | Static image export of interactive figures (`kaleido`) |
| `ssatk[video]` | MP4 output (`opencv-python`, `imageio-ffmpeg`) |
| `ssatk[notebook]` | IPython display and ipyvolume Earth/Moon meshes |
| `ssatk[pdf]` | Appending pages to existing PDF plots (`pypdf`) |
| `ssatk[browser]` | Browser capture of HTML figures (`selenium`) |
| `ssatk[monitoring]` | Process memory reporting (`psutil`) |
| `ssatk[validation]` | Extra external cross-checks (`spacepy`) |

---

## Quick start

Propagate a 51.6° low-Earth orbit for three hours and draw its ground track:

```python
import numpy as np
import ssatk
from ssatk.plots import orbit_plot

orbit = ssatk.Orbit.fromKeplerianElements(
    a=ssatk.EARTH_RADIUS + 420e3, e=0.0005, i=np.radians(51.6),
    pa=0.0, raan=0.0, trueAnomaly=0.0, t=1.4e9,   # t: GPS seconds
)
t = 1.4e9 + np.linspace(0.0, 3 * 3600.0, 541)
r, v = ssatk.rv(orbit, time=t)

orbit_plot(r, t, view="ground track", save_path="quickstart/ground_track.png")
```

Relative output paths are written under `~/ssatk_output/figures`; set
`SSATK_OUTPUT_DIR` to choose another root. Interactive versions of most
figures are written as self-contained HTML.

---

## What SSATK covers

| Area | What it provides | Start with |
|---|---|---|
| **Orbit propagation** | Adaptive DOP853 translational propagation with composable accelerations: point-mass and spherical-harmonic Earth gravity, Sun and Moon third bodies, and continuous-thrust steering laws (circularization, plane and inclination change, radial and along-track thrust); fixed-step RK4 and leapfrog helpers | `ssatk.propagators_orbit.propagate_orbit_state`, `ssatk.accelerations_orbit` |
| **6-DoF spacecraft dynamics** | Coupled translation and quaternion attitude; cannonball and facet drag and solar-radiation pressure; Sun, Moon and planetary third bodies; gravity-gradient and magnetic torques; thrusters, reaction wheels, magnetorquers, tanks with evolving mass properties; dry-mass stop events; attitude PD control; preset spacecraft buses | `ssatk.Spacecraft`, `ssatk.satellite_design`, `ssatk.accelerations_6dof` |
| **Orbital mechanics and transfers** | Keplerian conversions, ellipse fitting, Lagrange points, synthetic populations; Hohmann, bi-elliptic, Lambert, plane-change, rendezvous and continuous-thrust transfers; staged optimal-transfer search with a structured problem schema | `ssatk.orbital_mechanics` |
| **Frames and time** | GCRF, ITRF, geodetic, lunar, ecliptic and topocentric frames; NTW, RTN/LVLH, VNB and body frames; time-scale conversions | `ssatk.coordinates`, `ssatk.time_functions` |
| **Space environment** | Sun/Moon ephemerides, Earth/Moon eclipse fractions, atmosphere density and co-rotation, dipole and IGRF magnetic field, T89/T96 magnetosphere, AE-8/AP-8 radiation-belt fluxes, packaged IERS EOP and space-weather tables | `ssatk.SpaceEnvironment`, `ssatk.geomagnetics` |
| **SSA and navigation** | Coarse and refined conjunction screening, encounter frames, collision probability; ground-station and Cartesian measurements, extended Kalman filter, batch orbit fitting | `ssatk.ssa`, `ssatk.navigation` |
| **Observables** | Lambertian and faceted brightness and visual magnitudes, Earth-shadow illumination | `ssatk.compute` |
| **Data I/O** | CCSDS CDM (KVN 1.0) and OMM (XML 2.0), TLE/3LE parsing, HDF5/NPZ/CSV/JSON with extension-based `ssatk_save`/`ssatk_read` | `ssatk.io` |
| **Visualization** | Four-panel orbit views, ground tracks, textured globes, cislunar dashboards, eclipse and sensor field-of-view scenes, magnetosphere and Van Allen belts, GIF/MP4 animation, and a WebGL satellite viewer that applies supplied attitude quaternions | `ssatk.plots` (`orbit_plot`, `satellite_viewer`) |
| **Launch and propulsion** | Launch sites, gravity-turn ascent, engine catalogues, thrust-curve models | `ssatk.launch`, `ssatk.engines` |
| **Benchmarking** | Timing dashboard and an analytical validation profile | `ssatk-benchmark` |

Physical constants and the core orbit objects are available at the top level
(`ssatk.EARTH_MU`, `ssatk.constants.RGEO`, `ssatk.Orbit`, `ssatk.rv`); task
code imports the module it needs, for example
`from ssatk.orbital_mechanics import transfer_hohmann`.

---

## Validation

SSATK's test policy ([`docs/testing_policy.md`](docs/testing_policy.md))
requires every test to compare against an independent reference with a stated
tolerance; tests that only show code runs are not accepted. Representative
results:

| Quantity | Reference | Result |
|---|---|---|
| Two-body specific energy, DOP853 (rtol 10⁻¹⁰) | Analytic | 3.3 × 10⁻⁷ J/kg residual |
| Two-body angular momentum | Analytic | 5.8 × 10⁻¹⁵ relative residual |
| J2 nodal precession over 20 orbits | First-order secular theory | 2.1 × 10⁻⁴ rad |
| Torque-free rigid body | Angular-momentum and energy invariants | 1.5 × 10⁻¹¹ relative; 4.1 × 10⁻¹³ J |
| Finite burn Δv | Tsiolkovsky rocket equation | 1.5 × 10⁻¹³ m/s |
| GCRF→ITRF rotation | Astropy (IAU 2006/2000A) | ≤ 0.05″ in 2026 (test tolerance 0.5″) |
| Sun direction, 2000–2040 | Astropy `get_sun` | 10–37″ (≈ 20″ is annual aberration) |
| UT1−UTC, 1990–2026, including leap seconds | IERS EOP C04 | ≤ 0.25 ms |
| Star directions with proper motion | Astropy ICRS | < 1″ |

The first five rows are reproduced by
`ssatk-benchmark --profile validation --quick`; the rest by `pytest tests`.

---

## Demos

The `demos/` directory holds runnable examples in 13 categories, from getting
started and orbit visualization to 6-DoF dynamics, eclipses, photometry and
sensor coverage. Build the full gallery (55 demos) as an HTML report:

```bash
ssatk-demo-gallery --open
```

The report is written to `~/ssatk_output/documents/index.html`. Use
`--output PATH` to choose another directory, or `--demos-dir PATH` to run the
demos from a source checkout. CI also publishes the gallery to GitHub Pages.

---

## Documentation

The user guide, API reference, benchmarking study and 6-DoF design notes are
built with Sphinx:

- Hosted: <https://ssatk.readthedocs.io>
- Local build:

  ```bash
  python -m pip install -r docs/requirements.txt
  python -m sphinx -b html -W --keep-going docs /tmp/ssatk-docs
  ```

---

## Development

```bash
git clone https://github.com/LLNL/ssatk.git
cd ssatk
python -m pip install -e ".[dev,geomagnetics,atmosphere,propulsion]"
python -m pip install --no-deps -e packaging/ssapy-toolkit-compat   # tests the legacy import name
pytest tests
```

CI runs the tests on Python 3.10–3.13, builds the docs with warnings treated as
errors, runs the demo gallery, and enforces the repository policy in
[`CONTRIBUTING.md`](CONTRIBUTING.md). Its lint gate checks fatal Ruff errors
(`E9,F63,F7,F82`) in changed Python files.

An optional local hook maps the repository with Graphify after each commit:

```bash
python -m pip install graphifyy
bash scripts/install_graphify_hook.sh
```

Disable it for one commit with `SSATK_GRAPHIFY_HOOK=0 git commit ...`.

---

## Dependencies

NumPy, SciPy, pandas, Astropy and PyERFA, h5py, Matplotlib, Plotly, Pillow,
imageio, REBOUND, `llnl-ssapy` (orbit objects, SGP4 and Keplerian propagation,
Earth-orientation helpers), and the four `ssatk-data-*` packages above.

---

## Citing SSATK

Citation metadata is in [`CITATION.cff`](CITATION.cff); GitHub's
"Cite this repository" button formats it.

---

## License

SSATK is distributed under the BSD 3-Clause license. All new contributions must
be made under the same license; see [LICENSE](LICENSE).

SPDX-License-Identifier: BSD-3-Clause

LLNL-CODE-2015996
