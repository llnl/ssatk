# Demo Gallery

> Run all demos locally and build a presentation-style gallery.

---

## Running from a cloned repo

From the top of the repository:

```bash
python -m ssatk.run_all_demos
```

If you installed the repo in editable mode, you can also use the console script:

```bash
ssatk-demo-gallery
```

Example with explicit paths:

```bash
python -m ssatk.run_all_demos --demos-dir demos --output ./gallery_output
```

---

## Demo categories

The `demos/` package is organized by workflow area:

| Folder | Contents |
|--------|----------|
| `getting_started/` | first workflow, plotting quickstart, packaged data access, time helpers, optional 3LE parsing |
| `coordinates/` | frame transforms, surface state vectors, NTW/GCRF checks, rotation utilities |
| `data_io/` | sampling, sphere generation, and dictionary/HDF5 round-trip examples |
| `orbital_mechanics/` | orbit quantities, propagation/model comparisons, ellipse fitting, maneuvers, transfers, station keeping |
| `six_dof/` | rigid-body attitude, gravity-gradient torque, facet solar-radiation pressure, and finite-burn coupling |
| `orbit_visualization/` | orbit, cislunar, globe, Moon, and ground-track visualization workflows |
| `sensor_coverage/` | satellite viewer and sensor field-of-view coverage demos |
| `space_environment/` | Sun/Earth/Moon scenes, solar-system views, magnetic fields, and Van Allen belts |
| `eclipse/` | solar and lunar eclipse geometry and animation demos |
| `analysis_dashboards/` | dashboards, divergence plots, and population summary visualizations |
| `video_animation/` | GIF and video generation workflows |
| `photometry/` | Lambertian brightness, thermal/reflection models, ground-observer examples |
| `benchmarks/` | Reproducible Artemis/JPL and external-propagator comparison cases |

The gallery runner searches these subfolders recursively, so adding a new
`demo_*.py` file under the right category is enough for it to appear unless the
file sets `GALLERY_INCLUDE = False`.

---

## Running after `pip install ssatk`

If `ssatk` is installed and the console entry point is available, the
command auto-detects the packaged demos and can be run from any directory:

```bash
ssatk-demo-gallery
```

Or run it as a module:

```bash
python -m ssatk.run_all_demos
```

---

## Useful options

| Flag | Description |
|------|-------------|
| `--open` | Open the generated report after it is written |
| `--no-clean` | Don't clean the output directory before running |
| `--output ./gallery_output` | Write results to a custom output directory |
| `--demos-dir demos` | Specify the demos directory explicitly |

---

## Output

The gallery runner creates an **HTML report** at
`~/ssatk_output/documents/index.html` by default and saves generated demo
artifacts beside it in the selected output directory. The default output root is
the user's home directory, not the source checkout; set `SSATK_OUTPUT_DIR` or
pass `--output` for an explicit alternate location.

---

## Demo vs. test policy

- Gallery demos should create useful artifacts: figures, animations, logs, or
  other files that help a user understand the workflow.
- Validation-only examples should set `GALLERY_INCLUDE = False` and be covered
  by `pytest` instead of appearing in the default gallery.
- Duplicate gallery demos should be merged into the clearest user workflow; any
  deleted behavior should move into direct tests when it improves coverage.
- Optional-data demos should skip gracefully or report actionable missing-data
  messages when a package, local cache, or network source is unavailable.
