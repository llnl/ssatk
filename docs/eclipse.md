# Solar and lunar eclipses

The eclipse implementation is split across existing SSAPy-Toolkit packages:

- `compute/eclipse_*.py`: finite-source physics, event states, search, photometry, uncertainty.
- `coordinates/eclipse_*.py`: observers, horizons, lunar attitude, lunar geometry.
- `io/eclipse_*.py`: SSAPy-Data resolution, terrain/topography readers, provenance.
- `plots/eclipse_*.py`: scientific, interactive, and cinematic rendering.
- `ssatk/eclipse.py`: lazy canonical public API.
- `demos/eclipse/demo_eclipse_scientific.py`: reference solar and lunar demonstration.

Earth and Moon maps and ephemeris kernels come from the installed split
`ssa-data-*` packages. The Earth map falls back to `earth_day_2048.jpg` and
the DE430 kernel resolves as `de430_1900_2150.bsp` through SSAPy's existing
lookup. No nearby checkout is searched automatically. Select external assets
with `SSAPY_DATA_DIR` or the per-asset overrides reported by `ssatk-eclipse assets`.

The cinematic photometry correction LUT is generated deterministically in memory
when no external LUT is installed. An override requires both the binary and its
JSON metadata. No downloaded or generated assets are committed to SSATK.

Reference event vectors are GCRF: the NASA apparent equator-of-date geometry is
rotated before it is combined with SSAPy states. Earth-fixed transforms reuse
the existing Astropy-backed runtime helpers. Reference reconstruction remains
an approximation, not an independent precision ephemeris.

Use `solar_scope="local"` with `EventRequest.observer` for observer products.
`ssatk-eclipse promote` exits nonzero when its report is blocked or failed.

Discovery supports SSAPy, the two-event catalog, and an approximate analytic
preview backend. The unreviewed Swiss Ephemeris adapter and Playwright browser
matrix are not shipped. Terrain inputs support HGT and Terrarium PNG, and
lunar topography supports NPZ/PDS; GeoTIFF support is deferred pending a
maintainer-approved Rasterio dependency. The normal CI matrix covers eclipse
code and checks all three bundled JavaScript assets.

Rendering readiness is not scientific promotion. The strict promotion gate
still compares SSAPy predictions with NASA contact/path references under its
original thresholds; a nonzero exit must not be treated as a validated release.
