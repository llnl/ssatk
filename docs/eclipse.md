# Solar and lunar eclipses

The eclipse implementation is split across existing SSAPy-Toolkit packages:

- `compute/eclipse_*.py`: finite-source physics, event states, search, photometry, uncertainty.
- `coordinates/eclipse_*.py`: observers, horizons, lunar attitude, lunar geometry.
- `io/eclipse_*.py`: SSAPy-Data resolution, terrain/topography readers, provenance.
- `plots/eclipse_*.py`: scientific, interactive, and cinematic rendering.
- `ssatk/eclipse.py`: lazy canonical public API.
- `demos/eclipse/demo_eclipse_scientific.py`: reference solar and lunar demonstration.

No new repository directory is created. Immutable PNG/NPZ/LUT resources are kept in SSAPy-Data.
