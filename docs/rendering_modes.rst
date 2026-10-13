Surface maps and low-spec plots
===============================

Moon surfaces use the installed ``moon.png`` lunar photomosaic from
``ssa-data-lunar`` by default. This includes Matplotlib orbit/cislunar panels,
Moon close-ups, Plotly layers, solar-system scenes, eclipse views, notebook
meshes, and the spacecraft camera. The dedicated Moon WebGL viewer retains
its baked albedo, normal and horizon maps. No additional repository image
assets are required and ordinary plotting never downloads a surface map.
If a general-purpose surface map is unavailable, the renderer warns and
uses a plain sphere. The full Moon WebGL viewer still requires its prepared
cache; its low-spec mode does not.

Enable low-spec rendering for an entire group of plots:

.. code-block:: python

   from ssatk.plots import low_spec_mode, orbit_plot

   with low_spec_mode():
       orbit_plot(r, t, view="cislunar_3d", save_path="lunar.png")
       orbit_plot(r, t, view="globe", save_path="earth.png")

The principal plotting functions also accept a per-call override:

.. code-block:: python

   orbit_plot(r, t, view="cislunar_3d", low_spec=True)
   orbit_plot(r, t, view="cislunar_3d", low_spec=False)  # surface map

``SpacecraftView(..., low_spec=True)``, ``MoonLayer(low_spec=True)``,
``EarthLayer(low_spec=True)``, and the Sun-view shading layers capture the
setting when constructed. Resampling a spacecraft view preserves it.
``set_low_spec_mode(True)`` sets the default in the current Python context;
``set_low_spec_mode(None)`` restores the environment default. Nested contexts
restore their previous setting even when plotting raises an exception.

For a whole gallery or a program launched in another process, set
``SSATK_LOW_SPEC=1`` in its environment. The environment setting defaults to
false; an explicit Python override takes precedence. Low-spec browser exports
are generated with the setting; existing HTML files must be regenerated.
When using the standalone Moon server, pass ``show(path, low_spec=True)`` too.

Low-spec surfaces are blue for Earth and white for the Moon, with the
existing solar/eclipse shading where relevant. Otherwise a neutral display
light gives plain 3-D spheres depth; it does not represent a dated Sun. General Matplotlib surfaces
use at most 36 longitude by 18 latitude cells (648 quads, compared with
16,200 for the default cislunar surface). Browser spheres use 36 by 18
segments. Texture loading, lunar terrain textures, and decorative Earth
terrain/city-light generation are omitted. Geographic raster backgrounds
become plain blue, while coverage values, ground tracks and coordinate grids
remain available. Plots without body surfaces are unaffected.

The mode changes presentation only: it does not change trajectory samples,
propagation, epochs, coordinate systems, physical body radii, explicit display
scale factors or movie duration. WGS84 magnetic-environment plots retain their
oblate Earth reference ellipsoid. Eclipse diagnostic colours and occulting
silhouettes retain their physical meaning; a solar-eclipse silhouette remains
black. Cross markers denote unresolved body centres, not body diameters.

Orientation and interpretation
------------------------------

Dated maps use SSATK's Earth and DE440 Moon body rotations, transformed into
the same axes as their trajectories. An Earth–Moon rotating frame is not the
Moon's body-fixed frame. Still orbit/cislunar scenes show surfaces at the first
supplied epoch; animations update them at each sample. Undated low-level
surface helpers use body-fixed axes. Existing orbit helpers that accept no
time retain their legacy GPS-zero reference epoch. Pass a date for meaningful
surface orientation.

The map is north-up with longitude -180 to +180 degrees. Lunar map coordinates
approximate the DE440 principal-axis frame; the mean-Earth map-frame offset
is below one kilometre. Albedo contrast and lighting are illustrative, not
calibrated photometry. Generic mapped spheres add no invented radial terrain.
The dedicated WebGL viewer uses measured elevation-derived shading without
changing its reference sphere. Eclipse appearance strips without dates use a
fixed near-side view, without libration.

Validation
----------

``scripts/tests/test_body_rendering_modes.py`` checks literal spherical
coordinates and asymmetric map texels in both backends, real Moon positions
against Astropy's independent ERFA ephemeris (20 km model-difference budget),
and the prime meridian against the 9-degree optical-libration envelope.
Contract checks reject texture/terrain loading in low mode, exercise nested
mode restoration, preserve the mode across spacecraft resampling, and export
browser scenes without texture payloads or a Moon terrain cache. Existing
globe, Moon, cislunar and solar-system tests continue to check scale and frame
conventions. Browser and gallery validation check the rendered exports.
