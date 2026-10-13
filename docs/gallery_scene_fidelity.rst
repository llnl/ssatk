Solar System scenes and gallery sampling
========================================

The gallery names these views **Solar System — Planets and Orbits** and
**Solar System — Sun, Earth and Moon**. Their HTML exports are
``demo_solar_system_plot.html`` and ``demo_solar_system_earth_sun_moon.html``.
The existing Python module names ``ssatk.plots.solar_view_plot`` and
``ssatk.plots.earth_sun_plot`` remain available. Running these modules directly
exports ``solar_system_plot.html/.jpg`` and
``solar_system_earth_sun_moon.html/.jpg``, respectively.

Both views use physical spherical body radii and
geometric, simultaneous ephemeris positions by default. All scene coordinates,
stars, and trajectories use the mean J2000 ecliptic axes, in astronomical units
(AU). The default Astropy ``builtin`` ephemeris is an approximate model;
``sol_ephemeris`` may select an installed JPL ephemeris. Failures are reported
rather than mixing a Kepler approximation into one part of the scene.
Historical trails using ``builtin`` stop at 1900, the lower validated boundary
of its Earth/Sun model; Neptune's historical trail can therefore be partial.

Radii are the existing LLNL SSAPy spherical-body values: Sun 696,340 km, Earth
6,378.137 km, and Moon 1,738.1 km. The Sun value is SSAPy's adopted model radius,
not the IAU nominal solar conversion radius of 695,700 km. Earth is represented
as a sphere at its WGS84 equatorial radius. Surface colours, other planets'
spin orientations, and lighting remain illustrative; this is not a resolved
surface-navigation or radiometry tool.

At a one-AU viewing scale Earth is subpixel. Cross markers and labels identify
body centers and do not represent body diameters. ``planet_scale=1``,
``sun_scale=1``, and ``moon_orbit_scale=1`` select physical scale. Custom
positive factors remain available and appear explicitly in the figure title.
``planet_scale`` applies to planets and the Moon, while ``sun_scale`` applies
independently to the Sun. Decorative Sun glow is disabled by default.

The sky uses proper-motion-adjusted catalogue directions projected as a
background at infinity. It has no AU radius, parallax, or missing hemisphere.
Rotating an HTML scene changes the visible part of the sky; translating or
zooming the camera does not create stellar parallax. The angular projection
uses gl-plot3d's 45-degree perspective vertical field of view. Orthographic
plots keep this perspective sky as a visual backdrop. Missing catalogue data
is labeled; synthetic stars are not silently substituted.

Use the returned figure's ``write_html`` or ``to_html`` methods to retain the
camera-update script. ``write_image`` produces a PNG/JPEG/SVG snapshot with the
sky projected for that image's dimensions and camera. Export through the
lower-level ``plotly.io`` functions bypasses these figure methods and does not
install the interactive sky updater. Static export requires ``ssatk[static]``
and Chrome; the gallery workflow installs these dependencies explicitly.

Trajectory and playback sampling
--------------------------------

``orbit_animation(max_frames=...)`` limits playback frames while preserving
all input trajectory samples in the fading trail. ``tail``/``tail_points``
counts playback frames. The renderer does not invent a smooth orbit between
undersampled inputs. For the LEO gallery examples, propagated positions are
sampled at no more than 10-second intervals; full intervening samples remain
available even when the animation renders fewer frames.

The fancy lunar demo exports 720 frames at 12 frames per second: 60 seconds
of playback spanning 27.3217 days of physical time: one mean revolution of
the Moon around Earth. Fast mode lowers image
resolution, not duration. The camera moves during playback. This is a
perturbed cislunar trajectory initialized near Earth–Moon L1, not a designed
or station-kept halo orbit. Propagation failures fail the demo instead of
silently shortening its time span. Earth is drawn at its physical radius;
Moon and spacecraft markers identify their positions.

The magnetic-field gallery demo writes both interactive HTML and a PNG card
preview. It deliberately uses the internal IGRF field with illustrative
IGRF-traced L-shell belts, without requiring a network solar-wind lookup or
SpacePy/IRBEM. These belts do not claim measured particle flux. The gallery
workflow installs the geomagnetics and static-image extras; it must not
publish a missing magnetic-field dependency as a successful figure.

Validation
----------

Numerical tests compare the coordinate rotation with ERFA's IAU 2006 matrix,
the approximate body ephemerides with SSAPy's separate DE ephemeris, and
background projection with analytic pinhole rays. Export tests check physical
mesh radii, preserved double-precision AU coordinates, and a 7,000 km circle:
limiting playback to three frames must retain its 360 one-degree chords.
Browser checks cover opposite viewing directions and window resizing; encoded
MP4 duration is checked from the generated file during the handoff review.
