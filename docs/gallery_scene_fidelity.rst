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
resolution, not duration. The default camera is a static overview; select
``camera="orbit"`` for a moving sweep (see :doc:`camera_modes`). This is a
perturbed cislunar trajectory initialized near Earth–Moon L1, not a designed
or station-kept halo orbit. Propagation failures fail the demo instead of
silently shortening its time span. Earth is drawn at its physical radius;
Moon and spacecraft markers identify their positions.

The magnetic-field gallery demo writes interactive HTML, a PNG card preview,
and two 1280 × 960 MP4 camera tours: field lines alone and field lines with
belts. Fast mode exports 12 seconds at 12 frames per second; full mode exports
15 seconds at 24 frames per second. Only the camera moves: the field stays
at the 2025 epoch, and neither field evolution nor particle motion is simulated.
It deliberately uses the internal International Geomagnetic Reference Field
(IGRF) with illustrative
IGRF-traced L-shell belts, without requiring a network solar-wind lookup or
SpacePy/IRBEM. These belts do not claim measured particle flux. The gallery
workflow installs the geomagnetics, static-image, browser and video extras; it must not
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


Dated textured globe
--------------------

``orbit_plot(r, t, view="globe", save="earth.mp4")`` now rotates the textured
Earth at every rendered epoch. PNG/JPEG stills and GIF/MP4 animations share
one Earth surface implementation. The north-up, equirectangular texture maps
longitude -180 to +180 degrees from left to right, with pixels at cell centres.
The optional cached NASA 8K day map is preferred; otherwise the installed
Earth texture is used. Rendering never downloads imagery. The sphere uses
SSAPy's 6,378.137-km Earth radius; it does not represent terrain or oblateness.
The day map is static imagery, not weather or day/night illumination at the date.

Inputs are GCRF positions in metres and absolute Astropy ``Time`` values or
GPS seconds. ``frame="gcrf"`` displays Earth rotating in inertial space;
``frame="itrf"`` displays a fixed Earth with the trajectories transformed to
Earth-fixed axes. The surface orientation uses SSATK's single GCRF/ITRF route
(``ssatk.coordinates._earth_rotation``): CIO-based IAU 2006 precession / IAU
2000A nutation with the IERS celestial-pole offsets, Earth Rotation Angle at
UT1, and polar motion, all from SSATK's packaged EOP data. A UTC label
identifies the rendered instant.
The camera is orthographic, and the Earth hides far-side spacecraft and trails.
Trails omit segments with an occulted endpoint rather than bridging hidden arcs.

Animated globe tracks require the same strictly increasing, finite time grid
(to within one microsecond). Propagate tracks to common epochs before export;
the renderer does not synchronize different times by array index or extrapolate.
``max_frames`` selects playback samples without discarding intervening trail
geometry. Static ``globe_time`` selects the Earth orientation. A track lacking a sample
within one microsecond of that instant has no highlighted reference point. ``reference_index`` defaults
to the first track's last sample when no explicit globe time is supplied.

``earth_resolution=180`` gives 180 longitude by 90 latitude cells in animations;
static plots also accept it, overriding the older ``scale`` knob. Only the
Earth surface is meshed; satellites retain point markers and orbital trails.

The demo shows a synthetic 420-km LEO in a close-up, plus a separate LEO/MEO
overview. It is not an observed ISS trajectory. Fast gallery mode spans four
hours in 120 frames at 8 fps (15 seconds playback); full mode spans twelve
hours in 240 frames (30 seconds). Both retain ten-second trajectory samples
and use 240 by 120 texture cells in the video; the close-up still uses
720 by 360 cells. Set a different date with::

    from demos.orbit_visualization.demo_globe_plot import main
    main(start_time="2026-10-12T00:00:00", make_figures=True, make_video=True)

For a supplied trajectory::

    from ssatk.plots import orbit_plot
    orbit_plot(r_gcrf_m, times, view="globe", earth_resolution=240,
               save="earth.mp4")

The globe regression tests inspect the actual polygon coordinates delivered to
Matplotlib against Astropy's independent GCRS/ITRS transformation, with a
0.5-arcsecond model-difference allowance, and supplied satellite positions.
They also check north/south and longitude texture mapping, km and GEO axis units,
TT-to-UTC date handling, selected snapshot epochs, and actual raster pixels for
a spacecraft in front of and behind an opaque Earth. These validate geometry
and rendering contracts, not measured orbit accuracy.

Dated lunar plots
-----------------

``moon_plot_3d`` distinguishes Earth-centred GCRF input from
``r_frame="moon_centered"`` inertial positions. GCRF tracks become Moon-centred
Earth–Moon rotating coordinates at each sample; Moon-centred input retains
GCRF-aligned axes. The texture uses the separate DE440 lunar principal-axis
orientation at ``reference_time`` (scalar GPS seconds or Astropy Time), or the
midpoint of the supplied epochs. Surface-only ``t`` inputs also set the date.
An undated surface view uses the current time; an undated trajectory is rejected.
The date and frame appear on every export. Multi-epoch tracks have a frozen
reference-epoch surface and illumination, not a time-varying exposure.

The demo previously propagated with SSAPy's default Earth gravitational
parameter and then subtracted the Moon's position from an already Moon-centred
orbit. It now explicitly uses ``MOON_MU`` and ``r_frame="moon_centered"``.
Its synthetic ellipse has a 2,738.1-km semi-major axis, eccentricity 0.01 and
90-degree inclination in inertial equatorial axes. One revolution lasts
12,856.767 s; fast/full modes sample 361/721 positions. Mean altitude is
1,000 km above SSAPy's 1,738.1-km reference sphere, ranging from 972.619 to
1,027.381 km. This two-body example omits lunar harmonics and third-body forces;
its inertial polar plane is not exactly the Moon's equatorial polar plane.

Both orbit views and the surface close-up retain physical sizes. Limits include
the entire Moon as well as the track. A fourth Earth–Moon overview includes
both bodies, their ephemeris centre separation and explicit radius enlargement
factors. It omits the tiny lunar track rather than drawing it inside an enlarged
Moon. ``show_earth=True`` expands the view; automatic Earth display only occurs
if it fits. Earth uses its dated full GCRF/ITRF texture rotation; its distance
is no longer a fixed 384,400 km. Moon and Earth use their own Sun directions.
The Moon map approximates the DE440 principal-axis coordinates: the map's
mean-Earth-frame offset is sub-kilometre and is not resolved by this display.

The installed lunar map is north-up and east-positive, spanning -180 to +180
degrees; the mesh has 360 by 180 texture cells. Ambient plus Lambert shading
is illustrative, not calibrated radiometry. Missing textures give visible
plain spheres; available Earth split-package textures are supported. No new
binary assets are committed or downloaded by the renderer.

A catalogue sky is projected onto a separate 120-degree gnomonic backdrop in
the same reference-epoch axes. It is directional context for an orthographic
orbital diagram, not a sensor measurement. The former finite star sphere is
removed from this renderer. The Sun symbol is off by default; explicitly
requesting ``show_sun_graphic=True`` draws a labelled direction symbol with
schematic size and distance. Physical solar illumination remains enabled.

Cislunar scene framing
----------------------

``orbit_plot(..., view="cislunar_3d")`` fits its Moon-centred tracks and the
physical Moon with an 8% margin on each side. The former fixed one-GEO padding
added 42,164 km on every side even for a 1,000-km-altitude lunar orbit. Use
``scene_pad=0.05`` for a 5% margin. Bounds remain equal on all displayed axes;
body radii and trajectory coordinates are not enlarged to create the zoom.

All tracks are transformed before selecting the bounds and axis units.
Each panel chooses one common km or GEO unit from positions in that panel's
frame. This prevents both Earth-centred distances from inflating a lunar
close-up and mixtures of km/GEO coordinates for near-Earth and near-Moon
tracks of different lengths. Input order cannot change the axes. The entire
central body stays inside the bounds, even for a short arc; XY views ignore
out-of-plane excursions when choosing the displayed extent.

Paired views default to a 12-by-7-inch figure and keep both panels within the
canvas. Tick spacing supports fractional GEO values. Legends omit Earth and
Lagrange markers when they are outside the shown panels, and ``legend=False``
also applies to paired views. The Moon is an opaque sphere with its true
radius; samples behind its projected disc are hidden for the static camera.
This is an orthographic diagram, not a textured surface or photometric model.

Moon surface maps and the shared :doc:`low-spec mode <rendering_modes>`
are available in these views.
