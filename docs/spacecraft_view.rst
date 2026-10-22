Spacecraft mesh and attitude exports
====================================

``ssatk.geometry.SpacecraftMesh`` is the canonical mesh type used by
``ssatk.plots.spacecraft_view`` and the WebGL ``satellite_viewer``. The older
``ssatk.plots.spacecraft_view.SpacecraftMesh`` import remains an alias.
The viewers display a spacecraft mesh, body-frame axes,
the spacecraft-to-Sun direction, catalog stars, and Earth or Moon. The default camera
shows a static overview; select ``camera="follow"`` or ``camera="limb"`` for
a spacecraft close-up. Planet size and position follow perspective geometry
at the supplied position and epoch. A nearby planet can fill much of the
background. ``spacecraft_view`` rendering is offline and does not require a
browser or GPU; the exported ``satellite_viewer`` HTML uses browser WebGL.

Input conventions
-----------------

* ``r``: position samples ``(N, 3)``, in geocentric celestial reference frame
  (GCRF) axes. Units default to metres; use ``units="km"`` explicitly for km.
* ``t``: Astropy ``Time`` or **absolute GPS seconds**, strictly increasing.
  For an elapsed-time integration, supply ``epoch.gps + result.t``; do not
  label elapsed seconds as GPS time. If the result already uses absolute
  GPS seconds, pass those directly without adding the epoch twice.
* ``q``: body-to-GCRF quaternions ``[w, x, y, z]``, shape ``(N, 4)``, or
  one constant quaternion. Nonzero finite inputs are normalized. Attitude is
  supplied by the caller; the view does not infer it from orbital velocity.
* ``mesh``: an optional preset name (e.g. ``"starlink"``, ``"cubesat"``, ``"hubble"``),
  or ``SpacecraftMesh`` with body-frame vertices in **metres**, indexed
  faces, and optional per-face RGB colors. Mesh origin and scale are preserved.
  Triangular and convex polygon faces are supported. Alternatively, pass a
  ``SpacecraftBody`` with ``vertices_body`` on every facet through ``body=``.
  Meshes are always body-frame metres; construction does not normalize,
  recenter, or otherwise change the supplied origin or scale.
* ``frame="moon_centered_gcrf"``: Moon-relative positions in **GCRF axes**.
  This is neither the rotating Earth-Moon frame nor the lunar body-fixed frame.
  Quaternion axes remain GCRF. The default is ``frame="gcrf"``.
* Optional ``sun_position`` and ``moon_position`` are **geocentric GCRF**
  vectors, either constant or ``(N, 3)``, in the same units as ``r`` regardless
  of ``frame``. Without overrides, LLNL SSAPy's ``get_body(...).position``
  supplies ephemerides at the requested epochs. Earth is the GCRF origin.

Usage with a supplied time series
---------------------------------

.. code-block:: python

   from ssatk.plots.spacecraft_view import SpacecraftMesh, spacecraft_view

   view = spacecraft_view(
       r, t, q, camera="limb", environment="auto", units="m",  # Built-in Starlink-like mesh.
       title="Spacecraft attitude history",
   )
   view.save("spacecraft.png", index=0, width=1600, height=1000)
   view.save("spacecraft.jpg", index=10)
   view.save("attitude.gif", fps=15, duration_s=8, width=900, height=600)
   view.save("attitude.mp4", fps=30, duration_s=12, width=1600, height=1000)

Built-in meshes and custom geometry
----------------------------------------

No mesh file is required. Omitting both ``mesh`` and ``body`` uses the
``starlink`` preset. Select another shape with ``mesh="cubesat"`` or
``mesh="boxwing"``. ``SpacecraftMesh.presets()`` lists the available names.
The complete :doc:`spacecraft_presets` includes historical observatories, small
missions, NASA partner missions and hosted instruments, with sources and
per-model approximation notes. ``SpacecraftMesh.preset_info(name)`` exposes
those notes in Python; ``presets(group="small")`` filters the catalog.

.. list-table:: Illustrative preset geometry (body-frame metres)
   :header-rows: 1

   * - Name
     - Bus dimensions X x Y x Z
     - Solar arrays
   * - ``starlink`` (default)
     - 2.8 x 1.4 x 0.18
     - Two 2.8 x 4.0 wings extending along +/-Y
   * - ``cubesat``
     - 0.10 x 0.10 x 0.3405
     - Four corner rails and recessed body-mounted cells; no deployed wings
   * - ``boxwing``
     - 1.2 x 1.0 x 0.8
     - Two 2.0 x 1.4 wings extending along +/-X

These are schematic viewing templates. The ``starlink`` silhouette is inspired
by the flat bus and dual arrays shown in `SpaceX's Starlink technology overview
<https://www.starlink.com/technology>`_; the dimensions above are chosen for this
illustration, not measured or manufacturer-validated dimensions for a Starlink
generation. The revised CubeSat uses the 100 x 100 x 340.5 mm 3U envelope
from `NASA's SmallSat structures table
<https://www.nasa.gov/smallsat-institute/sst-soa/structures-materials-and-mechanisms/>`_.
Its rails and solar-cell details remain illustrative; this is not a qualified
deployer design. The physical ``SpacecraftBody.cubesat`` constructor is unchanged.
Presets do not define flight mass, inertia, drag, or radiation-pressure parameters.

The three generic templates use the bus center as origin. Their deployed
arrays lie in the body XY plane with +Z normals. Telescope presets have
mission-specific visual axes and construction origins documented in the catalog;
none provides a measured center of mass or flight attitude frame. All shapes
remain rigid and do not automatically track the Sun.
The supplied quaternion rotates the whole mesh. Selection of a preset does not
change the supplied orbit or attitude. To resize a preset, pass
``SpacecraftMesh.from_preset("starlink", scale=2.0)`` as ``mesh``; this doubles
all lengths, including thickness. Custom geometry replaces the default:

.. code-block:: python

   view = spacecraft_view(r, t, q, mesh="boxwing")
   mesh = SpacecraftMesh.from_file("spacecraft.obj")
   view = spacecraft_view(r, t, q, mesh=mesh)
   view = spacecraft_view(r, t, q, body=spacecraft.body)

``SpacecraftMesh.from_file(path, scale=None, units=None, linear_deflection=.1,
angular_deflection=.5)`` supports OBJ directly. ``units`` may be one of
``m``, ``cm``, ``mm``, ``in``, or ``ft`` and is mutually exclusive with
``scale``; neither is inferred for OBJ or mesh files. STL, PLY, OFF, GLTF,
GLB, and 3MF use the optional ``ssatk[mesh]`` extra (``trimesh``). STEP, STP,
IGES, and IGS use the optional ``ssatk[cad]`` extra (OpenCascade,
``cadquery-ocp``); OpenCascade
CAD coordinates are in OpenCascade's internal millimetres, so CAD uses a default
``.001`` SI conversion. Linear deflection is in those millimetres and angular
deflection is in radians. GLTF uses metres; 3MF reads embedded units and defaults
to millimetres. Unitless OBJ/STL/PLY/OFF default to metres: specify ``units`` or
``scale`` when the authoring tool uses different units.

``SpacecraftMesh.export(path)`` writes OBJ directly or uses ``trimesh`` for
mesh outputs. It does not export CAD. ``convert_mesh(source, destination,
**import_options)`` is the convenience form for importing and exporting.
Proprietary SolidWorks, Inventor, CATIA, and FBX inputs are not promised;
export those designs to a neutral CAD or mesh format first. RGB colors may be
retained where the backend supplies them, but textures and PBR materials are
not preserved.

Unsupported file extensions raise ``ValueError`` with conversion instructions.
For example, ``.sldprt``/``.sldasm`` identifies SolidWorks and recommends exporting
the 3D part or complete assembly to STEP, installing ``ssatk[cad]``, and loading
the exported file. Inventor, CATIA, Creo/NX, Fusion, FreeCAD, Parasolid and ACIS
inputs receive corresponding guidance. FBX and other scene formats recommend
exporting GLB/glTF or triangulated OBJ using a compatible external application.
Drawing formats are not treated as 3D bodies. Renaming an extension is not a
conversion. Verify units, component placements, origin and body axes afterward.
Unsupported export targets recommend mesh output instead; SSATK cannot recover
an editable CAD solid from its tessellated representation.

For interchange, ``mesh.to_dict()`` and ``SpacecraftMesh.from_dict(value)``
use schema ``ssatk.spacecraft_mesh.v1`` with ``units="m"``,
``frame="body"``, vertices, faces, RGB colors, and per-face part-name lists.

``SpacecraftMesh.from_body(spacecraft.body)`` uses the actual facet vertices
relative to ``body.current_center_of_mass``, so a propagated center-of-mass
position can be passed directly. This is one rigid geometry/mass-center snapshot;
articulation and a changing center of mass are not animated. For raw arrays or
OBJ files, ``r`` locates the supplied mesh origin; center those vertices on the
center of mass yourself when ``r`` represents the mass center.
It rejects facets defined only by area and normal, since those quantities do
not uniquely specify their shape. Supplying both ``mesh`` and ``body`` is an
error; a custom body alone always overrides the automatic preset.
OBJ materials and texture files are not imported.

The WebGL viewer accepts ``mesh=`` or ``body=`` for one shared model, or
``meshes=`` for one model per orbit. These choices are exclusive. Its
``mesh_scale="glyph"`` default exaggerates display size for visibility;
``mesh_scale="physical"`` uses the mesh's metre dimensions. WebGL presets are
the same Python presets used by this viewer, not independent procedural
shapes; display-only LOD glyphs remain available.

For dynamics, ``SpacecraftBody.from_file(path, mass=..., inertia=..., **...)``
uses the imported geometry while keeping mass, inertia, facet coefficients,
and other physical properties as caller inputs. If inertia is omitted, the
existing uniform bounding-box estimate is used. CAD import does not infer
mass, inertia, material, or optical coefficients and does not provide
self-occlusion physics.

See :doc:`camera_modes` for overview, follow, chase and orbit sweep options.

``view.figure(index)`` returns a Matplotlib figure for further annotation.
``view.render(index)`` returns an RGB image array. Stills use the exact selected
sample. ``view.geometry(index)`` exposes transformed vertices, body axes,
Sun/Earth/Moon unit vectors, and whether the Sun center is occulted, for
numerical inspection independently of image generation.

Timing and camera
-----------------

The gallery's **Spacecraft Mesh, Attitude, and Sun Direction** demo includes a
45-second MP4 showing **24 hours (86,400 seconds)** in a near-geostationary orbit.
LLNL SSAPy's ``Orbit`` and default Keplerian ``rv`` propagator sample every
movie epoch directly. The circular radius is derived from Earth's gravitational
parameter and the mean sidereal period, 86,164.0905 seconds: approximately
42,164 km from Earth's center, or 35,786 km altitude. A 24-hour movie therefore
covers slightly more than one revolution, consistent with the sidereal-day
distinction in `NASA's orbit catalogue
<https://science.nasa.gov/earth/earth-observatory/catalog-of-earth-satellite-orbits/>`_.

The initial plane is Earth's equator at the initial epoch, transformed into
GCRF with ``eci_to_ecf_matrix``. Body +X points along track, +Y opposite the
orbit normal and +Z toward Earth. The demo explicitly selects ``camera="limb"``, which moves with the spacecraft
and keeps Earth's limb behind it. Earth's geography should remain nearly fixed
in that view while the catalog sky, Sun direction and illumination change.
The scene starts at Greenwich on 2026-10-11 at 12:00 UTC. It is an unperturbed
two-body illustration with prescribed attitude, not a controlled flight orbit.

Fast gallery mode exports 540 frames at 12 fps and 960 by 640 pixels; normal
mode exports 1,080 frames at 24 fps and 1,280 by 720 pixels. Both retain the
full 24-hour span and 45-second playback. MP4 streams frames to the encoder.
The short Moon GIF is a separate one-minute lunar example.

Animation duration is **playback time**. The entire physical input interval is
resampled onto ``round(fps * duration_s)`` output frames, including both ends.
Positions interpolate linearly in actual elapsed seconds, and attitude uses
SciPy's shortest-path spherical linear interpolation (SLERP). Nonuniform input
timestamps therefore retain their relative timing. No orbit propagation,
extrapolation, or angular-rate estimation is performed. Sample fast enough to
resolve the intended motion; quaternions alone cannot encode full revolutions
between adjacent samples. A one-sample input supports stills only.

Default ephemerides are reevaluated at output epochs. Caller-supplied Sun/Moon
tracks interpolate linearly. GIF delays distribute 10 ms ticks across the
frames, keeping total timing within one tick of ``frame_count / fps`` instead
of accumulating per-frame rounding error. GIF rates are limited to 100 fps;
very short animations still contain at least two frames. Some players impose
additional minimum delays. The MP4 format has no GIF timing constraint;
MP4 requires ``imageio-ffmpeg`` (included in ``ssatk[video]``) and even image
dimensions. GIF export retains palette frames in memory; use MP4 for long or
large animations, whose frames stream to the encoder.

``environment`` accepts ``earth``, ``moon``, ``both``, or ``auto``. Auto selects
the nearest body surface at each frame. It can switch at a cislunar boundary;
choose an explicit body and camera offset for an uninterrupted camera path.
Environment changes only the displayed bodies, not coordinates or dynamics.
Earth and Moon both participate in Sun-center occultation checks.

The ``limb`` camera is five mesh radii from the body origin, oriented so the
selected body's limb lies in the background. Optional ``camera_offset`` is a
constant or ``(N, 3)`` GCRF vector **in metres from spacecraft to camera**.
The camera looks toward the spacecraft. ``camera_up`` is a GCRF up direction
and ``fov_deg`` is the vertical field of view. Camera and spacecraft positions
inside Earth or Moon are rejected. Body and Sun arrows have an illustrative
length of 1.75 mesh radii; their directions are physical. The arrows are
annotation overlays and remain visible through the mesh and foreground bodies.

Rendering limits and validation
-------------------------------

Earth and Moon are spherical visual surfaces with SSATK radii. Textures come
from the installed data packages; missing textures produce an explicit warning
and an untextured sphere. ``textures=False`` requests untextured spheres.
Earth texture orientation uses ``eci_to_ecf_matrix``. Lunar orientation uses
``lunar_body_rotation`` (DE440 principal axes); the small principal-axis versus
mean-Earth mapping difference is not corrected. This is not a lunar map product.

Lighting is illustrative Lambertian shading with an ambient floor, not a
calibrated radiance or solar-radiation-pressure result. Sun-center occultation
uses a point source and spherical bodies; no penumbra fraction, self-shadowing,
terrain relief, atmospheric refraction, or multiple reflection is modeled.
Mesh visibility uses a CPU depth buffer with perspective-correct reciprocal
depth interpolation at pixel centers. Polygon outlines exclude artificial
fan-triangulation diagonals. Displayed Earth/Moon spheres occlude the mesh
using the same pixel-center rays and physical distances. Only opaque surfaces are supported; exact coplanar
faces have ambiguous visibility, and sub-pixel features can alias. Celestial
backgrounds are ray-sampled at up to 720 pixels high
and enlarged for export; output dimensions are not a claim of surface detail.
The star catalog uses GCRF directions evaluated at the first input epoch;
proper motion across the animation and relativistic light effects are omitted.
Missing stars raise an error unless ``stars=False`` is chosen explicitly.

``tests/test_spacecraft_view.py`` checks exact quarter-turn attitude, physical
mesh size, a 3-4-5 Sun-direction triangle, spherical occultations, the 30-degree
limb of a radius-one sphere at distance two, pinhole projection, lunar origin
translation, and analytic constant-rate motion on nonuniform epochs. Image
decoding, timing, and foreground occlusion checks run separately in
``scripts/tests/test_spacecraft_exports.py`` as workflow contracts.
Preset tests compare envelopes and surface areas with the stated rectangular
prisms and panels, including squared area scaling. These checks validate the
template geometry, not its agreement with real spacecraft measurements.
``tests/test_spacecraft_presets.py`` separately checks the NASA 3U envelope and
selected published observatory dimensions/component counts. The depth buffer
is compared with independent analytic ray-box intersections and intersecting
planes in both drawing orders. Catalog build tests are software contracts,
not physical validation of the schematic mission family models.
``tests/test_viewer_star_reference.py`` compares the existing browser viewer's
exported sky with Astropy's independent GCRS-to-TEME transformation to 0.5 arcsec
at two epochs. It rejects the former duplicate TEME rotation.

Generate examples
-----------------

.. code-block:: bash

   python -m demos.six_dof.demo_spacecraft_view

The demo exports Earth and Moon PNG/GIF scenes with the default Starlink-like
mesh using specified circular arcs and body spin. ``example_view(mesh="cubesat")``
selects another preset; ``example_view(mesh=example_mesh())`` demonstrates a
custom bus and panels. These are analytic illustrations, not measured flight data.

.. toctree::
   :maxdepth: 1

   spacecraft_presets
