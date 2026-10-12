Spacecraft mesh and attitude exports
====================================

``ssatk.plots.spacecraft_view`` displays a spacecraft mesh, body-frame axes,
the spacecraft-to-Sun direction, catalog stars, and Earth or Moon. The camera
follows the spacecraft; planet size and position follow perspective geometry
at the supplied position and epoch. A nearby planet can fill much of the
background. Rendering is offline and does not require a browser or GPU.

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
* ``mesh``: ``SpacecraftMesh`` with body-frame vertices in **metres**, indexed
  faces, and optional per-face RGB colors. Mesh origin and scale are preserved.
  Triangular and convex polygon faces are supported. Alternatively, pass a
  ``SpacecraftBody`` with ``vertices_body`` on every facet through ``body=``.
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

   mesh = SpacecraftMesh.from_obj("spacecraft.obj", scale=1.0)  # m per OBJ unit
   view = spacecraft_view(
       r, t, q, mesh=mesh, environment="auto", units="m",
       title="Spacecraft attitude history",
   )
   view.save("spacecraft.png", index=0, width=1600, height=1000)
   view.save("spacecraft.jpg", index=10)
   view.save("attitude.gif", fps=15, duration_s=8, width=900, height=600)
   view.save("attitude.mp4", fps=30, duration_s=12, width=1600, height=1000)

``SpacecraftMesh.from_body(spacecraft.body)`` uses the actual facet vertices
relative to ``body.current_center_of_mass``, so a propagated center-of-mass
position can be passed directly. This is one rigid geometry/mass-center snapshot;
articulation and a changing center of mass are not animated. For raw arrays or
OBJ files, ``r`` locates the supplied mesh origin; center those vertices on the
center of mass yourself when ``r`` represents the mass center.
It rejects facets defined only by area and normal, since those quantities do
not uniquely specify their shape. If no mesh or body is supplied, the view
uses an illustrative SSATK box-wing model, not inferred vehicle geometry.
OBJ materials and texture files are not imported.

``view.figure(index)`` returns a Matplotlib figure for further annotation.
``view.render(index)`` returns an RGB image array. Stills use the exact selected
sample. ``view.geometry(index)`` exposes transformed vertices, body axes,
Sun/Earth/Moon unit vectors, and whether the Sun center is occulted, for
numerical inspection independently of image generation.

Timing and camera
-----------------

Animation duration is **playback time**. The entire physical input interval is
resampled onto ``round(fps * duration_s)`` output frames, including both ends.
Positions interpolate linearly in actual elapsed seconds, and attitude uses
SciPy's shortest-path spherical linear interpolation (SLERP). Nonuniform input
timestamps therefore retain their relative timing. No orbit propagation,
extrapolation, or angular-rate estimation is performed. Sample fast enough to
resolve the intended motion; quaternions alone cannot encode full revolutions
between adjacent samples. A one-sample input supports stills only.

Default ephemerides are reevaluated at output epochs. Caller-supplied Sun/Moon
tracks interpolate linearly. GIF delay is quantized to 10 ms by its format;
MP4 requires ``imageio-ffmpeg`` (included in ``ssatk[video]``) and even image
dimensions. GIF export retains palette frames in memory; use MP4 for long or
large animations, whose frames stream to the encoder.

``environment`` accepts ``earth``, ``moon``, ``both``, or ``auto``. Auto selects
the nearest body surface at each frame. It can switch at a cislunar boundary;
choose an explicit body and camera offset for an uninterrupted camera path.
Environment changes only the displayed bodies, not coordinates or dynamics.
Earth and Moon both participate in Sun-center occultation checks.

The default camera is five mesh radii from the body origin, oriented so the
selected body's limb lies in the background. Optional ``camera_offset`` is a
constant or ``(N, 3)`` GCRF vector **in metres from spacecraft to camera**.
The camera looks toward the spacecraft. ``camera_up`` is a GCRF up direction
and ``fov_deg`` is the vertical field of view. Camera and spacecraft positions
inside Earth or Moon are rejected. Body and Sun arrows have an illustrative
length of 1.75 mesh radii; their directions are physical. The arrows are
annotation overlays and remain visible through the mesh.

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
Mesh triangles use depth sorting, so intersecting geometry may render
incorrectly. Celestial backgrounds are ray-sampled at up to 720 pixels high
and enlarged for export; output dimensions are not a claim of surface detail.
The star catalog uses GCRF directions evaluated at the first input epoch;
proper motion across the animation and relativistic light effects are omitted.
Missing stars raise an error unless ``stars=False`` is chosen explicitly.

``tests/test_spacecraft_view.py`` checks exact quarter-turn attitude, physical
mesh size, a 3-4-5 Sun-direction triangle, spherical occultations, the 30-degree
limb of a radius-one sphere at distance two, pinhole projection, lunar origin
translation, and analytic constant-rate motion on nonuniform epochs. Image
decoding and timing tests are labeled export contracts, not physical validation.
``tests/test_viewer_star_reference.py`` compares the existing browser viewer's
exported sky with Astropy's independent GCRS-to-TEME transformation to 0.5 arcsec
at two epochs. It rejects the former duplicate TEME rotation.

Generate examples
-----------------

.. code-block:: bash

   python -m demos.six_dof.demo_spacecraft_view

The demo exports Earth and Moon PNG/GIF scenes using specified circular arcs
and body spin. These are analytic illustrations, not measured spacecraft data.
