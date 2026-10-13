Animation camera shots
======================

Animated 3-D scenes accept ``camera="overview"`` by default. The camera holds
one framing for the complete plotted trajectory interval. Camera selection
changes the view, never positions, attitudes, epochs, propagation, body scale,
or the number of animation samples. It works in both surface-map and
:doc:`low-spec rendering <rendering_modes>` modes.

Available shots
---------------

.. list-table:: Shared camera modes
   :header-rows: 1
   :widths: 16 44 40

   * - Name
     - Behaviour
     - Useful for
   * - ``overview``
     - Static full-scene framing over the supplied interval.
     - Comparing trajectories, conjunctions and eclipse geometry.
   * - ``follow``
     - Looks at one object from a constant offset in the displayed frame.
     - Watching a spacecraft against the changing Earth, Moon or sky.
   * - ``chase``
     - Follows behind the target's path tangent with a small upward offset.
     - Seeing turns, transfers and along-track motion.
   * - ``orbit``
     - Sweeps around the full scene, keeping its centre and scale fixed.
     - Revealing depth and orbital-plane geometry.
   * - ``limb``
     - Spacecraft-view only: keeps the nearest body's limb behind the mesh.
     - Close-ups of attitude, body axes and Sun direction.

A follow offset is fixed in the **plotted coordinate frame**. In an Earth-fixed
or lunar-fixed plot it rotates with that frame; it is not an inertial vector.
Chase follows the sampled displayed path, not the body's attitude. Stationary
Python targets cannot define chase and raise an error. Browser viewers show a
message until a moving target is available. A sampled pause retains the last
valid chase direction. Abruptly changing or sparsely sampled paths can give
abrupt camera motion; the camera does not smooth or repropagate them.

Usage
-----

.. code-block:: python

   from ssatk.plots import AnimationCamera, orbit_plot, spacecraft_view

   orbit_plot(r, t, view="globe", save_path="overview.mp4")
   orbit_plot(r, t, view="3d", save_path="following.mp4",
              camera=AnimationCamera("follow", offset=(2e6, -3e6, 1e6)))
   orbit_plot(r, t, view="3d", save_path="chase.mp4", camera="chase")
   orbit_plot(r, t, view="3d", save_path="sweep.mp4",
              camera=AnimationCamera("orbit", turns=0.5))
   view = spacecraft_view(r, t, q, camera="limb")
   view.save("attitude.mp4", duration_s=45)

``camera`` accepts a shot name, ``AnimationCamera`` or a dictionary with its
fields. ``target=0`` selects the first trajectory in a multi-orbit animation;
an invalid index raises an error. Single-target adapters require index zero.
The Earth and Moon browser viewers add camera and target selectors to their
exported pages. Their overview remains manually rotatable; animated shots take
over the camera while active. Regenerate older HTML to obtain these controls.

``distance`` sets eye-to-target separation. ``offset`` instead sets a constant
three-vector for follow, or the starting vector for orbit/overview. Do not
specify both. ``up`` supplies the preferred up vector; a parallel view chooses
a perpendicular fallback. ``elevation`` and ``azimuth`` are degrees;
``turns`` is revolutions over the input interval. Valid time samples control
sweep phase, including nonuniform sampling. Legacy undated orbit plots use
sample fraction instead. Browser TLE/catalogue shots use one day for a full
sweep; supplied state-vector tracks use their supplied time interval.

.. list-table:: Camera length units and entry points
   :header-rows: 1

   * - API
     - Distance/offset units
   * - ``orbit_plot(..., save_path="*.gif"/"*.mp4")``, ``orbit_animation``
     - Metres, matching their input trajectory convention; axes may display km or GEO radii.
   * - ``SpacecraftView`` / ``spacecraft_view``
     - Metres, even when trajectory input uses ``units="km"``.
   * - ``BasePlot3D.animate``
     - Kilometres.
   * - ``plot_sensor_fov`` (``fov_animate=True``)
     - Kilometres; ``camera`` also works in its configuration dictionary.
   * - Earth/Sun animated ``build_figure``
     - Astronomical units; use ``cfg["camera"]``. The target is Earth.
   * - ``plot_space_view_animated``, ``plot_2024_solar_eclipse_animated``
     - Kilometres; the target is the Moon.
   * - ``satellite_viewer`` / viewer ``build``
     - Kilometres in browser scene axes: (TEME X, TEME Z, -TEME Y).
   * - ``moon_webgl``
     - Kilometres in its lunar-fixed scene axes.
   * - Fancy cislunar video demo
     - Metres.
   * - Star-finder demo's orbit panel
     - Kilometres. Star-centroid, residual and error plots keep their own fixed axes.

The rotated-vector GIF demo also accepts the shared modes; its camera lengths
are dimensionless unit-vector coordinates.

Flat ground maps, divergence/error charts and sensor-image panels keep fixed
2-D framing; a 3-D chase camera has no meaning for those axes. A dashboard's
camera affects its 3-D panels. A purely 2-D orbit or ground-track animation
rejects a non-overview camera. Image-to-video/GIF encoders combine already
rendered frames and therefore have no scene camera of their own.

Framing and interpretation
--------------------------

Overview/orbit fit the scene's configured bounds over the full interval;
distant context bodies excluded by the plot's existing bounds remain outside
that view. Follow/chase intentionally show a local region, so other bodies
can leave the frame. Matplotlib uses equal-scale orthographic projection;
Plotly, WebGL and the spacecraft renderer use perspective. The same shot
therefore has the same tracking semantics, not identical pixels across engines.

``SpacecraftView`` now defaults to overview, where a metre-scale spacecraft is
usually unresolved against an orbit measured in kilometres. A labelled cross
marks its actual centre without enlarging the mesh. Use ``follow``, ``chase``
or ``limb`` for a mesh close-up. Existing explicit ``camera_offset`` arguments
retain their follow behaviour. Resampling preserves the original overview
bounds and orbit-sweep phase. The existing 24-hour GEO gallery example selects
``limb`` explicitly, retaining its close-up and 45-second duration.

The :mod:`demos.orbit_visualization.demo_camera_modes` example compares all four
shared shots over one SSAPy circular orbit in a 15-second, four-panel movie.
Camera regressions use literal translations, a quarter-turn Rodrigues
rotation, constant-velocity tangents and independently projected target pixels.
Browser tests validate actual rendered target centring, not just JSON fields.
The solar-system sky refreshes on scripted camera frames; the Moon viewer
centres its star layer on the camera so translation creates no false stellar
parallax. Catalogue directions remain evaluated at the viewer's reference epoch.
Plotly's domain-coordinate convention is described in its
`camera documentation <https://plotly.com/python/3d-camera-controls/>`_.
