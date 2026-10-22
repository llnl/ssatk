Solar and lunar eclipses
========================

``ssatk.eclipse`` builds solar and lunar eclipse events with a finite,
limb-darkened Sun, validates them, and renders four products. Events come
either from JPL DE430 states or from NASA/GSFC published elements:

.. code-block:: python

   from ssatk.eclipse import EventRequest, RenderRequest, build_event, render_product

   event = build_event(EventRequest(kind="solar", backend="de430"))
   render_product(RenderRequest(product="scientific-suite", output="eclipse_2024", event=event))

State backends
--------------

``de430`` (default)
   Sun and Moon from the JPL DE430 kernel in ``ssa-data-lunar``. The positions
   are light-time corrected, because an eclipse happens where the light is.
   Without the correction, the Sun's 499 s light time puts predicted contacts
   about 40 s late.

``reference``
   Replays the two bundled events: the 2024-04-08 total solar eclipse from the
   NASA/GSFC Besselian elements, and the 2025-03-14 total lunar eclipse from
   NASA/GSFC published circumstances. Both are converted to GCRF.

Both backends use the SSATK GCRF/ITRF transform. The DE430 backend orients the
Moon with the DE440 lunar PCK and the reference backend with the IAU 2009 model.
``discover_eclipses`` finds every solar and lunar eclipse in a UTC interval
from DE430 states, and ``build_discovered_event`` turns a result into an event
that any product can render.

Products
--------

``scientific-suite``
   Summary figure, timeline CSV, event state, validation report, and an
   interactive WebGL view.

``observer-suite``
   Contact times, local circumstances, sky view, and visibility maps for one
   WGS-84 observer of a solar eclipse.

``cinematic``
   Self-contained WebGL scene with physical Earth and Moon bodies,
   photospheric light paths, and contact-to-contact playback.

``plotly``
   Plotly scene of the exact shadow geometry.

The same operations are available from the ``ssatk-eclipse`` command
(``search``, ``materialize``, ``build``, ``validate``, ``render``). Derived map
grids are cached under ``$SSATK_CACHE/eclipse``, or ``~/.ssatk/eclipse`` when
``$SSATK_CACHE`` is unset.

Accuracy
--------

``tests/test_eclipse_integration.py`` compares these results with independent
references:

* DE430-based discovery reproduces the NASA/GSFC contacts of the 2025-03-14
  lunar eclipse within 15 s, and its umbral magnitude within 0.003. Lunar
  shadows use Danjon's rule: Earth's mean radius enlarged by 1/85.
* The 2024-04-08 global solar contacts agree within 25 s. The global contact
  equation treats Earth as a sphere of equatorial radius, so contacts near the
  polar limb are predicted late.
* The eclipse Sun matches Astropy's apparent Sun within 0.5 arcsec, and the
  reference backends match Astropy GCRS within 5 arcsec.
* The 2024-04-08 greatest-eclipse ground point is within 0.001 deg of NASA's
  (reference backend) and 0.03 deg (DE430 backend).
* Limb-darkened visible flux matches an independent adaptive radial
  integral within 2e-7.

Observer contacts are refined against the WGS-84 surface. Terrain affects the
horizon only when the observer supplies a horizon profile.
