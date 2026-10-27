Gamma and heading: conventions and validation
=============================================

The demo shows one **4.277617-hour** orbit with semimajor axis
**13378.137 km**, eccentricity **0.2**, and inclination **45°**, starting at
2025-01-01T00:00:00 UTC. SSAPy's ``KeplerianPropagator`` supplies both position
and velocity at **721 epochs** in gallery mode, spaced **21.388087 seconds**
apart; the standalone full mode uses 1441 epochs. This is an analytic two-body
example, not a perturbed orbit or a comparison with flight tracking.

Definitions
-----------

``calc_gamma_and_heading(r, t, v=v)`` accepts Geocentric Celestial Reference
Frame (GCRF) positions in metres and velocities in metres per second. It
transforms both into the International Terrestrial Reference Frame (ITRF),
including the velocity induced by the time-dependent Earth rotation.
``calc_gamma_and_heading_itrf(r, t, v=v)`` accepts states already in ITRF.
Omitting velocity retains position-only input, with an approximate numerical
derivative that cannot reconstruct motion unresolved by the samples.

Gamma is the angle above the **geocentric** local horizon: positive radially
outward, negative inward. This horizon is perpendicular to the geocentric
radius; it is not the World Geodetic System 1984 (WGS84) ellipsoid normal.
Heading is clockwise from local north, with north/east/south/west equal to
0°/90°/180°/270°. Both use **Earth-relative velocity**, so they need not equal
angles calculated from the inertial velocity. The dashboard labels its speed
separately as **GCRF speed**, and its altitude is **WGS84 geodetic height**.

Pure radial motion has gamma +90° or −90° and **undefined heading**. Stationary
motion leaves both angles undefined. Heading is also undefined at geographic
poles, where longitude/north azimuth is singular. These cases return NaN;
directionless components at floating-point roundoff scale are not interpreted
as meaningful compass bearings. The six-direction figure uses dot/cross
symbols for outward/inward motion, excludes the poles, and evaluates the
production ITRF angle functions. Its arrows denote direction, not speed.

Verified defects and corrections
--------------------------------

* The dashboard took position differences without dividing by elapsed time,
  then labeled them km/s. At the former 300-second cadence, an equivalent
  two-body reproduction displayed **1345–1991 km/s** instead of **4.491–6.669
  km/s** at the interior plotted epochs. The dashboard now uses supplied
  velocities, or time-aware derivatives when velocities are absent.
* The shared position derivative used a centered secant even for unequal
  spacing. An analytic quadratic sampled at 0, 1, 4 and 10 seconds produced
  component errors up to **18 m/s**. Three-point derivatives now account for
  unequal spacing and use second-order endpoint formulas; two-point input
  retains its secant. Repeated or reversed epochs raise an error.
* Coarse forward differences at the first epoch gave **−3.723864°** gamma at
  perigee in the reproduction, where radial velocity is zero. The demo now
  supplies the propagator's velocity states and covers the complete period.
* Radial velocity returned a numerical compass heading, including 0° for
  axis-aligned radial motion. The direction panel's subtraction roundoff also
  produced spurious Up/Down arrows. Singular headings now remain undefined.
* The previous printed circular-orbit check accepted **1.964° RMS** under a
  2.5° threshold using a separate inertial chord calculation. It did not test
  the plotted production Earth-relative angles. The replacement evaluates
  all demo states against an independent frame/angle path.
* The dashboard silently replaced angle-calculation failures with NaN curves.
  It now propagates those errors. Compass wraps break plotted lines rather
  than drawing an artificial sweep across nearly 360°.

References and tolerances
-------------------------

The independent path transforms GCRS states with an Astropy
``CartesianDifferential`` into ITRS and derives local velocities from
spherical-coordinate rates. The production code under test now shares
Astropy's IAU 2006/2000A convention (``ssatk.coordinates._earth_rotation``,
fed by SSATK's packaged EOP data rather than Astropy's own IERS table); the
**0.0005°** angle tolerance was sized to also cover the dashboard's earlier
IAU 1976/80 rotation and is now larger than the measured difference needs
(see ``tests/test_gamma_and_heading.py``'s matching R2 check, now agreeing to
about 2e-7 deg). The following measured maxima predate that change and have
not been rerun against it:
Across all 721 gallery epochs, the measured
maximum differences are **0.000000941°** in gamma and **0.000020662°** in
heading. These are implementation comparisons, not tracking uncertainties.

The analytic vis-viva equation checks the GCRF speed with a **10⁻⁸ m/s**
absolute tolerance. The demo ranges from **4.456824 to 6.685235 km/s**; its
measured maximum vis-viva difference is **2.73 × 10⁻¹² m/s**.

Core regressions in ``tests/test_gamma_and_heading.py`` use exact local
directions, radial/stationary/polar singularities, a differentiated analytic
quadratic, and Astropy spherical velocity differentials. Workflow tests in
``scripts/tests/test_gamma_heading_dashboard.py`` inspect plotted numerical
values against constant straight-track speed and analytic circular speed,
and verify radial symbols/cardinal arrows. They do not certify physical
correctness from image snapshots.

For the velocity transformation interface, see the
`Astropy velocity documentation <https://docs.astropy.org/en/stable/coordinates/velocities.html>`_.

Reproduction
------------

.. code-block:: bash

   python demos/orbital_mechanics/demo_gamma_and_heading.py
   python -m pytest tests/test_gamma_and_heading.py \
     tests/test_frame_and_state_references.py \
     scripts/tests/test_gamma_heading_dashboard.py -q

The demo exports its two figures, ``gamma_heading_series.csv`` with calculated
and reference angles, and ``gamma_heading_validation.json`` with parameters
and measured residuals. ``make_figures=False`` runs the numerical reference
comparison without creating figures or files.
