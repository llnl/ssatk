Star finder and optical navigation
===================================

The gallery demo **Star Finder — Angles-Only Attitude and Position** compares
simulated camera measurements with the installed bright-star catalogue for
one 700-km circular orbit (approximately 98.8 minutes). Run it with:

.. code-block:: bash

   python demos/navigation/demo_star_finder.py

The interactive HTML and PNG show a camera field, measured-minus-predicted
star centroids, the satellite orbit, and attitude/position errors over time.
Hover over measured stars for predicted apparent and measured right ascension
and declination. The accompanying CSV files retain each measured centroid,
its catalogue row/name, angular residual, fit/withheld status, and epoch.
The JSON summary records the exact catalogue SHA-256, seed and assumptions.
All measurements are **simulated**, not observed spacecraft data.

What the angles determine
--------------------------

Distant-star directions determine attitude. They do not determine translation
in the direction-at-infinity model. This agrees with NASA's description of
`star-tracker attitude determination
<https://www.nasa.gov/smallsat-institute/sst-soa/guidance-navigation-and-control/>`_.
The position solution instead uses bearings from a second, rigidly calibrated
camera to identified finite-distance Earth landmarks. These are ideal mapped
WGS84 grid points, not a real image-derived landmark data set. No ranges enter
the position solver.

The simulation uses ``ssatk.propagators_orbit.propagate_orbit_state`` with
DOP853 and checks the propagated path against an independent circular-orbit
solution. The star catalogue is J2000 ICRS RA/Dec with proper motion. SSATK propagates their Cartesian tangent vectors (zero radial motion)
and Astropy transforms them to apparent GCRS directions
with observer position/velocity, aberration and solar light deflection.
The coarse orbit supplied to that prediction differs from truth by 100 km
in position and 50 m/s in velocity; recovered position is not fed back into
these corrections in this demonstration.

Detection order is shuffled before matching. A coarse attitude prior and a
one-to-one angular assignment identify the stars. An equal-weight
Wahba/Kabsch rotation determines attitude from three quarters of the matches;
the remaining quarter tests the fitted solution. This is tracking-mode
acquisition, not a blind lost-in-space pattern recognizer. Each position fix
comes from a least-squares intersection of the landmark bearing lines. The demo
then passes those fixes into SSATK's Cartesian orbit extended Kalman filter
(EKF), which combines repeated observations and two-body dynamics to estimate
position and velocity. Raw fixes remain visible alongside the filtered orbit.
The CSV includes filtered position, velocity and their errors against simulation
truth; the JSON records the final errors.

The default assumes 3 arcsec of tangent-plane noise, a 2048-pixel square
sensor with a 30-degree field, and 181 epochs in fast gallery mode or 361 in
full mode. Playback spans approximately 60 seconds. Runtime parameters are
available through ``simulate_star_finder(start=..., duration_s=...,
samples=..., noise_arcsec=..., seed=...)``. These are model assumptions, not
performance claims about a particular flight camera.

Reusable functions
-------------------

``ssatk.navigation.star_tracker`` provides ``load_star_catalogue``,
``catalogue_directions``, ``match_star_directions`` and ``fit_star_attitude``.
Finite-distance geometry lives in ``ssatk.navigation.landmarks`` through
``position_from_landmark_bearings``; its previous ``star_tracker`` import is
retained for compatibility. The combined solve and orbit-filter adapter live
in ``ssatk.navigation.optical_navigation``. All are exported from
``ssatk.navigation`` and work without importing a demo or plotting module.

Camera axes have +z along the boresight;
attitude matrices rotate camera vectors into GCRF-aligned axes. Position
inputs are metres, velocity inputs m/s, and matching gates radians. Epochs
accept scalar Astropy ``Time`` or GPS seconds. Catalogue files use RA hours,
Dec degrees and proper motions in mas/year, with RA proper motion already
multiplied by cos(Dec). Explicit missing or invalid catalogues raise errors.

Optical fixes for station keeping
--------------------------------------

``solve_optical_navigation`` performs star matching, attitude fitting and
landmark triangulation. Its inputs are calibrated star-camera bearings,
predicted star directions, identified landmark positions in GCRF metres,
landmark-camera bearings, the observation epoch and a coarse attitude prior.
An optional rigid calibration rotates landmark-camera coordinates into
star-camera coordinates. Landmarks in Earth-fixed coordinates must first be
transformed at the observation epoch. ``fit_star_mask`` refers to input
detections, allowing withheld-star validation without a demo-specific solver.

The returned ``OpticalNavigationFix`` contains the epoch, camera-to-GCRF
rotation, GCRF position, matched indices, fit mask and angular residuals in
radians. Degenerate stars/landmarks and invalid calibrations fail explicitly.
One image provides no measured velocity. ``update_orbit_from_optical_fix``
connects fixes to ``CartesianOrbitEKF``, preserving the existing propagator,
force-model and process-noise interfaces. It returns ``EKFState`` with
``x = [x, y, z, vx, vy, vz]`` in metres and metres per second, a 6x6 covariance,
and a GPS epoch. Invalid/stale fixes or failed updates leave the filter intact.

For a prepared observation and a filter initialized from a coarse orbit:

.. code-block:: python

   import ssapy
   from ssatk.navigation import (
       solve_optical_navigation, update_orbit_from_optical_fix,
   )

   fix = solve_optical_navigation(
       measured_star_bearings, predicted_star_directions,
       landmark_positions_gcrf_m, measured_landmark_bearings,
       time=observation_time,
       prior_camera_to_gcrf=coarse_attitude,
       match_gate_rad=match_gate_rad,
       landmark_camera_to_star_camera=camera_calibration,
   )
   state = update_orbit_from_optical_fix(
       orbit_filter, fix, position_covariance_m2,
       propagation_kwargs={"acceleration": force_models},
   )
   estimated_orbit = ssapy.Orbit(
       r=state.x[:3], v=state.x[3:], t=state.time,
   )

``estimated_orbit`` can be supplied as ``initial`` to
``ssatk.orbital_mechanics.transfer_optimal`` or its state can initialize
``ssatk.propagators_6dof.Spacecraft``. Station-keeping code remains responsible
for selecting a reference orbit, checking estimate quality and planning burns.
After a burn, update the filter's state and covariance to reflect the maneuver
before processing further observations. This adapter supplies navigation; it
does not implement a station-keeping control law.

The filter requires an explicit positive-definite 3x3 position covariance in
GCRF metres squared. It must include attitude, calibration and landmark errors;
small residuals alone do not establish that uncertainty. Correlations shared
across observations require a suitable augmented error model. The demo uses
an explicitly prescribed isotropic 100-m standard deviation, configurable as
``filter_position_sigma_m``, and a coarse 100-km/50-m/s initial uncertainty.
These are simulation assumptions, not calibrated flight error statistics.
The demo retains its independent position fixes and does not feed recovered
states back into apparent-star corrections.

Limits and validation
----------------------

The installed catalogue extract lacks distances, radial velocities and
uncertainties. Stellar parallax, perspective acceleration and catalogue
errors are therefore not modeled. The ordinary gallery backgrounds are
less detailed: proper-motion-adjusted distant-star directions without
aberration or light deflection. Their decorative marker sizes are not angular
diameters. They must not be treated as calibrated images of the apparent sky.
The Moon WebGL sky now applies proper motion at its requested epoch before
rotating to the lunar body frame.

The demo generates calibrated centroids/bearings rather than extracting them
from a pixel image. It omits optical distortion, calibration drift, clouds,
terrain, illumination and landmark identification errors. Finite-landmark
bearings use instantaneous geometric rays without light-time or aberration
corrections; this bounds the interpretation of metre-level simulation errors. Synthetic truth
provides a controlled validation of geometry and inference; measured
navigation accuracy would require real images and independent truth data.

Tests use literal quarter-turn rotations, exact intersecting landmark rays,
analytic tangent-plane proper motion (including a pole), parallel-ray
observability failures, and an independently computed circular orbit. The
sky-coordinate transform delegates apparent astrometry to
`Astropy GCRS <https://docs.astropy.org/en/stable/api/astropy.coordinates.GCRS.html>`_.
Checks of the packaged catalogue's finiteness or agreement with another
propagation calculation do not independently certify the original measurements.
