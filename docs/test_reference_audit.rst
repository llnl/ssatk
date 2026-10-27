Test reference audit
====================

The initial audit covered all 103 ``tests/test_*.py`` files and their 389 test
functions, including methods in test classes. Parametrization produces more
executed cases than this function count. ``tests/conftest.py`` supplies the
shared GPS epochs; it contains no test functions. The reviewed starting commit
is ``f05e171fe4ccd2d0c693ffef9e9d6d23d85b70f5``. Later additions and their
references are recorded in the inventory and follow-up sections below.

The suite contains independent numerical references, conservation and
derivative checks, and software contracts. Passing all three categories does
not establish measured accuracy for every physical model. In particular,
agreement with a directly called SSAPy force validates an adapter, while
agreement with SSAPy's Kepler solver validates a separate integration method
for the same point-mass equations. Neither comparison validates atmospheric
density or spacecraft properties against measurements.

Changes from the audit
----------------------

* Attitude-error state transition matrices and covariance now use an analytic
  isotropic torque-free body solution, including the rotating body-error axes.
  The inertia-rate case prescribes consistent ``I(t)`` and ``dI/dt`` and checks
  conservation of angular momentum.
* Lunar position and velocity transformations now have a prescribed circular
  ephemeris with exact rotation and rotation-rate matrices. This tests frame
  kinematics independently of the ephemeris implementation. Real lunar
  ephemeris comparisons with Astropy remain separate.
* Inclination burns and the station-keeping example use separate Radau
  integrations. Tightening reference tolerances by a factor of ten must leave
  reference states or semimajor-axis changes within the stated error budget.
  Collision probability uses a noncentral chi-square distribution and a
  converged polar cubature independent of the production integral.
* Maneuver tests derive vis-viva burn speeds directly instead of calling the
  production maneuver helper. The unexplained stored Lambert delta-v is
  replaced by independently propagated position and velocity boundary checks.
* The extended Kalman filter checks the conditional-Gaussian mean and complete
  covariance. Flexible modes use exact harmonic-oscillator solutions; mass
  depletion, propulsion, reaction wheels, and radiation pressure use analytic
  balances or force equations.
* Atmosphere density comparisons disable the default absolute floor that
  previously allowed zero density. MSIS wrapper tests compare versions 0 and
  2.1 with direct pymsis calls, raw packaged weather records, and independent
  WGS84 geodetic conversion. Checking the MSIS input specification exposed
  another production defect: the toolkit supplied flux adjusted to 1 au,
  whereas MSIS requires observed flux at the actual Sun-Earth distance.
  ``SpaceWeatherTable.msis_inputs`` now selects the observed daily and
  observed centered-average fields. Winter and summer input tests also run
  without the optional pymsis package. The source convention is documented
  in `pymsis calculate Notes
  <https://swxtrec.github.io/pymsis/reference/generated/pymsis.msis.calculate.html>`_.
  These checks do not claim validation against measured atmospheric density.
* Earth orientation and CDM frame conversion gain independent ERFA matrix
  comparisons. Nadir and ecliptic tests check actual axis directions as well
  as orthogonality or inverse round trips.
* Spacecraft-view geometry is checked against exact rotations, translations,
  pinhole projection, spherical limbs and occultations, and prescribed linear
  and angular rates. Export checks are software contracts. The browser star
  catalog is compared with Astropy's GCRS-to-TEME chain at two epochs to reject
  a duplicate frame rotation. See :doc:`spacecraft_view` for rendering limits.

* Gallery publishing now routes all standard outputs into the selected site
  directory and validates local links. ZIP relocation tests compare HTTP-served
  bytes with literal assets. These are software contracts; they make no claim
  about physical models or the fidelity of rendered images.

Tolerance interpretation
------------------------

NumPy compares against ``atol + rtol*abs(reference)``. Tests stating only an
absolute tolerance now set ``rtol=0`` explicitly; for example, a millimetre
position budget must not quietly acquire a metre-scale allowance at orbital
distances. See `NumPy assert_allclose
<https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose.html>`_.

Purely relative ``pytest.approx`` checks now set ``abs=0``. Explicit
``pytest.approx(abs=...)`` already disables its default relative tolerance;
those assertions did not have the NumPy issue. See `pytest approx
<https://docs.pytest.org/en/stable/reference/reference.html#pytest-approx>`_.

Three approximation budgets need special treatment. The SSAPy-style
linearized polar-motion matrix has a squared-angle orthogonality error bounded
by ``xp**2 + yp**2``. Two-part Julian-date subtraction is allowed 10 ps in the
OEM time round trip. The bounded targeting optimizer has a stated
``1e-7 m/s`` terminal-velocity budget, while its constant-acceleration dynamics
are separately checked to ``1e-12 m/s``. These are model, representation, and
optimization budgets, respectively.

The photometry oracle uses the same *geometric* Sun position as a specified
input and computes phase and Planck-band flux independently. Astropy's
*apparent* Sun is checked with an angular allowance for aberration; treating
the two positions as identical had made the previous flux reference invalid.
The ERFA Earth comparison (``tests/test_environment_eop.py``) recomputes the
same IAU 2006/2000A model independently via ``erfa.c2t06a``; its 0.5 arcsec
budget predates SSATK's single-route refactor, when production used IAU
1976/1980, and is now far looser than the measured 0.0003 m difference. The
latter matrix is documented in
`ERFA c2t06a <https://github.com/liberfa/erfa/blob/master/src/c2t06a.c>`_.

Limits that remain explicit
---------------------------

* **Empirical asteroid models:** the LSST color table is verified against
  Chesley and Veres (2017), JPL Publication 16-11, Table 8, p. 18
  (`NASA report <https://ntrs.nasa.gov/api/citations/20170008014/downloads/20170008014.pdf>`_).
  That source does not validate reusing LSST colors for ZTF. A source or
  calibration data set for that approximation and for the two Granvik linear
  fits was not established. Existing checks of those coefficients are retained
  as model-contract regressions, explicitly marked **unverified as empirical
  calibrations**. No replacement coefficients are invented.
* **Observed orbit determination:** current batch and filter references use
  synthetic truth and known measurement models. No independent real tracking
  data set with a reference orbit was supplied. The tests validate equations,
  geometry, and inference for those constructed cases.
* **Optional tools:** Basilisk, Orekit/EIGEN-6S, and SpacePy/IRBEM checks count
  only when actually executed with their required tools/data. Report skips
  separately.
* **Conservation and broad bounds:** epoch invariance, round trips, angular
  momentum, libration bounds, and geographic magnetic-field bounds constrain
  specific properties. They are not independent measurements of every output.
  The inventory names paired numerical references where available.
* **Synthetic fixtures:** CCSDS messages, model spacecraft, prescribed
  ephemerides, and noise realizations are controlled test inputs. An
  Orekit-shaped text fixture is not an Orekit-computed trajectory. A passing
  simplified CubeSat inertia test does not certify a flight vehicle.

Complete file inventory
-----------------------

R1 denotes an analytic equation; R2 a separate implementation or backend;
R3 a published value; R4 an invariant; R5 a derivative comparison; R6 a
numerical regression. C denotes a software/model contract and U an unresolved
empirical reference. Multiple labels mean a file contains different kinds of
checks, not that every assertion has every kind of independent reference.

.. csv-table:: References and scope for every test file
   :header-rows: 1
   :widths: 24, 5, 10, 32, 29

   file,test_functions,reference_types,reference,scope_or_limit
   tests/test_6dof_elapsed_time_integration.py,4,R1 R4 R6,Constant mass loss and free flight; paired SSAPy/REBOUND trajectories,Epoch invariance and callback checks are not independent trajectory truth
   tests/test_6dof_extended.py,4,R1 R5,Harmonic oscillator sine/cosine solutions; integrated mode torques; finite differences,Synthetic reduced modes; not measured structural dynamics
   tests/test_6dof_variational.py,11,R1 R5,Finite-difference state/force/torque Jacobians; covariance matrix algebra,Derivative consistency; absolute attitude covariance anchored in review regressions
   tests/test_acceleration_helpers.py,1,R1,Newtonian exterior and uniform-sphere interior gravity,Specified spherical Earth model
   tests/test_adaptive_propagators_elapsed_time.py,3,R4 R6,Autonomous ODE epoch invariance; absolute callback epochs,Pair with propagators_6dof_truth for absolute trajectories
   tests/test_asteroid_magnitudes.py,1,R1 R3 C U,H-albedo diameter equation; Chesley and Veres 2017 JPL 16-11 Table 8,ZTF reuse and Granvik fit provenance unverified; contract checks only for those coefficients
   tests/test_basilisk_6dof_benchmark.py,1,R2,Basilisk independent rigid-body trajectory benchmark,Optional dependency; skip is not validation
   tests/test_body_ephemerides.py,2,R2,Astropy geocentric planetary and solar ephemerides,Model comparison within stated distance tolerance; no tracking data
   tests/test_burn_conversions.py,2,R1,Vis-viva semimajor axis and acceleration-duration impulse,Specified two-body burn case
   tests/test_catalog_conjunction_screen.py,1,R2 R1,Dense independently sampled SSAPy trajectories; known node offset,Dense sampling limits TCA resolution
   tests/test_ccsds_cdm.py,4,R1 R2 C,Literal SI fixture; RTN dyadic covariance; ERFA Earth/frame-bias matrices,Synthetic CDM; not a real conjunction data product
   tests/test_ccsds_omm.py,3,R2 C,Literal angular/rate SI conversions and native SGP4 Satrec,Synthetic CelesTrak-shaped record; shared TEME rotation checked separately
   tests/test_compute_helpers.py,9,R1 R2 R4,Fourier tone; analytic MEGNO/cat map; REBOUND; known probability distributions,Seeded statistical tests allow specified significance level
   tests/test_conic_sampling.py,4,R1 R2,SSAPy Kepler propagation and Barker parabolic equation,Specified Kepler model
   tests/test_constants.py,4,R3 R2,IAU unit definitions; CODATA 2018; JPL mean elements; REBOUND units,Rounded orbital constants have explicit relative tolerances
   tests/test_coverage_site_geometry.py,1,R1,WGS84 ellipsoid equation; east/north/up cardinal geometry,Geometric visibility; no refraction model
   tests/test_earth_zonal_harmonics.py,3,R1 R2 R5,Potential-gradient finite differences; SSAPy EGM96 coefficients; J2 limits,Finite difference budget and model constants explicit
   tests/test_eclipse_geometry.py,2,R3 R4,NASA 2024-04-08 Texas totality path; coordinate round trip,150 km and 90 s event bounds; not precision eclipse timing
   tests/test_eclipse_integration.py,9,R1 R2 R3,Closed-form disc overlap; lens area; shadow cones; centred limb-darkened occultation; SciPy radial integral; Astropy apparent and GCRS Sun; NASA/GSFC 2024-04-08 greatest point and contacts; 2025-03-14 contacts and magnitude,Discovery uses a spherical-Earth contact equation and Danjon shadow; 15-25 s contact bounds
   tests/test_eclipse_partial_shadow.py,1,R1,Constructed finite-Sun occultation geometry,Boolean geometry case only
   tests/test_ekf.py,4,R1 C,Conditional-Gaussian mean/covariance; circular orbit; known bias,Synthetic observations; not validation against orbit tracking data
   tests/test_ellipse_fit_helpers.py,5,R1 R2,Focal definition; vis-viva; SSAPy propagation,Conic geometry and boundary conditions
   tests/test_ellipse_inject.py,2,R2,SSAPy independent transfer propagation and arrival velocity,Point-mass transfer only
   tests/test_encounter_frame_roundoff.py,3,R1 R4,Orthogonal projection and collinear fallback axes,Conditioned geometric cases
   tests/test_engine_budgets.py,2,R1,Tsiolkovsky equation; force/impulse/time identities,Ideal rocket assumptions
   tests/test_environment.py,4,R1 R4,Exponential density; dipole limits; full/zero occultation; SRP masking,Partial-shadow check is a bound; detailed force truth in spacecraft_force_models
   tests/test_environment_eop.py,1,R2 R4,ERFA c2t06a IAU 2006/2000A independent recomputation (production now shares that model; the 0.5 arcsec budget predates that and the measured difference is 0.0003 m),0.5 arcsec model difference budget; round trip alone is not truth
   tests/test_environment_space_weather.py,2,R2 R3 C,MSIS observed-flux input specification; direct pymsis 0 and 2.1; raw weather CSV; Astropy WGS84,Wrapper parity only; shared Earth rotation checked separately; not measured density
   tests/test_epoch_comparison_tolerance.py,2,R6 C,IEEE floating-point ULP and absolute GPS epoch differences,Numerical time contract
   tests/test_faceted_magnitude.py,6,R1 R4,Lambert surface cosine/area factors; magnitude flux ratio; occultation,Ideal diffuse surfaces
   tests/test_frame_and_state_references.py,11,R1 R2 R4 C,J2000 obliquity; specified axes; Astropy Sun/velocity; Kepler quantities,Mixed geometry and viewer contracts; polar-motion linearization bounded explicitly
   tests/test_gamma_and_heading.py,8,R1/R2/C,Exact local directions singularities quadratic time derivatives and Astropy state differentials,1e-10 deg directions; 5e-9 m/s quadratic; 5e-4 deg frame comparison
   tests/test_horizon_frames.py,3,R1 R2,Cardinal hour-angle geometry; Astropy AltAz,0.5 degree approximate horizon-model budget
   tests/test_iers_global_state.py,1,R2 C,Astropy IERS UT1 before/after plotting,Global-state preservation; not independent IERS calibration
   tests/test_igrf_wrapper.py,1,R2,ppigrf geocentric magnetic synthesis,Same IGRF model; wrapper/frame validation
   tests/test_inertial_and_lunar_frames.py,2,R2 R3,IERS J2000 frame bias and Astropy Moon-Earth ephemeris derivatives,0.1 m/s and 10 km lunar model comparison
   tests/test_integration_callback_regressions.py,1,R1,Exact exponential linear-drag solution,Synthetic drag coefficient
   tests/test_integration_inertia_regressions.py,5,R1 R4,Variable-inertia momentum; mass depletion; analytic rate STM,Specified inertia and flux models
   tests/test_integration_leapfrog_impact_regressions.py,3,R1 R6,Line/sphere intersections; constant-acceleration impact; epoch identity,Analytic synthetic collision geometry
   tests/test_integration_photometry_regressions.py,2,R1 R4,Blocked line of sight gives zero received flux and infinite magnitude,Visibility cases; flux model anchored separately
   tests/test_integration_profile_regressions.py,1,R1 C,Known piecewise profile values on integer time grids,Profile contract
   tests/test_integration_rk4_regressions.py,2,R1 R6,Constant-acceleration position/velocity; linear interpolation,Synthetic force and ephemeris models
   tests/test_integration_segment_regressions.py,1,R1 C,Known piecewise dense interpolants in arbitrary query order,Composition contract; not a trajectory accuracy test
   tests/test_integration_state_continuity_regressions.py,2,R1 R4,Known segmented mass loss; selected-tank state and inertia continuity,Some inertia checks share body helpers; body formula tested separately
   tests/test_integration_terminal_epoch_regressions.py,2,R1 R6,Known event roots and GPS floating-point ULP,Stopping-time contract
   tests/test_integration_terminal_regressions.py,2,R1 R6,Free-flight terminal state; occurrence-count event roots,Synthetic event functions
   tests/test_integration_wheel_regressions.py,4,R1 R4 R5,Integrated torque; capacity; angular momentum; attitude and event-time STM,Specified ideal actuator; multiple integrators
   tests/test_irene.py,2,R1 C,Closed-form rad-to-mGy and km-to-m conversions; exact dose sum and unchanged integral flux,Synthetic fixtures; output-format and unit contracts only; no IRENE prediction check
   tests/test_j2_reference.py,1,R1 R4,First-order secular J2 RAAN rate; step convergence,Averaged first-order formula permits short-period and higher-order differences
   tests/test_keplerian_helpers.py,5,R1 R2,Kepler equation; vis-viva; orbital invariants; SSAPy states,Specified two-body model
   tests/test_lagrange_points.py,2,R1 R3,CR3BP equilibrium residual and collinear approximate fractions,Equilibrium residual is authoritative; rounded fractions are secondary
   tests/test_lagrange_rotating_frames.py,1,R1 R4,Equilateral L4/L5 geometry and collinear equilibrium fractions,Earth-Moon rotating convention; broad geometry tolerance
   tests/test_lambert_reference.py,2,R1 R2,Circular arc velocities and SSAPy Kepler boundary propagation,Zero-revolution cases
   tests/test_lambertian_photometry.py,3,R1 R2 R3,Independent Planck quadrature; Lambert phase geometry; solar magnitude,Shared geometric Sun input; apparent-Sun model difference checked separately
   tests/test_launch_gravity_turn.py,1,R1,Known vertical/azimuth/half-turn thrust directions plus gravity,Prescribed steering law
   tests/test_low_precision_moon.py,1,R2,Astropy lunar coordinates over 2000-2040,200 samples; 6/3.5 arcmin and 550 km budgets
   tests/test_lunar_rotating_frame.py,1,R1 R4,Earth at negative lunar X and constructed frame vectors,Paired exact circular-ephemeris kinematics in review regressions
   tests/test_magfield_physics.py,22,R1 R2 R3 R4,Geopack; ppigrf; dipole field theory; IRBEM when present; Astropy star frames,Contains approximate geographic/model sanity bounds; AE8/AP8 cache and IRBEM optional
   tests/test_moon_texture_rotation.py,1,R4,Near-side sub-Earth meridian within optical libration bound,Coarse 9 degree orientation check; not precision PA/ME alignment
   tests/test_moon_webgl_json.py,4,R1 R4 C,Prescribed terrain slopes; lunar radius/unit conversions; near-side geometry,"Libration bounds constrain gross orientation, not precision lunar cartography"
   tests/test_navigation_batch.py,2,R1 R5 C,Known linear free-flight state and Fisher covariance; finite differences,Synthetic measurement functions; not realistic observed orbit determination
   tests/test_navigation_measurements.py,4,R1 R4 R5,Known range/range-rate/angles; finite differences; station position and ENU basis share one Earth-rotation route,Geometry/derivatives; no atmospheric or instrument calibration
   tests/test_navigation_sensors.py,2,R1 R4,Known Gaussian/chi-square/binomial distributions and synthetic elevation mask,Seeded statistical tests; not measured sensor noise
   tests/test_optical_navigation.py,5,R1 C,Literal camera rotations and intersecting rays; batch Gaussian information solution; prescribed impulse,14 cases including invalid geometry/covariance/epochs; synthetic measurements not telemetry
   tests/test_numeric_mechanics_helpers.py,2,R1,Newtonian orbital equations; Kepler residuals; cardinal coordinates,Specified ideal geometry
   tests/test_orbit_absolute_epoch.py,2,C R6,Force callback epochs anchored to known GPS values,Time-interface contract; trajectory truth tested separately
   tests/test_orbit_acceleration_helpers.py,4,R1 R2,SSAPy third-body acceleration; uniform sphere; vector steering geometry,Shared body ephemerides; force-equation validation
   tests/test_orbit_initializer.py,1,R1 R4,Leading equilateral Earth-Moon L4 triangle,Geometric instantaneous initializer; no stability proof
   tests/test_orbital_maneuver_modes.py,5,R1 R2 C,Independent vis-viva burns; SSAPy Lambert boundary propagation; impulse budgets,Schema checks supplemental; continuous inclination independently integrated in review tests
   tests/test_orbital_maneuver_reference_cases.py,5,R1 R3,Curtis/Bate transfer equations; plane-change impulse; rocket equation; Gauss node rate,Textbook numeric anchors recomputed with configured Earth constants
   tests/test_orekit_j2_external.py,1,R2 R3,Orekit 10.3.1 Holmes-Featherstone EIGEN-6S J2 trajectory,Optional Java/data; unavailable reference is explicitly skipped
   tests/test_packaged_eop_reference.py,2,R2 R3,Packaged finals2000A versus bundled IERS-B/C04 UT1,0.5 ms separate-series budget; historical leap seconds
   tests/test_plot_earth_rotation.py,4,R2,Astropy GCRS/ITRS/WGS84; one test configures Astropy with the packaged finals2000A.all,0.5 arcsec generic-model budget for three tests; 2 cm packaged-EOP budget for the renamed groundtrack_enhanced test; no rendering correctness claim
   tests/test_plot_sun_and_stars.py,3,R2,Astropy apparent Sun and proper-motion/TEME transformations,Geometric/apparent Sun difference budget 60 arcsec
   tests/test_propagators.py,5,R1 R5,Circular orbit throughout arc; gravity Jacobian; rocket equation; wheel momentum,Specified ideal forces and simple body
   tests/test_propagators_6dof.py,37,R1 R2 R4,Rigid-body/point-mass dynamics; photon pressure; impulse/rocket equations; SSAPy,"Ideal model/preset validation, not measured spacecraft qualification"
   tests/test_propagators_6dof_truth.py,2,R2,SSAPy analytic Kepler propagator and REBOUND IAS15,Independent integrators for the same two-body model
   tests/test_proper_motion_radial.py,4,R1,Angular-rate projection and SI/REBOUND unit conversion,Specified kinematics
   tests/test_propulsion_catalog.py,1,R1 C,Rocket equation; F/m and F/(Isp*g0),Catalog configuration checks are not engine performance measurements
   tests/test_read_3le.py,1,R2 C,Native SGP4 twoline2rv parsing and epoch conversion,Parser parity
   tests/test_real_ssapy_force_parity.py,4,R2 C,Native SSAPy SRP/drag/harmonic/third-body models,Adapter parity; same backend cannot independently validate that backend
   tests/test_real_ssapy_parity.py,3,R2 C,SSAPy accelerations and independent Orbit.at propagation,Adapters share force backend; trajectory comparisons have explicit budgets
   tests/test_reference_case.py,8,R1 C,Literal CCSDS OEM SI/time/covariance fixtures and round trips,Synthetic Orekit-style text is not an independently generated Orekit trajectory
   tests/test_review_correctness_regressions.py,25,R1 R2 R4 R5,Noncentral chi-square; converged cubature/Radau; exact free-body STM; lunar kinematics; SSAPy orbit dynamics; Astropy with the packaged finals2000A.all for local tangent frames,Patch regressions plus consistency guards; no observed tracking data
   tests/test_satellite_frames.py,5,R1 R4,Known circular/local axes and 90 degree quaternions,Eccentric nadir/NTW independently anchored in review regressions
   tests/test_segment_dense_output.py,2,R1 C,Free-flight positions and pre/post impulse velocity jump,Known piecewise time convention
   tests/test_spacecraft_force_models.py,7,R1,Closed-form gravity gradient; drag; SRP; facet forces/torques; exponential density,Specified ideal force models
   tests/test_spacecraft_presets.py,11,R1 R3 C,Analytic ray-box intersections and crossing planes; NASA 3U envelope and selected published telescope dimensions/counts; selected published features,Only explicitly referenced features are checked; family models are schematic
   tests/test_spacecraft_view.py,12,R1 C,Quarter-turn and center-of-mass geometry; spherical occultation/limb; pinhole projection; constant-rate interpolation; preset prism and panel dimensions/areas,Illustrative spacecraft shapes and spherical lighting; export contracts are separate from physical references
   tests/test_ssa_conjunction.py,4,R1 R2 R4,Linear TCA; covariance algebra; dense cubic minimum; chi-square disk probability,Anisotropic absolute probability independently cubature-checked in review tests
   tests/test_starfield_epochs.py,1,R2 R3,Astropy leap-second time conversion,Epoch contract
   tests/test_state_to_kepler.py,3,R1 R2,SSAPy elements/state conversion and known conic invariants,Specified conics
   tests/test_station_keeping_burn_benchmark.py,1,R2,Converged independent Radau drag/impulse integration and orbital energy,Synthetic constant tangential drag; not operational station-keeping performance
   tests/test_synthetic_orbit_population.py,1,R1 R2 R4,Known Gaussian spread; SSAPy propagated trajectories,"Seeded population, not observed catalog calibration"
   tests/test_targeting.py,5,R1 R4,Constant-acceleration terminal states; control sum; mass/wheel/impulse continuity,1e-7 m/s bounded optimizer budget; ODE identity checked separately
   tests/test_thrust_profiles.py,5,R1 R3,Analytic ramp/pulse/exponential/triangle impulses; packaged certified motor impulse,Certification anchors depend on provenance of packaged motor curves
   tests/test_time_and_yastropy_helpers.py,1,R1 R2,WGS84 equatorial radius; Earth rotation surface velocity,Round trip supplemented by absolute geometric radius/velocity
   tests/test_time_functions.py,5,R1 R2 R3,Sexagesimal definitions; GPS/TT offsets; Meeus Julian dates; Astropy,Leap-second table/model versions matter
   tests/test_time_grids_and_orbit_quantities.py,3,R1 R2 R3,2016 leap second; SI time spans; SSAPy orbital elements,Time/element conventions
   tests/test_tle_fixed_width_helpers.py,1,R1 C,Literal ISS-like fixed-width TLE fields,"Synthetic formatting/parser fixture, not orbit truth"
   tests/test_tle_iter_pairs.py,1,C,Literal validated-checksum TLE line/name records,Serialization contract only
   tests/test_transfer_optimal.py,1,R1,Analytic Hohmann cost and transfer time,Selected coplanar circular case; not proof of general global optimality
   tests/test_two_body_closed_forms.py,2,R1,Vis-viva apsidal speeds and haversine great-circle distance,Ideal two-body/spherical models
   tests/test_solar_scene_geometry.py,5,R1 R2,ERFA ecm06; SSAPy DE ephemeris; analytic pinhole projection,Approximate built-in ephemeris; not a measured-sky calibration
   tests/test_vector_rotation_references.py,3,R1 R2,Literal right-hand turns; SciPy quaternion rotations; magnitude-independent directions,2e-14 dimensionless component budget; explicit reference-axis convention
   tests/test_viewer_star_reference.py,1,R2 R6,Astropy GCRS-to-TEME sky at 2008 and 2026 epochs,0.5 arcsec transform/serialization budget; catalog astrometry itself is not recalibrated

The inventory is a review record, not a proof of completeness over all
possible inputs. Revisit it when adding tests, changing conventions, updating
ephemerides or reference packages, or changing acceptance tolerances.

Workflow contracts outside the numerical suite
----------------------------------------------

The original testing policy excludes rendering and demo execution from
``tests/``. Publishing/export checks therefore run separately in the gallery
workflow with ``python -m pytest scripts/tests -q``. The default pytest command
selects ``tests/`` explicitly in ``pyproject.toml``. The current gallery follow-ups have
658 core cases (103 files, 389 functions) and 161 workflow cases (6 files,
33 functions): 819 total. The browser contract runs in gallery CI and skips
locally unless ``SSATK_BROWSER_TESTS=1`` is set. Relative to the preceding
792-case handoff, this adds eight numerical and ten rendering/export cases.
The subsequent gallery dependency fix adds three publishing cases: both CLI
skip policies (including stale archive removal) and discovery of the real
gallery membership with standalone GMAT excluded. These are deployment
contracts, not physical reference tests.

In the preceding audit, the former 775-case combined suite became 792 cases: one duplicated path
layout was removed and 18 targeted cases were added. Moving 125 cases out of
``tests/`` changes their classification, not the total number checked.

* ``scripts/tests/test_gallery_publishing.py`` checks a relocated ZIP under a
  project URL using literal HTTP asset bytes; missing and nonportable links;
  one execution per demo; named fast-mode arguments; captured failure output;
  deadlines including imports; source-preserving cleanup; and stale ZIP removal.
  Different failure modes are retained, but a redundant second directory layout
  was removed. The single HTTP case uses spaces and a dotted directory name.
* ``scripts/tests/test_spacecraft_exports.py`` checks format/dimensions, decoded
  GIF durations, mesh selection, all 107 catalog builds, aliases/metadata, and
  foreground sphere visibility. Catalog checks reject malformed geometry but
  do not establish measured spacecraft shape or flight-frame alignment.
  Generic unknown-name/error-message assertions were removed.

Analytic mesh visibility and selected published dimension checks remain in
``tests/``. No propagation or independent frame references were removed.
The former ground-track gallery demo's hardcoded residuals and unconditional
``run_tests(): return True`` were replaced by a computed comparison of
``gcrf_to_lat_lon`` (SSAPy) with Astropy GCRS/ITRS/WGS84. Its input orbit is
prescribed; it does not constitute flight-data calibration.

The handoff review also replaces 80 implicit ``assert_allclose`` tolerances
across 16 existing test files with explicit unit-aware budgets or exact
comparisons for copied data. In particular, a relative tolerance at a large
GPS epoch could accept a whole-second time error. Copied segment epochs now
require exact equality. Analytic frame arithmetic allows float64 rounding;
OEM elapsed-time conversions retain a 1 ns absolute budget. These budgets
validate specified computations, not measured model fidelity.

The solar scene follow-up adds analytic pinhole projection checks, an ERFA
J2000 ecliptic rotation reference, and four-epoch comparisons against SSAPy's
separate DE ephemeris. ``scripts/tests/test_solar_gallery.py`` checks physical
mesh radii, explicit scale labels, precise UTC dates and AU coordinates, dense
circular trails under playback decimation, and 60-second frame timing.
``scripts/tests/test_solar_sky_browser.py`` uses two prescribed opposite rays
and a green occluder: it checks actual browser pixels on camera reversal and
resize. It does not treat file existence as evidence of rendering correctness.

The full-period divergence example adds two demo regressions in
``scripts/tests/test_divergence_gallery.py``. A circular GEO solution checks
all nominal epochs analytically; SSAPy's separate Kepler solver checks every
perturbed DOP853 trajectory. Absolute budgets are 1 cm and 1 micrometre/s over
one period, with a 0.5 microsecond epoch-span allowance and at most 120 seconds
between samples. These validate synthetic point-mass dynamics, not flight data.

``scripts/tests/test_moon_data_sources.py`` adds three preparation contracts:
a real packaged-data bake with downloads blocked and source hashes unchanged,
and custom-elevation precedence with either packaged or opted-in NASA colour
sources. They check file selection and decodable textures, not terrain accuracy.

The existing movie-duration contract now checks both an explicit seven-day
span and the default 27.3217-day mean lunar revolution from NASA's Moon Fact
Sheet. Both decode to the requested playback length; the physical endpoint
allows 1 microsecond for GPS rounding and path sampling is at most one hour.
This adds one parametrized case, not another test function or propagation oracle.

``scripts/tests/test_gallery_max_textures.py`` adds two functions/three cases.
Four literal, asymmetric colours painted onto the quadrants of an 8192x4096
JPEG check that ``load_earth_file`` and ``load_moon_file`` keep that native
grid and reproduce each landmark's RGB within 2/255, parametrized over both
loaders (C). A separate case checks the satellite-viewer browser payload
decodes to the same byte-identical JPEG, at the same native grid and landmark
colours within 2/255 (C). These confirm gallery-mode texture loading and
browser export are not silently downsampling the prepared map; they make no
claim about geographic accuracy.

``scripts/tests/test_gallery_sphere_raster.py`` adds three functions/five
cases. Two R1 analytic references check ``_sphere_raster.sphere_intersections``
against a closed-form orthographic unit-circle intersection (1e-14 budget) and
a perspective tangent ray from a specified eye point (3e-8 budget, set by
square-root conditioning at the double root). A third, three-way parametrized
case (orthographic; rolled and inverted-axis perspective) paints four literal
colours onto the quadrants of an 8192x4096 map at closed-form latitude +/-30
degrees, longitude +/-30 degrees on a radius-two sphere, then projects those
coordinates through the real Matplotlib camera and checks the rendered pixel
colours match within 1/255 (R1/C). These validate the raster projection and
camera-axis handling added for raster body surfaces, not measured imagery.


Star-finder extension
----------------------

``tests/test_star_tracker.py`` adds five tests with literal quarter-turn
rotations, exactly intersecting finite-distance rays, loss of observability
for parallel rays, analytic tangent motion at a celestial pole, and the
Moon viewer's epoch/rotation contract. The new Cartesian proper-motion
calculation avoids division by cos(Dec); the former linear RA/Dec update
was displaced by up to 0.1492 arcsec from Astropy's zero-distance-information
space-motion model for the 83,123 stellar rows at 2026-10-12. This is a
comparison of calculations on the same catalogue, not independent
validation of the original astrometric measurements.

``scripts/tests/test_star_finder_demo.py`` adds two tests: an end-to-end
noise-free simulation compared with a separate analytic circular orbit and
prescribed attitude/position truth, and explicit catalogue-source/unit
contracts. The simulation's approximate aberration correction uses a
100-km/50-m/s coarse orbit offset, so a 0.1-arcsec attitude and 2-m position
budget is retained even with zero centroid noise. Full-cycle DOP853
positions must agree with the analytic circle within 2 mm. Rendering and
file metadata contracts are not counted as physical validation.

Satellite viewer Earth detail
-----------------------------

``scripts/tests/test_viewer_earth_assets.py`` adds two publishing contracts:
byte-exact embedding and offline fallback for an asymmetric image fixture,
and rejection of a checksum-mismatched source without replacing the previous
cached asset. Neither test claims to validate geography or orbit dynamics.
The prepared gallery uses a SHA-256-pinned NASA Blue Marble product, with
browser inspection of the embedded dimensions and landmark UV alignment.


Coverage-map validation
------------------------

``tests/test_coverage_references.py`` adds five functions/seven cases:
analytic crossing times, the spherical horizon-angle formula, stationary
WGS84 access against ellipsoid normals, latitude-band surface integrals,
and archived HST states against direct SGP4 with Astropy TEME-to-ITRS.
The latter checks model selection and coordinate handling, not flight truth.
``scripts/tests/test_coverage_analysis_contracts.py`` separately checks
no-output mode and Sun/satellite/site geometry wiring. The reproducible
full-grid and event-convergence audit is described in :doc:`coverage_validation`.

Gamma and heading follow-up
---------------------------

The gamma/heading module adds eight core cases: radial/stationary and polar
singularities, an analytic quadratic sampled at unequal intervals, invalid
epoch ordering, and an Astropy spherical-differential state comparison.
Two workflow cases separately check plotted speeds against straight/circular
motion and direction symbols against cardinal/radial geometry. The demo's
full-period comparison and explicit frame conventions are documented in
:doc:`gamma_heading_validation`.


Dated globe rendering follow-up
-------------------------------

``scripts/tests/test_globe_rendering.py`` contains seven test functions, fourteen
parametrized cases. R1/R2 references are the explicit spherical-coordinate map,
prescribed satellite positions, opaque-sphere visibility and Astropy GCRS/ITRS
coordinates. They inspect rendered polygon vertices and pixels, rather than
comparing two uses of the same SSATK transform. The frame comparison allows
0.5 arcsecond for IAU model differences; arithmetic-only coordinates allow
0.1 micrometre at Earth radius and 20 nanometres for marker placement. C cases
reject absent, repeated, reversed or asynchronous animation epochs and suppress
a highlighted satellite marker at an unsupplied time. They do not establish telemetry
accuracy or physical surface fidelity of the Earth imagery.

The batched-rendering contract uses 32 synthetic tracks: positions at
+7000 km along the viewing axis are visible, while positions at -7000 km
with lateral offsets no larger than 100 km are behind Earth. It draws the
initial frame with empty trails and the following frame, checking visible
head pixels and fixed overview limits. This catches Matplotlib 3.11's empty
collection autoscaling failure without treating an exported file as a
physical reference. Polygon tests observe public ``set_verts`` calls while
preserving the real renderer and the existing independent references; they
do not read Matplotlib's removed private ``_vec`` buffer. Pixel checks attach
an Agg canvas explicitly when the plotting function closes its pyplot manager.
These contracts also run in the Python CI matrix without a Matplotlib pin.

Lunar plot follow-up
--------------------

``scripts/tests/test_moon_plot_rendering.py`` adds seven functions/thirteen
cases. An independent scalar Kepler-equation solution checks the demo's lunar
ellipse and velocity to 1 mm and 1 micrometre/s, including GPS-time rounding.
An end-to-end renderer check catches a repeated origin subtraction, and literal
spherical coordinates plus an asymmetric texture catch longitude shifts and
latitude flips. Astropy ERFA Moon positions and apparent Sun directions provide
independent 20-km and 0.02-degree model-comparison budgets against DE440.
Tidal locking constrains the rendered prime meridian within the 9-degree optical
libration envelope. Radius checks include the analytically bounded second-order
error of the SSAPy-compatible first-order polar-motion matrix.

Separate C checks cover short-arc framing, visible/occluded raster pixels,
standalone TT-to-UTC epochs, gnomonic sky projection and invalid inputs. These
checks validate geometry and output contracts, not measured lunar flight states,
terrain elevation or calibrated surface photometry.

Cislunar framing follow-up
--------------------------

``scripts/tests/test_cislunar_framing.py`` adds three functions/seven cases.
A specified linear Moon ephemeris puts the real rotating-frame transform in
an exact Cartesian basis at the test epoch. Literal 3,000-km circles must
produce +/-3,480-km limits under 8% padding, with a 1e-8-km floating-point
budget and the physical 1,738.1-km Moon radius. Analytic orthographic visibility
selects the expected samples behind the lunar disc. Mixed near-Earth and
near-Moon tracks are checked in both input orders and all three layouts,
against supplied metre positions with a 0.1-mm arithmetic budget. XY bounds
must remain tied to the lunar radius despite a large vertical excursion.

Canvas containment, legend suppression and fractional tick placement are
explicit C contracts. These tests validate framing and coordinate/unit wiring;
they do not provide an independent real lunar ephemeris or propagation oracle.

Mesh catalogue demo follow-up
-----------------------------

``scripts/tests/test_mesh_catalog_demo.py`` adds three functions/four cases.
A literal 4-by-2-m rectangle in the camera plane must retain its 2:1 shape and
90% width fit in landscape and portrait thumbnails, within one raster pixel.
Translation and uniform scaling must leave those bounds unchanged. This R1
reference validates display geometry, not mission dimensions.

C contracts check the complete current registry, one embedded and decodable
actual-mesh preview per canonical name, escaped metadata and standalone assets.
The opt-in browser case runs with networking disabled and exercises an HST alias
search, combined collection filtering, preset selection, referenced CubeSat
dimensions, and local PNG/JPG downloads. Flight-model fidelity remains limited
to the separately documented referenced features in :doc:`spacecraft_presets`.

GEO chase-camera demo follow-up
-------------------------------

``scripts/tests/test_geo_spacecraft_demo.py`` adds three functions/four cases.
An analytic circular-orbit solution checks the propagated 24-hour track to
1 mm and 1 micrometre/s; the radius and speed correspond to the mean sidereal
period, rather than a 24-hour orbital period. Astropy independently transforms
the track to Earth-fixed coordinates, requiring longitude and latitude to
remain within 0.001 degree of the initial Greenwich subpoint. The angular budget
includes frame-model differences and the unperturbed orbit plane's small drift
relative to Earth's evolving equator over one day.

Scene geometry checks constrain nadir/along-track body axes to 1e-12 and camera
placement to 10 nm. Export contracts require 45-second playback in both fast
and normal modes and one propagated position per output frame, with no sparse
chord interpolation. These are model and display checks; no measured GEO
telemetry, attitude-control dynamics or station-keeping performance is claimed.

Vector rotation and animation-camera follow-up
----------------------------------------------

``tests/test_vector_rotation_references.py`` adds three numerical reference
tests for ``ssatk.vectors.rotate_vector``. Literal right-hand quarter-turns
constrain the signs and order of both rotations, including the formerly
undefined positive and negative Y directions. SciPy quaternion rotations about
specified axes independently check an oblique direction and directions within
1e-14 of either Y pole. The absolute component budget is 2e-14, dimensionless,
with zero relative tolerance. Known positive multiples of +X spanning
1e-200 to 1e200 check normalization without overflowing or underflowing squares.
The routine uses the original +Y reference except within 1e-12 of either pole,
where the documented +X reference defines the otherwise singular first axis.

Three new software-contract cases check invalid rotation input, JSON export
of NumPy camera scalars, and actual browser target replacement/reordering.
The last runs the shipped Three.js controller, including repeated names and
names containing delimiters. It requires a removed target's chase direction
to be cleared while a still-selected target retains its direction on reorder.
The existing Plotly frame test also requires titles, backgrounds and annotations
to survive camera attachment unchanged. These are C contracts, not physical
accuracy references. The previously added camera geometry cases separately
check fixed translations, nonuniform-time sweeps, constant-velocity tangents,
independent projection, full-scene containment and rendered pixel centring.

Transfer planning dashboard follow-up
-------------------------------------

``scripts/tests/test_structured_transfer_dashboard.py`` adds four cases.
An independent DOP853 integration of each selected Keplerian transfer must
reach a closed-form inclined circular target within 5 cm. Tightening integration
tolerances by ten must change the reference by less than 1 mm and 1 micrometre/s.
Reconstructing the departure and arrival impulses from those independent
states must match the reported cost within 5e-5 m/s. These are R1/R2 synthetic
two-body references; no measured accuracy or finite-thrust accuracy is claimed.

The flight-time curve finalizes each accepted search point because the coarse
Lambert grid and final plan use different solver tolerances. Its 60-minute
point must match the comparison plan within 1e-6 m/s. C contracts require
single-departure-burn plans to retain their full coast interval, stacked bars
to show every planned impulse even for an arrival-only objective, and rendered
text rectangles to stay inside the canvas without overlapping at 140 and
200 dots per inch. The existing Hohmann minimum-cost reference remains a
separate check on the solver itself.

The dashboard alignment follow-up extends the existing rendering contract:
all four plot rectangles must share row/column edges and dimensions to within
one pixel at both export resolutions, while equal x/y distances in the orbit
panel retain equal screen lengths. It adds no numerical test cases.

Three publishing contracts execute the command displayed in a gallery card
for a ``main`` entry point, a ``run`` entry point, and a standalone script.
They verify the command runs the selected module and uses the documented
standalone defaults, rather than implying the gallery's fast settings were
reproduced. These are software contracts, not physical validation.

Gallery paths, body maps, and dense animations
-----------------------------------------------

``scripts/tests/test_gallery_figure_calls.py`` adds seven C contracts: an
executed drawing statement with HTML-sensitive arguments survives publishing
and relocation, malformed/ambiguous annotations fail, the introductory demo
writes a single ``figures`` directory, and every bundled visual demo supplies
parseable output-specific source regions. These validate publishing behavior,
not the physics depicted by the figures.

``scripts/tests/test_gallery_render_repairs.py`` adds eight checks. R1 references
include literal circular Walker states (1 mm and 1 micrometre/s), a straight
trajectory with known 5 km/s speed at nonuniform times in both time directions,
an explicitly unknown speed at the initial single sample, and analytic sphere
occlusion/dateline cases for the batched renderer. The R2 lunar-map reference
contains 50 CSPICE matrices across 2000--2040 and weekly 2025 epochs. The
2 cm surface-displacement budget covers SSAPy's TT versus TDB evaluation and
epoch roundoff; source URLs, kernel hashes and reproduction steps accompany
the fixture. This tests orientation, not centimetre-resolution imagery.

C checks require north-up map legend keys, no map I/O in low-spec mode, actual
background pixels and non-overlapping panel bounds at three animation frames,
retention of 3-D trajectory coordinates, and faithful display of specified
residuals including exact zero in the force-model dashboard. The existing sky
epoch contract now mocks the map-frame helper used by the renderer; its
independent quarter-turn and expected direction remain unchanged.
