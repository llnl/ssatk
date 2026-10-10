6-DoF Architecture Study
========================

Purpose
-------

This document records the design rationale for six-degree-of-freedom (6-DoF)
spacecraft dynamics in Space Situational Awareness Toolkit (SSATK). It is a trade study, not a
benchmark, endorsement, or complete review of every flight-dynamics package.
The external tools listed here were reviewed to identify common modeling
patterns, API boundaries, and non-goals for SSATK.

Survey date: 2026-08-20.

Problem Statement
-----------------

SSATK propagates orbits and analyses them end to end. The 6-DoF layer adds
spacecraft attitude and body physics to that same workflow. It is not meant to
replace Basilisk, Tudat, GMAT, Orekit, STK, FreeFlyer, or launch-vehicle
flight-dynamics tools.

The target use case is an analyst who works with ``ssatk.Orbit`` objects or
inertial position/velocity arrays and needs to add:

* quaternion attitude state,
* body-frame angular rate,
* spacecraft mass and inertia,
* attitude- and angular-rate-dependent drag plus solar-radiation-pressure
  forces,
* body-frame torques,
* finite thrust and mass depletion,
* optional hardware components such as facets, tanks, thrusters, and wheels.

The 6-DoF layer should keep the same practical style as the rest of SSATK:
small Python objects, direct NumPy arrays, explicit units, and examples that can
be read as documentation.

External Tools Reviewed
-----------------------

NASA 42
^^^^^^^

NASA 42 is a direct reference for general-purpose spacecraft attitude and orbit
dynamics. Its public README describes multi-body spacecraft attitude dynamics,
rigid and flexible bodies, concurrent spacecraft, contact forces, two-body and
three-body orbital regimes, and visualization.

Design lesson for SSATK:

* Support a rigid-body hub first.
* Keep the API open to multiple bodies and flexible appendages later.
* Do not require a full simulation framework for simple concept studies.

Reference: https://github.com/ericstoneking/42

Basilisk
^^^^^^^^

Basilisk is the most relevant open-source architecture reference. Its
``spacecraft`` module is built around a hub plus attached ``stateEffectors`` and
``dynamicEffectors``. Public Basilisk docs describe reaction wheels, fuel tanks,
thrusters, hinged rigid bodies, radiation pressure, facet drag, and faceted
spacecraft models.

Design lesson for SSATK:

* Separate the spacecraft state from the force/torque providers.
* Treat hardware as components attached to a central spacecraft body.
* Distinguish ``state effectors`` that add integrated states from ``dynamic
  effectors`` that only contribute forces and torques.
* Start with rigid, fixed components; add articulated/flexible states only when
  a real SSATK workflow needs them.

Reference: https://github.com/AVSLab/basilisk

Tudat and TudatPy
^^^^^^^^^^^^^^^^^

Tudat/TudatPy provides the strongest astrodynamics modeling pattern for
combining translational, rotational, and mass propagation. Its documentation
exposes translational, rotational, mass, multi-type, multi-arc, torque,
aerodynamic, and radiation-pressure setup objects. Its force and torque
interfaces distinguish environmental models from propagated state settings.

Design lesson for SSATK:

* Keep translational, rotational, and mass states conceptually separate even
  when one integrator propagates them together.
* Make the force and torque model interface composable.
* Allow later multi-arc or estimation use without forcing that complexity into
  the first public API.

Reference: https://github.com/tudat-team/tudatpy

NASA JEOD and Trick
^^^^^^^^^^^^^^^^^^^

NASA JSC Engineering Orbital Dynamics (JEOD) is designed for use with the NASA
Trick Simulation Environment. Its public README describes environment,
dynamics, interaction, and utility models; standalone spacecraft trajectory and
attitude state; and coupling to effectors and guidance, navigation, and control
systems.

Design lesson for SSATK:

* Keep environment models, dynamics models, interaction models, and utilities
  separable.
* Do not make SSATK a general simulation executive. Python function calls and
  explicit object composition are enough for this package.

References:

* https://github.com/nasa/jeod
* https://github.com/nasa/trick

GMAT
^^^^

NASA GMAT is a mission-design and orbit-analysis reference, especially for
finite burns, tanks, thrusters, SPAD files, and operational maneuver workflows.
GMAT's torque-modeling documentation states that current torque outputs are
available for reporting and are not included in attitude propagation.

Design lesson for SSATK:

* Reuse the useful operational concepts: spacecraft hardware, tanks, thrusters,
  finite burns, and force-model reports.
* Do not treat GMAT as the architecture reference for coupled 6-DoF attitude
  propagation.

Reference: https://github.com/nasa/GMAT

Orekit
^^^^^^

Orekit is a mature astrodynamics library with strong force-model, maneuver,
attitude-provider, frame, and event-handling architecture. Its public docs
describe force models that access spacecraft state, attitude, mass, date, and
frame, plus impulse and continuous-thrust maneuvers. Orekit is not primarily a
spacecraft rigid-body dynamics simulator.

Design lesson for SSATK:

* Use explicit force-model interfaces.
* Preserve event and maneuver concepts for future finite-burn and control
  development.
* Keep attitude providers distinct from attitude dynamics.

Reference: https://www.orekit.org/

FreeFlyer and STK/Astrogator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

FreeFlyer and STK/Astrogator are operational mission-design references for
finite burns, attitude reference frames, maneuver targeting, and analyst-facing
workflows. Public FreeFlyer documentation describes finite burns with active
thrusters, tank mass depletion, burn duration, burn reference frames, and thrust
steering. These tools are not the best model for a lightweight open Python
6-DoF implementation.

Design lesson for SSATK:

* Use user-facing terms familiar to mission analysts: finite burn, burn frame,
  tank, thruster, thrust direction, and mass depletion.
* Keep the first SSATK implementation scriptable and inspectable instead of
  reproducing a full mission-control-sequence system.

References:

* https://ai-solutions.com/freeflyer/
* https://www.ansys.com/products/missions/ansys-stk

JSBSim, RocketPy, and OpenRocket
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

JSBSim, RocketPy, and OpenRocket are useful references for atmospheric vehicle
and launch-vehicle 6-DoF modeling. They emphasize configurable aerodynamics,
propulsion, atmosphere, and vehicle geometry. Their core problem is different
from SSATK's orbital and cislunar spacecraft focus.

Design lesson for SSATK:

* Borrow clean body/component/aerodynamic patterns where useful.
* Do not tune SSATK's 6-DoF API around launch-vehicle-specific concepts unless
  a rocket workflow explicitly needs them.

References:

* https://github.com/JSBSim-Team/jsbsim
* https://github.com/RocketPy-Team/RocketPy
* https://github.com/openrocket/openrocket

Capability Matrix
-----------------

.. list-table::
   :header-rows: 1
   :widths: 22 18 18 18 24

   * - Tool
     - Coupled 6-DoF
     - Spacecraft body model
     - Maneuver hardware
     - Best lesson for SSATK
   * - NASA 42
     - Yes
     - Rigid/flexible multi-body
     - Thrusters and FSW concepts
     - Keep the path open to multi-body and flexible dynamics.
   * - Basilisk
     - Yes
     - Hub plus effectors
     - Thrusters, tanks, wheels
     - Use componentized state and dynamic effectors.
   * - Tudat/TudatPy
     - Yes
     - Environment/body settings
     - Thrust and mass propagation
     - Separate translational, rotational, mass, force, and torque settings.
   * - JEOD/Trick
     - Yes
     - Simulation-framework components
     - Effectors via simulation coupling
     - Keep environment, dynamics, and interactions modular.
   * - GMAT
     - Partial
     - Spacecraft hardware and SPAD
     - Strong finite-burn workflow
     - Reuse operational concepts, not attitude-dynamics architecture.
   * - Orekit
     - Partial
     - State, mass, attitude providers
     - Strong maneuver/event model
     - Keep force models explicit and event-ready.
   * - FreeFlyer/STK
     - Partial
     - Analyst-facing spacecraft objects
     - Strong targeting workflows
     - Use familiar user terms and burn-frame concepts.
   * - JSBSim/RocketPy/OpenRocket
     - Yes, atmospheric focus
     - Configurable vehicle geometry
     - Propulsion and aero
     - Borrow geometry patterns, avoid launch-specific API bias.

Chosen SSATK Architecture
-------------------------

SSATK should implement a small, composable 6-DoF layer with these boundaries:

``Spacecraft``
    A user-facing state object with inertial position, inertial velocity,
    epoch, body-to-inertial quaternion, body-frame angular rate, mass, inertia,
    and optional attached components.

``SpacecraftBody`` or ``RigidBody``
    A physical body definition containing mass properties, center of mass,
    reference geometry, and body-frame component locations. This should become
    the source of truth for mass, inertia, reference area, center of pressure,
    and visual geometry.

``Facet``
    A fixed body-frame surface element with area, normal, center of pressure,
    drag coefficient, reflectivity coefficient, and optional optical properties.
    The first implementation should support fixed facets only.

``Thruster``
    A body-frame actuator with location, thrust direction, thrust magnitude or
    thrust schedule, specific impulse, and optional tank connection.

``Tank``
    A mass component with propellant mass, dry mass or hardware mass, body-frame
    location, and optional inertia contribution. The first implementation can
    use constant inertia or simple point-mass parallel-axis updates.

``ReactionWheel``
    A body-frame actuator with a torque axis, torque limit, and optional
    momentum-capacity metadata. The current SSATK implementation contributes
    saturated body-frame torque and can propagate wheel angular momentum as
    separate states when wheels are attached to a ``SpacecraftBody``. Commands
    are clipped at configured momentum-capacity limits.

``Environment``
    A provider for atmosphere density, wind, Sun position, third-body positions,
    eclipse fraction, and gravity constants. This keeps physical environment
    data out of individual force models.

``ForceTorqueModel``
    A callable object that receives the spacecraft state, body model, and
    environment, then returns inertial force, body-frame torque, and optional
    mass-flow contributions.

Near-Term Implementation Plan
-----------------------------

The practical first release should be deliberately limited:

1. Keep the existing ``Spacecraft`` and ``propagate_6dof`` user workflow.
2. Add a body-model object that can be attached to ``Spacecraft``.
3. Add fixed-facet geometry and one helper constructor for a box-wing
   spacecraft.
4. Convert flat-plate drag and solar-radiation-pressure models to operate over
   one or more facets.
5. Add thruster components with body-frame force, body-frame torque, and
   optional mass-flow output.
6. Add reaction wheels and a basic quaternion PD controller for simple attitude
   studies.
7. Add tests that check force direction, torque sign, quaternion normalization,
   mass-flow sign, and agreement with the existing single-flat-plate
   functions for the one-facet case.

Non-Goals for the First SSATK 6-DoF Release
-------------------------------------------

These features are useful, but should not be required for the initial
architecture:

* flexible bodies,
* fuel slosh,
* articulated solar arrays,
* contact dynamics,
* high-order geopotential torque,
* full finite-element or CAD-derived spacecraft geometry,
* a simulation executive,
* a GUI or mission-control-sequence language.

These can be added later without breaking the first API if the body, component,
environment, and force/torque boundaries remain clean.

Current SSATK State
-------------------

The current SSATK 6-DoF implementation already has the correct minimal
numerical backbone:

* ``ssatk.propagators_6dof.Spacecraft`` stores inertial position, inertial
  velocity, quaternion attitude, angular rate, inertia, mass, area, drag
  coefficient, reflectivity coefficient, and center of pressure.
* ``ssatk.propagators_6dof.propagate_6dof`` propagates position, velocity,
  quaternion, angular rate, and solver diagnostics with user-provided
  acceleration and torque models.
* ``ssatk.propagators_6dof.propagate_6dof`` forwards SciPy ``solve_ivp`` events
  and dense output for event-driven propagation segments without adding an
  SSATK-specific event framework.
* ``ssatk.propagators_6dof.radius_crossing_event``,
  ``altitude_crossing_event``, ``mass_floor_event``, and
  ``propellant_empty_event`` provide small reusable physical event helpers for
  altitude/radius crossings and burn mass limits.
* ``ssatk.accelerations_6dof`` contains point-mass gravity, J2,
  third-body gravity, cannonball drag, cannonball solar-radiation pressure,
  constant inertial/NTW/body acceleration, summed force/torque helpers, and
  flat-plate/facet drag/SRP, thruster, gravity-gradient, magnetic-dipole,
  reaction-wheel, and quaternion PD attitude-control torque models.
  Flat-plate and facet drag include the local surface velocity from body-frame
  angular rate and accept explicit atmosphere velocity inputs, so force and
  torque change when exposed surfaces spin relative to winds or rigid
  co-rotation.
* ``ssatk.accelerations_6dof.SpacecraftManeuverAccel`` adds finite
  maneuver acceleration with explicit ``frame`` selection and scalar, callable,
  analytical, or CSV-loaded thrust curves.
* ``ssatk.accelerations_6dof.SpacecraftAccelSSAPy`` adapts SSAPy
  translational accelerations into the SSATK 6-DoF force-model interface. This
  allows SSATK to reuse SSAPy's mature Earth harmonics, third-body gravity,
  Harris-Priester drag, solar radiation, and Earth radiation models without
  forcing all spacecraft body physics into SSAPy.
* ``ssatk.propagators_6dof.propagate_6dof`` can optionally propagate spacecraft
  mass as a 14th state when a mass-flow model is supplied. Thruster force models
  therefore see the current propagated mass instead of a fixed initial mass.
  When a ``SpacecraftBody`` has tanks, propagated mass updates tank propellant
  proportionally for state-dependent center of mass and inertia.
  ``Spacecraft.propagate`` continues tanked bodies at dry mass by default after
  depletion and disables propulsive acceleration, torque, and mass flow. Set
  ``stop_at_dry_mass=True`` for a terminal dry-mass event; lower-level
  ``propagate_6dof`` calls should pass ``propellant_empty_event`` or
  ``mass_floor_event`` explicitly.
  ``SpacecraftManeuverAccel(tank_name=...)`` can instead feed one named tank
  and stops that maneuver when the selected tank is empty.
* ``ssatk.propagators_6dof.propagate_spacecraft_high_accuracy`` provides
  the current analyst-facing high-accuracy entry point for combining SSATK
  body/torque/thrust models with optional SSAPy translational perturbations.
* ``ssatk.propagators_6dof.propagate_spacecraft_segments`` chains
  consecutive coast/burn/environment segments while preserving state, propagated
  mass, event diagnostics, and solver evaluation counts. Its
  ``ImpulseManeuver`` state reset supports inertial, body, NTW, RTN, and VNB
  delta-v components plus optional mass, quaternion, and angular-rate updates.
* ``ssatk.propagators_6dof.solve_6dof_multi_segment_target`` performs
  bounded single shooting over those segments and accepts normalized constraint
  and residual hooks.
* ``ssatk.propagators_6dof.propagate_6dof_variational`` propagates a
  coupled local-coordinate STM through the same rigid-body RHS, while
  ``propagate_6dof_covariance`` maps initial covariance and process-noise
  contributions through it. Covariances may use either the raw quaternion
  coordinates or the three-parameter local attitude-error coordinates.
* ``ssatk.propagators_6dof.propagate_6dof_extended`` propagates one or
  more linearized hinged-appendage, flexible-mode, and propellant-slosh
  coordinates with reduced-order body force/torque coupling.
* ``ssatk.propagators_6dof.propagate_6dof_extended_variational``
  propagates the same reduced-order states with a full finite-difference STM
  for sensitivity and covariance workflows.
* ``ssatk.environment.SpaceEnvironment`` centralizes epoch-aware
  Sun/Moon position, atmosphere density/velocity, magnetic-field,
  eclipse-fraction, and environment-backed SSATK force-model construction,
  including named third-body perturbations. The conical eclipse model applies
  disk-overlap visibility for Earth and Moon solar occultation by default so
  cislunar SRP cases do not remain fully illuminated behind the Moon. The
  default atmosphere velocity is rigid Earth co-rotation; the default magnetic
  field is a dependency-light centered Earth dipole; ``magnetic_field_model="igrf"``
  uses optional ``ppigrf`` for epoch-dependent internal-field synthesis.
* ``ssatk.environment_eop.load_packaged_eop`` loads the frozen IERS
  Finals2000A daily Earth-orientation snapshot from ``ssatk-data-core``. Its
  ``EarthOrientationTable`` preserves observed/predicted status and refuses
  predicted records unless the caller opts in. The explicit
  ``gcrf_to_itrf_eop`` and ``itrf_to_gcrf_eop`` transforms use UT1 and polar
  motion without depending on a live network table.
* ``ssatk.environment_space_weather.load_packaged_space_weather``
  loads the frozen CelesTrak F10.7 and Ap series. Setting
  ``SpaceEnvironment(atmosphere_density_model="nrlmsise00")`` activates the
  optional ``pymsis`` NRLMSISE-00 model using those drivers and the packaged
  Earth-orientation table; predicted solar/geomagnetic records require an
  explicit opt-in.
* ``ssatk.coordinates.attitude_quaternion_from_frame`` converts existing
  SSATK satellite-operation frame definitions, such as ``ntw``, ``vnb``, and
  ``nadir_velocity``, into body-to-GCRF target quaternions for attitude-control
  studies.
* ``tests/test_propagators_6dof_truth.py`` contains a fixed GPS-epoch,
  two-orbit comparison against SSAPy's ``Orbit.at`` Keplerian reference. The
  case currently constrains position to 1 cm and velocity to 10 µm/s using the
  analyst-facing DOP853 wrapper.
* ``tests/test_j2_reference.py`` checks integrated secular nodal precession
  against first-order J2 theory and max-step convergence; real SSAPy adapter
  parity and SciPyPropagator comparisons are in ``tests/test_real_ssapy_parity.py``.

The package boundary for new 6-DoF work is:

* ``ssatk.coordinates`` owns coordinate-frame and quaternion attitude
  transforms.
* ``ssatk.propagators_6dof`` owns spacecraft state containers, rigid-body
  equations of motion, and the low-level right-hand side.
* ``ssatk.accelerations_6dof`` owns 6-DoF force, acceleration, torque,
  thrust-curve, and mass-flow models. Do not move state propagation into this
  namespace; doing so would mix equations of motion with force models.
  Facet drag/SRP models support callable ``facet_transform`` hooks for
  prescribed articulated panels without adding extra propagated states.
* ``ssatk.propagators_6dof`` owns analyst-facing integration wrappers
  and segmented propagation.
* ``ssatk.accelerations_orbit`` and
  ``ssatk.propagators_orbit`` are the canonical names for translational
  orbit-only acceleration callbacks and propagators. The optional
  ``propagate_orbit_state_with_stm`` path integrates a 6x6 state transition
  matrix for covariance and sensitivity workflows while reusing those force
  callbacks.
* ``ssatk.engines`` owns propulsion catalogs, thrust profiles,
  propellant estimates, and stationkeeping/maneuver engine helpers.
* ``ssatk.launch`` owns launch-vehicle and launch-to-orbit utilities.
  It is intentionally small until launch-ascent modeling is implemented.
* ``ssatk.compute`` should stay as a lightweight numerical-helper
  namespace for now. If it grows, split by function rather than moving all at
  once: photometry to a photometry namespace, geometry helpers to geometry,
  time-series tools to numerics/signal, and chaos metrics to dynamics analysis.

Propagation speed is a first-class development goal. New force, torque,
environment, and propagator work should add representative cases to
``ssatk.benchmark`` before broad optimization. The initial benchmark
coverage includes point-mass 6-DoF propagation, thruster propagation with
mass depletion, environment-backed facet drag/SRP propagation, and a prescribed
articulated-facet SRP case. Optimization work should preserve the public API,
avoid duplicated solver logic, and move shared math into reusable helpers rather
than copying code between force models, propagators, demos, and tests.

The remaining major gap is higher-fidelity reduced-order body modeling. SSATK
now provides linear propagated hinge, flexible-mode, and propellant-slosh
states, bounded multi-segment targeting, coupled STM/covariance utilities, and
deterministic external-validation reporting through
``scripts/external_validation_report.py``. The extended models do not replace
nonlinear multibody, finite-element, or computational-fluid-dynamics solvers.
GMAT, STK/Astrogator, FreeFlyer, Orekit, Basilisk, and Tudat reference runs
still require an environment with those tools and their force-model data.

Recommended Direction
---------------------

SSATK should follow Basilisk's componentized spacecraft pattern and Tudat's
separation of propagated states, environment, forces, and torques, while
remaining smaller and more direct than either package.

The ``NTW`` convention remains useful because it is already part of the
``Orbit`` API and gives a compact normal/tangential/cross-track burn input. It is
not the only standard used in satellite operations. Mission-analysis tools also
use ``RTN``/``RSW``/``RIC``/``LVLH`` for radial-transverse-normal commands and
``VNB``/``VNC`` for velocity-normal-binormal commands. SSATK should therefore
avoid a new NTW-only interface and expose a single ``frame=...`` argument. This
keeps ``Orbit``-API compatibility through ``frame="ntw"`` while making operational
frame choices explicit.

The public workflow should remain:

.. code-block:: python

   import numpy as np
   import ssatk

   spacecraft = ssatk.Spacecraft(
       r=[7_000_000.0, 0.0, 0.0],
       v=[0.0, 7_500.0, 0.0],
       q=[1.0, 0.0, 0.0, 0.0],
       omega=[0.0, 0.0, 0.001],
       mass=100.0,
       inertia=np.diag([10.0, 12.0, 8.0]),
   )

   body = ssatk.satellite_design(
       "earth_observation",
       mass=500.0,
       solar_array_area=10.0,
   ).with_components(
       ssatk.Component(mass=25.0, position_body=[0.0, 0.0, 0.7], name="payload"),
   )

   trajectory = ssatk.propagate_spacecraft_high_accuracy(
       ssatk.Spacecraft(r=spacecraft.r, v=spacecraft.v, body=body),
       times=np.linspace(0.0, 600.0, 61),
       ssapy_perturbations=True,
       models=[
           ssatk.SpacecraftFacetDrag(density=1e-12),
           ssatk.SpacecraftFacetSolRad([ssatk.AU, 0.0, 0.0]),
       ],
   )

This keeps SSATK aligned with SSAPy users while adding real body physics in a
way that can grow toward higher-fidelity spacecraft modeling.
