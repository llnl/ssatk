Benchmarking SSATK
==================

Purpose
-------

This document reviews Space Situational Awareness Toolkit (SSATK) against adjacent astrodynamics,
spacecraft-dynamics, mission-design, and flight-dynamics software. It is meant
to answer two practical questions:

1. What capabilities should SSATK provide directly?
2. What capabilities should SSATK intentionally leave to more specialized tools?

This is a capability benchmark and benchmark plan. It is not yet a numerical
performance benchmark with measured run times or accuracy residuals for every
tool. Performance and accuracy numbers should only be added after running a
controlled benchmark suite with fixed versions, input data, tolerances, and
hardware.

Review date: 2026-08-20.

Scope
-----

The review focuses on capabilities relevant to SSATK:

* orbital mechanics and transfer design,
* SSAPy interoperability,
* frame and coordinate conversions,
* plotting and analyst-facing workflows,
* data I/O,
* finite burns and maneuver representation,
* coupled six-degree-of-freedom (6-DoF) spacecraft dynamics,
* spacecraft body, actuator, and environment modeling,
* suitability for Python-first research workflows.

The review does not attempt to rank all flight-dynamics software globally.
Several tools below solve different problems: Basilisk is a spacecraft
simulation framework, GMAT is a mission-design tool, Orekit is a Java
astrodynamics library, SPICE is an ephemeris and geometry toolkit, and JSBSim is
an atmospheric flight-dynamics model. SSATK should integrate useful patterns
from these tools without trying to replace them.

Review Criteria
---------------

Each package is evaluated against these criteria:

``Orbit propagation``
    Point-mass, perturbed, multi-body, numerical, analytical, or semi-analytical
    propagation.

``Maneuvers``
    Impulsive burns, finite burns, low-thrust arcs, targeting, optimization, and
    staged transfer workflows.

``Attitude and 6-DoF``
    Attitude representation, attitude providers, rigid-body dynamics,
    force/torque coupling, and mass-property coupling.

``Spacecraft body modeling``
    Mass, inertia, center of mass, facets, plates, tanks, thrusters, wheels,
    flexible bodies, or other hardware models.

``Environment``
    Gravity, atmosphere, drag, solar radiation pressure, third bodies, eclipses,
    albedo, magnetic fields, and ephemerides.

``Frames and time``
    Inertial/body-fixed frames, local orbital frames, Earth orientation,
    ephemeris time systems, and coordinate conversion breadth.

``Visualization and workflow``
    Plotting, interactive graphics, examples, analyst usability, and scripting.

``Validation posture``
    Public tests, documented mathematical models, benchmark cases, flight-data
    comparisons, or operational heritage.

Executive Summary
-----------------

SSATK is strongest when it acts as the Python analyst layer around SSAPy:

* It should keep SSAPy as the primary high-fidelity orbit engine.
* It should provide convenient transfer, plotting, data, and workflow functions
  that are faster to use than lower-level astrodynamics libraries.
* It should add lightweight 6-DoF spacecraft dynamics for research workflows
  where a user needs attitude-dependent forces, torques, finite burns, and
  simple body models in the same Python package.
* It should not attempt to become Basilisk, Tudat, GMAT, Orekit, STK, or
  FreeFlyer.

The most defensible 6-DoF direction is:

* Use Basilisk's hub-plus-components pattern as the body-model reference.
* Use Tudat's separation between propagated state, environment, forces, torques,
  and mass as the dynamics-interface reference.
* Use GMAT, FreeFlyer, STK/Astrogator, and Orekit terminology for user-facing
  maneuver workflows.
* Use SPICE/NAIF-style ephemeris rigor through dependencies rather than
  rebuilding kernel handling in SSATK.
* Use JSBSim/RocketPy/OpenRocket as secondary references only for atmospheric
  and launch-vehicle body/aerodynamic patterns.

High-Level Capability Matrix
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 17 12 12 12 14 13 20

   * - Software
     - Orbit
     - Maneuver
     - 6-DoF
     - Body model
     - Python-first
     - SSATK relevance
   * - SSAPy
     - Strong
     - Strong
     - Limited
     - Limited
     - Yes
     - Core propagation engine for SSATK.
   * - SSATK
     - Growing
     - Growing
     - Early
     - Early
     - Yes
     - Analyst-facing workflows, plotting, transfers, I/O, and lightweight 6-DoF.
   * - Basilisk
     - Strong
     - Strong
     - Strong
     - Strong
     - Python interface
     - Best open architecture reference for spacecraft body dynamics.
   * - NASA 42
     - Strong
     - Moderate
     - Strong
     - Strong
     - No
     - Reference for full spacecraft attitude/orbit simulation scope.
   * - Tudat/TudatPy
     - Strong
     - Strong
     - Strong
     - Strong
     - Yes
     - Best astrodynamics reference for coupled state/force/torque setup.
   * - JEOD/Trick
     - Strong
     - Framework-dependent
     - Strong
     - Strong
     - No
     - Reference for high-fidelity NASA simulation decomposition.
   * - GMAT
     - Strong
     - Strong
     - Partial
     - Moderate
     - Script/API
     - Reference for operational maneuver and hardware concepts.
   * - Orekit
     - Strong
     - Strong
     - Partial
     - Moderate
     - Java-first
     - Reference for force models, frames, events, and maneuvers.
   * - FreeFlyer
     - Strong
     - Strong
     - Partial
     - Moderate
     - Proprietary
     - Reference for analyst-facing finite-burn workflows.
   * - STK/Astrogator
     - Strong
     - Strong
     - Partial
     - Moderate
     - Proprietary
     - Reference for targeting, mission design, and user workflow.
   * - MATLAB/Simulink Aerospace
     - Moderate
     - Moderate
     - Strong
     - Moderate
     - MATLAB-first
     - Reference for block-diagram 6-DoF and controls workflows.
   * - SPICE/NAIF
     - Ephemeris
     - No
     - No
     - No
     - Multi-language
     - Reference for time, frames, kernels, and geometry.
   * - poliastro
     - Moderate
     - Moderate
     - Limited
     - Limited
     - Yes
     - Python convenience reference for classical orbital mechanics.
   * - Astropy
     - Limited
     - No
     - No
     - No
     - Yes
     - Reference for units, time, coordinates, and astronomy interoperability.
   * - JSBSim
     - Atmospheric
     - Vehicle-specific
     - Strong
     - Strong
     - Python bindings
     - Secondary reference for configurable flight-dynamics bodies.
   * - RocketPy/OpenRocket
     - Launch-focused
     - Rocket-focused
     - Strong
     - Strong
     - Python/Java
     - Secondary reference for launch-vehicle aerodynamics and staging.

SSAPy
-----

Role
^^^^

SSAPy is the base package for high-fidelity orbital modeling in the LLNL
ecosystem. SSATK should continue to treat SSAPy as the authoritative engine for
core orbit propagation, force models, observer geometry, and orbit
determination.

Strengths
^^^^^^^^^

* High-fidelity Earth-orbit through cislunar propagation workflows.
* Force-model support for gravity, solar radiation pressure, drag, third-body
  perturbations, and maneuvers.
* Core objects such as ``Orbit`` and ``rv`` already define the user mental
  model for many SSATK users.
* Existing validation and operational use inside the LLNL ecosystem.

Limitations for SSATK's 6-DoF goals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* SSAPy is not primarily a spacecraft body-dynamics simulator.
* It does not provide a Basilisk-style hub/facet/thruster/tank/wheel component
  model as the central public abstraction.
* Attitude-dependent body forces and torque propagation are better developed in
  SSATK as an extension layer than forced into base SSAPy.

SSATK benchmark implication
^^^^^^^^^^^^^^^^^^^^^^^^^^^

SSATK should benchmark against SSAPy for:

* identical two-body propagation for point-mass cases,
* agreement with SSAPy acceleration models where SSATK wraps or mirrors them,
* correct behavior when accepting SSAPy ``Orbit`` objects as inputs,
* no user-facing ambiguity about whether a common constant or helper comes from
  SSAPy or SSATK.

SSATK
-----

Current role
^^^^^^^^^^^^

SSATK is the high-level, Python-first workflow package around SSAPy. It provides
plotting, orbital mechanics helpers, transfer design, data I/O, demo workflows,
coordinate conversions, and early 6-DoF dynamics.

Current strengths
^^^^^^^^^^^^^^^^^

* Single import path for common SSAPy-adjacent research workflows.
* Rich plotting and demo-gallery workflows.
* Transfer-design helpers that accept either SSAPy objects or raw inertial
  states.
* Data-access strategy that keeps reusable datasets in the split ``ssatk-data-*`` packages rather
  than embedding large files in the Toolkit repository.
* Coupled 6-DoF propagation with quaternion attitude, body angular rate,
  gravity-gradient torque, user-provided acceleration/torque models, attached
  body components, finite burns, mass properties, reaction wheels, and
  environment-backed forces.

Current limitations
^^^^^^^^^^^^^^^^^^^

* The extended appendage and slosh models are linear reduced-order models.
* External GMAT, STK/Astrogator, FreeFlyer, Orekit, Basilisk, and Tudat
  reference runs remain optional because their runtimes and force-model data
  are not consistently available in CI.
* The orbit-only STM uses the analytic central-gravity Jacobian for the
  unperturbed Kepler problem and finite-difference Jacobians when arbitrary
  perturbation callbacks are present. The coupled 6-DoF STM uses analytic
  central-gravity and fixed-inertia rigid-body Jacobians, including
  gravity-gradient torque partials, with finite-difference fallback for
  additional forces, torques, wheels, and mass flow. A multiplicative
  three-parameter body-frame attitude-error STM is available alongside the
  original quaternion-coordinate STM.

Measured baseline
^^^^^^^^^^^^^^^^^

On 2026-08-26, the SSATK Python 3.12 environment completed ``611 passed, 18
skipped`` in 273 seconds. The focused orbit and 6-DoF variational checks
completed ``20 passed`` after the analytic and attitude-error STM updates. The
full suite is the regression gate; optional external-tool comparisons are
evidence for individual capability claims and do not silently substitute for
analytical checks.

The optional Basilisk coupled 6-DoF case also ran on the same date using
Basilisk's ``Spacecraft`` and ``ExtForceTorque`` models for a 20 s, 41-sample
zero-gravity case with constant body-frame force and torque. The maximum
SSATK-minus-Basilisk residuals were ``4.0e-16 m`` in position, ``7.1e-17 m/s``
in velocity, ``3.4e-12`` in quaternion distance, and ``9.2e-17 rad/s`` in body
rate.

Orekit 10.3.1 also ran for the matched two-body case using its
``KeplerianPropagator`` over 3600 s with 60 s output spacing. Relative to the
SSATK DOP853 trajectory, the maximum position and velocity residuals were
``1.22e-3 m`` and ``1.03e-6 m/s``; the RMS residuals were ``5.81e-4 m`` and
``5.37e-7 m/s``. These residuals establish a numerical baseline for the
propagator pair, not a general claim about all Orekit or SSATK force models.

A matched perturbation case used Orekit 10.3.1's
``HolmesFeatherstoneAttractionModel`` with the EIGEN-6S field truncated to
degree 2 and order 0. The SSATK ``SpacecraftAccelJ2`` instance used the same
``mu = 3.986004415e14 m^3/s^2``, ``R = 6378136.460 m``, and
``J2 = 1.0826265227440867e-3`` derived from the fully normalized EIGEN-6S
``C20`` coefficient. Over a 5400 s, 91-sample, 63-degree low-Earth-orbit case,
the maximum SSATK-minus-Orekit position and velocity residuals were
``1.11e-7 m`` and ``1.05e-10 m/s``.

GMAT R2026a also ran through the configured Ubuntu 24.04 Podman container for
the matched Earth degree/order-0 JGM2 point-mass case over 600 s with 60 s
output spacing. The maximum position and velocity residuals were
``2.89e-3 m`` and ``3.11e-6 m/s``; the RMS residuals were ``2.11e-3 m`` and
``2.28e-6 m/s``. The GMAT case uses GMAT's JGM2 Earth gravitational parameter,
so its residuals include the documented ``mu`` convention difference from
SSATK's default Earth constant.

Target benchmark identity
^^^^^^^^^^^^^^^^^^^^^^^^^

SSATK should be benchmarked as:

* easier to use than lower-level libraries for common SSAPy workflows,
* less comprehensive but lighter-weight than Basilisk/Tudat for spacecraft body
  simulation,
* more Python-native and package-integrated than GUI/mission-design tools,
* accurate enough for research prototyping when compared against analytical
  checks and SSAPy baseline propagation.

Basilisk
--------

Role
^^^^

Basilisk is a high-fidelity spacecraft simulation framework. It is the strongest
open-source architecture reference for SSATK's 6-DoF development.

Relevant capabilities
^^^^^^^^^^^^^^^^^^^^^

* Rigid spacecraft hub with translational and rotational states.
* ``stateEffector`` and ``dynamicEffector`` architecture.
* Reaction wheels, thrusters, fuel tanks, hinged rigid bodies, radiation
  pressure, and facet drag modules.
* Multiple spacecraft and GNC-oriented simulation workflows.
* Python interface over a compiled simulation core.

Strengths relative to SSATK
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Much deeper spacecraft component modeling.
* Mature attitude dynamics and actuator-effectors architecture.
* Better reference for GNC, reaction wheels, flexible appendages, and complex
  spacecraft simulations.

Limitations relative to SSATK
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Heavier simulation-framework mindset.
* Not centered on SSAPy ``Orbit`` objects or SSATK plotting/data workflows.
* More setup overhead for simple analyst tasks.

SSATK benchmark implication
^^^^^^^^^^^^^^^^^^^^^^^^^^^

SSATK should not claim parity with Basilisk. The useful benchmark is narrower:

* single rigid body,
* one-facet and multi-facet drag/SRP,
* simple finite burn from one or more body-mounted thrusters,
* gravity-gradient torque sign and stability,
* mass and inertia bookkeeping for simple components.

NASA 42
-------

Role
^^^^

NASA 42 is a general-purpose spacecraft attitude and orbit dynamics simulator.
Its documented scope includes multi-body spacecraft attitude dynamics, rigid and
flexible bodies, multiple spacecraft, contact forces, and two-body or
three-body orbital regimes.

Strengths relative to SSATK
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Broader full-spacecraft simulation scope.
* Flexible-body and contact concepts are within its design envelope.
* Strong reference for attitude/orbit simulation terminology.

Limitations relative to SSATK
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Not a Python-first analysis package.
* Not integrated with SSAPy or SSATK plotting/data workflows.
* More specialized setup and configuration style.

SSATK benchmark implication
^^^^^^^^^^^^^^^^^^^^^^^^^^^

NASA 42 is useful as a scope boundary. SSATK should implement rigid-body
spacecraft dynamics well before considering flexible bodies, contact, or
multi-body appendage coupling.

Tudat and TudatPy
-----------------

Role
^^^^

Tudat/TudatPy is a comprehensive astrodynamics framework with strong Python
bindings. It supports translational dynamics, rotational dynamics, mass
propagation, multi-type propagation, torque setup, aerodynamic models, and
radiation-pressure models.

Strengths relative to SSATK
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Mature environment and propagation setup architecture.
* Coupled translational/rotational/mass propagation concepts.
* Richer dynamics and estimation ecosystem.
* Strong separation between environment settings, body settings, acceleration
  settings, torque settings, and propagated states.

Limitations relative to SSATK
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Larger conceptual surface area.
* Not built around SSAPy's API.
* More setup overhead for simple Toolkit workflows.

SSATK benchmark implication
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Tudat is the best reference for API boundaries:

* keep propagated states explicit,
* keep environment providers separate,
* keep force and torque models composable,
* support mass as a propagated or updated quantity,
* allow multi-arc or estimation-oriented extension later without forcing it
  into the first public API.

JEOD and Trick
--------------

Role
^^^^

NASA JEOD, used with the Trick Simulation Environment, is a high-fidelity
simulation system for spacecraft trajectory and attitude state. JEOD separates
environment, dynamics, interaction, and utility models and can couple to larger
simulation spaces with effectors and GNC systems.

Strengths relative to SSATK
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* NASA-grade simulation decomposition.
* Strong framework for coupling spacecraft dynamics to other simulation
  systems.
* Suitable for larger integrated simulations.

Limitations relative to SSATK
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Not a lightweight Python package.
* Requires a simulation-framework workflow.
* Not intended as a simple SSAPy-adjacent analyst utility.

SSATK benchmark implication
^^^^^^^^^^^^^^^^^^^^^^^^^^^

SSATK should borrow the model separation but not the simulation-executive
architecture. The right SSATK interface remains direct Python composition:
``Spacecraft`` plus body components plus force/torque models.

GMAT
----

Role
^^^^

GMAT is an open-source mission-analysis, optimization, and navigation tool. It
is strong for orbit propagation, targeting, finite burns, thrusters, tanks, and
mission-design scripts.

Relevant capabilities
^^^^^^^^^^^^^^^^^^^^^

* Numerical propagation with mission-design force models.
* Impulsive and finite burns.
* Chemical and electric thrusters.
* Tanks and mass depletion.
* Spacecraft hardware concepts.
* SPAD files for drag/SRP workflows.
* Torque reporting for selected environmental and maneuver sources.

Important limitation
^^^^^^^^^^^^^^^^^^^^

GMAT's torque-modeling design documentation states that torque computations are
available for reporting and are not included in propagation modeling in the
current design. That means GMAT should not be used as the primary reference for
coupled attitude dynamics in SSATK.

SSATK benchmark implication
^^^^^^^^^^^^^^^^^^^^^^^^^^^

GMAT is still a strong reference for user-facing maneuver concepts:

* ``Tank``,
* ``Thruster``,
* ``FiniteBurn``,
* active thrusters,
* mass flow,
* burn frame,
* thrust direction,
* maneuver reports.

Orekit
------

Role
^^^^

Orekit is a mature Java astrodynamics library with broad support for frames,
time systems, force models, maneuvers, event detection, propagation, estimation,
and attitude providers.

Relevant capabilities
^^^^^^^^^^^^^^^^^^^^^

* Force models that use state, attitude, mass, date, and frame.
* Atmospheric drag and solar radiation pressure with attitude dependence when a
  spacecraft shape is supplied.
* Impulsive maneuvers and continuous thrust maneuvers.
* Event detectors and state resets.
* Strong frame and time handling.
* Attitude providers for pointing laws and maneuver-specific attitude.

Limitations relative to SSATK
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Java-first.
* Attitude providers are not the same as a lightweight Python rigid-body
  spacecraft simulation API.
* Not integrated with SSAPy plotting and data workflows.

SSATK benchmark implication
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Orekit is the clearest reference for:

* event-aware maneuvers,
* force-model interfaces,
* mass-aware propagation states,
* separating attitude laws from attitude dynamics,
* state-reset workflows for impulsive maneuvers, now provided by
  ``ImpulseManeuver`` and ``propagate_spacecraft_segments``.

FreeFlyer
---------

Role
^^^^

FreeFlyer is a proprietary mission-design and operations tool. Its public
documentation describes attitude systems, quaternion attitude, angular velocity,
finite burns, active thrusters, tanks, burn durations, thrust steering, and burn
reference frames.

Strengths relative to SSATK
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Strong analyst-facing workflow design.
* Clear operational terminology for finite burns.
* Rich mission-design and targeting environment.

Limitations relative to SSATK
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Proprietary.
* Not a Python package.
* Not designed as an SSAPy extension layer.

SSATK benchmark implication
^^^^^^^^^^^^^^^^^^^^^^^^^^^

FreeFlyer is most useful as a vocabulary benchmark. SSATK should use terms that
analysts already recognize: ``FiniteBurn``, ``Thruster``, ``Tank``, ``LVLH``,
``VNB``, ``ICRF/GCRF``, ``burn direction``, ``specific impulse``, and
``mass depletion``.

STK/Astrogator
--------------

Role
^^^^

STK and Astrogator are proprietary mission-design, targeting, visualization, and
analysis tools. They are relevant to SSATK because many mission analysts expect
targeting workflows, maneuver sequences, finite burns, and interactive
visualization patterns similar to STK.

Strengths relative to SSATK
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Mature GUI and operational workflow.
* Strong trajectory targeting and mission sequence design.
* Integrated visualization and scenario analysis.

Limitations relative to SSATK
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Proprietary and license-dependent.
* Not a lightweight package dependency.
* Public documentation access can be version- and license-dependent.

SSATK benchmark implication
^^^^^^^^^^^^^^^^^^^^^^^^^^^

SSATK should not try to become STK. The useful benchmark is whether common
scripted tasks are easy:

* define initial and target states,
* choose a transfer objective,
* specify burn constraints,
* plot the result,
* save artifacts reproducibly.

MATLAB, Simulink, and Aerospace Blockset
----------------------------------------

Role
^^^^

MATLAB/Simulink Aerospace tools are important references for controls,
simulation, block-diagram workflows, and 3-DoF/6-DoF equations of motion. They
are commonly used for GNC prototyping and vehicle simulation.

Strengths relative to SSATK
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Strong control-system workflow.
* Block-diagram integration with sensors, actuators, and flight software
  prototypes.
* Well-developed 6-DoF vehicle simulation patterns.

Limitations relative to SSATK
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Proprietary.
* MATLAB/Simulink-first rather than Python-first.
* Not designed around SSAPy objects or SSATK plotting/data conventions.

SSATK benchmark implication
^^^^^^^^^^^^^^^^^^^^^^^^^^^

MATLAB/Simulink is a useful reference for equations-of-motion clarity and GNC
interoperability, but not for SSATK's package architecture. SSATK should remain
plain Python/NumPy unless there is a specific need for external co-simulation.

SPICE and NAIF
--------------

Role
^^^^

SPICE is the standard reference for ephemerides, geometry, time systems, frames,
and kernels. It is not an orbit propagator or 6-DoF spacecraft simulator.

Strengths relative to SSATK
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Authoritative kernel-based ephemeris and geometry infrastructure.
* Mature time and frame conventions.
* Multi-language support and extensive technical references.

Limitations relative to SSATK
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Does not solve SSATK's high-level plotting, transfer, data, or 6-DoF workflow
  needs directly.
* Kernel management is a separate operational concern.

SSATK benchmark implication
^^^^^^^^^^^^^^^^^^^^^^^^^^^

SSATK should use validated ephemeris and time/frame sources through dependencies
instead of reimplementing SPICE. Benchmarks involving Sun, Moon, planets, and
eclipses should document whether the positions come from SSAPy, Astropy, SPICE,
or packaged data.

poliastro
---------

Role
^^^^

poliastro is a Python astrodynamics package focused on ease of use for orbital
mechanics, Lambert solutions, propagation, element/state conversions, and
plotting.

Strengths relative to SSATK
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Clean Python user experience.
* Useful examples for classical orbital mechanics workflows.
* Plotting and educational workflows.

Limitations relative to SSATK
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Not centered on SSAPy.
* Not a high-fidelity spacecraft body-dynamics package.
* Development status and dependency compatibility should be checked before any
  direct comparison.

SSATK benchmark implication
^^^^^^^^^^^^^^^^^^^^^^^^^^^

poliastro is a good usability reference for simple transfers and plotting. SSATK
should exceed it for SSAPy-specific workflows and cislunar/SSA-oriented plots,
not necessarily for all general orbital mechanics examples.

Astropy
-------

Role
^^^^

Astropy is a core astronomy infrastructure package. It is relevant for units,
time, coordinates, tables, and interoperability, not as a mission-design or 6-DoF
propagator.

Strengths relative to SSATK
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Time and coordinate infrastructure.
* Units and table ecosystem.
* Large scientific Python user base.

Limitations relative to SSATK
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Not a spacecraft dynamics package.
* Not a transfer-design package.
* Not specific to SSAPy.

SSATK benchmark implication
^^^^^^^^^^^^^^^^^^^^^^^^^^^

SSATK should interoperate cleanly with Astropy types where practical, but should
avoid forcing Astropy units into every low-level numerical path when plain NumPy
arrays are faster and clearer.

JSBSim
------

Role
^^^^

JSBSim is a configurable flight dynamics model for aircraft, rockets, and other
vehicles. It is a strong reference for aerodynamic and propulsion component
configuration, but its primary domain is atmospheric flight.

Strengths relative to SSATK
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Mature 6-DoF atmospheric flight dynamics.
* Configurable aircraft/rocket body, propulsion, landing gear, and flight
  control systems.
* Python bindings and simulation examples.

Limitations relative to SSATK
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Not an orbital/cislunar spacecraft toolkit.
* Not centered on SSAPy.
* Atmospheric aircraft abstractions are not the right default for satellite
  body dynamics.

SSATK benchmark implication
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use JSBSim as a reference only when SSATK develops atmospheric ascent,
reentry, or aerodynamics-heavy workflows.

RocketPy and OpenRocket
-----------------------

Role
^^^^

RocketPy and OpenRocket are launch-vehicle and model/high-power rocketry
simulation tools. They include 6-DoF flight simulation, aerodynamic models,
staging or motors, and atmosphere/wind concepts.

Strengths relative to SSATK
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Strong rocket-specific body and propulsion workflows.
* Launch and atmospheric flight validation examples.
* Useful patterns for staging and aerodynamic coefficients.

Limitations relative to SSATK
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Different primary domain.
* Not designed around long-duration orbital propagation or SSAPy.
* Rocket-specific abstractions should not dominate satellite 6-DoF APIs.

SSATK benchmark implication
^^^^^^^^^^^^^^^^^^^^^^^^^^^

These tools are useful if SSATK's ``launch`` helpers grow into a serious
launch/ascent module. They are secondary references for spacecraft 6-DoF.

Benchmark Cases SSATK Should Run
--------------------------------

SSATK should use a benchmark suite with analytical checks first, then SSAPy and
external-tool comparisons where possible.

Core propagation cases
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 32 23 23

   * - Case
     - Purpose
     - Expected check
     - Comparison target
   * - Two-body circular orbit
     - Validate translational integrator and units.
     - Radius, speed, energy, and angular momentum remain constant within tolerance.
     - Analytical solution and SSAPy.
   * - Two-body elliptical orbit
     - Validate eccentric propagation.
     - Perigee/apogee and orbital period match Keplerian expectation.
     - Analytical solution and SSAPy.
   * - J2 nodal precession
     - Validate perturbing acceleration sign and magnitude.
     - RAAN drift matches first-order J2 theory for a near-circular orbit.
     - Analytical J2 approximation and SSAPy.
   * - Third-body perturbation
     - Validate differential third-body acceleration.
     - Perturbation points in the expected direction and scales with distance.
     - SSAPy or Tudat-style reference case.
   * - Low-thrust constant NTW burn
     - Validate frame conversion and accumulated delta-v.
     - Integrated acceleration equals expected delta-v for short burn.
     - Analytical short-arc check.
   * - Variable finite burn
     - Validate operational thrust profiles and frame selection.
     - Integrated acceleration matches thrust impulse divided by mass.
     - Analytical constant/trapezoid/pulsed profile checks.

Attitude and 6-DoF cases
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 32 23 23

   * - Case
     - Purpose
     - Expected check
     - Comparison target
   * - Quaternion normalization
     - Ensure attitude state remains valid.
     - Quaternion norm remains one within tolerance.
     - Internal invariant.
   * - Torque-free rigid body
     - Validate Euler rigid-body equations.
     - Angular momentum and kinetic energy remain constant.
     - Analytical invariant.
   * - Constant body torque
     - Validate angular acceleration.
     - ``omega_dot = I^-1 tau`` for small rates.
     - Analytical short-time check.
   * - Gravity-gradient torque
     - Validate sign and stability behavior.
     - Torque is zero for spherical inertia and expected for asymmetric inertia.
     - Textbook gravity-gradient equation.
   * - Body-frame thrust
     - Validate attitude-dependent translation.
     - Rotating attitude rotates the applied acceleration into inertial frame.
     - Analytical frame check.

Body and component cases
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 32 23 23

   * - Case
     - Purpose
     - Expected check
     - Comparison target
   * - One flat plate drag
     - Validate projected area and drag direction.
     - Force opposes atmosphere-relative velocity and torque is ``r x F``.
     - Current SSATK single-plate helper.
   * - One flat plate SRP
     - Validate illumination and projected area.
     - Force direction and magnitude match solar pressure formula.
     - Current SSATK single-plate helper.
   * - Box-wing facet model
     - Validate multi-facet aggregation.
     - Symmetric body cancels torques when expected.
     - Internal symmetry check.
   * - Canted thruster
     - Validate force/torque coupling.
     - Force follows body direction; torque equals moment arm cross force.
     - Analytical check.
   * - Thruster mass flow
     - Validate mass-flow and sign convention.
     - Reported ``mdot`` is positive and equals thrust divided by ``Isp * g0``.
     - Analytical check.
   * - CSV thrust curve
     - Validate external/open-data thrust-curve ingestion without committing data.
     - Interpolated thrust and total impulse match the source table.
     - Tabulated open-data curve or synthetic table fixture.
   * - Parallel-axis inertia update
     - Validate component inertia bookkeeping.
     - Inertia contribution matches parallel-axis theorem.
     - Analytical check.

Workflow cases
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 32 23 23

   * - Case
     - Purpose
     - Expected check
     - Comparison target
   * - SSAPy ``Orbit`` input
     - Validate Toolkit entry-point compatibility.
     - Same propagation as raw ``r, v`` inputs.
     - Internal equivalence.
   * - Transfer accepts objects and states
     - Validate standardized transfer APIs.
     - ``Orbit`` and ``r1, v1, r2, v2`` forms produce equivalent boundary states.
     - Internal equivalence.
   * - Plot accepts time-series states
     - Validate visualization API.
     - Static and animated outputs use same trajectory.
     - File existence and geometry sanity checks.
   * - Demo gallery smoke run
     - Validate user-facing examples.
     - All included demos produce expected artifacts or skip gracefully.
     - CI smoke test.
   * - Repository policy
     - Validate data hygiene.
     - No generated figures, binary data, or large files enter the source repo.
     - Policy script.

Performance Metrics to Report
-----------------------------

Performance numbers should be reported only after a controlled run. Each table
should include:

* package version,
* Python version,
* platform and CPU,
* BLAS/SciPy stack if relevant,
* tolerances,
* number of state evaluations,
* wall-clock median over at least five runs,
* minimum and maximum run time,
* memory peak if available,
* final position/velocity residual against the reference,
* final attitude residual for 6-DoF cases,
* mass residual for finite-burn cases.

Validation profile
^^^^^^^^^^^^^^^^^^

``ssatk-benchmark --profile validation`` runs the four analytical cases in the
``validation`` registry group: two-body invariants, J2 nodal precession,
torque-free rigid-body invariants, and finite-burn mass/delta-v/impulse. These
cases retain timing output, but timing statistics never affect validation
acceptance. The profile exits with status 1 for an execution failure or a
failed validation metric. Use ``--fail-on-validation`` to apply the same gate
to another profile.

Each result contains a ``validation`` mapping whose metrics have the stable
machine-readable schema ``value``, ``unit``, ``tolerance``, ``pass``, and
``reference``. ``value`` is normally a residual compared with zero; the
validator records any non-residual comparison explicitly in ``pass``. The
result also contains ``validation_pass`` and, for analytical cases,
``validation_settings`` with solver method, tolerances, and step limits. JSON
metadata records the SSAPy and SciPy versions alongside the SSATK and NumPy
versions.

Continuous integration and release publishing run the reproducible gate as::

   python -m ssatk.benchmark --profile validation --quick --no-dashboard --fail-on-validation --output-dir .ci-output/benchmark-validation

The resulting ``benchmark_results.json`` is retained as a workflow artifact.

External validation report
^^^^^^^^^^^^^^^^^^^^^^^^^^

The optional Orekit, GMAT, and Basilisk comparisons share a deterministic JSON
reporter. It runs fixed short cases without downloading runtimes or generating
figures, records each model, constant, residual, tolerance, and skip reason,
and writes the report outside the source tree by default::

   python scripts/external_validation_report.py

Use ``--require-external`` in an environment that provides all three runtimes;
the command then exits nonzero for a skipped runtime as well as for a failed
residual gate. ``--full`` selects the longer fixed cases.

For SSATK, the first performance table should compare:

* fixed-step RK4,
* fixed-step leapfrog,
* adaptive DOP853 translational propagation,
* 6-DoF propagation with no torque,
* 6-DoF propagation with gravity-gradient torque,
* 6-DoF propagation with facet drag/SRP once implemented.

Benchmark Acceptance Criteria
-----------------------------

Initial acceptance criteria should be strict enough to catch sign and unit
errors but not so strict that they depend on one machine or one solver:

* two-body circular-orbit energy drift below a documented tolerance,
* J2 RAAN drift within a documented fractional tolerance of first-order theory,
* quaternion norm within numerical tolerance after propagation,
* torque-free angular momentum conserved within numerical tolerance,
* flat-plate drag/SRP force direction correct for multiple attitudes,
* finite-burn mass depletion agrees with the Tsiolkovsky or constant-mdot check
  for simple cases,
* benchmark scripts produce machine-readable JSON or CSV summaries,
* generated plots are written outside the repository by default.

SSATK Positioning
-----------------

SSATK should be described as:

* an SSAPy-centered research toolkit,
* a convenience and workflow layer for orbital mechanics, plotting, transfer
  design, data I/O, and demonstrations,
* a lightweight 6-DoF spacecraft dynamics layer for early-stage research and
  analysis,
* not a replacement for Basilisk, Tudat, GMAT, Orekit, STK, FreeFlyer, SPICE, or
  launch-vehicle flight-dynamics software.

Recommended Next Development Steps
----------------------------------

1. Extend analytic Jacobians to additional built-in attitude-dependent force
   models, wheel states, and variable mass properties.
2. Convert each benchmark case in this document into a machine-readable
   acceptance result with explicit units, tolerances, solver settings, and
   reference provenance.
3. Extend the existing optional GMAT, Orekit, and Basilisk comparisons to
   matched force-model cases when their runtimes are available; report residuals
   rather than capability labels alone.
4. Profile representative propagation cases on fixed hardware and optimize
   only measured bottlenecks. Preserve the current direct Python composition
   API and avoid adding a simulation executive.

References
----------

Primary public references reviewed:

* SSAPy: https://github.com/llnl/SSAPy
* SSATK: https://github.com/LLNL/ssatk
* NASA 42: https://github.com/ericstoneking/42
* Basilisk: https://github.com/AVSLab/basilisk
* TudatPy: https://github.com/tudat-team/tudatpy
* Tudat: https://github.com/tudat-team/tudat
* NASA JEOD: https://github.com/nasa/jeod
* NASA Trick: https://github.com/nasa/trick
* NASA GMAT: https://github.com/nasa/GMAT
* Orekit: https://www.orekit.org/
* FreeFlyer documentation: https://ai-solutions.com/_help_Files/
* Ansys STK: https://www.ansys.com/products/missions/ansys-stk
* SPICE/NAIF: https://naif.jpl.nasa.gov/naif/toolkit.html
* poliastro: https://github.com/poliastro/poliastro
* Astropy: https://github.com/astropy/astropy
* JSBSim: https://github.com/JSBSim-Team/jsbsim
* RocketPy: https://github.com/RocketPy-Team/RocketPy
* OpenRocket: https://github.com/openrocket/openrocket

This reference list should be updated when benchmark numbers are added. Tool
capabilities change over time, and any numerical comparison should record exact
versions and configuration files.
