Launch, departure and heliocentric flight roadmap
=================================================

SSATK is to simulate a flight from the launch pad to orbit, through Earth
departure, and onward in heliocentric space, with all astrodynamics in SSATK
and the data it needs in the ``ssa-data-*`` packages. This page records what
exists, what is pending review, and what remains to be built. A capability
counts as complete only when its reference tests meet
``docs/testing_policy.md``.

Status
------

.. list-table::
   :header-rows: 1
   :widths: 22, 18, 60

   * - Capability
     - State
     - Notes
   * - Steps 2-7 of :doc:`launch_baseline`
     - On ``main``
     - Mission contract and preflight, time/frames/pad initialization, stage
       and tank mass accounting, changing-mass dynamics, propulsion operating
       conditions, launch weather and aerodynamic loads.
   * - Step 7 follow-up
     - On ``main``
     - State-only aerodynamic acceleration calls and their force/mass check.
   * - Step 8: mission phases and event resets
     - On ``main``
     - Phase graph, roots, staging and separation with conservation ledger,
       pad/rail constraints, terrain clearance.
   * - Step 9: sampled guidance, sensors and actuators
     - On ``main``
     - Sensor/estimator/guidance/controller loop and bounded actuators.
   * - Integration qualification
     - On ``main``
     - State-component tolerances, raw quaternion diagnostics and synthetic
       event/control convergence references; see :doc:`launch_baseline`.
   * - Insertion and orbit-handoff mechanism
     - On ``main``
     - ``Spacecraft.to_orbit`` retains the current mass, area and force
       coefficients. A synthetic ascent/cutoff/coast example checks overlap
       against native SSAPy propagation; mission-level insertion remains open.
   * - Integrated synthetic ascent and LEO acceptance
     - Implemented on a topic branch
     - ``demo_integrated_launch`` combines the existing pad, gravity, weather,
       aerodynamics, mass, staging and control functions in one propagation.
       Flight limits and final orbital elements are checked against declared
       synthetic requirements; source-backed flight acceptance remains open.
   * - Minor Planet Center small-body orbits
     - On ``main``
     - :mod:`ssatk.io.mpc`: catalogs, parsing, heliocentric two-body states.
   * - Earth-centred propagation and transfers
     - On ``main``
     - ``propagate_orbit_state`` (with STM), Hohmann/bi-elliptic/optimal
       transfers, zero-revolution Lambert in ``transfer_ssapy``.

The documented limits remain: the event/control examples use synthetic phase
boundaries, and the orbit-handoff reference starts above the atmosphere. The
handoff example includes native SSAPy gravity and optional perturbations, but
does not establish a pad-to-orbit mission. The integrated example supplies a
synthetic pad-to-LEO reference with bounded external attitude control and
noiseless state sensors. Vehicle-specific guidance, an attitude estimator,
dispersions, draining-tank slosh, flexibility and measured flight acceptance
remain separate work.

Remaining launch steps
----------------------

Step 10: integrated ascent dynamics
    One propagation from the pad that combines gravity (central body plus
    zonal terms), Earth rotation, the Step 7 atmosphere and winds,
    aerodynamics, thrust, Step 8 events and Step 9 control. Reference: closed
    forms for reduced cases and an independent integration of the same
    equations.
    The integrated synthetic reference is implemented in
    ``demos.launch.demo_integrated_launch``; see :doc:`launch_baseline`.

Step 11: launch vehicle definitions as data
    A schema and loader in :mod:`ssatk.launch` for complete vehicles: stages,
    engines (vacuum and sea-level thrust and specific impulse, throttle range,
    restarts), propellant loads and residuals, mass properties, aerodynamic
    coefficient tables, fairing and adapter, and the separation sequence.
    Vehicle data moves out of ``ssatk.launch.vehicles`` into a data package
    with per-value sources. Reference: published vehicle user guides.

Step 12: ascent guidance to a target orbit
    Open-loop pitch programs and gravity turns for the atmospheric phase, and
    closed-loop upper-stage guidance (for example powered explicit guidance)
    targeting orbital elements, in three axes, under dynamic-pressure,
    axial-acceleration and heating limits. Reference: closed-form optimal
    steering cases and published ascent profiles.

Step 13: insertion and orbit handoff
    Apply the existing six-degree-of-freedom (6-DoF) ascent-to-SSAPy handoff
    to a complete launch mission with target-orbit acceptance criteria.
    ``Spacecraft.to_orbit`` and the synthetic shared-arc overlap reference
    already exist; achieving the requested insertion orbit from the pad
    is demonstrated for the integrated synthetic LEO scenario. Source-backed
    vehicle performance and measured acceptance missions remain open.

Step 14: Earth departure
    Parking-orbit coast, launch window and azimuth from the site and the
    target asymptote, and a finite departure burn (upper-stage restart) to a
    target hyperbolic excess velocity. Report C3, declination and right
    ascension of the departure asymptote, and hand the state to heliocentric
    propagation at the sphere of influence or by continuous integration.

Step 15: performance envelopes
    Optimize trajectories to compute deliverable mass against C3 and against
    circular-orbit altitude and inclination for each vehicle. Reference:
    published performance curves stored as data, with stated agreement.

Step 16: dispersions and acceptance
    Monte Carlo over declared uncertainties, and acceptance against public
    launches with published insertion orbits or departure energies.

Supporting astrodynamics
------------------------

Heliocentric propagation
    Sun-centred propagation with planets and the Moon from the packaged
    DE440s/DE430 kernels (optional relativistic term), for spacecraft and
    small bodies, starting from :mod:`ssatk.io.mpc` epoch states. Reference:
    JPL Horizons vectors.

Interplanetary transfer design
    Lambert solutions for any central body with multiple revolutions,
    low/high path and prograde/retrograde branches; departure/arrival grids
    reporting departure C3 and asymptote, arrival relative velocity, phase and
    Sun-probe-Earth angles and Sun-distance extremes. Reference: independent
    Lambert implementations and propagated arcs.

Encounters
    Close-approach search between bodies, B-plane coordinates and minimum
    orbit intersection distance. Reference: JPL close-approach data.

Impulsive state changes on any body
    Apply a velocity change to a body at an epoch, re-propagate, and report
    the change in a later close-approach distance; state transition matrices
    for heliocentric propagation. Reference: zero-change identity, linear
    scaling for small changes, finite differences.

Constants
    ``EARTH_OBLIQUITY_J2000_DEG`` (23.439291) is rounded from the IAU value
    84381.448 arcsec; the difference moves ecliptic-to-equatorial positions by
    about 200 m at 1.4 au.

Deferred launch improvements
----------------------------

The integrated ascent and mission acceptance checks take priority. Retain
these next tasks in the order of the launch review:

3. Complete sourced vehicle configurations using the existing stage assembly,
   propulsion and data resolvers. Include geometry, inertia, aerodynamic
   coefficients and per-value provenance in SSAPy-Data (Step 11 above). The
   integrated Falcon 9 catalog sample still declares synthetic assumptions
   and an unresolved catalog mass difference; it does not complete this task.
4. Correct altitude interpolation in the local launch replay prototype.
   Sample telemetry through the existing propagator and
   coordinate functions while preserving staging discontinuities.
5. Preserve the staging example's body-to-GCRF attitude history and pass it
   through the existing satellite viewer's ``q`` argument.

Data packages
-------------

.. list-table::
   :header-rows: 1
   :widths: 25, 75

   * - Package
     - Needed content
   * - new ``ssa-data-launch``
     - Vehicle definitions (Step 11), aerodynamic coefficient tables,
       published performance curves (Step 15), acceptance-mission records
       (Step 16), each with a source.
   * - ``ssa-data-propulsion``
     - Liquid-engine thrust and specific impulse against ambient pressure and
       throttle for the engines the vehicle definitions name.
   * - ``ssa-data-core``, ``ssa-data-lunar``
     - Already ship Earth orientation and space weather (core) and the
       DE440s/DE430 ephemerides (lunar); confirm time coverage for departure
       epochs.
   * - ``ssa-data-benchmarks``
     - Reference trajectories for heliocentric propagation and departure
       checks.
