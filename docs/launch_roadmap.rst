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
     - Branch ``feature/launch-atmosphere-aerodynamics``
     - State-only aerodynamic acceleration calls and their force/mass check;
       not yet on ``main``.
   * - Step 8: mission phases and event resets
     - Pull request #88
     - Phase graph, roots, staging and separation with conservation ledger,
       pad/rail constraints, terrain clearance.
   * - Step 9: sampled guidance, sensors and actuators
     - Pull request #91
     - Sensor/estimator/guidance/controller loop and bounded actuators. Carries
       a different copy of the Step 8 commit; update it after #88 merges.
   * - Minor Planet Center small-body orbits
     - On ``main``
     - :mod:`ssatk.io.mpc`: catalogs, parsing, heliocentric two-body states.
   * - Earth-centred propagation and transfers
     - On ``main``
     - ``propagate_orbit_state`` (with STM), Hohmann/bi-elliptic/optimal
       transfers, zero-revolution Lambert in ``transfer_ssapy``.

After Steps 8 and 9 merge, the documented limits remain: the demo insertion
boundary is not an achieved orbit; there is no three-axis launch guidance or
attitude estimator; the integrated demos exclude gravity, weather and Earth
rotation; dispersions, slosh, flexibility and an acceptance mission are
separate work.

Remaining launch steps
----------------------

Step 10: integrated ascent dynamics
    One propagation from the pad that combines gravity (central body plus
    zonal terms), Earth rotation, the Step 7 atmosphere and winds,
    aerodynamics, thrust, Step 8 events and Step 9 control. Reference: closed
    forms for reduced cases and an independent integration of the same
    equations.

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
    End the 6-DoF ascent at insertion and continue with
    ``propagate_orbit_state``, with an overlap check that both models agree
    on a shared arc.

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
