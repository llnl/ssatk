SSA Workflows
=============

This guide documents the small, deterministic workflow layer in
``ssatk.ssa``.  The examples under ``demos/sensor_coverage``,
``demos/orbital_mechanics``, and ``demos/navigation`` use synthetic states and
run without a catalog download or a raw-data file.  They are API examples, not
validated operational screening, collision-risk, or orbit-determination
products.  Public-release approval is still required before treating a demo,
figure, or derived result as a release artifact.

Conventions
-----------

States in these APIs are six-component or position/velocity GCRF vectors in
SI units: metres and metres per second.  Epochs are strictly increasing GPS
seconds (Astropy ``Time`` values are accepted where the time API accepts
them).  ``OrbitStateProvider`` adapts existing ``Orbit`` objects and requires
an explicit propagator, for example ``KeplerianPropagator()``.  Its replay
contract is deterministic: a detailed metric pass must use the same provider,
observer callback, and illumination callback used for analysis.

``pair_metrics`` defines relative quantities as B minus A.  Its RTN frame is
centered on A.  Apparent-sky angles use the observer-to-object lines of sight;
the along-track direction is the target A apparent velocity after removing its
line-of-sight component.  If that projected velocity is singular, the
corresponding sky angles are invalid even when scalar separation is finite.
Observer velocity is used for apparent sky motion, not for the pair velocity
cosine.  RTN relative velocity is the instantaneous B-minus-A vector projected
onto A's RTN axes; it is not the time derivative of components in a rotating
RTN frame.

Pair screening and workflows
----------------------------

``screen_pairs`` is a one-epoch angular/velocity candidate screen.
``analyze_pairs`` samples a trajectory provider, applies the screen, and
aggregates trigger samples into padded ``EventWindow`` objects.  Angular
acceptance is inclusive (``<= angular_limit_rad``); velocity-cosine acceptance
is strict (``> velocity_threshold``).  Absolute GCRF object velocities are
used for the velocity cosine.  An angular event is not a conjunction and does
not imply collision risk.

Access and illumination restrictions are opt-in.  A minimum elevation gate
requires observer ENU geometry.  The optional occultation gate is a spherical
Earth line-of-sight test; it is not an ellipsoid, terrain, atmosphere,
refraction, or detectability model.  An illumination callback must return an
object-by-time boolean mask and must be replayable exactly.  Invalid state or
observer geometry is excluded from triggering; padded context can still
contain invalid or gated-out samples, so a window is not a continuous
detection interval.

Irregular sampling is supported.  Set ``max_gap_s`` when a sampling gap must
close an active event.  Padding and windows are clipped to the declared
analysis span.  Chunk boundaries are processing boundaries, not event
boundaries; the accumulator preserves continuity across chunks.  A cancelled
run returns ``complete=False`` and must not be presented as complete.

Candidate generation uses velocity-direction or sightline spatial trees, with
an adaptive choice, and has a brute-force reference strategy.  Tree screening
does not materialize all :math:`O(N^2)` candidates, but the output itself has
inherent :math:`K` memory and active event windows require memory proportional
to their number.  ``workers>1`` opts into ordered thread screening; use it
only when the workload benefits.  Deterministic providers, callbacks, and
serial aggregation preserve replayable results.

Minimal workflow
----------------

Construct orbit objects with existing SSATK readers or synthetic elements;
the following example uses no downloaded catalog:

.. code-block:: python

   import numpy as np
   import ssatk
   from ssatk.navigation import GroundStation
   from ssatk.ssa import (
       OrbitStateProvider, PairAnalysisConfig, analyze_pairs, from_ground_station,
   )

   times = 1.4e9 + np.arange(0.0, 601.0, 30.0)
   orbits = {
       name: ssatk.Orbit.fromKeplerianElements(
           a=7e6, e=0, i=0, pa=0, raan=0, trueAnomaly=phase, t=times[0],
       )
       for name, phase in (("A", 0.0), ("B", 0.01))
   }
   provider = OrbitStateProvider(
       orbits, times, propagator=ssatk.KeplerianPropagator(),
       provenance={"input": "synthetic circular orbits"},
   )
   station = GroundStation(-106.5, 35.1, elevation_m=1600.0)
   observer = lambda block: from_ground_station(station, block)
   result = analyze_pairs(
       provider, observer, config=PairAnalysisConfig(), chunk_size=10,
   )

Detailed metrics and I/O
------------------------

``iter_event_metrics`` replays the provider's full declared timeline in time
blocks to verify exact coverage and deterministic replay, but yields bounded
metric batches only for sampled rows inside event context windows.
``estimate_event_rows`` gives the exact number of sampled rows before
extraction, including the window endpoints that are present in the declared
time vector.  The I/O helpers are separate from analysis:

.. code-block:: python

   from ssatk.io.ssa_events import write_event_metrics, write_event_summary
   from ssatk.ssa.workflows import estimate_event_rows, iter_event_metrics

   write_event_summary(result, "runs/events.json")
   batches = iter_event_metrics(result, provider, observer)
   write_event_metrics(batches, "runs/event_metrics.csv")

Both writers use atomic replacement.  Summary JSON records completeness,
configuration, provenance, and windows.  Metrics CSV is streamed and contains
valid/trigger flags plus scalar and three-component vector columns.  These
helpers write files under the normal SSATK output root for relative paths.  A
failed write, including a replay whose schema changes part way through, removes
the temporary file and leaves the previous target untouched.  A replay ending
before the provider's declared epochs, or otherwise failing its exact coverage
contract, must be treated as incomplete rather than summarized as complete.
The declared timeline is checked before yielding; a chunk-independent SHA-256
fingerprint of states, observer geometry and applied masks is checked when the
metric iterator is exhausted. A consumer must exhaust it successfully before
accepting its output. Cancelled analyses replay only their processed prefix,
using the same originally declared source timeline. The atomic CSV writer
does not replace its target if fingerprint verification fails.
The demos deliberately do not call these writers.

Conjunction tools
-----------------

For actual trajectory conjunction analysis, use
``coarse_conjunction_screen`` followed by ``refine_closest_approach`` or use
``catalog_conjunction_screen`` for an insertion-ordered mapping of
trajectories.  These tools refine a B-minus-A miss distance over overlapping
trajectory intervals.  Their output is separate from angular/velocity events;
the orbital-mechanics demo shows both results side by side without equating
them.

Navigation example
------------------

``GroundStationSensor`` produces ``StationObservation`` objects from the
existing ``GroundStation`` prediction model.  ``solve_batch_orbit`` consumes
valid station observations, propagates a perturbed initial GCRF state, and
returns a fit plus STM-based diagnostics.  The synthetic example uses
actual ``GroundStation`` Earth-orientation geometry and noise-free values with
positive definite reported covariance only to make the least-squares weighting
well-defined.  The reported fitting weights are not a physical covariance
truth.  It does not infer or validate a real
covariance budget, sensor calibration, observability claim, or operational
accuracy.

Run the examples
----------------

.. code-block:: bash

   python -m demos.sensor_coverage.demo_observer_pair_screening
   python -m demos.orbital_mechanics.demo_pair_screening_and_conjunctions
   python -m demos.navigation.demo_synthetic_orbit_estimation

For offline validation without writing PNGs, call each module's
``main(make_figures=False)``.  The first demo also has
``benchmark=True`` to compare velocity, sightline, adaptive, and brute-force
candidate strategies on the same small synthetic state batch.  The benchmark
uses ``perf_counter`` and is a local comparison, not a performance guarantee.
