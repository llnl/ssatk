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

Joint orbit-and-parameter estimation
------------------------------------

``solve_batch_orbit`` also accepts ``parameters0``, a mapping of named scalar
guesses. The six Cartesian state components are always fitted. Parameters
follow the mapping's insertion order; ``parameter_scale`` contains their
positive characteristic magnitudes and ``parameter_bounds`` contains lower
and upper arrays in physical units. ``parameter_step`` optionally supplies
absolute numerical-differentiation steps in those physical units. Choose steps
large enough for their observation changes to exceed integration error.

Supply ``propagation_kwargs`` as a callable of the parameter mapping to rebuild
existing force models with candidate drag/SRP coefficients or finite-burn
thrust, duration, or direction parameters. It returns the usual propagation
options and cannot override the initial state, epoch, STM, or dense output.
The existing propagator handles force-model ``time_breakpoints`` so a fitted
burn boundary is evaluated again for every candidate. Keep fixed force
components and all physical data explicit. Reuse SSAPy/SSATK models rather
than implementing a second force-model engine.

``measurement_bias(station, observation, parameters)`` returns an additive
bias with the observable's dimension, in metres, metres/second, radians, or
hertz as appropriate. The observation's known ``bias`` is corrected first;
this callback models an additional unknown bias. It can select parameters by
station, observable, and observation epoch. Parameter mappings are read-only.

.. code-block:: python

   import numpy as np
   from ssapy import AccelDrag, AccelSolRad
   from ssatk.accelerations_6dof import make_finite_burn_acceleration
   from ssatk.navigation import solve_batch_orbit

   # Ground-station observations, initial state, epoch, and burn epochs are inputs.
   drag = AccelDrag(area=10, mass=100)
   srp = AccelSolRad(area=10, mass=100)

   def propagation(parameters):
       def drag_force(t, r, v):
           return drag(r, v, t, CD=parameters["cd"])

       def srp_force(t, r, v):
           return srp(r, v, t, CR=parameters["cr"])

       burn = make_finite_burn_acceleration(
           parameters["thrust_n"], mass=100, frame="inertial",
           direction=[0, 0, 1], start=burn_start, stop=burn_stop,
       )
       return {"acceleration": (drag_force, srp_force, burn)}

   def bias(station, observation, parameters):
       if observation.measurement == "range":
           return parameters["range_bias_m"]
       return np.zeros_like(np.atleast_1d(observation.value))

   fit = solve_batch_orbit(
       observations, state_guess, t0=state_epoch,
       parameters0={"cd": 2.2, "cr": 1.3, "thrust_n": 0.2, "range_bias_m": 0},
       parameter_scale=[2, 1, 0.2, 10],
       parameter_step=[0.002, 0.002, 0.0002, 0.1],
       parameter_bounds=([0.1, 0.1, 0, -100], [5, 3, 1, 100]),
       propagation_kwargs=propagation, measurement_bias=bias,
   )
   print(fit.parameters, fit.estimate_names, fit.correlation)

``fit.joint_covariance`` includes state/parameter cross-covariances in the
order given by ``fit.estimate_names``. ``fit.covariance`` remains the 6-by-6
state block for existing callers. ``fit.correlation`` is the dimensionless
joint matrix. The covariance is local inverse Fisher information using the
supplied observation noise; it is not multiplied by the fitted residual
variance. Insufficient numerical rank makes ``success`` false and leaves
both covariance matrices and correlation unavailable. ``active_mask`` reports
lower-bound (-1), interior (0), and upper-bound (1) estimates; formal Gaussian
covariance does not model truncation at active bounds.

State derivatives use the existing STM. Geometric residuals and perturbed
parameter trials use the existing state-only propagator. Named derivatives use
bounded finite differences that repropagate the orbit and re-solve radio
signal times, including Doppler count intervals. They require more propagation
work than a state-only fit. Set propagation accuracy and parameter scales so
parameter-induced changes exceed integration roundoff; verify that smaller
steps give consistent uncertainties. Identifiability still
depends on the observation geometry and arc, physical assumptions, and noise.
Synthetic references do not establish measured coefficient or flight accuracy.

Run the controlled example with:

.. code-block:: bash

   python -m demos.navigation.demo_joint_orbit_parameter_estimation

Coherent radio tracking
-----------------------

``GroundStation.predict_radio`` computes vacuum ``two_way_range`` in metres
and ``two_way_doppler`` in hertz. Both use a ground transmit event, separate
spacecraft receive/transmit events, and a ground receive event. A positive
fixed transponder delay moves the spacecraft reception before retransmission;
the spacecraft is propagated at both events. Existing ``range`` and
``range_rate`` observables retain instantaneous geometry.
Signal paths include Earth's barycentric position and velocity at each event,
using the DE430 kernel already supplied through SSAPy's planetary ephemerides.
GPS epochs and signal offsets are converted to two-part TDB kernel epochs.

.. code-block:: python

   import numpy as np
   from ssatk.navigation import GroundStation, RadioTrackingConfig
   from ssatk.propagators_orbit import propagate_orbit_state_with_stm

   t0 = 1.4e9
   station = GroundStation(-104.9, 39.7, elevation_m=1600)
   radio = RadioTrackingConfig(
       uplink_frequency_hz=7.2e9, count_interval_s=10,
       turnaround_ratio=880 / 749, transponder_delay_s=2e-6,
   )
   trajectory = propagate_orbit_state_with_stm(
       r0=[7e6, 0, 0], v0=[0, 7500, 1200], t0=t0,
       times=[t0, t0 + 60], dense_output=True,
   )
   prediction = station.predict_radio(trajectory, t0 + 60, "two_way_doppler", radio)

The timestamp is ground reception. Range is half the full round-trip distance,
including ``c * transponder_delay_s / 2``. Doppler is mean received frequency
minus the coherent reference ``turnaround_ratio * uplink_frequency_hz``, over
the interval ending at that timestamp; recession gives a negative shift.
It is calculated from the change in solved signal paths, not from a single
instantaneous range rate. GMAT's generic ``Range`` is a full path in km;
GMAT ``DSN_TCP`` uses a different carrier/reference convention. Convert those
conventions explicitly when comparing outputs.

``GroundStationSensor(..., measurement="two_way_doppler", radio=radio)``
accepts a dense state/STM trajectory as ``truth`` in ``measure(truth, time)``.
Its ``StationObservation`` retains the immutable radio configuration, including
the count interval. Sensor covariance and bias use metres for range and hertz
for Doppler. ``solve_batch_orbit`` consumes these observations alongside the
existing geometric types, sharing one dense propagation per candidate state.
Supply an explicit ``t0`` early enough to cover all spacecraft events and the
beginning of every Doppler count. Out-of-coverage signal events are rejected;
trajectories are never extrapolated. ``trajectory.at(time, offset_s=...)`` keeps
signal offsets separate from the large absolute GPS epoch.

For an EKF, ``GroundStationMeasurement(station, receive_time, measurement,
radio=radio, t0=state_epoch, propagation_kwargs=...)`` returns a prediction and
its Jacobian with respect to the state at ``state_epoch``. That epoch must
precede the signal/count interval. Update an EKF holding a state at that epoch,
then propagate it forward; setting the filter timestamp to receive time alone
would not propagate its state. Batch fitting likewise includes the dependence
of the solved signal times on the initial state in its derivatives.

These are constant-frequency, single-station, two-way vacuum links. They
exclude three-way links, frequency ramps, media corrections, relativistic
terms, clocks, and proper-time hardware calibration. Visibility checks use
station elevations on both legs and, for Doppler, both count endpoints; they
do not search for an interruption within a count interval. The synthetic demo
uses an all-sky mask and specified noise-free observations with artificial
fitting weights. Analytic references and finite differences validate this
model; observed tracking data is still needed for operational accuracy.

Tracking Data Message ingestion
-------------------------------

``ssatk.io.read_tdm`` reads CCSDS TDM 1.0 and 2.0 KVN or XML from a path,
stream, or message text. It retains header/segment metadata, comments,
original epochs and lexical observable values. Calendar and ordinal dates
use the existing CCSDS time conversion; GPS, UTC, TAI and other Astropy-backed
scales are supported. Spacecraft clock (SCLK) tags require external correlation
and are rejected.

.. code-block:: python

   from ssatk.io import read_tdm
   from ssatk.navigation import GroundStation, RadioTrackingConfig, solve_batch_orbit

   message = read_tdm("tracking.tdm")
   station = GroundStation(-104.9, 39.7, elevation_m=1600)
   hardware = RadioTrackingConfig(7.2e9, 10, 880 / 749, 2e-6)
   observations = message.to_station_observations(
       {"DSS-EXAMPLE": station},
       radio={"DSS-EXAMPLE": hardware},
       covariance={"two_way_range": 4.0, "two_way_doppler": 0.01},
       range_convention="round_trip",
   )
   # state_guess and state_epoch are the caller's initial orbit estimate.
   fit = solve_batch_orbit(observations, state_guess, t0=state_epoch)

Station/radio mappings use the ground ``PARTICIPANT_n`` identifier from the
file. Station coordinates, uplink frequency, fixed coordinate-time transponder
delay, and measurement noise must be supplied explicitly. Turnaround ratios
and constant transmit-frequency records, when present, must agree with that
hardware configuration. The TDM's integration interval replaces the caller's
count interval for Doppler. Covariances and optional biases use SSATK units:
metres squared/metres for ``two_way_range`` and hertz squared/hertz for
``two_way_doppler``. No weighting is inferred from a tracking file.

Supported navigation observations are ``RANGE`` in km or seconds,
``DOPPLER_INTEGRATED`` in km/s, and ``RECEIVE_FREQ``/``RECEIVE_FREQ_1`` in Hz.
The source interface control document (ICD) must specify whether range and
integrated range rate describe the full round-trip path or half of it.
Accordingly, ``range_convention`` and ``doppler_convention`` must explicitly be
``"round_trip"`` or ``"half_round_trip"`` for those data types. Frequency data
needs no path convention: ``FREQ_OFFSET`` reconstructs the received carrier,
then the coherent reference is subtracted to obtain SSATK's frequency shift.
Recession gives a negative shift, opposite to the standard's
transmitted-reference-minus-received Doppler convention.

``TIMETAG_REF`` must be RECEIVE (the standard default for received data).
START/MIDDLE/END ``INTEGRATION_REF`` tags move to the receive-interval end for
Doppler; range tags retain their original epoch. Scalar range, Doppler and
receive-frequency corrections marked ``CORRECTIONS_APPLIED=NO`` are added in
the source units before conversion; YES avoids applying them twice.
``DATA_QUALITY=DEGRADED`` records remain invalid. Imported elevation and
visibility are unknown (``None``); ingestion does not invent a visibility
check or discard observations by the configured elevation mask.

Conversion requires single-station two-way SEQUENTIAL paths and one spacecraft
across all segments. It rejects unsupported observables, differenced/three-way
links, ambiguous range, RU units, varying uplink frequencies, nonzero equipment
delays, and unsupported corrections. Constant uplink records and zero frequency
rates are checked and do not become orbit-fit observations. Other raw data can
be inspected in the parsed message, but are never silently dropped into a fit.
The reader validates complete sections, unique metadata, and chronological,
unique epochs per observable, without reordering the file.

The reference is `CCSDS 503.0-B-2, June 2020 with Corrigendum 1
<https://public.ccsds.org/Pubs/503x0b2c1.pdf>`_, particularly table 3-3 and
sections 3.4.9, 3.4.15, 3.5.2.3, 3.5.2.7 and 3.5.2.8. The radio demo now
passes synthetic TDM records through ingestion before fitting. Analytic
free-flight files validate prediction, fitted state and Fisher covariance;
they are synthetic references, not measured tracking calibration. Persistent
external tracking/reference datasets belong in SSAPy-Data; user-provided
tracking files are explicit inputs, not bundled data.

Run the examples
----------------

.. code-block:: bash

   python -m demos.sensor_coverage.demo_observer_pair_screening
   python -m demos.orbital_mechanics.demo_pair_screening_and_conjunctions
   python -m demos.navigation.demo_synthetic_orbit_estimation
   python -m demos.navigation.demo_radio_tracking

For offline validation without writing PNGs, call each module's
``main(make_figures=False)``.  The first demo also has
``benchmark=True`` to compare velocity, sightline, adaptive, and brute-force
candidate strategies on the same small synthetic state batch.  The benchmark
uses ``perf_counter`` and is a local comparison, not a performance guarantee.
