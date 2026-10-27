Launch preflight baseline
==========================

``ssatk.launch.preflight`` checks the inputs a launch simulation needs before
an ascent is propagated. It does not propagate a trajectory or claim that a
preflight pass qualifies a mission.

A preflight run evaluates three gates against a JSON mission contract:

* **G0 -- contract completeness.** :func:`~ssatk.launch.preflight.validate_mission_contract`
  checks that the contract has exactly the expected top-level fields, a
  recognized launch site and vehicle, a valid phase/transition sequence, and
  that every requirement carries a numeric value, unit, tolerance, check name,
  evidence field and source. :data:`~ssatk.launch.preflight.LAUNCH_CONVENTIONS`
  fixes the frames, epoch scale and units the contract must declare.
* **G1 -- software and data baseline.** :func:`~ssatk.launch.preflight.collect_software_baseline`
  records installed package versions and, for repositories passed in by the
  caller, their Git revision and worktree cleanliness, without modifying
  either. :func:`~ssatk.launch.preflight.verify_data_resources` checks each
  declared resource's SHA-256 and byte count against the installed wheel's
  ``RECORD`` manifest and rejects an unresolved Git LFS pointer.
* **G2 -- launch-pad state.** :func:`~ssatk.launch.preflight.launch_site_state`
  builds the stationary pad position and the inertial velocity caused by
  Earth's rotation in GCRF. Both the epoch-coverage check and the transform
  use SSATK's packaged Earth-orientation table
  (:func:`ssatk.environment_eop.load_packaged_eop`) through SSATK's IAU
  2006/2000A Earth-rotation route, with no network access. An epoch that falls in the packaged table's predicted
  region raises :class:`~ssatk.launch.preflight.EOPCoverageError` unless the
  caller passes ``allow_predicted_eop=True`` (or the mission contract sets
  ``allow_predicted_eop``); an epoch past the end of the table always raises
  it, since no opt-in can produce data the table does not have. In
  :func:`~ssatk.launch.preflight.run_launch_preflight`, that error BLOCKS
  G2 with a readable reason rather than failing the run -- a preflight for a
  future launch legitimately needs predicted EOP, so the contract's opt-in
  controls whether that is acceptable. The contract's
  ``frame_position_closure`` and ``frame_velocity_closure`` requirements
  bound a self-consistency check: transforming the computed GCRF state back
  to ITRS and comparing it with the starting ITRF state. An internally
  consistent but wrong rotation (for example, a sign error that cancels
  going out and back) still closes, so this closure is not independent
  evidence of accuracy; ``tests/test_launch_preflight.py`` pairs
  ``launch_site_state`` with a closed-form WGS84 reference and with
  Astropy's own ITRS->GCRS implementation fed the same packaged table for
  that.

:func:`~ssatk.launch.preflight.run_launch_preflight` runs all three gates and
returns a JSON-serializable report; it does not raise, so a caller can inspect
every gate even when an earlier one fails.

Mission contract
-----------------

``demos/launch/leo_preflight_template.json`` is an unconfigured LEO template:
every mission-specific value is ``null`` or ``"UNSET"``. Replace those values
with the mission's actual site, epoch, vehicle, payload, targets, limits and
tolerances before using it; loading it unmodified reports gate G0 as
``BLOCKED``. The EOP resource entry is pre-filled because the Earth-orientation
data it names is a fixed dependency of the frame-closure checks, independent
of the mission. The template's ``allow_predicted_eop`` defaults to ``false``;
set it to ``true`` only when the mission epoch is known to need predicted
Earth-orientation data (for example, a launch date beyond the latest IERS
bulletin), and record why in the contract's risk entries.

``ssatk.launch.sites`` and ``ssatk.launch.vehicles`` hold the launch pad and
vehicle catalogs that ``launch_site`` and ``vehicle`` are checked against.

Readable real-launch preflight check
-------------------------------------

``demos/launch/demo_real_launch_preflight.py`` compares ``launch_site_state``
for SpaceX Demo-2 with the independent SpacePy coordinate backend (the
``validation`` extra). NASA records LC-39A and 3:22 p.m. EDT on May 30, 2020:
https://www.nasa.gov/image-article/demo-2-launch-setting-forth-historic-journey/.
The input epoch is the published launch minute, not subsecond flight
telemetry.

Run the comparison test with its printed output visible::

   python3 -m pytest -q -s tests/test_launch_preflight.py::test_real_demo2_launch_preflight_matches_spacepy

Or generate an HTML report at an explicit output location outside the
repository::

   python3 -m demos.launch.demo_real_launch_preflight --output /tmp/demo2_preflight.html

The check compares independent WGS84 conversions of the same catalog pad
coordinates and the inertial speed from Earth's rotation. The position
tolerance is numerical, not a surveyed pad uncertainty. The speed allowance
comes from the floating-point Julian-day spacing in SpacePy's
central-difference reference. J2000/GCRF direction differences are reported
but not scored: SpacePy omits Earth-orientation-parameter corrections and
assumes UT1 equals UTC. This is not an ascent or telemetry match.
