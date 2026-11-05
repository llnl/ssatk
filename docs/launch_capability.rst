Reproducing the demonstrated launch capability
==============================================

This is Step 15 of the launch literature-review development sequence:
**publish a reproducible capability statement**. It follows the verification
and calibration work in :doc:`launch_baseline`. The separate
:doc:`launch_roadmap` uses different step numbers for future mission features.
This statement does not release a new package version or qualify a mission.

Supported domain
----------------

The demonstrated domain is isolated launch subsystems and a specified
Earth-centred upper-stage burn followed by orbital coast. The examples cover
rigid six-degree-of-freedom motion, bounded changing-mass assumptions,
propulsion, weather/aerodynamic loads, synthetic staging and sampled control,
numerical refinement, constant-mass linear slosh, and the exact-state handoff
to native SSAPy. The calibration example fits thrust and velocity-sensor bias
against the rocket equation and a rotating, filtered sensor. Its deliberately
faulty held-out sensor remains a failed case.

These examples do **not** demonstrate a complete launch from the pad to LEO,
final GEO insertion, measured-flight prediction, or flight qualification.
The LEO mission contract still needs an approved coherent vehicle, measured
inputs and acceptance budgets. GEO additionally needs transfer, circularization
and final-orbit evidence. Published vehicle catalogs are not calibrated flight
datasets. Cislunar and interplanetary missions are outside this statement.

The verification references use packaged Earth orientation, gravity and
ephemerides. Synthetic inputs, frame conventions, solver settings, state units,
sample times, seed and model limits are retained in each case's JSON output.
During verification, Astropy and native SSAPy use the recorded packaged
``finals2000A.all`` table with automatic Earth-orientation downloads disabled.
The runner isolates SSAPy's cached interpolation and restores the caller's
table/cache configuration afterward, so earlier process state cannot select
a different Earth-orientation input for the retained gravity/drag references.
Queries outside that snapshot's coverage fail instead of extrapolating it.
Integration settings are distinct from requested output spacing. Numerical
refinement does not bound model discrepancy or measurement uncertainty.
Constant-mass slosh does not establish coupled draining-tank dynamics; the
native SSAPy overlap shares force implementations and is integration evidence.

Generate a handoff bundle
-------------------------

Use a source checkout with the normal development dependencies and all source
changes committed or staged. No additional dependency is needed. Keep the output
outside the source checkout, in a data/evidence location of your choosing::

   python -m demos.launch.demo_launch_capability --output ../launch-evidence/run-1

The command refuses to overwrite an existing directory. It runs the existing
launch verification ladder with external comparisons disabled, then the
existing synthetic calibration example. No external bridge is imported or
executed. Their comparison rows remain ``NOT RUN``; unavailable NASA, launch
package, measured-flight and other references remain ``UNAVAILABLE``. It does
not run unrelated demos, browser checks, or a full mission suite.

The bundle contains:

* ``capability.json`` and ``CAPABILITY.md``: supported mission class, fidelity,
  data conditions, run ID, per-case statuses, limitations, software versions,
  hardware/platform information and measured elapsed wall time. Timing covers
  these serial examples only and makes no speed-comparison claim.
* ``calibration-input.json`` and ``calibration.json``: exact synthetic input,
  fit settings, frozen split, uncertainty, predictions, residuals and failed
  holdout. Other case inputs, source/model assumptions, solver settings,
  histories, event ledgers, reference metrics and units/tolerances live in
  ``verification/*.json``.
* ``requirements.txt``: exact installed distribution versions, except SSATK,
  which must come from the recorded source revision and patch. The inventory
  includes optional/development packages in the producing environment; use the
  recorded Python version and compatible operating system. Package artifacts
  must remain available from their original distribution sources.
* ``source.patch``: staged changes against the recorded Git revision, empty for
  a clean committed checkout. ``source.tree`` is the canonical Git tree for the
  executed source, so a package version alone cannot substitute for these edits.
  Source changes during execution invalidate the bundle.
* ``run.log`` and ``manifest.json``: captured progress/diagnostics and SHA-256
  plus byte count for every artifact except the manifest itself. Preserve or
  transmit the manifest's own hash separately when authenticity matters.

The bundle is generated evidence and must not be committed to the Toolkit.
It contains no packaged binary datasets: each required resource is identified
by its installed data distribution and checked against the wheel's ``RECORD``
and SHA-256. Retain the pinned distribution artifacts in the data archive if
long-term availability is required. Review source patches and environment
records before sharing; they are a reproducibility record, not anonymization.

Status interpretation
---------------------

The bundle's ``status`` and exit code describe the **named software
demonstrations**. Code 0 requires every executed verification case and resource
check to pass, the synthetic fit to recover the independent thrust/bias
references within their stated SI budgets, and the injected holdout to remain
``FAIL``. Code 2 means these checks failed. The complete per-case ledger is
retained even when some rows fail or lack evidence. Exceptions during bundle
creation leave an incomplete directory without a final manifest; never treat
that as a finished handoff.

``mission_acceptance_status`` remains ``BLOCKED``,
``measurement_validation_status`` remains ``UNAVAILABLE``, and
``qualification_status`` remains ``NOT RUN``. No missing prerequisite is waived
by a synthetic pass. Integrity verification and numerical reproduction are
separate checks. No claim of accuracy superiority over another tool is made.

Verify and reproduce
--------------------

Verify a copied bundle without loading bridges or rerunning dynamics::

   python -m demos.launch.demo_launch_capability --verify ../launch-evidence/run-1

This checks the exact file inventory, sizes and hashes, including detection of
modified or missing evidence. It does not certify the claimed physical result.

To reproduce on another machine, install the recorded Python version, create
a fresh virtual environment, and obtain the repository named in
``capability.json``. Check out ``source.revision``, apply the bundled patch if
it is nonempty, and stage its additions::

   git checkout <source.revision>
   git apply --index ../launch-evidence/run-1/source.patch
   python -m pip install -r ../launch-evidence/run-1/requirements.txt
   python -m pip install --no-deps -e .
   python -m demos.launch.demo_launch_capability --replay ../launch-evidence/run-1 --output ../launch-evidence/run-2

Skip ``git apply`` for an empty patch. Use the source's recorded Git revision,
not whichever revision ``main`` currently names. A staged source snapshot is
supported so review work can be reproduced before a release is published.

Replay checks bundle integrity, the canonical source tree, Python and installed
distribution versions, and data hashes **before** running. It loads the saved
calibration input instead of generating replacement observations. It then
requires the same case inventory, outcomes and independent-reference budgets.
Metric values may vary inside their original tolerance; timestamps, run IDs,
elapsed time and floating-point histories are not required to be byte-identical.
The new bundle records the original run ID and a separate reproduction status.
All solver/model inputs for the verification cases are frozen by the checked
source tree and retained case records. Failed fits and absent flight evidence
cannot be converted into measured validation by replay.

Previous-step evidence boundaries
---------------------------------

Steps 1-2 supply software/data checks and the mission-contract mechanism;
mission data and acceptance remain open. Steps 3-7 supply the documented
frame, mass, propulsion and environment references. Steps 8-11 demonstrate
synthetic event/control sequences, numerical refinement and orbit handoff.
Step 12 demonstrates a bounded optional slosh model, not every flexible or
recovery subsystem. Step 13 retains unavailable independent comparisons.
Step 14 supplies calibration and editable synthetic data; it lacks measured
held-out flights. Step 15 freezes and reruns those exact supported cases
without promoting their status to a completed Earth-orbit mission.
