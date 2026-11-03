Artemis III radiation demo
==========================

The self-contained HTML opens with a brief mission overview. Click **Start demo**
to reveal the Earth-to-Moon trajectory and synchronized cumulative absorbed-dose
chart. The scene stays paused and its controls stay hidden until you start.
**Mission overview** returns to the opening screen. Playback, pause and scrubbing
work without a server or network connection. Optional recording produces a
42-second movie, holding the overview for its first ten seconds; interactive
playback skips that hold when you click Start demo.

.. code-block:: powershell

   python -m demos.space_environment.demo_artemis_radiation --movie

Use a Python environment with the toolkit and its existing ``browser`` and
``video`` extras (``python -m pip install "ssatk[browser,video]"``).
Recording requires Microsoft Edge and uses its headless mode through Selenium and
the toolkit's ``write_video``. ``--fast`` records at 15 fps instead of 24 fps.
Outputs include ``artemis_radiation.html``, ``artemis_radiation.mp4``, selected
preview frames, ``demo.json``, and a separate fresh IRENE run directory. Generated
artifacts default to ``~/ssatk_output/figures/artemis_radiation`` through the
toolkit's ``figpath`` resolver and ``figsave``; ``SSATK_OUTPUT_DIR`` overrides the output root.
Use ``--output`` to choose another directory outside the Toolkit checkout.
Keep HTML, JSON, media, raw model outputs and IRENE databases out of the source
repository; reusable persistent data belongs in the appropriate split ssa-data package.

Official IRENE integration
--------------------------

IRENE is AFRL's **AE9/AP9/SPM radiation belt and space plasma model**. The current
official public distribution listed is **V1.58.001**, containing Windows
executables, databases, Python/C/C++ APIs, and sample files. As checked on
October 22, 2026, its download requires an AFRL public or VDL account; the public
download URL redirects to a login page. The PyPI package called ``irene`` is an
unrelated polynomial optimization project.

* `Official downloads and account registration <https://www.vdl.afrl.af.mil/programs/ae9ap9/downloads.php>`_
* `Current user guide <https://www.vdl.afrl.af.mil/programs/ae9ap9/files/package/IRENE_Users_Guide.pdf>`_
* `Current Python API <https://www.vdl.afrl.af.mil/programs/ae9ap9/files/package/IRENE_Python_API.pdf>`_
* `Known issues and limits <https://www.vdl.afrl.af.mil/programs/ae9ap9/issues-and-limits.php>`_
* `Recommended time sampling <https://www.vdl.afrl.af.mil/programs/ae9ap9/time-sampling.php>`_

After obtaining and extracting the official Windows distribution, pass its
``CmdLineIrene.exe`` and ``modelData`` directory explicitly:

.. code-block:: powershell

   python -m demos.space_environment.demo_artemis_radiation --movie --irene-executable "<IRENE>/win64/bin/CmdLineIrene.exe" --irene-data "<IRENE>/modelData"

The adapter in ``ssatk.io.irene`` converts the SSAPy trajectory to
Earth-fixed **GEO Cartesian kilometers**, writes time as **UTC MJD**, and
prepares two documented command-line inputs. Each calls IRENE with ``-i``,
without a shell.

.. list-table:: Radiation calculation assumptions
   :header-rows: 1
   :widths: 25 75

   * - Setting
     - Demo assumption
   * - Particle models
     - AE9 electrons, AP9 trapped protons
   * - Flux
     - Omnidirectional differential mean; 100 logarithmic energies per species
   * - Energy ranges
     - AE9 0.04–10 MeV; AP9 0.1–2000 MeV
   * - Dose calculation
     - IRENE SHIELDOSE2, including electron bremsstrahlung
   * - Detector
     - Tissue
   * - Shield
     - Aluminum sphere covering 4π; 1/3/10 mm thickness
   * - Accumulation
     - Cumulative at each ephemeris time
   * - Display
     - Sum of electron and proton absorbed dose; 3 mm curve in mGy

The adapter reads IRENE's ``_mean_doseaccum.txt`` files, verifies both match the
input ephemeris, rejects missing/nonfinite/negative/decreasing results and
incompatible units, and adds the two contributions. **1 rad = 10 mGy**.
All three depth curves are exported to ``dose_vs_mission_day.csv``. Absorbed
dose is not converted to effective dose in mSv. Failed runs preserve logs and
inputs. Existing run results are never overwritten.

The demo uses the same runtime discovery as the public toolkit API. Set
``IRENE_HOME`` to the extracted ``Irene`` directory, or set
``IRENE_EXECUTABLE`` and ``IRENE_DB``. A ``CmdLineIrene`` executable on ``PATH``
is also supported; its sibling distribution's ``modelData`` is used by default.
Explicit function arguments or command-line flags override environment settings.
Multiple runtimes under ``IRENE_HOME`` require an explicit selection. A configured
but incomplete installation raises an error and preserves diagnostic logs.

Without an installed or configured runtime, the demo still prepares inputs and makes the
movie, but the dose chart and absent belts explicitly say **Pending IRENE**. No numerical
radiation prediction is manufactured. Adapter regression tests use clearly
labeled synthetic output fixtures; they do not execute IRENE or independently
validate its physical predictions.

Reusable toolkit API
~~~~~~~~~~~~~~~~~~~~

The adapter is available through ``ssatk``, ``ssatk.io``, and
``ssatk.io.irene``. It accepts any sampled SSAPy trajectory as GCRF
positions in meters and an array of Astropy times; it is independent of the
Artemis viewer. The adapter calculates mean AE9/AP9 dose at the three
documented shield depths and single-epoch integral-flux grids.
SPM plasma and Monte Carlo/percentile modes are not
exposed by this adapter.

.. code-block:: python

   import numpy as np
   import astropy.units as u
   from astropy.time import Time
   import ssatk as ssatk

   epoch = Time("2027-06-01T00:00:00", scale="utc")
   times = epoch + np.arange(121) * 30 * u.s
   orbit = ssatk.Orbit.fromKeplerianElements(
       10_000_000, 0.1, 0.3, 0.0, 0.0, 0.0, epoch.gps)
   positions, velocities = ssatk.rv(orbit, times.gps)
   run = ssatk.prepare_irene_run(positions, times, ssatk.figpath("irene_example/run"))
   dose = ssatk.run_irene(run)
   print(dose["total_mgy"][-1])  # 1, 3, 10 mm aluminum, absorbed tissue mGy

Use a fresh run directory each time. ``read_irene_dose(run)`` reimports a
completed run without executing the model again. ``find_irene_runtime()``
returns validated executable/database paths, or ``None`` when no installation
is found. Numerical values are never substituted for missing model output.

The official run serializes MJD to eight decimal places and GEO/km positions to
six. The reader compares the input rounded to that exact format. One final
printed dose digit of numerical reversal is accepted without modifying the
source dose values; larger decreases still fail validation.

Mission and scientific scope
----------------------------

This is the user's requested **hypothetical Artemis III lunar scenario**.
`NASA's current Artemis III page <https://www.nasa.gov/mission/artemis-iii/>`_
describes a 2027 low-Earth-orbit systems demonstration, with lunar landings
beginning with Artemis IV. This demo is not NASA's Artemis III flight plan.

The illustrative epoch is June 1, 2027 UTC. Mission day zero is translunar
injection from a 200 km altitude position, not liftoff. The existing toolkit
Lambert solver and SSAPy Keplerian propagator produce a four-day two-body
outbound arc to a point 2,000 km Earthward of the Moon's ephemeris position at
arrival. Launch, lunar orbit insertion, landing, and return are excluded.
Sampling is 30 seconds through the first six hours and 600 seconds afterward;
a prediction requires time-step and energy-grid convergence studies.

Visible belts reuse the toolkit's official IRENE mean integral flux grid:
AP9 protons above 10 MeV and AE9 electrons above 1 MeV. Each surface follows
18% of that species' maximum sampled flux; the numerical contour levels in
cm\ :sup:`−2` s\ :sup:`−1` are displayed and saved in ``demo.json``. These
display levels are not hazard thresholds. A 48-cubed GEO grid (36-cubed with
``--fast``) is triangulated by linear marching tetrahedra. The grid is evaluated
at departure and the contours rotate with Earth; this is not a time-varying
space-weather simulation. Initial GEO-to-GCRF orientation uses the same
IAU 2006/2000A transform as the dose ephemeris, followed by the toolkit's constant Earth
rotation rate. The independent trajectory-dose calculation still
integrates the differential energy spectra at every mission sample.
Missing IRENE leaves belts absent; invalid output fails without substitute
geometry. Earth/Moon positions are at physical scale; the spacecraft marker is
enlarged. The movie
slows the first six hours to show the crossing, then accelerates the coast.

The starfield reuses the toolkit's packaged bright-star catalog and existing
``load_star_catalog`` helper in GCRF at the departure epoch. Catalog directions,
magnitudes and colors drive the sky; stars keep their inertial orientation
while Earth and the belts rotate. The sky follows the camera's position to
avoid artificial stellar parallax as the view pulls back. Earth and Moon
occlude stars. Markers are enlarged for visibility in compressed or downscaled
video, with the renderer's display color conversion applied. Marker brightness
and size are for visualization, not a camera exposure simulation. A missing
catalog raises an error rather than generating
random stars. ``demo.json`` records the star count, frame and epoch.

IRENE V1.58 excludes untrapped solar energetic particles and galactic cosmic
rays. The sum is therefore the **trapped-particle contribution**, not total
crew radiation over a lunar mission. Shielding is illustrative, not an Orion
mass model. The official known-issues page also documents geomagnetic epoch
limitations: the supplied V1.58.001 magnetic field is fixed for all dates after
January 1, 2020. Thus the 2027 output is a calculation under that fixed field,
not a validated forecast. Invalid or out-of-domain output is rejected rather
than zero-filled.

Verification scope
------------------

Numerical contour, trajectory endpoint and catalog-reference checks live in
``tests/test_irene.py`` and ``tests/test_viewer_star_reference.py``. Demo and
recording contracts live in ``scripts/tests/test_artemis_radiation.py``; they
check disabled output and screenshot failures, not physical radiation accuracy.
