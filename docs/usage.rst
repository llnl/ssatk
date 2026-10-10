Usage Guide
===========

Installation
------------

Install in editable mode with development extras:

.. code-block:: bash

   python -m pip install -e .[dev]

Plotting installs the Python packages needed for HTML, image, and GIF outputs,
including Plotly, Matplotlib, Pillow, imageio, and split SSATK data packages.
Install ``ssatk[static]`` for Plotly static-image export through Kaleido.
Install ``ssatk[pdf]`` to append pages to existing PDF plots.
Install ``ssatk[notebook]`` for IPython display and ipyvolume
Earth/Moon meshes.
Install ``ssatk[video]`` for OpenCV MP4 output and a bundled FFmpeg
fallback, and ``ssatk[browser]`` for Selenium browser capture.
Install ``ssatk[monitoring]`` to enable the optional current-process RSS
memory helper.
Node.js 20+ is only needed for validating the JavaScript satellite-viewer source;
GitHub Actions installs it with ``actions/setup-node``.

Basic Example
-------------

.. code-block:: python

   import ssatk

   from ssatk.orbital_mechanics import keplerian

   orbit = ssatk.Orbit.fromKeplerianElements(
       a=ssatk.constants.RGEO,
       e=0.0,
       i=0.0,
       pa=0.0,
       raan=0.0,
       trueAnomaly=0.0,
       t=0.0,
   )
   r, v = ssatk.rv(orbit, time=[0.0, 60.0])

   # Use Toolkit keplerian routines and plotting helpers around SSAPy objects.
   from ssatk.plots import orbit_plot

   orbit_plot(r, view="xy", frame="gcrf")
   orbit_plot(r, view=("xy", "xz", "3d"), frame="itrf")
   orbit_plot(r, view="lunar_yz")
   orbit_plot(r, view="lunar_xy", coordinate="gcrf")
   orbit_plot(r, view="ground track")
   orbit_plot(r, view=("groundtrack", "globe"))
   orbit_plot(r, view="dashboard")
   orbit_plot(r, view="cislunar_3d")
   orbit_plot(r, view="cislunar_dashboard")

Plot saving
------------

All plotting helpers accept ``save``, ``savefig``, ``save_fig``,
``save_figure``, ``savepath``, and ``save_path`` as equivalent save-path
keywords. Relative filenames are saved under ``~/ssatk_output/figures`` and
absolute paths are honored exactly as provided:

.. code-block:: python

   from ssatk.plots import orbit_plot, ssatk_fig, ssatk_path

   orbit_plot(r, view="xy", save="quicklook/orbit_xy")
   orbit_plot(r, view="globe", save_fig="/tmp/orbit_globe.png")
   orbit_plot(r, t, view="xy", save="quicklook/orbit_xy.mp4")
   orbit_plot(r, t, view="xy", save="quicklook/orbit_xy.gif")

   figure_path = ssatk_path("reports/summary")
   saved_path = ssatk_fig(fig, save_path=figure_path)

For :func:`ssatk.plots.orbit_plot`, ``.mp4`` and ``.gif`` save paths
create animated quicklooks with short fading tails. Static extensions such as
``.png`` and ``.jpg`` save the full time-series figure.


Relationship to SSAPy
----------------------

Space Situational Awareness Toolkit (SSATK) is designed as an extension library for `SSAPy <https://github.com/llnl/SSAPy/tree/main>`_, which provides high-fidelity orbital modeling and analysis across LEO through the cislunar regime. SSAPy handles orbit propagation, force models, integrators, and rich coordinate/frame support; SSATK builds on top of that to provide convenience utilities for data IO, plotting (including ground tracks and cislunar visualizations), and higher-level orbital mechanics helpers.

Use ``import ssatk`` as the main user-facing entry point.
Shared astrodynamics constants are available through ``ssatk.constants`` and as
lazy top-level attributes such as ``ssatk.EARTH_MU``. Core SSAPy classes and
functions such as ``ssatk.Orbit``, ``ssatk.rv``, ``ssatk.groundTrack``, and
``ssatk.AccelKepler`` are also lazily available through the Toolkit. If a name
exists in both packages, Toolkit helpers and submodules take precedence; direct
base-package access remains available through ``ssatk.ssapy``.
Earth/Moon helpers formerly provided by ``ssapy.plotUtils`` are available as
``ssatk.draw_earth``, ``ssatk.draw_moon``, ``ssatk.load_earth_file``, and
``ssatk.load_moon_file``.
Wildcard imports are intentionally unsupported; use ``import ssatk as
ssatk`` or import individual names explicitly.

CCSDS conjunction messages
---------------------------

Read and write CCSDS 508.0-B-1 Conjunction Data Message (CDM) KVN 1.0 files
with :func:`ssatk.io.ccsds_cdm.read_cdm` and
:func:`ssatk.io.ccsds_cdm.write_cdm`:

.. code-block:: python

   from ssatk.io.ccsds_cdm import read_cdm, write_cdm

   cdm = read_cdm("conjunction.cdm")
   object1_gcrf = cdm.object1.state_gcrf()
   write_cdm(cdm, "canonical.cdm")

States and covariances use SI units in memory. The KVN reader accepts calendar
or ordinal Coordinated Universal Time (UTC), ``GCRF``, ``EME2000``, and
``ITRF`` state frames, and the mandatory RTN covariance plus complete optional
drag, solar-radiation-pressure, and thrust rows. Alternate XYZ covariance
extensions from later message specifications are rejected explicitly.

Satellite operation frames
--------------------------

Use :mod:`ssatk.coordinates.satellite_frames` for common satellite
operation frames. Matrices are returned as frame-to-GCRF rotations: columns are
the requested frame axes expressed in GCRF, so ``matrix @ vector_in_frame``
returns a GCRF vector.

.. code-block:: python

   from ssatk.coordinates.satellite_frames import (
       frame_to_gcrf_matrix,
       transform_from_gcrf,
       transform_to_gcrf,
   )

   # SSAPy maneuver convention: [N, T, W].
   ntw_to_gcrf = frame_to_gcrf_matrix("ntw", r=r_gcrf, v=v_gcrf)
   thrust_gcrf = transform_to_gcrf([0.0, 1e-7, 0.0], "ntw", r=r_gcrf, v=v_gcrf)

   # Other supported selectors include "rtn", "rsw", "ric", "lvlh", "vnb",
   # "body", "nadir_velocity", "enu", "ned", "sez", "los", and "sun".
   thrust_ntw = transform_from_gcrf(thrust_gcrf, "ntw", r=r_gcrf, v=v_gcrf)

High-accuracy propagation
-------------------------

Use :func:`ssatk.propagators_orbit.propagate_orbit_state` for adaptive
high-accuracy translational propagation. It defaults to SciPy's eighth-order
``DOP853`` method and accepts inertial perturbing acceleration callbacks.

.. code-block:: python

   import numpy as np
   from ssatk.constants import EARTH_MU
   from ssatk.propagators_orbit import propagate_orbit_state

   radius = 7_000_000.0
   speed = np.sqrt(EARTH_MU / radius)

   traj = propagate_orbit_state(
       r0=[radius, 0.0, 0.0],
       v0=[0.0, speed, 0.0],
       times=np.linspace(0.0, 3600.0, 121),
   )

For covariance or sensitivity propagation, use
:func:`ssatk.propagators_orbit.propagate_orbit_state_with_stm`; its
``stm`` output has shape ``(N, 6, 6)`` and maps initial Cartesian perturbations
to each sampled state.

For coupled spacecraft covariance or sensitivity propagation, use
:func:`ssatk.propagators_6dof.propagate_6dof_variational`. The returned
``stm`` retains the solver's quaternion coordinates. Use its
``attitude_error_stm`` property for the local multiplicative form with state
ordering ``[r, v, δθ_body, omega, mass, wheel_momentum]``; ``δθ_body`` is the
three-parameter body-frame error defined by
``q_perturbed = q_nominal ⊗ δq_body``.
``propagate_6dof_covariance`` accepts either the raw quaternion covariance or
the local covariance; use ``coordinates="attitude_error"`` to select the
latter explicitly, or leave the default ``"auto"`` shape selection enabled.

Export an independent-tool reference case with
:func:`ssatk.io.write_reference_case`. It writes a CCSDS Orbit
Ephemeris Message (OEM) in km and km/s plus a JSON sidecar in SI units with the
epoch, frame, force-model labels, constants, integration settings, and numeric
precision. GMAT, STK, Orekit, and similar tools can consume the OEM without
guessing the SSATK conventions.
Pass an SI ``covariance`` array with optional ``covariance_t`` and
``covariance_reference_frame`` to include standard triangular OEM covariance
blocks; covariance frames are preserved rather than implicitly rotated.
Pass an SI ``acceleration`` array with shape ``(N, 3)`` to write standard
nine-component OEM records (position, velocity, and acceleration); optional
``acceleration_t`` must match the state epochs. The JSON sidecar records the
acceleration count, epochs, and SI units.

.. code-block:: python

   from ssatk.constants import EARTH_MU
   from ssatk.io import write_reference_case

   write_reference_case(
       traj,
       "reference_cases/leo",
       epoch="2025-01-01T00:00:00Z",
       force_models=["point_mass_earth", "J2"],
       constants={"mu_m3_s2": EARTH_MU},
       integrator={"method": "DOP853", "rtol": 1e-10, "atol": 1e-9},
   )

Compare a trajectory against an OEM or JSON reference with
:func:`ssatk.io.compare_reference_case`. It evaluates the candidate at
the reference epochs and returns maximum, root-mean-square, and final position
and velocity residuals in SI units.

Read a standalone CCSDS OEM 2.0 KVN message with
:func:`ssatk.io.read_oem`. The reader accepts standard multi-segment
messages and legacy ``DATA_START``/``DATA_STOP`` wrappers, preserves each
segment's metadata and comments, and converts the km/km/s state records to
meters and meters per second. Standard triangular 6x6 OEM covariance blocks
are preserved per epoch and covariance frame and converted to SI units; no
implicit RTN/RIC/frame rotation is applied. Optional standard three-component
acceleration records are preserved per epoch and segment and converted from
km/s^2 to m/s^2. Incomplete, nonfinite, mixed, or out-of-order acceleration
records, and inconsistent sidecar acceleration metadata, are rejected
explicitly. Absolute UTC, TAI, TT, TDB, TCB, TCG, UT1, and GPS
calendar or ordinal epochs are supported; mission-relative time systems such
as MRT and SCLK require an external epoch or clock-correlation contract not
represented by the current OEM API and are rejected explicitly.

.. code-block:: python

   from ssatk.io import compare_reference_case

   residuals = compare_reference_case(traj, "reference_cases/leo/ssatk_reference_case.json")

For terminal maneuver targeting, use
:func:`ssatk.propagators_6dof.solve_6dof_target`. This performs bounded
single shooting around the same ``Spacecraft.propagate`` force and attitude
models used for the final trajectory.

.. code-block:: python

   import numpy as np
   import ssatk

   sat = ssatk.Spacecraft(
       r=[0.0, 0.0, 0.0],
       v=[0.0, 0.0, 0.0],
       inertia=np.eye(3),
   )
   target = ssatk.solve_6dof_target(
       sat,
       times=np.linspace(0.0, 10.0, 11),
       target_v=[2.0, 0.0, 0.0],
       control_scale=[1.0, 1.0, 1.0],
       propagation_kwargs={"mu": 0.0},
   )
   assert target.success

For a sequence of coast and burn arcs, use
:func:`ssatk.propagators_6dof.solve_6dof_multi_segment_target`. Each
segment supplies its own ``times`` and can override models, burn frame, bounds,
and control scaling. ``constraints`` and ``residual_hook`` append normalized
residuals to the bounded least-squares solve.

For an exact impulsive maneuver between arcs, add an
:class:`ssatk.propagators_6dof.ImpulseManeuver` to a segment's
``impulses``. ``dv`` accepts inertial, body, NTW, RTN, or VNB components;
``mass_change``, ``q_reset``, and ``omega_reset`` optionally update the
spacecraft state at that epoch. The combined trajectory retains both samples
at the maneuver epoch so the state jump remains observable.

.. code-block:: python

   trajectory = ssatk.propagate_spacecraft_segments(sat, [
       {"times": [0.0, 600.0]},
       {
           "times": [600.0, 3600.0],
           "impulses": ssatk.ImpulseManeuver(
               dv=[0.0, 25.0, 0.0], frame="ntw", mass_change=-0.5
           ),
       },
   ])

For coupled sensitivity and uncertainty workflows,
:func:`ssatk.propagators_6dof.propagate_6dof_variational` returns the
nominal trajectory and state-transition matrices. Pass that result to
:func:`ssatk.propagators_6dof.propagate_6dof_covariance` to map an
initial covariance and optional per-epoch process-noise contributions.

For an optional NRLMSISE-00 atmosphere driven by packaged solar and
geomagnetic indices, install ``ssatk[atmosphere]`` and configure
``SpaceEnvironment(atmosphere_density_model="nrlmsise00")``. The adapter
rejects predicted space-weather and Earth-orientation records by default.

6-DoF dynamics
--------------

Use :class:`ssatk.propagators_6dof.Spacecraft` for an ``Orbit``-like object
with attitude, angular rate, inertia, and mass attached. Use
:func:`ssatk.propagators_6dof.propagate_6dof` directly for lower-level coupled
translational and rigid-body attitude propagation. The state uses inertial
``r``/``v`` vectors, a quaternion ``q=[w, x, y, z]`` that rotates body-frame
vectors into the inertial frame, and body-frame angular rates ``omega`` in
rad/s. Use ``Spacecraft.from_orbit(orbit, ...)`` to attach attitude/body state
to an SSAPy ``Orbit`` and ``spacecraft.to_orbit()`` to return the translational
state to SSAPy workflows.

.. code-block:: python

   import numpy as np
   import ssatk
   from ssatk.accelerations_6dof import SpacecraftAccelJ2, constant_body_thrust
   from ssatk.plots import orbit_plot

   sat = ssatk.Spacecraft(
       r=[7_000_000.0, 0.0, 0.0],
       v=[0.0, 7_500.0, 0.0],
       q=[1.0, 0.0, 0.0, 0.0],
       omega=[0.0, 0.0, 0.001],
       inertia=np.diag([10.0, 12.0, 8.0]),
       mass=100.0,
   )

   traj = sat.propagate(
       times=np.linspace(0.0, 600.0, 61),
       acceleration=SpacecraftAccelJ2(),
       body_acceleration=constant_body_thrust([0.0, 0.01, 0.0], sat.mass),
       gravity_gradient=True,
   )

   orbit_plot(traj.r, traj.t, view="3d")

The high-accuracy convenience wrapper accepts the same environment presets:

.. code-block:: python

   traj = ssatk.propagate_spacecraft_high_accuracy(
       sat,
       times=np.linspace(0.0, 3600.0, 121),
       environment=ssatk.SpaceEnvironment(epoch="2025-01-01T00:00:00"),
       environment_models="leo",
   )

Without an attitude-dependent ``acceleration`` callback, attitude does not feed
back into the orbital trajectory. Provide ``acceleration(t, r, v, q, omega)``
for inertial/GCRF m/s², ``ntw_acceleration(t, r, v, q, omega)`` for SSAPy
``[N, T, W]`` m/s², or ``body_acceleration(t, r, v, q, omega)`` for body-frame
m/s² when thrust, drag, solar-radiation pressure, or another orientation-
dependent force should change ``r`` and ``v``. The propagated quaternion remains
body-to-GCRF; use NTW acceleration for orbit-frame maneuvers rather than
satellite attitude.

Reusable models live in :mod:`ssatk.accelerations_6dof` and include
``SpacecraftAccelKepler``, ``SpacecraftAccelJ2``,
``SpacecraftAccelThirdBody``, ``SpacecraftAccelDrag``,
``SpacecraftAccelSolRad``, ``SpacecraftAccelConstInertial``,
``SpacecraftAccelConstNTW``, ``SpacecraftAccelConstBody``,
``SpacecraftFlatPlateDrag``, ``SpacecraftFlatPlateSolRad``,
``SpacecraftFacetDrag``, ``SpacecraftFacetSolRad``, ``SpacecraftThrusterAccel``,
``SpacecraftGravityGradientTorque``, ``SpacecraftMagneticTorque``,
``SpacecraftReactionWheelTorque``,
``SpacecraftAttitudePD``, ``SpacecraftAccelSum``, ``SpacecraftTorqueSum``, and
constant thrust/torque callback helpers. Flat-plate models use spacecraft
``mass``, ``area``, ``cd``/``cl``/``cr``, and body-frame ``center_of_pressure``
when those values are not provided directly to the model; ``cl`` adds signed
lift and ``cl=0`` preserves drag-only behavior. Thruster models report
positive propellant mass flow from thrust and specific impulse, and
``propagate_6dof`` can propagate mass when a mass-flow model is supplied. For
``SpacecraftBody`` objects with tanks, propagated mass is distributed across
tanks in proportion to their configured propellant mass, so center of mass and
inertia can evolve during finite burns. ``Spacecraft.propagate`` continues
tanked spacecraft at dry mass by default after propellant depletion, with
propulsive acceleration, torque, and mass flow set to zero. Set
``stop_at_dry_mass=True`` for terminal depletion, or use
``propellant_empty_event``/``mass_floor_event`` directly for lower-level
``propagate_6dof`` calls.
Set ``tank_name`` on ``SpacecraftManeuverAccel`` to draw from one named tank;
that tank alone is depleted and the maneuver stops when it is empty.
If the body defines reaction wheels, ``Spacecraft`` initializes wheel momentum
from ``wheel_inertia * speed`` when available, ``propagate_6dof`` appends those
momenta to the numerical state, and ``SixDOFTrajectory.wheel_momentum`` returns
the propagated wheel angular momenta in wheel order. Configured
``momentum_capacity`` values clip torque commands that would drive a wheel past
its stored angular-momentum limit.
Facet drag/SRP models accept ``facet_transform=...`` for time- or
state-dependent articulated panels; use ``rotate_facets(...)`` for simple
hinged-panel rotations. Facet and flat-plate drag use each surface point's
local rigid-body velocity, including the ``omega × r`` contribution from the
spacecraft angular rate, so spinning appendages can change both aerodynamic
force and torque. Pass ``atmosphere_velocity=...`` to drag models for explicit
GCRF wind/corotation velocity, or set
``SpaceEnvironment(atmosphere_velocity_model=...)`` when assembling
environment-backed drag models.
For linearized appendage and propellant-slosh states, use
:func:`ssatk.propagators_6dof.propagate_6dof_extended` with
``HingedAppendage``, ``FlexibleMode``, and ``SloshMode``. These models are
linear reduced-order couplings, not finite-element or computational-fluid-
dynamics replacements. Each mode argument may be one mode or a sequence;
single-mode results retain shape ``(samples, 2)`` and multi-mode results use
``(samples, 2, modes)``.
For sensitivity or covariance propagation through those modes, use
:func:`ssatk.propagators_6dof.propagate_6dof_extended_variational`.
Its ``stm`` uses the full ``[r, v, q, omega, mass, modes]`` state ordering and
can be passed to :func:`ssatk.propagators_6dof.propagate_6dof_covariance`.
The extended STM uses relative central differences because hinge cubic
stiffness is nonlinear.
``SpaceEnvironment.force_models(...)`` can assemble environment-backed drag,
solar-radiation pressure, magnetic torque, and named third-body perturbations
such as ``third_bodies=("moon", "sun")``. Use ``third_bodies=True`` for
Moon/Sun, ``third_bodies="planets"`` for Mercury through Neptune except Earth,
or ``third_bodies="all"`` for Moon, Sun, and planets. For common setups, pass
``preset="leo"``, ``preset="earth_orbit"``, ``preset="cislunar"``, or
``preset="all"`` and override individual options as needed. Pass
``gravity_gradient=True`` for central-Earth gravity-gradient torque or
``gravity_gradient="all"`` for Earth/Moon/Sun torque models. Its default conical
eclipse model uses
disk-overlap geometry for Earth and Moon occultation of the Sun; set
``solar_occulting_bodies=("earth",)`` or ``eclipse_model=None`` for simpler SRP
studies. Its default magnetic field is a centered Earth dipole in GCRF using
``EARTH_DIPOLE_EQUATOR_FIELD``; use ``magnetic_field_model="igrf"`` for
optional ``ppigrf``-backed IGRF field synthesis. Set ``epoch=...`` when
propagation times are relative seconds that should map onto an absolute
calendar date.

For normal finite satellite maneuvers, use ``SpacecraftManeuverAccel`` with an
explicit ``frame``. ``frame="rtn"``/``"lvlh"``/``"ric"`` maps to the common
radial-transverse-normal operations convention, ``frame="vnb"`` maps to
velocity-normal-binormal, ``frame="body"`` maps body-mounted thrust through the
current attitude, and ``frame="ntw"`` preserves exact SSAPy ``[N, T, W]``
convention. Thrust can be constant, trapezoidal, smoothstep, exponential,
pulsed, callable, or loaded from a CSV file through ``ThrustCurve``. Citable
engine curves should live in the propulsion data package, not this source repository, and can be
loaded with ``load_thrust_curve_data(...)`` once packaged.

Representative propulsion presets live in :mod:`ssatk.engines` and
launch-vehicle presets live in :mod:`ssatk.launch`. Use
``available_thruster_families()``, ``available_thruster_specs(...)``, and
``thruster_spec(...)`` to select cold-gas, monopropellant, bipropellant, solid
kick-motor, liquid, Hall-effect, gridded-ion, resistojet, arcjet,
electrospray, or dual-mode chemical/electric propulsion classes.  Each
``ThrusterSpec`` stores representative thrust and specific-impulse ranges,
power and dry-mass ranges where applicable, and builders for
``Thruster`` body components or ``SpacecraftManeuverAccel`` finite burns.
These values are engineering-scale defaults for analysis setup; replace them
with vendor or mission thrust curves from ``ssatk-data-propulsion`` when flight-specific
validation is required.

Preset body designs live in :mod:`ssatk.satellites`. Use
``satellite_design(...)`` to start from a common bus, override dimensions or
mass, then add components, tanks, facets, thrusters, magnetic dipoles, or
reaction wheels as needed. ``Spacecraft`` uses the body's aggregate mass,
center of mass, and inertia when those values are not provided directly.

.. code-block:: python

   from ssatk.accelerations_6dof import (
       SpacecraftFacetDrag,
       SpacecraftFacetSolRad,
       SpacecraftManeuverAccel,
       SpacecraftThrusterAccel,
   )

   body = ssatk.satellite_design(
       "earth_observation",
       mass=500.0,
       solar_array_area=10.0,
   ).with_thrusters(
       ssatk.Thruster(thrust=0.2, direction_body=[1, 0, 0], position_body=[0, 0.5, 0]),
   ).with_components(
       ssatk.Component(mass=25.0, position_body=[0.0, 0.0, 0.7], name="payload"),
   ).with_magnetic_dipoles(
       ssatk.MagneticDipole(moment_body=[0.2, 0.0, 0.0], name="x_magnetorquer"),
   ).with_reaction_wheels(
       *ssatk.reaction_wheel_triplet(max_torque=0.02),
   )

   hall = ssatk.thruster_spec("hall_effect", scale="small")
   body = body.with_thrusters(
       hall.to_thruster(direction_body=[1, 0, 0], position_body=[0, 0.7, 0]),
   )

   sat = ssatk.Spacecraft(r=[7e6, 0, 0], v=[0, 7500, 0], body=body)
   q_target = ssatk.attitude_quaternion_from_frame("nadir_velocity", r=sat.r, v=sat.v)
   burn = hall.maneuver_acceleration(
       start=120.0,
       burn_time=60.0,
       rise_time=5.0,
       frame="rtn",
       direction=[0, 1, 0],
       mass=body.current_mass,
   )
   traj = sat.propagate(
       times=np.linspace(0.0, 600.0, 61),
       models=[
           SpacecraftFacetDrag(density=1e-12),
           SpacecraftFacetSolRad([ssatk.AU, 0, 0]),
           burn,
           ssatk.SpacecraftMagneticTorque([0, 2e-5, 0]),
           ssatk.SpacecraftReactionWheelTorque([0, 0, 0.01]),
           ssatk.SpacecraftAttitudePD(q_target=q_target, kp=0.05, kd=0.2, max_torque=0.02),
           SpacecraftThrusterAccel(),
       ],
   )

Packaged data
-------------

SSATK should not commit reusable datasets, generated figures, or other
binary artifacts. Toolkit functions that require reusable data should read it
from the installed split data dependencies instead. These packages expose
the ``ssapy_data`` import package with resources below ``ssapy_data/data``.

Use :mod:`ssatk.data` when a toolkit function needs a packaged data
file:

.. code-block:: python

   from ssatk.data import data_path, read_data_text

   catalog_text = read_data_text("catalogs/example.csv")

   with data_path("catalogs/example.csv") as catalog_path:
       # Pass catalog_path to libraries that require a filesystem path.
       print(catalog_path)

This keeps ``SSATK`` source-only while allowing users to get required
data through normal ``pip`` installation.

Optional demo data
------------------

Demo-only files should also stay out of the repository. Demos that need public
sample data can call :func:`ssatk.io.demo_data.ensure_demo_data_file`,
which first checks the local ``ssatk_output`` cache, then downloads from a known
public source when internet access is available. When the file cannot be found
or fetched, the helper emits ``DemoDataUnavailableWarning`` and returns
``None`` so demos and tests can skip gracefully instead of failing hard.
