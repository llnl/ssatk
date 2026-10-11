Test reference audit
====================

This audit covers all 99 ``tests/test_*.py`` files and their 359 test
functions, including methods in test classes. Parametrization produces more
executed cases than this function count. ``tests/conftest.py`` supplies the
shared GPS epochs; it contains no test functions. The reviewed starting commit
is ``f05e171fe4ccd2d0c693ffef9e9d6d23d85b70f5``.

The suite contains independent numerical references, conservation and
derivative checks, and software contracts. Passing all three categories does
not establish measured accuracy for every physical model. In particular,
agreement with a directly called SSAPy force validates an adapter, while
agreement with SSAPy's Kepler solver validates a separate integration method
for the same point-mass equations. Neither comparison validates atmospheric
density or spacecraft properties against measurements.

Changes from the audit
----------------------

* Attitude-error state transition matrices and covariance now use an analytic
  isotropic torque-free body solution, including the rotating body-error axes.
  The inertia-rate case prescribes consistent ``I(t)`` and ``dI/dt`` and checks
  conservation of angular momentum.
* Lunar position and velocity transformations now have a prescribed circular
  ephemeris with exact rotation and rotation-rate matrices. This tests frame
  kinematics independently of the ephemeris implementation. Real lunar
  ephemeris comparisons with Astropy remain separate.
* Inclination burns and the station-keeping example use separate Radau
  integrations. Tightening reference tolerances by a factor of ten must leave
  reference states or semimajor-axis changes within the stated error budget.
  Collision probability uses a noncentral chi-square distribution and a
  converged polar cubature independent of the production integral.
* Maneuver tests derive vis-viva burn speeds directly instead of calling the
  production maneuver helper. The unexplained stored Lambert delta-v is
  replaced by independently propagated position and velocity boundary checks.
* The extended Kalman filter checks the conditional-Gaussian mean and complete
  covariance. Flexible modes use exact harmonic-oscillator solutions; mass
  depletion, propulsion, reaction wheels, and radiation pressure use analytic
  balances or force equations.
* Atmosphere density comparisons disable the default absolute floor that
  previously allowed zero density. MSIS wrapper tests compare versions 0 and
  2.1 with direct pymsis calls, raw packaged weather records, and independent
  WGS84 geodetic conversion. They do not claim empirical density validation.
* Earth orientation and CDM frame conversion gain independent ERFA matrix
  comparisons. Nadir and ecliptic tests check actual axis directions as well
  as orthogonality or inverse round trips.

Tolerance interpretation
------------------------

NumPy compares against ``atol + rtol*abs(reference)``. Tests stating only an
absolute tolerance now set ``rtol=0`` explicitly; for example, a millimetre
position budget must not quietly acquire a metre-scale allowance at orbital
distances. See `NumPy assert_allclose
<https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose.html>`_.

Purely relative ``pytest.approx`` checks now set ``abs=0``. Explicit
``pytest.approx(abs=...)`` already disables its default relative tolerance;
those assertions did not have the NumPy issue. See `pytest approx
<https://docs.pytest.org/en/stable/reference/reference.html#pytest-approx>`_.

Three approximation budgets need special treatment. The SSAPy-style
linearized polar-motion matrix has a squared-angle orthogonality error bounded
by ``xp**2 + yp**2``. Two-part Julian-date subtraction is allowed 10 ps in the
OEM time round trip. The bounded targeting optimizer has a stated
``1e-7 m/s`` terminal-velocity budget, while its constant-acceleration dynamics
are separately checked to ``1e-12 m/s``. These are model, representation, and
optimization budgets, respectively.

The photometry oracle uses the same *geometric* Sun position as a specified
input and computes phase and Planck-band flux independently. Astropy's
*apparent* Sun is checked with an angular allowance for aberration; treating
the two positions as identical had made the previous flux reference invalid.
The ERFA Earth comparison similarly permits the documented IAU 1976/1980
versus IAU 2006/2000A model difference. The latter matrix is documented in
`ERFA c2t06a <https://github.com/liberfa/erfa/blob/master/src/c2t06a.c>`_.

Limits that remain explicit
---------------------------

* **Empirical asteroid models:** the LSST color table is verified against
  Chesley and Veres (2017), JPL Publication 16-11, Table 8, p. 18
  (`NASA report <https://ntrs.nasa.gov/api/citations/20170008014/downloads/20170008014.pdf>`_).
  That source does not validate reusing LSST colors for ZTF. A source or
  calibration data set for that approximation and for the two Granvik linear
  fits was not established. Existing checks of those coefficients are retained
  as model-contract regressions, explicitly marked **unverified as empirical
  calibrations**. No replacement coefficients are invented.
* **Observed orbit determination:** current batch and filter references use
  synthetic truth and known measurement models. No independent real tracking
  data set with a reference orbit was supplied. The tests validate equations,
  geometry, and inference for those constructed cases.
* **Optional tools:** Basilisk, Orekit/EIGEN-6S, and SpacePy/IRBEM checks count
  only when actually executed with their required tools/data. Report skips
  separately. Installing an import alias validates compatibility only.
* **Conservation and broad bounds:** epoch invariance, round trips, angular
  momentum, libration bounds, and geographic magnetic-field bounds constrain
  specific properties. They are not independent measurements of every output.
  The inventory names paired numerical references where available.
* **Synthetic fixtures:** CCSDS messages, model spacecraft, prescribed
  ephemerides, and noise realizations are controlled test inputs. An
  Orekit-shaped text fixture is not an Orekit-computed trajectory. A passing
  simplified CubeSat inertia test does not certify a flight vehicle.

Complete file inventory
-----------------------

R1 denotes an analytic equation; R2 a separate implementation or backend;
R3 a published value; R4 an invariant; R5 a derivative comparison; R6 a
numerical regression. C denotes a software/model contract and U an unresolved
empirical reference. Multiple labels mean a file contains different kinds of
checks, not that every assertion has every kind of independent reference.

.. csv-table:: References and scope for every test file
   :file: test_reference_audit.csv
   :header-rows: 1
   :widths: 24, 5, 10, 32, 29

The inventory is a review record, not a proof of completeness over all
possible inputs. Revisit it when adding tests, changing conventions, updating
ephemerides or reference packages, or changing acceptance tolerances.
