# Testing Policy

Numerical tests must compare computed quantities with named analytic,
published, or independently implemented references under stated tolerances.
Conservation and derivative tests constrain particular properties; neither a
round trip nor agreement between two SSATK runs establishes absolute accuracy.
Pair those checks with an independent numerical reference for the same model.

Compatibility, serialization, callback, and input-contract tests have a
different purpose. Keep them clearly identified as contract checks; they do
not count as physical validation. Returning an object or writing a file alone
does not validate a numerical result. The complete test-file inventory and
known limitations are recorded in `test_reference_audit.rst`.

## Accepted references

| Code | Reference | Examples in `tests/` |
|---|---|---|
| R1 | Closed-form or analytic solution | Hohmann and bi-elliptic delta-v, J2 RAAN rate, dipole field limits, Lambert on a circular orbit |
| R2 | Independent implementation | astropy, SGP4 `Satrec`, geopack, IGRF, MSIS, REBOUND IAS15, Orekit, Basilisk, native SSAPy |
| R3 | Published or real-world value | textbook LEO-to-GEO transfer, ISS TLE fields, 2024-04-08 eclipse path, IAU and IGRF constants |
| R4 | Conservation law or physical invariant | angular momentum with reaction wheels, propellant mass across segments, rotation orthonormality, epoch invariance |
| R5 | Analytic derivative against finite differences | STM and Jacobian checks in `test_6dof_variational.py` |
| R6 | Regression for a shipped numerical bug | terminal-event roots within one ULP, absolute epochs reaching force models |

Name the reference in the test name or docstring, and state the tolerance with
its unit.

Use `rtol=0` for an absolute NumPy tolerance and `abs=0` for a purely relative
`pytest.approx` comparison, especially for densities, fluxes, accelerations,
and other small quantities. Explicit `pytest.approx(abs=...)` already disables
its default relative tolerance. Explain model approximation, discretization,
or floating-point error budgets; do not enlarge tolerances merely to pass.
Converge numerical reference integrations before using them as an oracle.
Synthetic trajectories and specified physical parameters are valid analytic
test cases, but must not be described as measured flight or laboratory data.

## Not tested in `tests/`

- Plot, viewer, and HTML rendering, and demo execution. Demos run in the
  gallery workflow, on pull requests and on pushes to `main`.
  Selected geometry and rendering contracts in `scripts/tests` also run in
  the Python CI matrix with its installed Matplotlib version.
- Argument validation and error messages, unless the rejected input is
  physically meaningful.
- Code paths exercised only to raise a coverage percentage.

## External references in CI

CI installs the `geomagnetics` and `atmosphere` extras, so the geopack, IGRF,
and MSIS comparisons run. The spacepy/IRBEM, Basilisk, and Orekit comparisons
skip unless those tools are installed locally.

## Coverage

Coverage is a diagnostic, not a gate. `scripts/audit_public_api_coverage.py`
still reports which functions no test reaches. Use it to find untested physics,
then add a reference test for that physics rather than a test that only
executes the code.
