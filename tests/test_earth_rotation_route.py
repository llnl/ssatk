"""Reference tests for the single IAU 2006/2000A GCRS<->ITRS route.

See ``ssatk/coordinates/_earth_rotation.py`` for the model this replaces
(IAU 1976/1980 via ``ssapy.groundTrack``, and a separate Astropy-based IAU
2006/2000A path) and why the two used to disagree.
"""

import numpy as np
import pytest
from astropy.coordinates import GCRS, ITRS, CartesianDifferential, CartesianRepresentation
from astropy.time import Time
import astropy.units as u

from ssatk.constants import WGS84_EARTH_OMEGA
from ssatk.coordinates import gcrf_to_itrf, itrf_to_gcrf
from ssatk.coordinates.geodetic import llh_to_gcrf
from ssatk.data import data_path
from ssatk.environment_eop import EarthOrientationExtrapolationWarning

iers = pytest.importorskip("astropy.utils.iers")

# Epochs at exact packaged-EOP table midnights, all on the observed (not
# predicted) side of the snapshot as of this change (see
# ssatk/environment_eop.py's `predicted` flag), so no interpolation or
# allow_predicted opt-in is needed on either side of the comparison.
_EPOCHS = Time(["2026-07-20T00:00:00", "2026-07-25T00:00:00", "2026-08-01T00:00:00"], scale="utc")
_POSITIONS_M = np.array([
    [7_000e3, 1_000e3, -500e3],
    [-4_000e3, 5_000e3, 3_000e3],
    [6_878e3, 0.0, 0.0],
])
_VELOCITIES_M_S = np.array([
    [-1_000.0, 7_300.0, 1_500.0],
    [-3_000.0, -4_000.0, 5_000.0],
    [0.0, 7_612.6, 0.0],
])

# R2: measured maxima over _POSITIONS_M/_EPOCHS are 0.0147 m and 1.3e-4 m/s.
# Astropy's built-in GCRS<->ITRS transform does not apply the IERS celestial
# pole offsets (dX, dY) that ssatk.coordinates._earth_rotation adds from the
# same finals2000A.all file (confirmed by zeroing dX/dY in a standalone
# comparison, which brings agreement to ~0.1 mm); that correction, not a
# model or unit error, is almost all of this residual, and it scales with
# distance from Earth's center as a fixed small-angle rotation would.
_POSITION_TOLERANCE_M = 0.02
_VELOCITY_TOLERANCE_M_S = 2e-4


def _astropy_reference():
    """Configure Astropy's active EOP table to SSATK's packaged snapshot."""
    with data_path("environment/eop/finals2000A.all") as path:
        table = iers.IERS_A.open(str(path))

    representation = CartesianRepresentation(
        _POSITIONS_M.T * u.m,
        differentials=CartesianDifferential(_VELOCITIES_M_S.T * u.m / u.s),
    )
    # Scoped so the packaged table does not leak into later tests.
    with iers.conf.set_temp("auto_download", False), iers.earth_orientation_table.set(table):
        transformed = GCRS(representation, obstime=_EPOCHS).transform_to(ITRS(obstime=_EPOCHS))
    position = transformed.cartesian.xyz.to_value(u.m).T
    velocity = transformed.cartesian.differentials["s"].d_xyz.to_value(u.m / u.s).T
    return position, velocity


def test_gcrf_to_itrf_matches_astropy_configured_with_the_packaged_eop_table():
    """R2: Astropy's own GCRS->ITRS transform, same packaged finals2000A.all.

    Tolerances are the measured maxima (see the module-level comment); they
    are not loosened beyond what reproducibly occurs from the dX/dY
    difference explained there.
    """
    expected_position, expected_velocity = _astropy_reference()

    position, velocity = gcrf_to_itrf(
        _POSITIONS_M, _EPOCHS.gps, v=_VELOCITIES_M_S, allow_predicted=False
    )

    np.testing.assert_allclose(position, expected_position, rtol=0, atol=_POSITION_TOLERANCE_M)
    np.testing.assert_allclose(velocity, expected_velocity, rtol=0, atol=_VELOCITY_TOLERANCE_M_S)


def test_position_only_conversions_match_astropy_in_both_directions():
    """R2: the position-only branches (no velocity argument) use a separate
    code path from the state conversion above, so they are checked against
    the same Astropy reference: GCRF->ITRF directly, and ITRF->GCRF by
    mapping Astropy's ITRS positions back to the original GCRF inputs.
    """
    expected_position, _ = _astropy_reference()

    position = gcrf_to_itrf(_POSITIONS_M, _EPOCHS.gps, allow_predicted=False)
    np.testing.assert_allclose(position, expected_position, rtol=0, atol=_POSITION_TOLERANCE_M)

    recovered = itrf_to_gcrf(expected_position, _EPOCHS.gps, allow_predicted=False)
    np.testing.assert_allclose(recovered, _POSITIONS_M, rtol=0, atol=_POSITION_TOLERANCE_M)


def test_gcrf_to_itrf_round_trips_through_itrf_to_gcrf():
    """Paired with the Astropy comparison above: applying the inverse rotation
    to its own output must recover the original GCRF position. This alone
    would also pass for any orthogonal rotation (even a wrong one), which is
    why it is paired with an independent reference rather than used alone.
    """
    position, _ = gcrf_to_itrf(_POSITIONS_M, _EPOCHS.gps, v=_VELOCITIES_M_S, allow_predicted=False)
    recovered = itrf_to_gcrf(position, _EPOCHS.gps, allow_predicted=False)

    np.testing.assert_allclose(recovered, _POSITIONS_M, rtol=0, atol=1e-6)


@pytest.mark.parametrize("lat_deg", [0.0, 30.0, 60.0, -45.0])
def test_stationary_wgs84_surface_point_speed_matches_omega_times_rho(lat_deg):
    """R1: a WGS84 surface point's GCRF speed is omega_E * rho, rho the
    distance from the spin axis (not the full radius, off the equator).

    The ~1e-4 m/s tolerance is the measured accuracy of the central-difference
    dR/dt in ssatk.coordinates._earth_rotation.gcrs_to_itrs_matrix_and_rate
    (see its module-level comment on ``_RATE_STEP_S``), not the originally
    targeted 1e-6 m/s: that finite difference is limited by float64
    precision in erfa.era00's own Julian-date arguments, measured to bottom
    out at a few 1e-5 to 1e-4 m/s at Earth's surface regardless of step size.
    """
    lon_deg = 10.0
    r_gcrf, v_gcrf = llh_to_gcrf(lon_deg, lat_deg, _EPOCHS[0], height=0.0, allow_predicted=False)

    # rho is the ITRF cylindrical radius (distance from the spin axis), which
    # WGS84 geodetic geometry gives directly at height=0; the GCRF position
    # above has already been rotated, so its own x/y do not give rho without
    # also knowing the (unrotated) spin-axis orientation in GCRF.
    import erfa

    r_itrf = np.asarray(erfa.gd2gc(1, np.radians(lon_deg), np.radians(lat_deg), 0.0))
    rho = np.hypot(r_itrf[0], r_itrf[1])

    expected_speed = WGS84_EARTH_OMEGA * rho
    assert np.linalg.norm(v_gcrf) == pytest.approx(expected_speed, abs=5e-4)


# ---------------------------------------------------------------------------
# Epochs outside the packaged EOP table's span.
# ---------------------------------------------------------------------------
#
# Strict functions (gcrs_to_itrs_matrix*, gcrf_to_itrf_eop, itrf_to_gcrf_eop)
# raise outside the table. Lenient public functions (gcrf_to_itrf,
# itrf_to_gcrf, gcrf_to_itrf_astropy, eci_to_ecf_matrix, and the
# geodetic/yastropy conversions) hold the nearest end record's values and
# warn once with EarthOrientationExtrapolationWarning instead -- matching
# SSAPy's iers_interp, which held the table's end values via numpy.interp
# rather than raising.


def _far_future_gps(eop):
    return eop.end_gps + 10.0 * 365.25 * 86400.0  # ten years past the table


def test_strict_core_functions_raise_outside_the_eop_table():
    from ssatk.coordinates._earth_rotation import gcrs_to_itrs_matrix, gcrs_to_itrs_matrix_and_rate
    from ssatk.coordinates.earth_fixed import gcrf_to_itrf_eop, itrf_to_gcrf_eop
    from ssatk.environment_eop import load_packaged_eop

    eop = load_packaged_eop()
    future = _far_future_gps(eop)

    with pytest.raises(ValueError, match="outside"):
        gcrs_to_itrs_matrix(future, allow_predicted=True)
    with pytest.raises(ValueError, match="outside"):
        gcrs_to_itrs_matrix_and_rate(future, allow_predicted=True)
    with pytest.raises(ValueError, match="outside"):
        gcrf_to_itrf_eop(_POSITIONS_M[:1], [future], allow_predicted=True)
    with pytest.raises(ValueError, match="outside"):
        itrf_to_gcrf_eop(_POSITIONS_M[:1], [future], allow_predicted=True)


def test_core_extrapolation_holds_the_table_end_record_exactly():
    """The extrapolated rotation (and its rate) must equal evaluating the
    core with a synthetic table that simply continues the real table's last
    record's UT1-UTC/polar motion/dX,dY out to the requested epoch -- an
    exact match, not merely "close", since both use the same held values and
    the same (true, un-held) TT/UT1-date arguments for precession-nutation
    and the Earth Rotation Angle.
    """
    from dataclasses import replace

    from ssatk.coordinates._earth_rotation import gcrs_to_itrs_matrix, gcrs_to_itrs_matrix_and_rate
    from ssatk.environment_eop import EarthOrientationTable, load_packaged_eop

    eop = load_packaged_eop()
    future = _far_future_gps(eop)
    end_record = eop.records[-1]
    held_table = EarthOrientationTable((
        replace(end_record, gps_seconds=eop.start_gps),
        replace(end_record, gps_seconds=future + 86400.0),
    ))

    with pytest.warns(EarthOrientationExtrapolationWarning, match="outside the packaged EOP table"):
        actual = gcrs_to_itrs_matrix(future, allow_predicted=True, extrapolate=True)
    expected = gcrs_to_itrs_matrix(future, eop=held_table, allow_predicted=True)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)

    with pytest.warns(EarthOrientationExtrapolationWarning):
        actual_r, actual_rate = gcrs_to_itrs_matrix_and_rate(future, allow_predicted=True, extrapolate=True)
    expected_r, expected_rate = gcrs_to_itrs_matrix_and_rate(future, eop=held_table, allow_predicted=True)
    np.testing.assert_allclose(actual_r, expected_r, rtol=0, atol=0)
    np.testing.assert_allclose(actual_rate, expected_rate, rtol=0, atol=0)


def test_lenient_public_functions_warn_and_hold_outside_the_eop_table():
    from ssatk.coordinates import (
        eci_to_ecf_matrix,
        gcrf_to_itrf,
        gcrf_to_itrf_astropy,
        gcrf_to_llh,
        itrf_to_gcrf,
        llh_to_gcrf,
    )
    from ssatk.environment_eop import load_packaged_eop
    from ssatk.yastropy import astropy_gcrf_to_llh, astropy_llh_to_gcrf

    eop = load_packaged_eop()
    future = _far_future_gps(eop)
    r = _POSITIONS_M[:1]

    with pytest.warns(EarthOrientationExtrapolationWarning):
        gcrf_to_itrf(r, [future])
    with pytest.warns(EarthOrientationExtrapolationWarning):
        itrf_to_gcrf(r, [future])
    with pytest.warns(EarthOrientationExtrapolationWarning):
        gcrf_to_itrf_astropy(r, [future])
    with pytest.warns(EarthOrientationExtrapolationWarning):
        eci_to_ecf_matrix(future)
    with pytest.warns(EarthOrientationExtrapolationWarning):
        gcrf_to_llh(r, [future])
    with pytest.warns(EarthOrientationExtrapolationWarning):
        llh_to_gcrf(0.0, 0.0, future)
    with pytest.warns(EarthOrientationExtrapolationWarning):
        astropy_llh_to_gcrf(lon=[0.0], lat=[0.0], t=future)
    with pytest.warns(EarthOrientationExtrapolationWarning):
        astropy_gcrf_to_llh(r, [future])
