
import numpy as np
import pytest

from ssatk.io.ccsds_cdm import (
    format_cdm,
    read_cdm,
)

_COVARIANCE = """\
CR_R                          = 1.0 [m**2]
CT_R                          = 0.0 [m**2]
CT_T                          = 4.0 [m**2]
CN_R                          = 0.0 [m**2]
CN_T                          = 0.0 [m**2]
CN_N                          = 9.0 [m**2]
CRDOT_R                       = 0.0 [m**2/s]
CRDOT_T                       = 0.0 [m**2/s]
CRDOT_N                       = 0.0 [m**2/s]
CRDOT_RDOT                    = 0.01 [m**2/s**2]
CTDOT_R                       = 0.0 [m**2/s]
CTDOT_T                       = 0.0 [m**2/s]
CTDOT_N                       = 0.0 [m**2/s]
CTDOT_RDOT                   = 0.0 [m**2/s**2]
CTDOT_TDOT                    = 0.04 [m**2/s**2]
CNDOT_R                       = 0.0 [m**2/s]
CNDOT_T                       = 0.0 [m**2/s]
CNDOT_N                       = 0.0 [m**2/s]
CNDOT_RDOT                    = 0.0 [m**2/s**2]
CNDOT_TDOT                    = 0.0 [m**2/s**2]
CNDOT_NDOT                    = 0.09 [m**2/s**2]
"""


def _object(label, designator, frame="GCRF", epoch=None):
    epoch_line = "" if epoch is None else f"EPOCH                         = {epoch}\n"
    return f"""OBJECT                        = {label}
OBJECT_DESIGNATOR             = {designator}
CATALOG_NAME                  = SATCAT
OBJECT_NAME                   = TEST {label}
INTERNATIONAL_DESIGNATOR      = 2020-001A
EPHEMERIS_NAME                = TEST
COVARIANCE_METHOD             = CALCULATED
MANEUVERABLE                  = NO
REF_FRAME                     = {frame}
TIME_SYSTEM                   = UTC
{epoch_line}X                             = 7000.0 [km]
Y                             = 100.0 [km]
Z                             = 50.0 [km]
X_DOT                         = -0.1 [km/s]
Y_DOT                         = 7.5 [km/s]
Z_DOT                         = 0.2 [km/s]
USER_DEFINED                  = retained
{_COVARIANCE}"""


def _fixture(frame="GCRF", epoch=None):
    return f"""CCSDS_CDM_VERS                = 1.0
CREATION_DATE                 = 2025-01-01T00:00:00.000
ORIGINATOR                    = TEST
MESSAGE_ID                    = TEST-MESSAGE-1
COMMENT header comment
TCA                           = 2025-01-02T00:00:00.000
MISS_DISTANCE                 = 715 [m]
RELATIVE_SPEED                = 12.5 [m/s]
RELATIVE_POSITION_R           = 1 [m]
RELATIVE_POSITION_T           = 2 [m]
RELATIVE_POSITION_N           = 3 [m]
RELATIVE_VELOCITY_R           = 4 [m/s]
RELATIVE_VELOCITY_T           = 5 [m/s]
RELATIVE_VELOCITY_N           = 6 [m/s]
COLLISION_PROBABILITY         = 1.0e-4
CONJUNCTION_ID                = TEST-1
MESSAGE_EXTRA                 = preserved
{_object("OBJECT1", "ONE", frame, epoch)}
{_object("OBJECT2", "TWO", frame, epoch).replace("Y                             = 100.0", "Y                             = 101.0")}"""


def test_read_standard_layout_and_si_conversion():
    message = read_cdm(_fixture())

    assert message.object1.state.tolist() == [7_000_000.0, 100_000.0, 50_000.0, -100.0, 7_500.0, 200.0]
    assert message.object2.state[1] == 101_000.0
    assert message.object1.epoch == message.tca
    assert message.miss_distance_m == 715.0
    assert message.relative_position_rtn_m.flags.writeable is False
    assert message.extra_fields["MESSAGE_EXTRA"] == "preserved"
    assert message.object1.extra_fields["USER_DEFINED"] == "retained"


def test_gcrf_rtn_covariance_rotation_and_immutable_arrays():
    message = read_cdm(_fixture())
    obj = message.object1

    assert obj.covariance_ref_frame == "RTN"
    rotated = obj.position_covariance_gcrf()
    assert np.allclose(rotated, rotated.T, atol=1e-12, rtol=0)
    assert np.allclose(np.linalg.eigvalsh(rotated), [1.0, 4.0, 9.0], atol=1e-12, rtol=0)
    r, v = np.array([7e6, 1e5, 5e4]), np.array([-100., 7500., 200.])
    radial = r/np.linalg.norm(r)
    normal = np.cross(r, v)/np.linalg.norm(np.cross(r, v))
    transverse = np.cross(normal, radial)
    expected = np.outer(radial, radial) + 4*np.outer(transverse, transverse) + 9*np.outer(normal, normal)
    np.testing.assert_allclose(rotated, expected, atol=1e-12, rtol=0)
    with pytest.raises(ValueError):
        obj.state[0] = 0.0
    with pytest.raises(TypeError):
        message.extra_fields["new"] = "value"


@pytest.mark.parametrize("frame", ["EME2000", "ITRF"])
def test_supported_frames_transform_to_gcrf(frame):
    message = read_cdm(_fixture(frame=frame))
    obj = message.object1
    assert obj.reference_frame == frame
    import erfa
    from astropy import units as u
    from astropy.time import TimeDelta
    from ssatk.environment_eop import load_packaged_eop
    # The ITRF reference is assembled here from low-level ERFA routines
    # (CIP X,Y plus the IERS dX,dY offsets, s06, c2ixys, era00, sp00, pom00,
    # c2tcio) with SSATK's packaged EOP, independent of the reader's code
    # path. Check both positions and rotating velocities.
    eop = load_packaged_eop()
    arcsec = np.pi / (180.0 * 3600.0)

    def rotation(time):
        if frame == "EME2000":
            return erfa.bp00(2451545.0, 0.0)[0].T
        record = eop.at(time, allow_predicted=True)
        tt, ut1 = time.tt, time.utc + record.ut1_minus_utc_s * u.s
        x, y, _ = erfa.xys06a(tt.jd1, tt.jd2)
        x += (record.dx_mas or 0.0) * arcsec / 1000.0
        y += (record.dy_mas or 0.0) * arcsec / 1000.0
        c2i = erfa.c2ixys(x, y, erfa.s06(tt.jd1, tt.jd2, x, y))
        pom = erfa.pom00(record.polar_motion_x_arcsec * arcsec, record.polar_motion_y_arcsec * arcsec,
                         erfa.sp00(tt.jd1, tt.jd2))
        return erfa.c2tcio(c2i, erfa.era00(ut1.jd1, ut1.jd2), pom).T
    matrix = rotation(obj.epoch)
    dt = TimeDelta(.5, format="sec")
    rate = rotation(obj.epoch+dt) - rotation(obj.epoch-dt)
    expected_r = matrix @ obj.state[:3]
    expected_v = matrix @ obj.state[3:] + rate @ obj.state[:3]
    np.testing.assert_allclose(obj.state_gcrf()[:3], expected_r, atol=.002, rtol=0)
    np.testing.assert_allclose(obj.state_gcrf()[3:], expected_v, atol=1e-4, rtol=0)


def test_optional_epoch_and_writer_round_trip():
    epoch = "2025-01-01T12:34:56.000"
    message = read_cdm(_fixture(epoch=epoch))
    text = format_cdm(message)
    for field in ("META_START", "META_STOP", "DATA_START", "DATA_STOP", "COVARIANCE_START", "EPOCH", "TIME_SYSTEM", "COV_REF_FRAME"):
        assert field not in text
    assert "[1]" not in text
    assert "COMMENT header comment" in text
    assert "COMMENT                        =" not in text
    result = read_cdm(text)
    assert result.object1.epoch == result.tca
    assert np.allclose(result.object2.covariance_rtn, message.object2.covariance_rtn)
