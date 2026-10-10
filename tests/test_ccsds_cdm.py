
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
    assert np.allclose(rotated, rotated.T, atol=1e-12)
    assert np.allclose(np.linalg.eigvalsh(rotated), [1.0, 4.0, 9.0], atol=1e-12)
    with pytest.raises(ValueError):
        obj.state[0] = 0.0
    with pytest.raises(TypeError):
        message.extra_fields["new"] = "value"


@pytest.mark.parametrize("frame", ["EME2000", "ITRF"])
def test_supported_frames_transform_to_gcrf(frame):
    message = read_cdm(_fixture(frame=frame))
    obj = message.object1
    assert obj.reference_frame == frame
    assert not np.allclose(obj.state_gcrf(), obj.state, rtol=0.0, atol=1e-8)


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
