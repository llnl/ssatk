"""Epoch comparisons must not carry a scale-dependent relative tolerance.

The 6-DoF propagators compare absolute epochs in three places: the terminal
event de-duplication in ``propagate_6dof``, the segment-start check in
``_propagate_spacecraft_segment``, and the completeness check in
``_require_complete_segment``. All three used ``np.isclose`` defaults, whose
``rtol=1e-5`` is a tolerance of hours once the epochs are absolute GPS
seconds.
"""

import numpy as np
import pytest

from ssatk.propagators_6dof.sixdof import _epochs_close

# GPS seconds. 1.4e9 is 2024-05-13; 3.8e9 is 2100-06-20.
GPS_2024 = 1_400_000_000.0
GPS_2100 = 3_800_000_000.0

EPOCHS = pytest.mark.parametrize("epoch", [0.0, GPS_2024, GPS_2100])


@EPOCHS
def test_millisecond_mismatch_is_rejected(epoch):
    assert not _epochs_close(epoch, epoch + 1.0e-3)


def test_ulp_term_is_what_makes_year_2100_work():
    """Pin the reason the ULP term exists, not just its effect."""
    nudged = GPS_2100 + 4.0 * np.spacing(GPS_2100)
    assert not np.isclose(GPS_2100, nudged, rtol=0.0, atol=1.0e-6)
    assert _epochs_close(GPS_2100, nudged)
