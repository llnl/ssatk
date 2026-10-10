import numpy as np
from astropy.time import Time

from ssatk.coordinates.earth_fixed import (
    gcrf_to_itrf_eop,
    itrf_to_gcrf_eop,
)
from ssatk.environment_eop import (
    EarthOrientationRecord,
    EarthOrientationTable,
)


def test_eop_frame_transform_round_trips_a_batch_of_states():
    epoch = float(Time(60310.0, format="mjd", scale="utc").gps)
    table = EarthOrientationTable(
        (
            EarthOrientationRecord(
                mjd_utc=60310.0,
                gps_seconds=epoch,
                polar_motion_x_arcsec=0.1,
                polar_motion_y_arcsec=0.2,
                ut1_minus_utc_s=0.3,
                polar_motion_flag="I",
                ut1_flag="I",
                nutation_flag="I",
            ),
            EarthOrientationRecord(
                mjd_utc=60311.0,
                gps_seconds=epoch + 86_400.0,
                polar_motion_x_arcsec=0.3,
                polar_motion_y_arcsec=0.4,
                ut1_minus_utc_s=0.5,
                polar_motion_flag="I",
                ut1_flag="I",
                nutation_flag="I",
            ),
        )
    )
    times = epoch + np.linspace(0.0, 21_600.0, 256)
    angles = np.linspace(0.0, 2.0 * np.pi, times.size, endpoint=False)
    positions = 7_000_000.0 * np.column_stack(
        (np.cos(angles), np.sin(angles), np.zeros_like(angles))
    )

    itrf = gcrf_to_itrf_eop(positions, times, eop=table)
    recovered = itrf_to_gcrf_eop(itrf, times, eop=table)

    assert itrf.shape == positions.shape
    assert np.all(np.isfinite(itrf))
    np.testing.assert_allclose(recovered, positions, rtol=0.0, atol=2.0e-5)
