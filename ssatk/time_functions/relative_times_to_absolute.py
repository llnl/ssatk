import numpy as np
from astropy.time import Time, TimeDelta


def time_rel_to_abs(times, ref, anchor='start'):
    """
    Convert relative times (seconds) into absolute astropy Time stamps.

    Parameters
    ----------
    times : array-like of float
        Sequence of time stamps in seconds. These are treated as relative
        offsets along a time series (not necessarily starting at 0).
    ref : astropy.time.Time or str or float/int (optional)
        The reference timestamp to anchor the series.
        - If Time: used directly.
        - If str: parsed by astropy as UTC, e.g. "2025-01-01 00:00:00".
        - If float/int: interpreted as GPS seconds.
    anchor : {'start', 'end'}, default 'start'
        Which sample the reference corresponds to:
        - 'start': ref is the absolute time of times[0]
        - 'end'  : ref is the absolute time of times[-1]

    Returns
    -------
    astropy.time.Time
        Vector Time object of the same length as `times`, with absolute stamps.

    Notes
    -----
    - This mirrors the behavior of a typical get_times helper: evenly applies
      offsets in seconds to a reference Time.
    - The relative sequence does not need to start at 0. Offsets are computed
      against the chosen anchor sample.
    """
    # Normalize times to a 1D numpy array of float
    times = np.asarray(times, dtype=float).ravel()
    if times.size == 0:
        return Time([], format='iso', scale='utc')  # empty

    # Normalize ref into an astropy Time
    if isinstance(ref, Time):
        t_ref = ref
    elif isinstance(ref, str):
        # Interpret strings as UTC calendar times
        t_ref = Time(ref, scale='utc')
    elif isinstance(ref, (float, int)):
        # Interpret numeric as GPS seconds
        t_ref = Time(float(ref), format='gps', scale='utc')
    else:
        raise TypeError("ref must be astropy.time.Time, str, or float/int (GPS seconds)")

    # Compute offsets (in seconds) relative to the chosen anchor sample
    if anchor == 'start':
        offsets_sec = times - times[0]
    elif anchor == 'end':
        offsets_sec = times - times[-1]
    else:
        raise ValueError("anchor must be 'start' or 'end'")

    # Build TimeDelta vector and add to reference
    deltas = TimeDelta(offsets_sec, format='sec')
    return t_ref + deltas


# ---------------------------------------------------------------------------
# Example usage:
# abs_times = time_rel_to_abs([0, 1, 2, 3], "2025-01-01 00:00:00", anchor='start')
# abs_times_end = time_rel_to_abs([10.0, 12.5, 20.0], "2025-01-01 12:00:00", anchor='end')
# gps_based = time_rel_to_abs(np.linspace(0, 30, 4), 1356091218.0, anchor='start')  # GPS seconds
# ---------------------------------------------------------------------------
