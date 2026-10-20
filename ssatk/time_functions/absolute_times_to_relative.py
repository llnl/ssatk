import numpy as np
from astropy.time import Time


def time_abs_to_rel(times, anchor='start'):
    """
    Convert absolute timestamps into relative seconds, choosing which sample is 0.

    Parameters
    ----------
    times : array-like of astropy.time.Time, str, or float/int
        Sequence of absolute timestamps.
        - If Time: used directly.
        - If str: parsed by astropy as UTC strings.
        - If float/int: interpreted as GPS seconds.
    anchor : {'start', 'end'}, default 'start'
        Which sample is treated as t=0 in the output.
        - 'start': output[0] == 0
        - 'end'  : output[-1] == 0 (earlier values will be <= 0)

    Returns
    -------
    numpy.ndarray
        1D float array of relative seconds.

    Notes
    -----
    - Designed to mirror `time_rel_to_abs(..., anchor=...)`.
    - With anchor='end', earlier samples become negative so that the last is 0.
    """
    # Empty input
    if np.size(times) == 0:
        return np.array([], dtype=float)

    # Normalize to an astropy Time array
    if isinstance(times, Time):
        t = times
    else:
        # Try parsing as UTC-like strings first; if that fails, try GPS seconds
        try:
            t = Time(times, scale='utc')
        except (TypeError, ValueError):
            t = Time(times, format='gps', scale='utc')

    # Choose reference sample based on anchor
    if anchor == 'start':
        t_ref = t[0]
    elif anchor == 'end':
        t_ref = t[-1]
    else:
        raise ValueError("anchor must be 'start' or 'end'")

    # Compute relative seconds
    rel_sec = (t - t_ref).to_value('sec')
    return np.asarray(rel_sec, dtype=float)


# ---------------------------------------------------------------------------
# Example usage:
# t_abs = ["2025-01-01 00:00:00", "2025-01-01 00:00:02.5", "2025-01-01 00:00:05"]
# time_abs_to_rel(t_abs, anchor='start')  # -> array([0. , 2.5, 5. ])
# time_abs_to_rel(t_abs, anchor='end')    # -> array([-5. , -2.5, 0. ])
#
# Round-trip sketch:
# rel = np.array([10.0, 12.5, 20.0])
# abs_end = time_rel_to_abs(rel, "2025-01-01 12:00:00", anchor='end')
# time_abs_to_rel(abs_end, anchor='end')  # -> array close to rel (floating tolerance)
# ---------------------------------------------------------------------------
