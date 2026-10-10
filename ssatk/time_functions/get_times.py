import numpy as np
from astropy.time import Time
import astropy.units as u
import warnings


def _interval_count(dur_seconds, freq_seconds):
    """Number of whole steps of ``freq_seconds`` spanning ``dur_seconds``.

    ``duration / freq`` is an exact integer whenever a caller derives the step
    from a sample count, but the quotient of two binary floats lands one unit
    in the last place below that integer often enough to matter: for
    ``duration = 23288.287991979163`` and ``freq = 195.6998990922619`` it is
    118.99999999999999, so truncation drops the final sample and returns a
    grid one element shorter than the caller asked for. Snap the quotient to a
    neighbouring integer when it is within a relative 1e-9 of one, then
    truncate, so an intended whole number of steps survives the division.
    """
    ratio = dur_seconds / freq_seconds
    nearest = round(ratio)
    if nearest > 0 and abs(ratio - nearest) <= 1e-9 * nearest:
        return int(nearest)
    return int(ratio)


def get_times(
    duration,
    freq=(1, 's'),
    t0=Time(0, format='gps'),
    tf=None,
    tm=None,  # middle time (optional)
):
    """
    Calculate a list of times spaced equally apart over a specified duration.

    Parameters
    ----------
    duration : int, float, or tuple
        A duration in seconds (float/int), or a tuple like (30, 'day').
        Must be non-negative.
    freq : int, float, or tuple
        A time step in seconds (float/int), or a tuple like (10, 'min').
        Must be positive.
    t0 : str, float, int, or Time, optional
        Initial time (first element) if tf and tm are not provided.
        If str, interpreted as UTC.
        If numeric, interpreted as GPS seconds.
    tf : str, float, int, or Time, optional
        Final time (last element). If provided (and tm is None), the time array
        is built so that the last element equals tf and the span equals `duration`
        with spacing `freq`.
    tm : str, float, int, or Time, optional
        Middle time. If provided, the time array is built so that:
          - the total span is exactly `duration`,
          - tm is exactly the middle element,
          - the effective step is adjusted slightly from `freq` if needed.

    Returns
    -------
    astropy.time.Time
        Time array of equally spaced times over the specified duration.
    """

    # --- Helper to normalize times ---
    def _to_time_obj(t_in):
        if isinstance(t_in, Time):
            return t_in
        if isinstance(t_in, str):
            return Time(t_in, scale='utc')
        if isinstance(t_in, (float, int)):
            return Time(t_in, scale='utc', format='gps')
        return None

    # Normalize anchors
    t0_obj = _to_time_obj(t0)
    tf_obj = _to_time_obj(tf) if tf is not None else None
    tm_obj = _to_time_obj(tm) if tm is not None else None

    unit_dict = {
        'second': 1, 'sec': 1, 's': 1,
        'minute': 60, 'min': 60,
        'hour': 3600, 'hr': 3600, 'h': 3600,
        'day': 86400, 'd': 86400,
        'week': 604800,
        'month': 2630016, 'mo': 2630016,
        'year': 31557600, 'yr': 31557600
    }

    def to_seconds(value):
        if isinstance(value, (int, float)):
            return float(value)
        else:
            val, unit = value
            unit = unit.lower()
            if len(unit) > 1:
                unit = unit.rstrip('s')

            if unit not in unit_dict:
                raise ValueError(
                    f'Error, {unit} is not a valid time unit. '
                    f'Valid options are: {", ".join(unit_dict.keys())}.'
                )
            return float(val) * unit_dict[unit]

    dur_seconds = to_seconds(duration)
    freq_seconds = to_seconds(freq)

    if dur_seconds < 0:
        raise ValueError("duration must be non-negative.")
    if freq_seconds <= 0:
        raise ValueError("freq must be positive.")

    if t0_obj is None and tf_obj is None and tm_obj is None:
        raise ValueError("At least one of t0, tf, or tm must be provided.")

    # --- Case 1: tm is provided (middle time) ---
    if tm_obj is not None:
        if dur_seconds == 0:
            # Single time equal to tm
            return Time([tm_obj])

        # Ideal number of intervals
        N_float = dur_seconds / freq_seconds

        # Choose an even integer N close to N_float so tm is exactly center
        N_candidate = int(round(N_float))
        if N_candidate <= 0:
            N_candidate = 2  # minimal even positive
        if N_candidate % 2 != 0:
            # make it even by adjusting to nearby even integer
            if N_candidate < N_float:
                N = N_candidate + 1
            else:
                N = N_candidate - 1 if N_candidate > 1 else 2
        else:
            N = N_candidate

        # Ensure still positive
        if N <= 0:
            N = 2

        # Effective step size and diagnostics
        effective_freq = dur_seconds / N
        if not np.isclose(effective_freq, freq_seconds):
            warnings.warn(
                "get_times adjusted frequency to keep duration exact and tm centered. "
                f"  requested freq = {freq_seconds:.6f} s\n"
                f"  effective freq = {effective_freq:.6f} s\n"
                f"  intervals N    = {N}\n"
                f"  total span     = {dur_seconds:.6f} s",
                UserWarning,
                stacklevel=2,
            )

        timesteps = N + 1
        half_span = dur_seconds / 2.0   # exact
        start_offset = -half_span
        end_offset = +half_span
        anchor = tm_obj

    # --- Case 2: tf is provided (end time) ---
    elif tf_obj is not None:
        if dur_seconds == 0:
            return Time([tf_obj])

        timesteps = _interval_count(dur_seconds, freq_seconds) + 1
        start_offset = -dur_seconds
        end_offset = 0.0
        anchor = tf_obj

    # --- Case 3: default t0 (start time) ---
    else:
        if dur_seconds == 0:
            return Time([t0_obj])

        timesteps = _interval_count(dur_seconds, freq_seconds) + 1
        start_offset = 0.0
        end_offset = dur_seconds
        anchor = t0_obj

    # Build offsets in days
    offsets = np.linspace(start_offset, end_offset, timesteps) / unit_dict['day'] * u.day
    times = anchor + offsets

    return times
