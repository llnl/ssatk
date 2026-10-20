import numpy as np
from astropy.time import Time


def to_gps(t):
    """
    Convert astropy Time objects to GPS seconds.

    Parameters
    ----------
    t : astropy.time.Time or array-like of Time
        Input time(s). If already numeric, they are returned unchanged.

    Returns
    -------
    float, array-like, or same type as input
        GPS seconds corresponding to the input.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    if np.size(t) > 1:
        if isinstance(t[0], Time):
            try:
                t = t.gps
            except AttributeError:
                t = np.array([time.gps for time in t])
    else:
        if isinstance(t, Time):
            t = t.gps
    return t
