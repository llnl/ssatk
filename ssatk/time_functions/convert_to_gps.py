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
    if isinstance(t, Time):
        return t.gps
    if np.size(t) and np.ndim(t):
        values = np.asarray(t, dtype=object)
        if isinstance(values.flat[0], Time):
            return np.array([epoch.gps for epoch in values.flat]).reshape(values.shape)
    return t
