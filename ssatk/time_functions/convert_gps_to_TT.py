def _gpsToTT(t: float) -> float:
    """
    Convert GPS time in seconds to Terrestrial Time (TT) in days.

    Parameters
    ----------
    t : float
        GPS time in seconds.

    Returns
    -------
    float
        The corresponding TT time in days.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    return 44244.0 + (t + 51.184) / 86400
