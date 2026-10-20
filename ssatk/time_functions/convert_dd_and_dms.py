import numpy as np


def dms_to_dd(dms):
    """
    Convert Degree-Minute-Second (DMS) to Decimal Degrees (DD).

    Parameters
    ----------
    dms : str or list of str
        A string or a list of strings representing degrees, minutes, and seconds.

    Returns
    -------
    float or list of float
        The decimal degree equivalent(s) of the input DMS.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    dms, out = [[dms] if isinstance(dms, str) else dms][0], []
    for i in dms:
        text = i.strip()
        # The sign belongs to the whole angle: "-00:30:00" is -0.5 deg.
        sign = -1.0 if text.startswith('-') else 1.0
        deg, minute, sec = [abs(float(j)) for j in text.lstrip('+-').split(':')]
        out.append(sign * (deg + minute / 60 + sec / 3600))
    return out[0] if isinstance(dms, str) or len(dms) == 1 else out


def dd_to_dms(degree_decimal):
    """
    Convert Decimal Degrees (DD) to Degree-Minute-Second (DMS).

    Parameters
    ----------
    degree_decimal : float
        The decimal degree value.

    Returns
    -------
    str
        The corresponding DMS string in the format 'deg:min:sec'.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    # Work on the magnitude and prefix the sign, so angles between -1 and 0 deg
    # keep it: -0.5 deg is "-0:30:0", not "0:30:0".
    value = abs(float(degree_decimal))
    _d, __d = np.trunc(value), value - np.trunc(value)
    _m, __m = np.trunc(__d * 60), __d * 60 - np.trunc(__d * 60)
    _s = round(__m * 60, 4)
    if _s >= 60:
        _m, _s = _m + 1, _s - 60
    if _m >= 60:
        _d, _m = _d + 1, 0
    _s = int(_s) if int(_s) == _s else _s
    sign = '-' if degree_decimal < 0 and (_d, _m, _s) != (0, 0, 0) else ''
    return f'{sign}{int(_d)}:{int(_m)}:{_s}'
