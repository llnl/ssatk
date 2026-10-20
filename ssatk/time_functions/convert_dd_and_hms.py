import numpy as np

from .convert_dd_and_dms import dms_to_dd


def hms_to_dd(hms):
    """
    Convert Hour-Minute-Second (HMS) to Decimal Degrees (DD).

    Parameters
    ----------
    hms : str or list of str
        A string or a list of strings representing hours, minutes, and seconds.

    Returns
    -------
    float or list of float
        The decimal degree equivalent(s) of the input HMS.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    _type = type(hms)
    hms, out = [[hms] if _type == str else hms][0], []
    for i in hms:
        if i[0] == '-':
            raise ValueError('hms cannot be negative.')
        hour, minute, sec = i.split(':')
        hour, minute, sec = float(hour), float(minute), float(sec)
        out.append(hour * 15 + (minute / 4) + (sec / 240))

    return out[0] if _type == str or len(hms) == 1 else out


def dd_to_hms(degree_decimal):
    """
    Convert Decimal Degrees (DD) to Hour-Minute-Second (HMS).

    Parameters
    ----------
    degree_decimal : float or str
        The angle in decimal degrees, or a 'deg:min:sec' string. Angles
        outside [0, 360) are wrapped into it, so -15 deg is 23h.

    Returns
    -------
    str
        The corresponding HMS string in the format 'hour:minute:second', with
        hours in [0, 24).

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    if isinstance(degree_decimal, str):
        degree_decimal = dms_to_dd(degree_decimal)
    _dd = float(np.mod(degree_decimal, 360.0)) / 15
    _h, __h = np.trunc(_dd), _dd - np.trunc(_dd)
    _m, __m = np.trunc(__h * 60), __h * 60 - np.trunc(__h * 60)
    _s = round(__m * 60, 4)
    if _s >= 60:
        _m, _s = _m + 1, _s - 60
    if _m >= 60:
        _h, _m = _h + 1, 0
    if _h >= 24:
        _h = 0
    _s = int(_s) if int(_s) == _s else _s

    return f'{int(_h)}:{int(_m)}:{_s}'
