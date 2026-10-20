import numpy as np
from ._state import call_acceleration, state


def accel_add(*accel_funcs):
    """
    Returns a function that sums multiple acceleration functions.

    Parameters
    ----------
    accel_funcs : list of functions
        Each must take (r) or (r, t) depending on your usage.

    Returns
    -------
    combined : function
        A function that evaluates and sums all input accelerations.
    """
    def combined(r, v=None, t=None):
        r, v, t = state(r, v, t)
        total = np.zeros(3)
        for f in accel_funcs:
            total += call_acceleration(f, r, v, t)
        return total
    return combined
