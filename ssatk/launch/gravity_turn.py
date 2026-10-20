import numpy as np

from ..constants import EARTH_MU  # gravitational parameter


def accel_gravity_turn(r, t_idx, t_array, thrust_mags, turn_time, launch_az=0.0):
    """
    Point-mass gravity plus thrust on a linear pitch-over schedule.

    The thrust starts along the local vertical (r_hat) and pitches over
    linearly to the local horizontal at ``turn_time``. The horizontal
    direction is the launch azimuth measured clockwise from local north
    toward local east, the convention of ``launch.sites.launch_pads``
    (north = z_hat projected onto the local horizontal).

    Parameters
    ----------
    r : array-like, shape (3,)
        Position (m), Earth-centred; the z axis is Earth's spin axis.
    t_idx : int
        Index of the current step in ``t_array``.
    t_array : array-like
        Full time array (s).
    thrust_mags : array-like
        Thrust acceleration magnitude at each step (m/s^2).
    turn_time : float
        Time over which the pitch goes from vertical (90 deg) to horizontal (0 deg).
    launch_az : float
        Launch azimuth (rad), clockwise from local north: 0 = north,
        pi/2 = east. Previously the vertical was the global +z axis and the
        azimuth was measured in the equatorial plane from +x, which is only
        right for a launch from the North Pole.

    Returns
    -------
    numpy.ndarray
        Total acceleration (m/s^2).
    """
    r = np.asarray(r, dtype=float)
    r_norm = np.linalg.norm(r)
    a_grav = -EARTH_MU * r / r_norm**3

    t = t_array[t_idx] - t_array[0]
    pitch = np.clip(np.pi / 2 * (1 - t / turn_time), 0, np.pi / 2)

    up = r / r_norm
    east = np.cross([0.0, 0.0, 1.0], up)
    if np.linalg.norm(east) < 1e-12:  # at a pole, take +y as "east"
        east = np.array([0.0, 1.0, 0.0])
    east /= np.linalg.norm(east)
    north = np.cross(up, east)
    horizontal = np.cos(launch_az) * north + np.sin(launch_az) * east
    thrust_dir = np.sin(pitch) * up + np.cos(pitch) * horizontal
    return a_grav + thrust_mags[t_idx] * thrust_dir
