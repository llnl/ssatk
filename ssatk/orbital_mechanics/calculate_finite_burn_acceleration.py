import numpy as np


def calculate_finite_burn_acceleration(delta_v, t_impulsive, a_magnitude):
    """
    Calculate the acceleration to approximate an instantaneous orbital transfer with a finite burn.

    Parameters:
    - delta_v: np.array, shape (3,), impulsive delta-v vector in inertial frame (m/s)
    - t_impulsive: float, time of the impulsive burn (s)
    - a_magnitude: float, constant acceleration magnitude from specific thrust (m/s^2)

    Returns:
    - a_vector: np.array, shape (3,), constant acceleration vector (m/s^2)
    - t_burn: float, duration of the burn (s)
    - t_start: float, start time of the burn (s)
    - t_end: float, end time of the burn (s)

    Author:
    Travis Yeager
    """
    # Compute the magnitude of delta-v
    dv_norm = np.linalg.norm(delta_v)
    if dv_norm == 0:
        raise ValueError("Delta-v is zero; no burn is required.")

    # Direction of acceleration is the same as delta-v
    direction = delta_v / dv_norm
    a_vector = a_magnitude * direction

    # Burn duration to achieve the required delta-v
    t_burn = dv_norm / a_magnitude

    # Center the burn around the impulsive burn time
    t_start = t_impulsive - t_burn / 2
    t_end = t_impulsive + t_burn / 2

    return a_vector, t_burn, t_start, t_end
