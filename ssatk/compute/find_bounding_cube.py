import numpy as np


def find_smallest_bounding_cube(r: np.ndarray, pad: float = 0.0) -> np.ndarray:
    """
    Find the smallest bounding cube for a set of 3D coordinates, with optional padding.

    Parameters:
    r (np.ndarray): An array of shape (n, 3) containing the 3D coordinates.
    pad (float): Amount to increase the bounding cube in all dimensions.

    Returns:
    tuple: A tuple containing the lower and upper bounds of the bounding cube.
    """
    min_coords = np.min(r, axis=0)
    max_coords = np.max(r, axis=0)
    ranges = max_coords - min_coords
    max_range = np.max(ranges)
    center = (max_coords + min_coords) / 2
    half_side_length = max_range / 2 + pad
    lower_bound = center - half_side_length
    upper_bound = center + half_side_length

    return lower_bound, upper_bound
