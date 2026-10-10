"""Observer-to-target visibility shared by the photometry models."""

import numpy as np

from ..constants import EARTH_RADIUS


def line_of_sight_blocked(r_object, r_observer, r_earth=EARTH_RADIUS) -> bool:
    """Test the sightline segment for Earth occultation.

    Limit the spherical radius to the nearer endpoint, so a geodetic station
    below the equatorial radius does not occult itself when looking outward.
    """
    r_object = np.asarray(r_object, dtype=float).ravel()
    r_observer = np.asarray(r_observer, dtype=float).ravel()
    r_earth = min(float(r_earth), float(np.linalg.norm(r_observer)),
                  float(np.linalg.norm(r_object)))
    segment = r_object - r_observer
    length_squared = float(np.dot(segment, segment))
    if length_squared == 0.0:
        return False
    parameter = float(np.clip(-np.dot(r_observer, segment) / length_squared, 0.0, 1.0))
    closest = r_observer + parameter * segment
    return bool(np.linalg.norm(closest) < r_earth)
