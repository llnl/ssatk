"""Helpers for constructing common SSAPy orbit initial conditions."""

import numpy as np
from ssapy import Orbit, get_body


class OrbitInitialize:
    """Factory for SSAPy orbits with common initial conditions."""

    def __init__(self):
        pass

    @staticmethod
    def DRO(t, delta_r=7.52064e7, delta_v=344):
        """Construct a distant retrograde orbit (DRO) around the Moon.

        Parameters
        ----------
        t :
            SSAPy time object specifying the epoch.
        delta_r : float, optional
            Radial offset from the Moon in meters, by default 7.52064e7.
        delta_v : float, optional
            Velocity offset in m/s, by default 344.

        Returns
        -------
        ssapy.Orbit
            Initialized DRO orbit at the requested epoch.
        """
        moon = get_body("moon")

        unit_vector_moon = moon.position(t) / np.linalg.norm(moon.position(t))
        moon_v = (moon.position(t.gps) - moon.position(t.gps - 1)) / 1
        unit_vector_moon_velocity = moon_v / np.linalg.norm(moon_v)

        r = (np.linalg.norm(moon.position(t)) - delta_r) * unit_vector_moon
        v = (np.linalg.norm(moon_v) + delta_v) * unit_vector_moon_velocity

        orbit = Orbit(r=r, v=v, t=t)
        return orbit

    @staticmethod
    def Lunar_L4(t, delta_r=0.0, delta_v=0.0):
        """Construct an orbit at the Earth-Moon L4 point.

        L4 leads the Moon by 60 deg in the Moon's instantaneous orbital plane,
        forming an equilateral triangle with Earth and the Moon. The state is
        the Moon's geocentric state rotated by +60 deg about the Moon's orbit
        normal, i.e. co-moving with the Earth-Moon line.

        Parameters
        ----------
        t :
            Epoch (astropy Time or GPS seconds).
        delta_r : float, optional
            Radial offset outward from L4 (m), default 0.
        delta_v : float, optional
            Speed added along the L4 velocity (m/s), default 0.

        Returns
        -------
        ssapy.Orbit
            Orbit at (or offset from) L4 at the requested epoch.
        """
        moon = get_body("moon")
        t_gps = t.gps if hasattr(t, "gps") else float(t)
        r_moon = np.asarray(moon.position(t_gps), dtype=float).reshape(3)
        v_moon = (np.asarray(moon.position(t_gps + 1.0), dtype=float).reshape(3)
                  - np.asarray(moon.position(t_gps - 1.0), dtype=float).reshape(3)) / 2.0
        normal = np.cross(r_moon, v_moon)
        normal /= np.linalg.norm(normal)

        def rotate(vector, angle):
            # Rodrigues rotation about the orbit normal.
            return (vector * np.cos(angle) + np.cross(normal, vector) * np.sin(angle)
                    + normal * np.dot(normal, vector) * (1.0 - np.cos(angle)))

        r = rotate(r_moon, np.pi / 3.0)
        v = rotate(v_moon, np.pi / 3.0)
        r = r + delta_r * r / np.linalg.norm(r)
        v = v + delta_v * v / np.linalg.norm(v)
        return Orbit(r=r, v=v, t=t)

# Usage example:
# orbit = OrbitInitialize.DRO(t)
