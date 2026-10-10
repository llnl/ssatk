"""
core/accel_thrust.py
---------------------
Custom SSAPy acceleration class for finite-duration constant-thrust burns.

Implements a finite burn model:

- Constant thrust magnitude and direction during ``[t_start, t_end]``.
- Continuous Tsiolkovsky mass decrease.
- Zero acceleration outside the burn window.
- Hashable state for SSAPy's internal LRU cache.

Usage
-----

.. code-block:: python

    from ssatk.plots.accel_thrust import AccelConstantThrust
    burn = AccelConstantThrust(
        direction_unit=dv_vec / np.linalg.norm(dv_vec),
        thrust_n=110_100.0,    # ICPS RL10B-2
        isp_s=465.5,
        wet_mass_kg=54_700.0,
        t_start_gps=1352623509.0,
        t_end_gps=1352624589.0,   # t_start + burn_duration_s
    )
"""

import numpy as np
from ssapy.accel import Accel

from ssatk.constants import G0


class AccelConstantThrust(Accel):
    """
    Constant-thrust finite burn over a GPS-second time window.

    The spacecraft is treated as a point mass (standard for preliminary
    design). Thrust direction is fixed in inertial space throughout the
    burn — a valid approximation for short burns where attitude control
    keeps the engine pointed in the same inertial direction.

    Parameters
    ----------
    direction_unit : array_like (3,)
        Unit vector of the thrust direction in GCRF (metres or km —
        only the direction matters, magnitude is normalised internally).
    thrust_n : float
        Engine thrust in Newtons.
    isp_s : float
        Specific impulse in seconds.
    wet_mass_kg : float
        Total spacecraft mass at burn ignition (kg). Decreases during
        the burn as propellant is consumed.
    t_start_gps : float
        Burn ignition time in GPS seconds.
    t_end_gps : float
        Burn cutoff time in GPS seconds. Computed externally from
        Tsiolkovsky: t_end = t_start + m_prop / mdot, where
        m_prop = wet_mass * (1 - exp(-abs(dv) / (Isp * g0))).
    """

    def __init__(self, direction_unit, thrust_n, isp_s, wet_mass_kg,
                 t_start_gps, t_end_gps):
        d = np.asarray(direction_unit, dtype=float)
        norm = np.linalg.norm(d)
        if norm < 1e-12:
            raise ValueError("direction_unit must be a non-zero vector")
        self._dir    = d / norm
        self._F      = float(thrust_n)
        self._isp    = float(isp_s)
        self._m0     = float(wet_mass_kg)
        self._mdot   = self._F / (self._isp * G0)   # kg/s
        self._t0     = float(t_start_gps)
        self._t1     = float(t_end_gps)
        # SSAPy's AccelSum concatenates time_breakpoints from all child
        # accelerations to schedule exact integrator steps at discontinuities.
        # For a finite burn the force is discontinuous at ignition and cutoff.
        self.time_breakpoints = np.array([self._t0, self._t1])

    def __call__(self, r, v, t):
        if t < self._t0 or t > self._t1:
            return np.zeros(3)
        elapsed = t - self._t0
        mass    = self._m0 - self._mdot * elapsed
        if mass <= 0.0:
            return np.zeros(3)
        # Acceleration in m/s² — SSAPy expects m/s² in GCRF
        return (self._F / mass) * self._dir

    def __hash__(self):
        return hash(("AccelConstantThrust", self._t0, self._t1,
                     self._F, self._isp, self._m0))

    def __eq__(self, other):
        return (isinstance(other, AccelConstantThrust)
                and self._t0 == other._t0
                and self._t1 == other._t1
                and self._F  == other._F)


def burn_from_dv(dv_vec_kms, wet_mass_kg, thrust_n, isp_s, t_start_gps):
    """
    Convenience constructor: build an AccelConstantThrust from a
    delta-v vector (km/s) and engine parameters.

    Returns (burn_accel, burn_duration_s, m_prop_kg).
    """
    dv_ms  = np.linalg.norm(dv_vec_kms) * 1e3   # km/s → m/s
    if dv_ms < 1e-6:
        return None, 0.0, 0.0

    # Tsiolkovsky: propellant mass
    m_prop = wet_mass_kg * (1.0 - np.exp(-dv_ms / (isp_s * G0)))
    mdot   = thrust_n / (isp_s * G0)
    dur_s  = m_prop / mdot
    t_end  = t_start_gps + dur_s

    direction = dv_vec_kms / np.linalg.norm(dv_vec_kms)
    burn = AccelConstantThrust(
        direction_unit=direction,
        thrust_n=thrust_n,
        isp_s=isp_s,
        wet_mass_kg=wet_mass_kg,
        t_start_gps=t_start_gps,
        t_end_gps=t_end,
    )
    return burn, dur_s, m_prop
