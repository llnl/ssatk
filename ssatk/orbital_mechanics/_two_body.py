"""Internal two-body propagation helpers."""

from __future__ import annotations

import numpy as np


def _keplerian_two_body_rhs(t, state, mu):
    """Return Cartesian two-body dynamics for ``solve_ivp`` callbacks."""
    del t
    r = state[:3]
    v = state[3:]
    r_norm = np.linalg.norm(r)
    acceleration = -mu * r / r_norm**3
    return np.concatenate((v, acceleration))
