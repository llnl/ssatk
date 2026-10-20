"""Distant-star proper motion in fixed ICRS axes."""

import numpy as np


def proper_motion_directions(ra_hours, dec_deg, pmra_mas_yr, pmdec_mas_yr, years=0.):
    """Propagate directions by constant tangential space motion, then normalize.

    RA is in hours; Dec in degrees; pmra is mu_alpha* (includes cos Dec),
    pmdec in mas/Julian year. ``years`` is elapsed Julian years since J2000.
    No parallax or radial motion is modeled. Cartesian tangent vectors avoid
    dividing by cos(Dec), including exactly at the celestial poles.
    """
    ra, dec, pmra, pmdec = np.broadcast_arrays(
        np.asarray(ra_hours, float), np.asarray(dec_deg, float),
        np.asarray(pmra_mas_yr, float), np.asarray(pmdec_mas_yr, float))
    if not np.isfinite([ra, dec, pmra, pmdec]).all() or not np.isfinite(years):
        raise ValueError("stellar astrometry and elapsed years must be finite")
    a, d = np.radians(ra*15), np.radians(dec)
    unit = np.stack([np.cos(d)*np.cos(a), np.cos(d)*np.sin(a), np.sin(d)], axis=-1)
    east = np.stack([-np.sin(a), np.cos(a), np.zeros_like(a)], axis=-1)
    north = np.stack([-np.sin(d)*np.cos(a), -np.sin(d)*np.sin(a), np.cos(d)], axis=-1)
    moved = unit + float(years)*np.pi/(180*3.6e6)*(pmra[..., None]*east + pmdec[..., None]*north)
    return moved/np.linalg.norm(moved, axis=-1, keepdims=True)
