import numpy as np
from ..constants import EARTH_RADIUS

def segment_intersects_sphere(p0, p1, radius=EARTH_RADIUS, center=(0.0, 0.0, 0.0), atol=0.0):
    """
    Vectorized check: does the segment p0->p1 intersect a sphere?
    p0, p1: shape (..., 3); center: (3,) or broadcastable; units match radius.
    Returns a bool or array of bools.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    c  = np.asarray(center, dtype=float)
    r  = float(radius) + float(atol)

    d   = p1 - p0                         # segment direction
    m   = p0 - c                          # from center to p0
    a   = np.sum(d*d, axis=-1)            # ||d||^2
    b   = 2.0 * np.sum(m*d, axis=-1)      # 2 m·d
    cc  = np.sum(m*m, axis=-1) - r*r      # ||m||^2 - r^2

    # Degenerate segment (p0 == p1): point-in-sphere
    deg = (a == 0.0)
    out = np.zeros_like(a, dtype=bool)
    if np.any(deg):
        out = np.where(deg, cc <= 0.0, out)

    # Proper segment: quadratic test + root interval overlap with [0, 1]
    mask = ~deg
    if np.any(mask):
        a_m   = a[mask]
        b_m   = b[mask]
        c_m   = cc[mask]
        disc  = b_m*b_m - 4.0*a_m*c_m
        hit_q = disc >= 0.0
        if np.any(hit_q):
            sd    = np.sqrt(np.maximum(disc[hit_q], 0.0))
            denom = 2.0 * a_m[hit_q]
            t1    = (-b_m[hit_q] - sd) / denom
            t2    = (-b_m[hit_q] + sd) / denom
            # Segment intersects if roots overlap [0,1]
            hit_seg = (t2 >= 0.0) & (t1 <= 1.0)
            tmp = np.zeros_like(hit_q, dtype=bool)
            tmp[hit_q] = hit_seg
            out[mask] = tmp

    return out
