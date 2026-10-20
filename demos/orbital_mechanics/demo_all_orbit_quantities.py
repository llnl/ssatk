# demo_all_orbital_quantities.py
"""
Demo for ssatk.orbital_mechanics.all_orbit_quantities.all_orbital_quantities

Shows three usage modes:
  1) From Cartesian state (r, v)
  2) From (periapsis, apoapsis)
  3) From (a, e) plus mean anomaly (ma)
"""

import os
import sys

import numpy as np
from astropy.time import Time

from ssatk.constants import EARTH_MU
from ssatk.orbital_mechanics.all_orbit_quantities import all_orbital_quantities

UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None

# Smoke/validation module: exercised by pytest, not rendered in the demo gallery.
GALLERY_INCLUDE = False


def print_summary(tag, out):
    print(f"\n=== {tag} ===")
    for k in sorted(out.keys()):
        v = out[k]
        if isinstance(v, np.ndarray):
            continue
        print(f"{k:>16s}: {v}")
    for k in sorted(out.keys()):
        v = out[k]
        if isinstance(v, np.ndarray):
            v_arr = np.asarray(v)
            if v_arr.ndim == 1 and v_arr.size == 3:
                print(f"{k:>16s}: {v_arr}   |{k}|={np.linalg.norm(v_arr):.6g}")
            else:
                print(f"{k:>16s}: shape={v_arr.shape}\n{v_arr}")
    if "r" in out:
        print(f"{'|r|':>16s}: {np.linalg.norm(out['r']):.3f} m")
    if "v" in out:
        print(f"{'|v|':>16s}: {np.linalg.norm(out['v']):.6f} m/s")


def main(verbose=None):
    if verbose is None:
        verbose = not UNDER_PYTEST

    t = Time.now().gps
    mu = EARTH_MU

    r = np.array([7000e3, 0.0, 0.0])
    v = np.array([0.0, 7.546e3, 1.0e2])
    out_rv = all_orbital_quantities(r=r, v=v, t=t, mu=mu)

    periapsis = 7000e3
    apoapsis = 12000e3
    out_rpra = all_orbital_quantities(
        periapsis=periapsis,
        apoapsis=apoapsis,
        i=np.deg2rad(28.5),
        raan=np.deg2rad(40.0),
        pa=np.deg2rad(15.0),
        ta=np.deg2rad(10.0),
        t=t,
        mu=mu,
    )

    a = 9000e3
    e = 0.1
    ma = np.deg2rad(25.0)
    out_ae_ma = all_orbital_quantities(
        a=a,
        e=e,
        i=np.deg2rad(10.0),
        raan=np.deg2rad(80.0),
        pa=np.deg2rad(5.0),
        ma=ma,
        t=t,
        mu=mu,
    )

    if verbose:
        print_summary("From (r, v)", out_rv)
        print_summary("From (periapsis, apoapsis)", out_rpra)
        print_summary("From (a, e) with mean anomaly (ma)", out_ae_ma)

    return {"rv_case": out_rv, "rpra_case": out_rpra, "ae_ma_case": out_ae_ma}


if __name__ == "__main__":
    main(verbose=True)
