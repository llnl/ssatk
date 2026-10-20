# generate_sphere_of_vectors.py
# Purpose: generate N 3D vectors that point in uniformly random directions on the
#          sphere and all have the same magnitude. Returns a NumPy array (N, 3).
#
# Module use:
#   from ssatk.compute import generate_sphere_vectors
#   V = generate_sphere_vectors(1000, 13.0)  # example call
#
# Script use (hard-coded example at bottom):
#   python3 -m ssatk.compute.generate_sphere_of_vectors
#
# Notes:
#   - Uses NumPy only (no math, no typing).
#   - Sampling method: draw standard normals in R^3 and normalize each row.

import numpy as np


def generate_sphere_vectors(n, magnitude, seed=None, distribution="uniform"):
    """
    Generate N three-dimensional vectors with identical magnitude whose directions
    are area-uniform on the unit sphere.

    Parameters
    ----------
    n : int
        Number of vectors to generate. If n <= 0, returns an empty (0, 3) array.
    magnitude : float
        Desired magnitude for every vector.
    seed : int or None
        Optional seed for reproducibility.
    distribution : {"uniform", "random"}
        * ``"uniform"``: normalize i.i.d. Gaussian vectors (isotropic and
          area-uniform).
        * ``"random"``: area-uniform using spherical coordinates with
          ``z ~ U[-1, 1]`` and ``phi ~ U[0, 2*pi)``.

    Returns
    -------
    numpy.ndarray
        Array of shape (n, 3). Each row has norm approximately equal to `magnitude`.
    """
    n = int(n)
    if n <= 0:
        return np.zeros((0, 3), dtype=float)

    rng = np.random.default_rng(seed)
    dist = str(distribution).lower()

    if dist == "uniform":
        # Isotropic Gaussian -> normalize -> uniform on S^2
        V = rng.normal(size=(n, 3))
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        # Guard against the (extremely unlikely) all-zero row to avoid divide-by-zero
        norms = np.where(norms == 0.0, 1.0, norms)
        U = V / norms

    elif dist == "random":
        # Area-uniform using z ~ U[-1,1] and phi ~ U[0,2*pi)
        z = rng.uniform(-1.0, 1.0, size=n)
        phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
        # Clip very slightly to avoid rare negative inside sqrt from numerical noise
        zz = np.clip(1.0 - z * z, 0.0, 1.0)
        r = np.sqrt(zz)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        U = np.column_stack((x, y, z))

    else:
        raise ValueError('distribution must be either "uniform" or "random"')

    return U * float(magnitude)


def _summarize(V, label="vectors"):
    norms = np.linalg.norm(V, axis=1)
    print(f"{label}: shape={V.shape}")
    if V.size == 0:
        return
    print(f"norm range: [{norms.min():.6f}, {norms.max():.6f}]  mean={norms.mean():.6f}")
    m = min(5, V.shape[0])
    if m > 0:
        print("first few rows:")
        for i in range(m):
            v = V[i]
            print(f"  {i:03d}: [{v[0]: .6f}, {v[1]: .6f}, {v[2]: .6f}]  |v|={np.linalg.norm(v):.6f}")


if __name__ == "__main__":
    # ----------------------------
    # Hard-coded example parameters
    # ----------------------------
    N = 1000         # number of vectors to generate
    MAGNITUDE = 13.0 # target magnitude for each vector
    SEED = 42        # set to None for non-deterministic sampling
    SAVE_PATH = ""   # set to e.g. "vectors.npy" to save the array

    V = generate_sphere_vectors(N, MAGNITUDE, seed=SEED)
    _summarize(V, label=f"vectors (mag={MAGNITUDE})")

    if SAVE_PATH:
        np.save(SAVE_PATH, V)
        print(f"Saved to: {SAVE_PATH}")
