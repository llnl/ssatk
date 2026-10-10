"""
Sampling utilities.

This module provides a collection of small helper functions for random
sampling, including:

  * Sampling from sequences (with/without replacement)
  * Uniform scalar and array sampling
  * Gaussian (normal) scalar and array sampling
  * Log-normal, exponential, Poisson, binomial, Dirichlet, and
    multivariate normal sampling
  * Cached sigma perturbation generator for 6D state vectors (get_sigmas)

All random draws are implemented via NumPy's RNG.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Sequence, Optional

import numpy as np


# ---------------------------------------------------------------------
# Cache settings for get_sigmas
# ---------------------------------------------------------------------
ENV_VAR = "SSATK_CACHE"   # legacy SSAPY_TOOLKIT_CACHE / ssapy_toolkit_CACHE also read
DEFAULT_FILE = "covariance_sigmas.npy"


# ---------------------------------------------------------------------
# Sequence sampling / simple random numbers
# ---------------------------------------------------------------------
def sample_from_sequence(seq: Sequence, n: int, replacement: bool = False) -> np.ndarray:
    """
    Uniformly sample `n` elements from a sequence.

    Parameters
    ----------
    seq : Sequence
        The sequence to sample from.
    n : int
        Number of elements to sample.
    replacement : bool, default False
        If True, sample with replacement; otherwise without replacement.

    Returns
    -------
    np.ndarray
        Array of sampled elements.

    Notes
    -----
    This is a thin wrapper around numpy.random.choice with `p=None`
    (uniform discrete sampling over the sequence indices).
    """
    arr = np.array(seq)
    return np.random.choice(arr, size=n, replace=replacement)


def rand_num(low: float = 0.0, high: float = 1.0) -> float:
    """
    Draw a single uniform random float in [low, high).

    Parameters
    ----------
    low : float, default 0.0
        Lower bound (inclusive).
    high : float, default 1.0
        Upper bound (exclusive).

    Returns
    -------
    float
        Uniform random number in [low, high).
    """
    return float(np.random.uniform(low, high))


def shuffle(x: list) -> None:
    """
    Shuffle a list in-place using Python's built-in RNG.

    Parameters
    ----------
    x : list
        List to shuffle.

    Returns
    -------
    None
    """
    return random.shuffle(x)


def random_arr(
    low: float = 0,
    high: float = 1,
    size: tuple = (1, 10),
    dtype: str = "float64",
) -> np.ndarray:
    """
    Generate a random array with specified bounds and size.

    Parameters
    ----------
    low : float, default 0
        Lower bound.
    high : float, default 1
        Upper bound (for floats) or inclusive upper bound (for ints).
    size : tuple, default (1, 10)
        Shape of the array.
    dtype : str, default 'float64'
        Data type. If 'int' is in the string, integer sampling is used.

    Returns
    -------
    np.ndarray
        Random array.
    """
    if "int" in dtype:
        return np.random.randint(low, high + 1, size, dtype=dtype)
    else:
        return np.random.uniform(low, high, size).astype(dtype)


# ---------------------------------------------------------------------
# More specific distribution helpers
# ---------------------------------------------------------------------
def uniform_scalar(low: float = 0.0, high: float = 1.0) -> float:
    """
    Uniform( low, high ) scalar sample.

    Returns
    -------
    float
    """
    return rand_num(low, high)


def uniform_array(
    low: float = 0.0,
    high: float = 1.0,
    size: tuple = (1, 10),
    dtype: str = "float64",
) -> np.ndarray:
    """
    Uniform( low, high ) array sample.

    Returns
    -------
    np.ndarray
    """
    return np.random.uniform(low, high, size).astype(dtype)


def normal_scalar(mean: float = 0.0, std: float = 1.0) -> float:
    """
    Gaussian N(mean, std^2) scalar sample.

    Returns
    -------
    float
    """
    return float(np.random.normal(loc=mean, scale=std))


def normal_array(
    mean: float = 0.0,
    std: float = 1.0,
    size: tuple = (1, 10),
    dtype: str = "float64",
) -> np.ndarray:
    """
    Gaussian N(mean, std^2) array sample.

    Returns
    -------
    np.ndarray
    """
    return np.random.normal(loc=mean, scale=std, size=size).astype(dtype)


def lognormal_array(
    mean: float = 0.0,
    sigma: float = 1.0,
    size: tuple = (1, 10),
    dtype: str = "float64",
) -> np.ndarray:
    """
    Log-normal array sample: exp( N(mean, sigma^2) ).

    Returns
    -------
    np.ndarray
    """
    return np.random.lognormal(mean=mean, sigma=sigma, size=size).astype(dtype)


def exponential_array(
    scale: float = 1.0,
    size: tuple = (1, 10),
    dtype: str = "float64",
) -> np.ndarray:
    """
    Exponential(scale) array sample.

    Parameters
    ----------
    scale : float
        1 / lambda, the mean of the distribution.

    Returns
    -------
    np.ndarray
    """
    return np.random.exponential(scale=scale, size=size).astype(dtype)


def poisson_array(
    lam: float = 1.0,
    size: tuple = (1, 10),
    dtype: str = "int64",
) -> np.ndarray:
    """
    Poisson(lam) array sample.

    Parameters
    ----------
    lam : float
        Expected value (lambda).

    Returns
    -------
    np.ndarray
    """
    return np.random.poisson(lam=lam, size=size).astype(dtype)


def binomial_array(
    n: int = 1,
    p: float = 0.5,
    size: tuple = (1, 10),
    dtype: str = "int64",
) -> np.ndarray:
    """
    Binomial(n, p) array sample.

    Parameters
    ----------
    n : int
        Number of trials.
    p : float
        Success probability.

    Returns
    -------
    np.ndarray
    """
    return np.random.binomial(n=n, p=p, size=size).astype(dtype)


def dirichlet_array(alpha: Sequence[float], size: int = 1) -> np.ndarray:
    """
    Dirichlet(alpha) sampling.

    Parameters
    ----------
    alpha : sequence of float
        Concentration parameters (> 0).
    size : int
        Number of samples.

    Returns
    -------
    np.ndarray
        Shape (size, len(alpha)).
    """
    alpha_arr = np.asarray(alpha, dtype="float64")
    return np.random.dirichlet(alpha_arr, size=size)


def multivariate_normal_array(
    mean: Sequence[float],
    cov: np.ndarray,
    size: int = 1,
    dtype: str = "float64",
) -> np.ndarray:
    """
    Multivariate normal N(mean, cov) sampling.

    Parameters
    ----------
    mean : sequence of float
        Mean vector.
    cov : np.ndarray
        Covariance matrix.
    size : int
        Number of samples.

    Returns
    -------
    np.ndarray
        Shape (size, dim).
    """
    mean_arr = np.asarray(mean, dtype="float64")
    cov_arr = np.asarray(cov, dtype="float64")
    return np.random.multivariate_normal(mean_arr, cov_arr, size=size).astype(dtype)


# ---------------------------------------------------------------------
# Cached sigma perturbations for 6D state vectors
# ---------------------------------------------------------------------
def get_sigmas(n: int = 25, path: Optional[str] = None) -> np.ndarray:
    """
    Return an (n, 6) array of random sigma perturbations, cached on disk.

    Each row is:
        [dx, dy, dz, dvx, dvy, dvz]

    Generation logic (when cache is recomputed):
      * Positions are uniformly sampled inside a 3D ball of radius 10.
      * Velocities are uniformly sampled inside a 3D ball of radius 1.

    Cache path resolution when `path` is None:
      1) Use directory from env var SSATK_CACHE (legacy SSAPY_TOOLKIT_CACHE)
         (file name: covariance_sigmas.npy)
      2) Fallback to ~/.cache/ssatk/covariance_sigmas.npy

    If `path` is provided, it is treated as an explicit file path.

    Parameters
    ----------
    n : int, default 25
        Number of sigma samples (rows).
    path : str or None
        Optional explicit path for the cache file.

    Returns
    -------
    np.ndarray
        Array of shape (n, 6).
    """
    if path is None:
        from ssatk.io.datapath import datapath

        # Prefer env-provided directory, then fallback to ~/.cache/ssatk
        from ssatk._paths import cache_env_dir

        env_dir = cache_env_dir() or ""
        dirs = []
        if env_dir:
            dirs.append(Path(env_dir).expanduser())
        dirs.append(Path.home() / ".cache" / "ssatk")

        # Use datapath to choose/create the first usable directory
        path = Path(datapath(DEFAULT_FILE, dirs=dirs))
    else:
        path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    compute_new = True
    sigmas: Optional[np.ndarray] = None

    if path.exists():
        print("Loading sigmas.")
        try:
            sigmas = np.load(path)
            compute_new = not (
                sigmas.ndim == 2 and sigmas.shape[0] == n and sigmas.shape[1] == 6
            )
        except (OSError, ValueError, EOFError):
            # Corrupt or incompatible file -> recompute
            compute_new = True

    if compute_new:
        print("Computing sigmas.")
        sigmas = np.zeros((n, 6), dtype="float64")
        for i in range(n):
            x_dist, v_dist = 10.0, 1.0
            while True:
                # Sample uniformly in a cube, then reject if outside sphere
                tmpx, tmpy, tmpz = np.random.uniform(-x_dist, x_dist, 3)
                tmpvx, tmpvy, tmpvz = np.random.uniform(-v_dist, v_dist, 3)
                if (
                    np.sqrt(tmpx**2 + tmpy**2 + tmpz**2) <= x_dist
                    and np.sqrt(tmpvx**2 + tmpvy**2 + tmpvz**2) <= v_dist
                ):
                    break
            sigmas[i, :] = [tmpx, tmpy, tmpz, tmpvx, tmpvy, tmpvz]
        np.save(path, sigmas)

    return sigmas


def _sample_3d_offset(
    scale: float,
    distribution: str = "uniform",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Internal helper: draw a 3D offset vector with isotropic direction.

    Parameters
    ----------
    scale : float
        A characteristic length/velocity scale (e.g. max radius or sigma).
    distribution : {'uniform', 'normal', 'gaussian', 'shell', 'laplace'}
        - 'uniform'  : uniform inside a solid ball of radius = scale
        - 'normal'/'gaussian' : each component ~ N(0, scale^2)
        - 'shell'    : isotropic direction with fixed radius = scale
        - 'laplace'  : i.i.d Laplace(0, scale) per component
    rng : np.random.Generator or None
        Optional RNG. If None, uses np.random.default_rng().

    Returns
    -------
    np.ndarray
        Shape (3,), the sampled 3D offset.
    """
    if scale <= 0:
        return np.zeros(3, dtype="float64")

    if rng is None:
        rng = np.random.default_rng()

    distribution = distribution.lower()

    if distribution == "uniform":
        # Uniform in a solid ball of radius = scale
        # Radius ~ U(0,1)^(1/3) * scale, direction isotropic
        direction = rng.normal(size=3)
        norm = np.linalg.norm(direction)
        if norm == 0.0:
            direction = np.array([1.0, 0.0, 0.0], dtype="float64")
            norm = 1.0
        direction = direction / norm
        u = rng.random()
        radius = scale * u ** (1.0 / 3.0)
        return radius * direction

    if distribution in ("normal", "gaussian"):
        # Isotropic Gaussian: components i.i.d. N(0, scale^2)
        return rng.normal(loc=0.0, scale=scale, size=3)

    if distribution in ("shell", "surface"):
        # Uniform direction on sphere, fixed radius = scale
        direction = rng.normal(size=3)
        norm = np.linalg.norm(direction)
        if norm == 0.0:
            direction = np.array([1.0, 0.0, 0.0], dtype="float64")
            norm = 1.0
        direction = direction / norm
        return scale * direction

    if distribution == "laplace":
        # Components i.i.d. Laplace(0, scale)
        return rng.laplace(loc=0.0, scale=scale, size=3)

    raise ValueError(
        f"Unknown distribution '{distribution}'. "
        "Choose from: 'uniform', 'normal', 'gaussian', 'shell', 'surface', 'laplace'."
    )


def perturb_state_3d(
    r: np.ndarray,
    v: np.ndarray,
    pos_scale: float = 1.0,
    vel_scale: float = 0.1,
    pos_distribution: str = "uniform",
    vel_distribution: str = "uniform",
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perturb a 3D position and velocity with isotropic spherical uncertainty.

    Parameters
    ----------
    r : array-like, shape (3,)
        Nominal 3D position vector.
    v : array-like, shape (3,)
        Nominal 3D velocity vector.
    pos_scale : float, default 1.0
        Characteristic scale of the position perturbation.
        Interpretation depends on ``pos_distribution``: uniform is the
        maximum radius of a solid ball, normal is the per-component standard
        deviation, shell is a fixed radius, and laplace is the per-component
        Laplace scale.
    vel_scale : float, default 0.1
        Same idea as `pos_scale`, but for the velocity perturbation.
    pos_distribution : {'uniform', 'normal', 'gaussian', 'shell', 'surface', 'laplace'}
        Distribution for position offset (see `_sample_3d_offset`).
    vel_distribution : {'uniform', 'normal', 'gaussian', 'shell', 'surface', 'laplace'}
        Distribution for velocity offset.
    rng : np.random.Generator or None
        Optional RNG for reproducibility.

    Returns
    -------
    (r_pert, v_pert) : tuple of np.ndarray
        Perturbed position and velocity, each shape (3,).

    Examples
    --------
    >>> r = np.array([7000e3, 0.0, 0.0])
    >>> v = np.array([0.0, 7.5e3, 0.0])
    >>> r_p, v_p = perturb_state_3d(r, v, pos_scale=100.0, vel_scale=0.1,
    ...                             pos_distribution='uniform',
    ...                             vel_distribution='normal')
    """
    r = np.asarray(r, dtype="float64")
    v = np.asarray(v, dtype="float64")

    if r.shape != (3,) or v.shape != (3,):
        raise ValueError(
            f"r and v must be 3-vectors of shape (3,), got {r.shape} and {v.shape}"
        )

    if rng is None:
        rng = np.random.default_rng()

    delta_r = _sample_3d_offset(
        scale=pos_scale,
        distribution=pos_distribution,
        rng=rng,
    )
    delta_v = _sample_3d_offset(
        scale=vel_scale,
        distribution=vel_distribution,
        rng=rng,
    )

    return r + delta_r, v + delta_v
