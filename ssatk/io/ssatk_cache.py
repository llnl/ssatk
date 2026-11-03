"""SSATK HDF5 cache convenience helpers."""

from .h5cache import h5cache, h5load


# h5cache inspects its caller's frame when no explicit workspace is supplied.
ssatk_save_cache = h5cache


def ssatk_load_cache(*args, **kwargs):
    """Load an HDF5 cache via :func:`ssatk.io.h5cache.h5load`."""
    return h5load(*args, **kwargs)


__all__ = ["h5cache", "h5load", "ssatk_save_cache", "ssatk_load_cache"]
