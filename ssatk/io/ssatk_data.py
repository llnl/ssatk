"""SSATK local data path helper."""

from .datapath import datapath


def ssatk_data(filename="data", dirs=None):
    """Return a local data path under the shared SSATK output root."""
    return datapath(filename, dirs=dirs)


__all__ = ["datapath", "ssatk_data"]
