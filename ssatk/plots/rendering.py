"""Shared presentation settings; never change trajectories or physical scales.

Use ``with low_spec_mode(): ...`` for a batch of plots, or set
``SSATK_LOW_SPEC=1`` before running a standalone demo. Supported plotting entry points
also accept ``low_spec=True``. The default is textured rendering.
Published galleries require the prepared maximum-resolution body maps.
"""
from contextlib import contextmanager as _contextmanager
from contextvars import ContextVar as _ContextVar
from functools import wraps as _wraps
import inspect as _inspect
import os as _os

__all__ = ["low_spec_mode", "set_low_spec_mode", "is_low_spec"]

_override = _ContextVar("ssatk_low_spec", default=None)


def gallery_max_textures():
    """Whether this process is rendering a gallery with required 8K body maps."""
    return _os.environ.get("SSATK_GALLERY_MAX_TEXTURES", "") == "1"


def body_mesh_resolution(latitude_cells, *, plotly=False):
    """Keep fast gallery demos above the body-surface geometry floor.

    Matplotlib sorts individual polygons on the CPU; Plotly can use a denser
    GPU mesh. Larger caller-supplied meshes are retained in either backend.
    """
    if gallery_max_textures():
        return max(latitude_cells, 180 if plotly else 90)
    return latitude_cells


def is_low_spec(low_spec=None):
    """Resolve presentation settings, with full textures required in galleries."""
    value = _override.get() if low_spec is None else low_spec
    if value is not None and not isinstance(value, bool):
        raise TypeError("low_spec must be True, False, or None")
    if gallery_max_textures():
        return False
    if value is None:
        return _os.environ.get("SSATK_LOW_SPEC", "").strip().lower() in {"1", "true", "yes", "on"}
    return value


def set_low_spec_mode(enabled=None):
    """Set this context's default; None restores the environment setting."""
    if enabled is not None:
        is_low_spec(enabled)
    _override.set(enabled)


@_contextmanager
def low_spec_mode(enabled=True):
    """Temporarily use plain blue Earth / white Moon with fewer surface cells.

    Nested overrides restore the previous mode, including after exceptions.
    This does not decimate scientific data or alter units, frames, or epochs.
    """
    token = _override.set(is_low_spec(enabled))
    try:
        yield
    finally:
        _override.reset(token)


def _with_low_spec(function):
    """Add an inspectable keyword to existing plotting APIs, preserving wraps."""
    @_wraps(function)
    def wrapped(*args, low_spec=None, **kwargs):
        with low_spec_mode(is_low_spec(low_spec)):
            return function(*args, **kwargs)
    signature = _inspect.signature(function)
    parameters = list(signature.parameters.values())
    index = next((i for i, p in enumerate(parameters) if p.kind == p.VAR_KEYWORD), len(parameters))
    parameters.insert(index, _inspect.Parameter("low_spec", _inspect.Parameter.KEYWORD_ONLY, default=None))
    wrapped.__signature__ = signature.replace(parameters=parameters)
    return wrapped
