"""Shared state parsing for 6-DoF acceleration helpers."""

from __future__ import annotations

import numpy as np


def vector3(value, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if value.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector.")
    return value


def position(value) -> np.ndarray:
    return vector3(value.r if hasattr(value, "r") else value, "r")


def velocity(value, fallback=None) -> np.ndarray:
    if hasattr(value, "v"):
        return vector3(value.v, "v")
    if fallback is None:
        return vector3(value, "v")
    return vector3(fallback, "v")


def time(value, fallback=None):
    return getattr(value, "t", 0.0 if fallback is None else fallback)


def state(value, v=None, t=None):
    if hasattr(value, "r") and hasattr(value, "v"):
        return position(value), velocity(value), time(value, t)
    return position(value), np.zeros(3) if v is None else velocity(v), 0.0 if t is None else t


def call_acceleration(func, r, v=None, t=None):
    try:
        return vector3(func(r, v, t), "acceleration")
    except TypeError:
        try:
            return vector3(func(r, t), "acceleration")
        except TypeError:
            try:
                return vector3(func(r, v), "acceleration")
            except TypeError:
                return vector3(func(r), "acceleration")
