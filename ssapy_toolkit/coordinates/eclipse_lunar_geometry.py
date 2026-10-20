"""Lunar attitude and optional limb-profile geometry.

The strict LLNL path uses SSAPy's DE440 binary lunar PCK through
``Body.orientation``.  The deterministic reference path implements the NAIF
IAU_MOON trigonometric model from ``pck00011.tpc`` (2009 IAU constants).  The
text-PCK model is intentionally described as a lower-accuracy fallback; it is
not relabelled as the DE440 principal-axis solution.

All public rotation matrices map Moon body-fixed column vectors into GCRF.
Angles are degrees unless noted otherwise.  Lunar limb profiles are optional:
the default remains a circular prediction limb, while a user-supplied CSV or
NPZ can provide radius as a function of position angle.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math

import numpy as np

from ssapy_toolkit.constants import MOON_MEAN_RADIUS_KM as R_MOON_MEAN_KM

# Lunar attitude is owned by the single provider in lunar_attitude.py.
# This module retains the public helper names for the canonical API while
# concentrating its own implementation on optional topographic limb profiles.
from ssapy_toolkit.coordinates.eclipse_lunar_attitude import iau_moon_attitude


def iau_moon_orientation_angles_deg(jd_utc) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the canonical reference lunar pole and prime-meridian angles.

    This public compatibility function delegates to :mod:`lunar_attitude`,
    which owns the UTC-to-TDB path and the NAIF/IAU 2009 series used by the
    actual event renderer.  Keeping one implementation prevents a serialized
    event and a direct API call from disagreeing by their time argument.
    """
    from ssapy_toolkit.coordinates.eclipse_lunar_attitude import iau_moon_attitude
    jd = np.asarray(jd_utc, dtype=float)
    scalar = jd.ndim == 0
    values = [iau_moon_attitude(float(value)) for value in jd.reshape(-1)]
    ra = np.asarray([value.pole_ra_deg for value in values], dtype=float).reshape(jd.shape)
    dec = np.asarray([value.pole_dec_deg for value in values], dtype=float).reshape(jd.shape)
    pm = np.asarray([value.prime_meridian_deg for value in values], dtype=float).reshape(jd.shape)
    if scalar:
        return float(ra), float(dec), float(pm)
    return ra, dec, pm


def iau_moon_body_to_gcrf(jd_utc) -> np.ndarray:
    """Return canonical NAIF/IAU Moon body-fixed-to-GCRF matrices."""
    from ssapy_toolkit.coordinates.eclipse_lunar_attitude import iau_moon_attitude
    jd = np.asarray(jd_utc, dtype=float)
    scalar = jd.ndim == 0
    result = np.stack([
        np.asarray(iau_moon_attitude(float(value)).body_to_gcrf, dtype=float)
        for value in jd.reshape(-1)
    ], axis=0)
    return result[0] if scalar else result.reshape(jd.shape + (3, 3))


def lunar_pole_gcrf(jd_utc) -> np.ndarray:
    """Return the canonical lunar +Z body axis in GCRF."""
    matrix = iau_moon_body_to_gcrf(jd_utc)
    return np.asarray(matrix)[..., :, 2]


@dataclass(frozen=True)
class LunarLimbProfile:
    """Periodic lunar limb radius versus position angle.

    Position angle is measured in degrees, north through east in the apparent
    sky.  ``radius_km`` contains absolute geocentric radii, not height offsets.
    """

    position_angle_deg: np.ndarray
    radius_km_values: np.ndarray
    source: str = "circular mean-radius limb"

    def __post_init__(self) -> None:
        angle = np.asarray(self.position_angle_deg, dtype=float).reshape(-1)
        radius = np.asarray(self.radius_km_values, dtype=float).reshape(-1)
        if len(angle) < 2 or len(angle) != len(radius):
            raise ValueError("a limb profile requires matching angle/radius arrays")
        if not np.all(np.isfinite(angle)) or not np.all(np.isfinite(radius)):
            raise ValueError("limb profile values must be finite")
        if np.any(radius <= 0.0):
            raise ValueError("lunar limb radii must be positive")
        order = np.argsort(np.mod(angle, 360.0))
        object.__setattr__(self, "position_angle_deg", np.mod(angle[order], 360.0))
        object.__setattr__(self, "radius_km_values", radius[order])

    @classmethod
    def circular(cls, radius_km: float = R_MOON_MEAN_KM, *, source: str | None = None) -> "LunarLimbProfile":
        return cls(
            np.array([0.0, 180.0]),
            np.array([float(radius_km), float(radius_km)]),
            source or f"circular limb, radius={float(radius_km):.6f} km",
        )

    @classmethod
    def from_file(cls, path: str | Path, *, reference_radius_km: float = R_MOON_MEAN_KM) -> "LunarLimbProfile":
        path = Path(path).expanduser().resolve()
        if path.suffix.lower() == ".npz":
            with np.load(path) as data:
                angle = np.asarray(data["position_angle_deg"], dtype=float)
                if "radius_km" in data:
                    radius = np.asarray(data["radius_km"], dtype=float)
                elif "height_km" in data:
                    radius = float(reference_radius_km) + np.asarray(data["height_km"], dtype=float)
                else:
                    raise ValueError("NPZ limb profile needs radius_km or height_km")
            return cls(angle, radius, source=str(path))

        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fields = {str(name).strip().lower(): name for name in (reader.fieldnames or [])}
            angle_key = fields.get("position_angle_deg") or fields.get("angle_deg") or fields.get("pa_deg")
            radius_key = fields.get("radius_km")
            height_key = fields.get("height_km") or fields.get("elevation_km")
            if angle_key is None or (radius_key is None and height_key is None):
                raise ValueError(
                    "CSV limb profile needs position_angle_deg plus radius_km or height_km"
                )
            angles: list[float] = []
            radii: list[float] = []
            for row in reader:
                angles.append(float(row[angle_key]))
                radii.append(
                    float(row[radius_key]) if radius_key is not None
                    else float(reference_radius_km) + float(row[height_key])
                )
        return cls(np.asarray(angles), np.asarray(radii), source=str(path))

    def radius_km(self, position_angle_deg) -> np.ndarray | float:
        query = np.asarray(position_angle_deg, dtype=float)
        angles = self.position_angle_deg
        radii = self.radius_km_values
        extended_angles = np.concatenate([angles[-1:] - 360.0, angles, angles[:1] + 360.0])
        extended_radii = np.concatenate([radii[-1:], radii, radii[:1]])
        result = np.interp(np.mod(query, 360.0), extended_angles, extended_radii)
        return float(result) if query.ndim == 0 else result

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "sample_count": int(len(self.position_angle_deg)),
            "minimum_radius_km": float(np.min(self.radius_km_values)),
            "maximum_radius_km": float(np.max(self.radius_km_values)),
            "peak_to_peak_km": float(np.ptp(self.radius_km_values)),
            "position_angle_deg": np.asarray(self.position_angle_deg, dtype=float).tolist(),
            "radius_km": np.asarray(self.radius_km_values, dtype=float).tolist(),
        }


def apparent_position_angle_deg(reference_direction, target_direction, north_reference=(0.0, 0.0, 1.0)) -> float:
    """Position angle of ``target`` around ``reference``, north through east."""
    # Copy caller-owned arrays before normalization. ``np.asarray`` can return
    # a view; in-place division previously collapsed the observer's full
    # topocentric Sun/Moon vectors to unit length before the subsequent
    # angular-radius calculation, making arbitrary-site obscuration nearly
    # total. Position-angle calculation must be pure.
    center = np.array(reference_direction, dtype=float, copy=True)
    center /= np.linalg.norm(center)
    target = np.array(target_direction, dtype=float, copy=True)
    target /= np.linalg.norm(target)
    north = np.array(north_reference, dtype=float, copy=True)
    north -= np.dot(north, center) * center
    if np.linalg.norm(north) < 1e-12:
        north = np.array([1.0, 0.0, 0.0]) - center[0] * center
    north /= np.linalg.norm(north)
    east = np.cross(center, north)
    east /= np.linalg.norm(east)
    delta = target - np.dot(target, center) * center
    if np.linalg.norm(delta) < 1e-15:
        return 0.0
    return math.degrees(math.atan2(np.dot(delta, east), np.dot(delta, north))) % 360.0
