"""LOLA/LDEM-backed lunar limb profiles.

The visual Moon remains a mean-radius sphere.  This module reads a real LOLA
Global Lunar Digital Elevation Model and derives a separate observer-dependent
prediction limb for contact timing, Baily's-bead studies, and occultations.
That separation prevents topography from deforming the public body mesh.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import math
import re

import numpy as np

from ssapy_toolkit.coordinates.eclipse_lunar_geometry import LunarLimbProfile
from ssapy_toolkit.compute.eclipse_reference_events import R_MOON_MEAN_KM, ReferenceEvent


def parse_pds3_label(path: str | Path) -> dict[str, object]:
    """Parse the scalar fields needed from a PDS3 label."""
    text = Path(path).expanduser().read_text(encoding="latin-1")
    result: dict[str, object] = {}
    for raw in text.splitlines():
        line = raw.split("/*", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        key = key.upper()
        if value.startswith('"') and value.endswith('"'):
            result[key] = value[1:-1]
            continue
        # Retain units separately but normalize the numeric value.
        numeric = re.match(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)", value)
        if numeric:
            number = float(numeric.group(1))
            result[key] = int(number) if number.is_integer() else number
        else:
            result[key] = value.strip()
    return result


def _dtype_from_label(label: Mapping[str, object]) -> np.dtype:
    bits = int(label.get("SAMPLE_BITS", 16))
    sample_type = str(label.get("SAMPLE_TYPE", "LSB_INTEGER")).upper()
    if bits != 16:
        raise ValueError(f"only 16-bit LDEM samples are currently supported, got {bits}")
    if "UNSIGNED" in sample_type:
        kind = "u2"
    else:
        kind = "i2"
    endian = ">" if any(token in sample_type for token in ("MSB", "SUN", "MAC")) else "<"
    return np.dtype(endian + kind)


@dataclass(frozen=True)
class LolaGlobalDem:
    raw: np.ndarray
    radius_scale_m: float
    radius_offset_m: float
    source: str
    label: Mapping[str, object]

    @classmethod
    def from_pds(cls, label_path: str | Path, image_path: str | Path | None = None,
                 *, memory_map: bool = True) -> "LolaGlobalDem":
        label_path = Path(label_path).expanduser().resolve()
        label = parse_pds3_label(label_path)
        rows = int(label.get("LINES", 0))
        cols = int(label.get("LINE_SAMPLES", 0))
        if rows <= 0 or cols <= 0:
            raise ValueError("PDS label must contain positive LINES and LINE_SAMPLES")
        if image_path is None:
            pointer = label.get("^IMAGE")
            if pointer is None:
                candidates = list(label_path.parent.glob(label_path.stem + ".IMG"))
                if not candidates:
                    raise ValueError("PDS label does not identify an image file")
                image_path = candidates[0]
            else:
                image_path = label_path.parent / str(pointer).strip('"')
        image_path = Path(image_path).expanduser().resolve()
        dtype = _dtype_from_label(label)
        expected = rows * cols
        if memory_map:
            data = np.memmap(image_path, dtype=dtype, mode="r", shape=(rows, cols))
        else:
            flat = np.fromfile(image_path, dtype=dtype, count=expected)
            if flat.size != expected:
                raise ValueError(f"expected {expected} DEM samples, found {flat.size}")
            data = flat.reshape(rows, cols)
        scale = float(label.get("SCALING_FACTOR", label.get("SCALE", 1.0)))
        offset = float(label.get("OFFSET", label.get("BASE", R_MOON_MEAN_KM * 1000.0)))
        return cls(data, scale, offset, f"{image_path} ({label_path.name})", label)

    @classmethod
    def from_array(cls, radius_m: np.ndarray, *, source: str = "in-memory LOLA-compatible DEM") -> "LolaGlobalDem":
        values = np.asarray(radius_m, dtype=float)
        if values.ndim != 2:
            raise ValueError("radius_m must be a two-dimensional global grid")
        return cls(values, 1.0, 0.0, source, {
            "LINES": values.shape[0], "LINE_SAMPLES": values.shape[1],
            "MAP_PROJECTION_TYPE": "SIMPLE CYLINDRICAL",
        })

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.raw.shape[0]), int(self.raw.shape[1])

    @property
    def radius_m(self) -> np.ndarray:
        return np.asarray(self.raw, dtype=float) * self.radius_scale_m + self.radius_offset_m

    def coordinate_grids_deg(self) -> tuple[np.ndarray, np.ndarray]:
        rows, cols = self.shape
        resolution_lat = rows / 180.0
        resolution_lon = cols / 360.0
        lat = 90.0 - (np.arange(rows, dtype=float) + 0.5) / resolution_lat
        lon = -180.0 + (np.arange(cols, dtype=float) + 0.5) / resolution_lon
        return np.meshgrid(lat, lon, indexing="ij")

    def sample_radius_km(self, latitude_deg: float, longitude_east_deg: float) -> float:
        rows, cols = self.shape
        lat = float(np.clip(latitude_deg, -90.0 + 90.0/rows, 90.0 - 90.0/rows))
        lon = (float(longitude_east_deg) + 180.0) % 360.0 - 180.0
        row = (90.0 - lat) / 180.0 * rows - 0.5
        col = (lon + 180.0) / 360.0 * cols - 0.5
        r0, c0 = int(math.floor(row)), int(math.floor(col))
        fr, fc = row-r0, col-c0
        r0 = int(np.clip(r0, 0, rows-1)); r1 = min(r0+1, rows-1)
        c0 %= cols; c1 = (c0+1) % cols
        grid = self.radius_m
        value = ((1-fr)*(1-fc)*grid[r0,c0] + (1-fr)*fc*grid[r0,c1]
                 + fr*(1-fc)*grid[r1,c0] + fr*fc*grid[r1,c1])
        return float(value / 1000.0)

    def derive_limb_profile(
        self,
        observer_direction_body: np.ndarray,
        observer_distance_km: float,
        *,
        n_position_angles: int = 1440,
        mean_radius_km: float = R_MOON_MEAN_KM,
        source: str | None = None,
    ) -> LunarLimbProfile:
        """Project the full DEM into the observer's apparent lunar limb.

        ``observer_direction_body`` points from the Moon centre toward the
        observer in the lunar body frame.  Perspective is retained at the
        supplied observer distance rather than assuming an infinitely distant
        orthographic observer.
        """
        direction = np.asarray(observer_direction_body, dtype=float).reshape(3)
        direction /= np.linalg.norm(direction)
        distance = float(observer_distance_km)
        if distance <= mean_radius_km:
            raise ValueError("observer_distance_km must exceed the lunar radius")
        north = np.array([0.0, 0.0, 1.0])
        north = north - direction * float(np.dot(north, direction))
        if np.linalg.norm(north) < 1.0e-10:
            north = np.array([1.0, 0.0, 0.0])
            north = north - direction * float(np.dot(north, direction))
        north /= np.linalg.norm(north)
        east = np.cross(north, direction)
        east /= np.linalg.norm(east)

        lat, lon = self.coordinate_grids_deg()
        lat_r, lon_r = np.radians(lat), np.radians(lon)
        radius = self.radius_m / 1000.0
        cos_lat = np.cos(lat_r)
        points = np.stack([
            radius*cos_lat*np.cos(lon_r),
            radius*cos_lat*np.sin(lon_r),
            radius*np.sin(lat_r),
        ], axis=-1).reshape(-1, 3)
        observer = direction * distance
        q = points - observer
        forward = -direction
        depth = q @ forward
        x = q @ east
        y = q @ north
        valid = depth > 0.0
        tangent = np.hypot(x[valid], y[valid]) / depth[valid]
        equivalent_radius = distance * tangent
        pa = np.mod(np.degrees(np.arctan2(x[valid], y[valid])), 360.0)
        bins = np.floor(pa / 360.0 * int(n_position_angles)).astype(int) % int(n_position_angles)
        maxima = np.full(int(n_position_angles), -np.inf, dtype=float)
        np.maximum.at(maxima, bins, equivalent_radius)

        # A global DEM samples latitude/longitude, not apparent position angle.
        # When the requested limb profile is finer than the source grid, some
        # narrow PA bins can contain only interior surface points even though
        # they are technically non-empty.  Treating those interior maxima as
        # the silhouette produced kilometre-to-hundreds-of-kilometres false
        # depressions at 720/1440 PA samples on coarse validation grids.
        #
        # Resolve the silhouette over the angular support of one source cell:
        # every output PA receives the maximum projected radius in the nearest
        # source-grid-width neighbourhood.  On real 64-pixel/degree LOLA data
        # this window is only a few hundredths of a degree; on coarse fixtures
        # it prevents under-sampling without inventing radial displacement.
        rows, cols = self.shape
        source_spacing_deg = max(180.0 / rows, 360.0 / cols)
        bin_width_deg = 360.0 / int(n_position_angles)
        half_window = max(1, int(math.ceil(source_spacing_deg / bin_width_deg)))
        maxima = np.maximum.reduce([
            np.roll(maxima, shift) for shift in range(-half_window, half_window + 1)
        ])

        missing = ~np.isfinite(maxima)
        if np.any(missing):
            good = np.flatnonzero(~missing)
            if len(good) < 2:
                raise RuntimeError("DEM projection did not populate enough limb position angles")
            extended_x = np.concatenate([good[-1:]-len(maxima), good, good[:1]+len(maxima)])
            extended_y = np.concatenate([maxima[good[-1:]], maxima[good], maxima[good[:1]]])
            maxima[missing] = np.interp(np.flatnonzero(missing), extended_x, extended_y)
        position_angle = (np.arange(int(n_position_angles), dtype=float)+0.5) * 360.0/int(n_position_angles)
        return LunarLimbProfile(
            position_angle, maxima,
            source=source or f"LOLA/LDEM perspective limb from {self.source}",
        )


def limb_profile_for_event(
    event: ReferenceEvent,
    dem: LolaGlobalDem,
    *,
    jd_utc: float | None = None,
    n_position_angles: int = 1440,
) -> LunarLimbProfile:
    """Derive the terrestrial observer's lunar limb for an event instant."""
    from ssapy_toolkit.compute.eclipse_state import event_moon_body_to_gcrf, event_positions_gcrf
    jd = float(event.greatest_jd if jd_utc is None else jd_utc)
    _, moon = event_positions_gcrf(event, jd)
    body_to_gcrf = np.asarray(event_moon_body_to_gcrf(event, jd), dtype=float)
    earth_direction_gcrf = -np.asarray(moon, dtype=float)
    distance = float(np.linalg.norm(earth_direction_gcrf))
    direction_body = body_to_gcrf.T @ (earth_direction_gcrf / distance)
    return dem.derive_limb_profile(
        direction_body, distance,
        n_position_angles=n_position_angles,
        source=f"LOLA/LDEM limb at {event.definition.key} greatest eclipse from {dem.source}",
    )


def write_limb_profile(profile: LunarLimbProfile, output: str | Path) -> str:
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".npz":
        np.savez_compressed(
            path,
            position_angle_deg=np.asarray(profile.position_angle_deg, dtype=float),
            radius_km=np.asarray(profile.radius_km_values, dtype=float),
            source=np.asarray(profile.source),
        )
    else:
        import csv
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["position_angle_deg", "radius_km"])
            writer.writerows(zip(profile.position_angle_deg, profile.radius_km_values))
    return str(path)
