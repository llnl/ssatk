"""Data-backed terrestrial horizon profiles for eclipse observers.

The module reads standard SRTM ``.HGT`` tiles, georeferenced rasters supported
by Rasterio, and Terrarium-encoded PNG tiles.  It computes an azimuth-dependent
horizon using WGS-84 ECEF geometry, so Earth curvature and observer elevation
are handled without a flat-Earth approximation.

No terrain dataset is bundled.  This keeps the eclipse wheel small and avoids
silently substituting decorative terrain for a dated scientific input.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence
import csv
import math
import re

import numpy as np
from PIL import Image

WGS84_A_M = 6_378_137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


class TerrainSource(Protocol):
    source: str

    def contains(self, latitude_deg: float, longitude_east_deg: float) -> bool: ...
    def elevation_m(self, latitude_deg: float, longitude_east_deg: float) -> float: ...


def _wrap_lon(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _bilinear(grid: np.ndarray, row: float, col: float, *, nodata: float | None = None) -> float:
    rows, cols = grid.shape
    row = float(np.clip(row, 0.0, rows - 1.0))
    col = float(np.clip(col, 0.0, cols - 1.0))
    r0, c0 = int(math.floor(row)), int(math.floor(col))
    r1, c1 = min(r0 + 1, rows - 1), min(c0 + 1, cols - 1)
    fr, fc = row - r0, col - c0
    values = np.array([grid[r0, c0], grid[r0, c1], grid[r1, c0], grid[r1, c1]], dtype=float)
    weights = np.array([(1-fr)*(1-fc), (1-fr)*fc, fr*(1-fc), fr*fc], dtype=float)
    valid = np.isfinite(values)
    if nodata is not None:
        valid &= values != float(nodata)
    if not np.any(valid):
        return float("nan")
    return float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]))


@dataclass(frozen=True)
class SRTMHgtTile:
    data_m: np.ndarray
    south_lat_deg: int
    west_lon_deg: int
    source: str
    nodata: int = -32768

    @classmethod
    def from_file(cls, path: str | Path) -> "SRTMHgtTile":
        path = Path(path).expanduser().resolve()
        match = re.search(r"([NS])(\d{2})([EW])(\d{3})", path.name.upper())
        if match is None:
            raise ValueError("SRTM filename must contain a tile name such as N37W105")
        south = int(match.group(2)) * (1 if match.group(1) == "N" else -1)
        west = int(match.group(4)) * (1 if match.group(3) == "E" else -1)
        raw = np.fromfile(path, dtype=">i2")
        side = int(round(math.sqrt(raw.size)))
        if side * side != raw.size or side < 2:
            raise ValueError(f"{path.name} is not a square 16-bit SRTM HGT tile")
        return cls(raw.reshape(side, side).astype(float), south, west, str(path))

    @property
    def samples(self) -> int:
        return int(self.data_m.shape[0])

    def contains(self, latitude_deg: float, longitude_east_deg: float) -> bool:
        lon = _wrap_lon(longitude_east_deg)
        return (self.south_lat_deg <= float(latitude_deg) <= self.south_lat_deg + 1.0
                and self.west_lon_deg <= lon <= self.west_lon_deg + 1.0)

    def elevation_m(self, latitude_deg: float, longitude_east_deg: float) -> float:
        if not self.contains(latitude_deg, longitude_east_deg):
            return float("nan")
        lon = _wrap_lon(longitude_east_deg)
        # HGT row zero is the north edge and column zero is the west edge.
        row = (self.south_lat_deg + 1.0 - float(latitude_deg)) * (self.samples - 1)
        col = (lon - self.west_lon_deg) * (self.samples - 1)
        return _bilinear(self.data_m, row, col, nodata=self.nodata)


@dataclass
class RasterTerrain:
    data_m: np.ndarray
    transform: object
    crs: object
    nodata: float | None
    source: str
    bounds: object

    @classmethod
    def from_file(cls, path: str | Path, *, band: int = 1) -> "RasterTerrain":
        try:
            import rasterio
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Rasterio is required for GeoTIFF/DEM terrain input") from exc
        path = Path(path).expanduser().resolve()
        with rasterio.open(path) as dataset:
            data = dataset.read(int(band)).astype(float)
            return cls(data, dataset.transform, dataset.crs, dataset.nodata, str(path), dataset.bounds)

    def _xy(self, latitude_deg: float, longitude_east_deg: float) -> tuple[float, float]:
        lon, lat = _wrap_lon(longitude_east_deg), float(latitude_deg)
        if self.crs is None or str(self.crs).upper() in {"EPSG:4326", "OGC:CRS84"}:
            return lon, lat
        try:
            from rasterio.warp import transform
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Rasterio coordinate transformation is unavailable") from exc
        x, y = transform("EPSG:4326", self.crs, [lon], [lat])
        return float(x[0]), float(y[0])

    def contains(self, latitude_deg: float, longitude_east_deg: float) -> bool:
        x, y = self._xy(latitude_deg, longitude_east_deg)
        return self.bounds.left <= x <= self.bounds.right and self.bounds.bottom <= y <= self.bounds.top

    def elevation_m(self, latitude_deg: float, longitude_east_deg: float) -> float:
        if not self.contains(latitude_deg, longitude_east_deg):
            return float("nan")
        try:
            from rasterio.transform import rowcol
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Rasterio transform helpers are unavailable") from exc
        x, y = self._xy(latitude_deg, longitude_east_deg)
        row, col = rowcol(self.transform, x, y, op=float)
        return _bilinear(self.data_m, row, col, nodata=self.nodata)


@dataclass(frozen=True)
class TerrariumTile:
    elevation_grid_m: np.ndarray
    zoom: int
    x: int
    y: int
    source: str

    @classmethod
    def from_file(cls, path: str | Path, *, zoom: int, x: int, y: int) -> "TerrariumTile":
        path = Path(path).expanduser().resolve()
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=float)
        elevation = rgb[..., 0] * 256.0 + rgb[..., 1] + rgb[..., 2] / 256.0 - 32768.0
        return cls(elevation, int(zoom), int(x), int(y), str(path))

    @property
    def size(self) -> int:
        return int(self.elevation_grid_m.shape[0])

    def _pixel(self, latitude_deg: float, longitude_east_deg: float) -> tuple[float, float]:
        n = 2 ** self.zoom
        lon = _wrap_lon(longitude_east_deg)
        lat = float(np.clip(latitude_deg, -85.05112878, 85.05112878))
        gx = (lon + 180.0) / 360.0 * n
        gy = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) * 0.5 * n
        return (gy - self.y) * self.size, (gx - self.x) * self.size

    def contains(self, latitude_deg: float, longitude_east_deg: float) -> bool:
        row, col = self._pixel(latitude_deg, longitude_east_deg)
        return 0.0 <= row <= self.size - 1 and 0.0 <= col <= self.size - 1

    def elevation_m(self, latitude_deg: float, longitude_east_deg: float) -> float:
        row, col = self._pixel(latitude_deg, longitude_east_deg)
        if not (0.0 <= row <= self.size - 1 and 0.0 <= col <= self.size - 1):
            return float("nan")
        return _bilinear(self.elevation_grid_m, row, col)


@dataclass(frozen=True)
class TerrainMosaic:
    sources: tuple[TerrainSource, ...]
    source: str = "terrain mosaic"

    def __init__(self, sources: Sequence[TerrainSource], source: str = "terrain mosaic"):
        object.__setattr__(self, "sources", tuple(sources))
        object.__setattr__(self, "source", source)
        if not self.sources:
            raise ValueError("TerrainMosaic requires at least one terrain source")

    def contains(self, latitude_deg: float, longitude_east_deg: float) -> bool:
        return any(item.contains(latitude_deg, longitude_east_deg) for item in self.sources)

    def elevation_m(self, latitude_deg: float, longitude_east_deg: float) -> float:
        for item in self.sources:
            if item.contains(latitude_deg, longitude_east_deg):
                value = float(item.elevation_m(latitude_deg, longitude_east_deg))
                if np.isfinite(value):
                    return value
        return float("nan")


def load_terrain(path: str | Path, **kwargs) -> TerrainSource:
    path = Path(path).expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix == ".hgt":
        return SRTMHgtTile.from_file(path)
    if suffix in {".tif", ".tiff", ".img", ".vrt", ".nc"}:
        return RasterTerrain.from_file(path, band=int(kwargs.get("band", 1)))
    if suffix == ".png" and {"zoom", "x", "y"}.issubset(kwargs):
        return TerrariumTile.from_file(path, zoom=int(kwargs["zoom"]), x=int(kwargs["x"]), y=int(kwargs["y"]))
    raise ValueError("terrain input must be .hgt, a Rasterio-supported raster, or a Terrarium PNG with zoom/x/y")


def geodetic_to_ecef_m(latitude_deg: float, longitude_east_deg: float, height_m: float) -> np.ndarray:
    lat, lon = math.radians(float(latitude_deg)), math.radians(float(longitude_east_deg))
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    n = WGS84_A_M / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    return np.array([
        (n + height_m) * cos_lat * math.cos(lon),
        (n + height_m) * cos_lat * math.sin(lon),
        (n * (1.0 - WGS84_E2) + height_m) * sin_lat,
    ])


def local_up_east_north(latitude_deg: float, longitude_east_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat, lon = math.radians(float(latitude_deg)), math.radians(float(longitude_east_deg))
    up = np.array([math.cos(lat)*math.cos(lon), math.cos(lat)*math.sin(lon), math.sin(lat)])
    east = np.array([-math.sin(lon), math.cos(lon), 0.0])
    north = np.cross(up, east)
    return up, east, north / np.linalg.norm(north)


def destination_point(latitude_deg: float, longitude_east_deg: float,
                      azimuth_deg: float, distance_m: float) -> tuple[float, float]:
    """Spherical geodesic direct solution, adequate for horizon sampling."""
    radius = 6_371_008.8
    lat1, lon1 = math.radians(latitude_deg), math.radians(longitude_east_deg)
    az = math.radians(azimuth_deg)
    delta = float(distance_m) / radius
    lat2 = math.asin(np.clip(
        math.sin(lat1)*math.cos(delta) + math.cos(lat1)*math.sin(delta)*math.cos(az), -1.0, 1.0
    ))
    lon2 = lon1 + math.atan2(
        math.sin(az)*math.sin(delta)*math.cos(lat1),
        math.cos(delta)-math.sin(lat1)*math.sin(lat2),
    )
    return math.degrees(lat2), _wrap_lon(math.degrees(lon2))


def build_horizon_profile(
    *,
    latitude_deg: float,
    longitude_east_deg: float,
    observer_elevation_m: float,
    terrain: TerrainSource,
    azimuth_step_deg: float = 1.0,
    max_distance_km: float = 250.0,
    min_distance_m: float = 30.0,
    radial_samples: int = 700,
    missing: str = "skip",
):
    """Compute an azimuth-dependent WGS-84 terrain horizon.

    The returned object is the canonical :class:`HorizonProfile`.  Missing DEM
    samples can either be skipped or treated as sea level.
    """
    from ssapy_toolkit.coordinates.eclipse_observer_geometry import HorizonProfile

    if azimuth_step_deg <= 0.0 or max_distance_km <= 0.0 or radial_samples < 2:
        raise ValueError("horizon sampling parameters must be positive")
    observer = geodetic_to_ecef_m(latitude_deg, longitude_east_deg, observer_elevation_m)
    up, _, _ = local_up_east_north(latitude_deg, longitude_east_deg)
    # Dense nearby sampling and logarithmic long-range coverage.
    distances = np.geomspace(float(min_distance_m), float(max_distance_km)*1000.0, int(radial_samples))
    azimuths = np.arange(0.0, 360.0, float(azimuth_step_deg))
    horizon = np.full(azimuths.shape, -90.0, dtype=float)
    for index, azimuth in enumerate(azimuths):
        best = -90.0
        for distance in distances:
            lat, lon = destination_point(latitude_deg, longitude_east_deg, float(azimuth), float(distance))
            elevation = float(terrain.elevation_m(lat, lon))
            if not np.isfinite(elevation):
                if missing == "sea-level":
                    elevation = 0.0
                else:
                    continue
            target = geodetic_to_ecef_m(lat, lon, elevation)
            line = target - observer
            vertical = float(np.dot(line, up))
            horizontal = math.sqrt(max(float(np.dot(line, line)) - vertical*vertical, 0.0))
            angle = math.degrees(math.atan2(vertical, horizontal))
            best = max(best, angle)
        horizon[index] = 0.0 if best <= -89.0 else best
    return HorizonProfile(azimuths, horizon, source=f"WGS-84 terrain horizon from {terrain.source}")


def write_horizon_profile(profile, output: str | Path) -> str:
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".npz":
        np.savez_compressed(path, azimuth_deg=profile.azimuth_deg, altitude_deg=profile.altitude_deg,
                            source=np.asarray(profile.source))
    else:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["azimuth_deg", "altitude_deg"])
            writer.writerows(zip(profile.azimuth_deg, profile.altitude_deg))
    return str(path)
