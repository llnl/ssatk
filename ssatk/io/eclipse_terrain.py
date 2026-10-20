"""Terrain-backed astronomical horizon profiles.

The terrain subsystem is optional and data-provider neutral.  It accepts small
portable NPZ grids, SRTM-style HGT tiles, and GeoTIFF rasters when ``rasterio``
is installed.  The output is the canonical :class:`HorizonProfile` consumed by
observer contact and visibility calculations.

Terrain elevation is never mixed into WGS-84 body geometry.  It only changes
practical line-of-sight clearance at a named observer, preserving a separate
reproducible geometric-horizon result.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ssatk.coordinates.eclipse_observer_geometry import HorizonProfile

EARTH_MEAN_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class TerrainGrid:
    latitude_deg: np.ndarray
    longitude_east_deg: np.ndarray
    elevation_m: np.ndarray
    source: str
    vertical_datum: str = "unspecified terrain elevation"

    def __post_init__(self) -> None:
        lat = np.asarray(self.latitude_deg, dtype=float).reshape(-1)
        lon = np.asarray(self.longitude_east_deg, dtype=float).reshape(-1)
        z = np.asarray(self.elevation_m, dtype=float)
        if z.shape != (len(lat), len(lon)):
            raise ValueError("terrain elevation_m must have shape (N_lat, N_lon)")
        if len(lat) < 2 or len(lon) < 2:
            raise ValueError("terrain grid must contain at least 2x2 samples")
        if not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)):
            raise ValueError("terrain coordinates must be finite")
        # Keep latitude ascending and longitude monotonically increasing.
        lat_order = np.argsort(lat)
        lon_unwrapped = np.degrees(np.unwrap(np.radians(lon)))
        lon_order = np.argsort(lon_unwrapped)
        object.__setattr__(self, "latitude_deg", lat[lat_order])
        object.__setattr__(self, "longitude_east_deg", lon_unwrapped[lon_order])
        object.__setattr__(self, "elevation_m", z[np.ix_(lat_order, lon_order)])

    @classmethod
    def from_npz(cls, path: str | Path) -> "TerrainGrid":
        path = Path(path).expanduser().resolve()
        with np.load(path, allow_pickle=False) as data:
            lat = data["latitude_deg"] if "latitude_deg" in data else data["lat_deg"]
            lon = data["longitude_east_deg"] if "longitude_east_deg" in data else data["lon_deg"]
            if "elevation_m" in data:
                elevation = data["elevation_m"]
            elif "height_m" in data:
                elevation = data["height_m"]
            elif "elevation_km" in data:
                elevation = np.asarray(data["elevation_km"], dtype=float)*1000.0
            else:
                raise ValueError("terrain NPZ needs elevation_m, height_m, or elevation_km")
            vertical = str(data["vertical_datum"].item()) if "vertical_datum" in data else "NPZ terrain elevation"
        return cls(lat, lon, elevation, source=str(path), vertical_datum=vertical)

    @classmethod
    def from_hgt(cls, path: str | Path) -> "TerrainGrid":
        """Load a standard big-endian signed-16-bit SRTM HGT tile."""
        path = Path(path).expanduser().resolve()
        match = re.match(r"^([NS])(\d{2})([EW])(\d{3})", path.stem.upper())
        if match is None:
            raise ValueError("HGT filename must begin with N/S latitude and E/W longitude, e.g. N40W075.hgt")
        raw = np.fromfile(path, dtype=">i2")
        side = int(round(math.sqrt(len(raw))))
        if side*side != len(raw):
            raise ValueError("HGT byte count is not a square 16-bit raster")
        south = int(match.group(2))*(1 if match.group(1) == "N" else -1)
        west = int(match.group(4))*(1 if match.group(3) == "E" else -1)
        # HGT rows run north to south; convert to ascending latitude.
        elevation = raw.reshape(side, side).astype(float)[::-1, :]
        elevation[elevation <= -32768] = np.nan
        lat = np.linspace(south, south+1.0, side)
        lon = np.linspace(west, west+1.0, side)
        return cls(lat, lon, elevation, source=str(path), vertical_datum="SRTM orthometric height")

    @classmethod
    def from_geotiff(cls, path: str | Path, *, band: int = 1) -> "TerrainGrid":
        path = Path(path).expanduser().resolve()
        try:
            import rasterio
            from rasterio.warp import transform as warp_transform
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("GeoTIFF terrain loading requires rasterio") from exc
        with rasterio.open(path) as dataset:
            array = dataset.read(int(band)).astype(float)
            if dataset.nodata is not None:
                array[array == float(dataset.nodata)] = np.nan
            rows = np.arange(dataset.height)
            cols = np.arange(dataset.width)
            # Rectilinear geographic rasters are the common case.  For a
            # projected raster, transform the center row/column back to WGS84.
            xs = np.array([dataset.xy(dataset.height//2, int(col))[0] for col in cols])
            ys = np.array([dataset.xy(int(row), dataset.width//2)[1] for row in rows])
            if dataset.crs and not dataset.crs.is_geographic:
                lon, _ = warp_transform(dataset.crs, "EPSG:4326", xs.tolist(), [ys[len(ys)//2]]*len(xs))
                _, lat = warp_transform(dataset.crs, "EPSG:4326", [xs[len(xs)//2]]*len(ys), ys.tolist())
                lon = np.asarray(lon); lat = np.asarray(lat)
            else:
                lon, lat = xs, ys
        return cls(lat, lon, array, source=str(path), vertical_datum="GeoTIFF elevation")

    @classmethod
    def from_file(cls, path: str | Path) -> "TerrainGrid":
        suffix = Path(path).suffix.lower()
        if suffix == ".npz":
            return cls.from_npz(path)
        if suffix == ".hgt":
            return cls.from_hgt(path)
        if suffix in {".tif", ".tiff"}:
            return cls.from_geotiff(path)
        raise ValueError("terrain file must be NPZ, HGT, or GeoTIFF")

    def elevation_at(self, latitude_deg, longitude_east_deg) -> np.ndarray | float:
        latq, lonq = np.broadcast_arrays(
            np.asarray(latitude_deg, dtype=float), np.asarray(longitude_east_deg, dtype=float)
        )
        lat = self.latitude_deg
        lon = self.longitude_east_deg
        center = 0.5*(lon[0]+lon[-1])
        lon_values = lonq + 360.0*np.round((center-lonq)/360.0)
        lat_values = np.clip(latq, lat[0], lat[-1])
        lon_values = np.clip(lon_values, lon[0], lon[-1])
        i = np.clip(np.searchsorted(lat, lat_values)-1, 0, len(lat)-2)
        j = np.clip(np.searchsorted(lon, lon_values)-1, 0, len(lon)-2)
        fy = (lat_values-lat[i])/np.maximum(lat[i+1]-lat[i], 1e-15)
        fx = (lon_values-lon[j])/np.maximum(lon[j+1]-lon[j], 1e-15)
        z = self.elevation_m
        result = (
            z[i, j]*(1-fx)*(1-fy) + z[i, j+1]*fx*(1-fy)
            + z[i+1, j]*(1-fx)*fy + z[i+1, j+1]*fx*fy
        )
        # Nearest finite fallback for voids.
        invalid = ~np.isfinite(result)
        if np.any(invalid):
            nearest = z[np.where(fy < 0.5, i, i+1), np.where(fx < 0.5, j, j+1)]
            result = np.where(invalid, nearest, result)
        return float(result) if result.ndim == 0 else result

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "vertical_datum": self.vertical_datum,
            "shape": list(self.elevation_m.shape),
            "latitude_range_deg": [float(self.latitude_deg[0]), float(self.latitude_deg[-1])],
            "longitude_range_east_deg": [float(self.longitude_east_deg[0]), float(self.longitude_east_deg[-1])],
            "minimum_elevation_m": float(np.nanmin(self.elevation_m)),
            "maximum_elevation_m": float(np.nanmax(self.elevation_m)),
        }


def _forward_sphere(lat_deg: float, lon_deg: float, azimuth_deg, distance_km):
    lat1 = math.radians(float(lat_deg))
    lon1 = math.radians(float(lon_deg))
    az = np.radians(np.asarray(azimuth_deg, dtype=float))
    sigma = np.asarray(distance_km, dtype=float)/EARTH_MEAN_RADIUS_KM
    sin_lat2 = math.sin(lat1)*np.cos(sigma)+math.cos(lat1)*np.sin(sigma)*np.cos(az)
    lat2 = np.arcsin(np.clip(sin_lat2, -1.0, 1.0))
    lon2 = lon1+np.arctan2(
        np.sin(az)*np.sin(sigma)*math.cos(lat1),
        np.cos(sigma)-math.sin(lat1)*np.sin(lat2),
    )
    return np.degrees(lat2), (np.degrees(lon2)+180.0)%360.0-180.0


def build_horizon_profile(
    terrain: TerrainGrid,
    *,
    latitude_deg: float,
    longitude_east_deg: float,
    observer_elevation_m: float | None = None,
    azimuth_step_deg: float = 1.0,
    maximum_distance_km: float = 250.0,
    radial_samples: int = 600,
    effective_earth_radius_factor: float = 1.0,
) -> HorizonProfile:
    """Trace the maximum terrain elevation angle in each azimuth direction.

    ``effective_earth_radius_factor`` may be set to 7/6 or 4/3 for a
    terrestrial-survey refraction convention.  Astronomical refraction is
    normally handled separately by :func:`observer_geometry.atmospheric_refraction_deg`,
    so the default is the geometric value 1.0.
    """
    from ssatk.coordinates.eclipse_observer_geometry import HorizonProfile

    if azimuth_step_deg <= 0.0 or maximum_distance_km <= 0.0 or radial_samples < 2:
        raise ValueError("horizon sampling arguments must be positive")
    ground = float(terrain.elevation_at(latitude_deg, longitude_east_deg))
    eye = ground if observer_elevation_m is None else float(observer_elevation_m)
    azimuths = np.arange(0.0, 360.0, float(azimuth_step_deg))
    # Log spacing resolves nearby ridges while still reaching a distant horizon.
    distances = np.geomspace(0.03, float(maximum_distance_km), int(radial_samples))
    horizon = np.empty_like(azimuths)
    radius_m = EARTH_MEAN_RADIUS_KM*1000.0*float(effective_earth_radius_factor)
    for index, azimuth in enumerate(azimuths):
        lat, lon = _forward_sphere(latitude_deg, longitude_east_deg, azimuth, distances)
        terrain_m = np.asarray(terrain.elevation_at(lat, lon), dtype=float)
        distance_m = distances*1000.0
        curvature_drop_m = distance_m*distance_m/(2.0*radius_m)
        angle = np.degrees(np.arctan2(terrain_m-eye-curvature_drop_m, distance_m))
        horizon[index] = float(np.nanmax(angle))
    return HorizonProfile(
        azimuths, horizon,
        source=(f"terrain horizon from {terrain.source}; max_distance={maximum_distance_km:g} km; "
                f"earth_radius_factor={effective_earth_radius_factor:g}"),
    )


def write_horizon_profile(profile: HorizonProfile, output: str | Path) -> str:
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".npz":
        np.savez_compressed(path, azimuth_deg=profile.azimuth_deg,
                            altitude_deg=profile.altitude_deg, source=profile.source)
    else:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["azimuth_deg", "altitude_deg"])
            writer.writerows(zip(profile.azimuth_deg, profile.altitude_deg))
    return str(path)
