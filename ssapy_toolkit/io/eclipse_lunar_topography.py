"""LOLA/Kaguya-compatible lunar topography and apparent limb extraction.

Large mission datasets are not embedded in the wheel.  This module consumes
user-supplied radius/elevation grids and turns them into the periodic
position-angle/radius profile used by local solar-contact prediction.  The
source path and dataset metadata are retained in every generated profile.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import json
import math
import re

import numpy as np

from ssapy_toolkit.coordinates.eclipse_lunar_geometry import LunarLimbProfile
from ssapy_toolkit.compute.eclipse_reference_events import R_MOON_MEAN_KM, ReferenceEvent
from ssapy_toolkit.compute.eclipse_state import event_moon_body_to_gcrf, event_positions_gcrf


LOLA_DATA_SOURCES = {
    "NASA_PGDA_MOON_PA_64": {
        "description": "LOLA principal-axis global radius grid, 64 pixels/degree",
        "reference_radius_km": R_MOON_MEAN_KM,
        "landing_page": "https://pgda.gsfc.nasa.gov/products/78",
    },
    "PDS_LOLA_GDR": {
        "description": "LRO LOLA gridded data record archive",
        "reference_radius_km": R_MOON_MEAN_KM,
        "landing_page": "https://pds-geosciences.wustl.edu/missions/lro/lola.htm",
    },
}


@dataclass(frozen=True)
class LunarTopographyGrid:
    latitude_deg: np.ndarray
    longitude_east_deg: np.ndarray
    radius_km: np.ndarray
    source: str
    body_frame: str = "Moon principal-axis frame"
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        lat = np.asarray(self.latitude_deg, dtype=float).reshape(-1)
        lon = np.asarray(self.longitude_east_deg, dtype=float).reshape(-1)
        radius = np.asarray(self.radius_km, dtype=float)
        if radius.shape != (len(lat), len(lon)):
            raise ValueError("lunar radius grid must have shape (N_lat, N_lon)")
        if np.any(~np.isfinite(lat)) or np.any(~np.isfinite(lon)):
            raise ValueError("lunar coordinates must be finite")
        if np.nanmin(radius) <= 0.0:
            raise ValueError("lunar radius values must be positive kilometres")
        lat_order = np.argsort(lat)
        lon_unwrapped = np.degrees(np.unwrap(np.radians(lon)))
        lon_order = np.argsort(lon_unwrapped)
        object.__setattr__(self, "latitude_deg", lat[lat_order])
        object.__setattr__(self, "longitude_east_deg", lon_unwrapped[lon_order])
        object.__setattr__(self, "radius_km", radius[np.ix_(lat_order, lon_order)])
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def from_npz(cls, path: str | Path, *, reference_radius_km: float = R_MOON_MEAN_KM) -> "LunarTopographyGrid":
        path = Path(path).expanduser().resolve()
        with np.load(path, allow_pickle=False) as data:
            lat = data["latitude_deg"] if "latitude_deg" in data else data["lat_deg"]
            lon = data["longitude_east_deg"] if "longitude_east_deg" in data else data["lon_deg"]
            if "radius_km" in data:
                radius = data["radius_km"]
            elif "elevation_km" in data:
                radius = float(reference_radius_km)+np.asarray(data["elevation_km"], dtype=float)
            elif "elevation_m" in data:
                radius = float(reference_radius_km)+np.asarray(data["elevation_m"], dtype=float)/1000.0
            else:
                raise ValueError("lunar NPZ needs radius_km, elevation_km, or elevation_m")
            frame = str(data["body_frame"].item()) if "body_frame" in data else "Moon principal-axis frame"
        return cls(lat, lon, radius, source=str(path), body_frame=frame)

    @staticmethod
    def _pds_label(path: Path) -> dict[str, str]:
        text = path.read_text(encoding="latin-1", errors="ignore")
        result: dict[str, str] = {}
        for raw in text.splitlines():
            line = raw.split("/*", 1)[0].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            result[key.strip().upper()] = value.strip().strip('"')
        return result

    @classmethod
    def from_pds_img(
        cls,
        label_path: str | Path,
        *,
        image_path: str | Path | None = None,
        reference_radius_km: float = R_MOON_MEAN_KM,
    ) -> "LunarTopographyGrid":
        label_path = Path(label_path).expanduser().resolve()
        label = cls._pds_label(label_path)
        if image_path is None:
            pointer = label.get("^IMAGE", "").strip().strip('()').split(",")[0].strip().strip('"')
            image_path = label_path.with_name(pointer) if pointer else label_path.with_suffix(".IMG")
        image_path = Path(image_path).expanduser().resolve()
        lines = int(float(label["LINES"]))
        samples = int(float(label["LINE_SAMPLES"]))
        bits = int(float(label.get("SAMPLE_BITS", "16")))
        sample_type = label.get("SAMPLE_TYPE", "MSB_INTEGER").upper()
        if bits not in {16, 32}:
            raise ValueError("only 16- and 32-bit PDS raster samples are supported")
        if bits == 16:
            dtype = ">i2" if ("MSB" in sample_type or "BIG" in sample_type) else "<i2"
        else:
            is_float = "REAL" in sample_type or "FLOAT" in sample_type
            dtype = (">f4" if is_float else ">i4") if ("MSB" in sample_type or "BIG" in sample_type) else ("<f4" if is_float else "<i4")
        offset_bytes = int(float(label.get("IMAGE_HEADER_BYTES", "0")))
        raw = np.fromfile(image_path, dtype=dtype, count=lines*samples, offset=offset_bytes).reshape(lines, samples).astype(float)
        scale = float(label.get("SCALING_FACTOR", label.get("SCALE", "1")))
        offset = float(label.get("OFFSET", "0"))
        values = raw*scale+offset
        units = label.get("UNIT", label.get("UNITS", "M")).upper()
        # LOLA GDR products commonly store elevation relative to 1737.4 km.
        if "KM" in units:
            elevation_km = values
        else:
            elevation_km = values/1000.0
        minimum_lat = float(label.get("MINIMUM_LATITUDE", "-90"))
        maximum_lat = float(label.get("MAXIMUM_LATITUDE", "90"))
        western_lon = float(label.get("WESTERNMOST_LONGITUDE", label.get("MINIMUM_LONGITUDE", "0")))
        eastern_lon = float(label.get("EASTERNMOST_LONGITUDE", label.get("MAXIMUM_LONGITUDE", "360")))
        lat = np.linspace(maximum_lat, minimum_lat, lines)
        lon = np.linspace(western_lon, eastern_lon, samples, endpoint=False)
        radius = float(reference_radius_km)+elevation_km
        return cls(lat, lon, radius, source=f"{label_path} + {image_path}", metadata={"pds_label": label})

    @classmethod
    def from_geotiff(cls, path: str | Path, *, reference_radius_km: float = R_MOON_MEAN_KM) -> "LunarTopographyGrid":
        path = Path(path).expanduser().resolve()
        try:
            import rasterio
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("lunar GeoTIFF loading requires rasterio") from exc
        with rasterio.open(path) as dataset:
            values = dataset.read(1).astype(float)
            if dataset.nodata is not None:
                values[values == float(dataset.nodata)] = np.nan
            rows = np.arange(dataset.height)
            cols = np.arange(dataset.width)
            lon = np.array([dataset.xy(dataset.height//2, int(c))[0] for c in cols])
            lat = np.array([dataset.xy(int(r), dataset.width//2)[1] for r in rows])
            units = str(dataset.tags(1).get("units", "m")).lower()
        elevation_km = values if units.startswith("km") else values/1000.0
        return cls(lat, lon, float(reference_radius_km)+elevation_km, source=str(path))

    @classmethod
    def from_file(cls, path: str | Path, **kwargs) -> "LunarTopographyGrid":
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".npz":
            return cls.from_npz(path, **kwargs)
        if suffix in {".lbl", ".lab"}:
            return cls.from_pds_img(path, **kwargs)
        if suffix in {".tif", ".tiff"}:
            return cls.from_geotiff(path, **kwargs)
        raise ValueError("lunar topography file must be NPZ, PDS LBL/IMG, or GeoTIFF")

    def limb_profile(
        self,
        observer_direction_body: np.ndarray,
        *,
        n_angles: int = 720,
        maximum_points: int = 2_000_000,
    ) -> LunarLimbProfile:
        """Extract the apparent topographic silhouette for one view direction."""
        view = np.asarray(observer_direction_body, dtype=float).reshape(3)
        view /= np.linalg.norm(view)
        north = np.array([0.0, 0.0, 1.0])-view[2]*view
        if np.linalg.norm(north) < 1e-10:
            north = np.array([1.0, 0.0, 0.0])-view[0]*view
        north /= np.linalg.norm(north)
        east = np.cross(view, north)
        east /= np.linalg.norm(east)

        total = self.radius_km.size
        stride = max(1, int(math.ceil(math.sqrt(total/max(int(maximum_points), 1)))))
        lat = np.radians(self.latitude_deg[::stride])
        lon = np.radians(self.longitude_east_deg[::stride])
        radius = self.radius_km[::stride, ::stride]
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        unit = np.stack([
            np.cos(lat_grid)*np.cos(lon_grid),
            np.cos(lat_grid)*np.sin(lon_grid),
            np.sin(lat_grid),
        ], axis=-1)
        points = unit.reshape(-1, 3)*radius.reshape(-1, 1)
        finite = np.all(np.isfinite(points), axis=1)
        points = points[finite]
        north_coordinate = points@north
        east_coordinate = points@east
        projected_radius = np.hypot(north_coordinate, east_coordinate)
        angle = np.mod(np.degrees(np.arctan2(east_coordinate, north_coordinate)), 360.0)
        bins = np.floor(angle/360.0*int(n_angles)).astype(int)%int(n_angles)
        result = np.full(int(n_angles), -np.inf)
        np.maximum.at(result, bins, projected_radius)
        centers = (np.arange(int(n_angles))+0.5)*360.0/int(n_angles)
        valid = np.isfinite(result)
        if np.count_nonzero(valid) < max(8, int(n_angles)//4):
            raise RuntimeError("topography grid is too sparse to form a complete limb profile")
        extended_angle = np.concatenate([centers[valid]-360.0, centers[valid], centers[valid]+360.0])
        extended_radius = np.tile(result[valid], 3)
        filled = np.interp(centers, extended_angle, extended_radius)
        return LunarLimbProfile(
            centers, filled,
            source=(f"topographic limb from {self.source}; frame={self.body_frame}; "
                    f"input_stride={stride}; samples={len(points)}"),
        )

    def limb_profile_for_event(
        self,
        event: ReferenceEvent,
        jd_utc: float,
        *,
        observer_gcrf_km: np.ndarray | None = None,
        n_angles: int = 720,
    ) -> LunarLimbProfile:
        _, moon = event_positions_gcrf(event, float(jd_utc))
        observer = np.zeros(3) if observer_gcrf_km is None else np.asarray(observer_gcrf_km, dtype=float)
        direction_gcrf = observer-moon
        direction_gcrf /= np.linalg.norm(direction_gcrf)
        body_to_gcrf = np.asarray(event_moon_body_to_gcrf(event, float(jd_utc)), dtype=float)
        direction_body = body_to_gcrf.T@direction_gcrf
        return self.limb_profile(direction_body, n_angles=n_angles)

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "body_frame": self.body_frame,
            "shape": list(self.radius_km.shape),
            "minimum_radius_km": float(np.nanmin(self.radius_km)),
            "maximum_radius_km": float(np.nanmax(self.radius_km)),
            "metadata": dict(self.metadata or {}),
        }


def write_lola_source_manifest(output: str | Path) -> str:
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "$schema": "ssapy-toolkit.eclipse.lunar-topography-sources/1.8",
        "embedded_data": False,
        "sources": LOLA_DATA_SOURCES,
        "note": "Large mission topography files are user-supplied and must retain their PDS/NASA provenance.",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)
