"""Regression tests for Moon WebGL orbit JSON input."""

import importlib.util
from pathlib import Path


_MODULE_PATH = Path(__file__).parents[1] / "ssatk" / "plots" / "moon_webgl.py"
_SPEC = importlib.util.spec_from_file_location("moon_webgl_test_module", _MODULE_PATH)
MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(MODULE)

_BAKER_PATH = Path(__file__).parents[1] / "ssatk" / "io" / "moon_maps.py"
_BAKER_SPEC = importlib.util.spec_from_file_location("moon_maps_test_module", _BAKER_PATH)
BAKER = importlib.util.module_from_spec(_BAKER_SPEC)
assert _BAKER_SPEC.loader is not None
_BAKER_SPEC.loader.exec_module(BAKER)


def test_bake_normals_tilt_downslope_in_east_north_frame():
    import numpy as np

    # Same grid as the LOLA source: row 0 at the north pole, columns eastward
    # from -180 degrees.  The shader reads red as east and green as north.
    height, width = 180, 360
    latitude = np.radians(np.linspace(90.0, -90.0, height))[:, None]
    longitude = np.radians(np.linspace(-180.0, 180.0, width, endpoint=False))[None, :]
    slope = 0.01
    rises_north = slope * BAKER.R_MOON_KM * latitude * np.ones((1, width))
    rises_east = slope * BAKER.R_MOON_KM * np.cos(latitude) * longitude

    mid_latitudes = slice(30, 151)
    north = BAKER.bake_normals(rises_north)[mid_latitudes]
    east = BAKER.bake_normals(rises_east)[mid_latitudes]

    # Ground rising northward faces south; ground rising eastward faces west.
    assert np.all(north[..., 1] < 0.0)
    assert np.all(east[..., 0] < 0.0)
    np.testing.assert_allclose(north[..., 1], -slope, rtol=0.02)
    np.testing.assert_allclose(east[..., 0], -slope, rtol=0.02)
    np.testing.assert_allclose(north[..., 0], 0.0, atol=1e-12)


_FRAME_EPOCH_ISO = "2025-01-01T00:00:00"
# Optical libration keeps the sub-Earth point within about 8 deg of 0 deg
# selenographic longitude and latitude. The Earth-Moon rotating frame instead
# places it at 180 deg longitude, which these bounds reject.
_LIBRATION_BOUND_DEG = 9.0


def _frame_epoch_gps():
    from astropy.time import Time

    return Time(_FRAME_EPOCH_ISO, scale="utc").gps


def _lon_lat_deg(xyz):
    import numpy as np

    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    lon = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    lat = np.degrees(np.arcsin(xyz[:, 2] / np.linalg.norm(xyz, axis=1)))
    return lon, lat


def test_lunar_body_frame_keeps_earth_near_zero_longitude():
    # R2: SSAPy MoonOrientation (DE440 principal axes) puts the sub-Earth point
    # within the libration bounds of 0 deg longitude over a month.
    import numpy as np
    from ssatk.coordinates import gcrf_to_lunar_body

    times = _frame_epoch_gps() + np.linspace(0.0, 27.32 * 86400.0, 12)
    lon, lat = _lon_lat_deg(gcrf_to_lunar_body(np.zeros((len(times), 3)), times))

    assert np.all(np.abs(lon) < _LIBRATION_BOUND_DEG)
    assert np.all(np.abs(lat) < _LIBRATION_BOUND_DEG)


def test_gcrf_tracks_register_on_the_lunar_near_side_in_every_unit_mode():
    # R2: a point 100 km above the sub-Earth side must be drawn over the near
    # side, at 1837.4 km from the Moon's centre, whether given in m, km, or auto.
    import numpy as np
    import ssapy

    t = _frame_epoch_gps()
    moon = ssapy.get_body("moon")
    r_moon = np.asarray(moon.position(t), dtype=float).reshape(3)
    point_m = r_moon - r_moon / np.linalg.norm(r_moon) * (moon.radius + 100.0e3)
    for units, scale in (("m", 1.0), ("km", 1.0e-3), ("auto", 1.0), ("auto", 1.0e-3)):
        xyz = MODULE._orbit_xyz((point_m * scale)[None, :], np.array([t]), "gcrf", units=units)
        lon, lat = _lon_lat_deg(xyz)
        assert abs(lon[0]) < _LIBRATION_BOUND_DEG, (units, scale, lon[0])
        assert abs(lat[0]) < _LIBRATION_BOUND_DEG, (units, scale, lat[0])
        assert abs(np.linalg.norm(xyz) - 1837.4) < 0.01, (units, scale)  # km


def test_moon_webgl_stars_share_the_textured_moon_frame(monkeypatch):
    # R2: a star in the direction of Earth must sit over the near side, in the
    # same body frame as the textured Moon.
    import numpy as np
    import ssapy
    from ssatk.plots import starfield

    t = _frame_epoch_gps()
    moon = ssapy.get_body("moon")
    r_moon = np.asarray(moon.position(t), dtype=float).reshape(3)
    toward_earth = -r_moon / np.linalg.norm(r_moon)
    monkeypatch.setattr(
        starfield,
        "star_directions",
        lambda **kwargs: (toward_earth[None, :], np.array([1.0]), np.ones((1, 3))),
    )

    lon, lat = _lon_lat_deg(starfield.moon_fixed_webgl_stars(epoch=t)["p"])

    assert abs(lon[0]) < _LIBRATION_BOUND_DEG
    assert abs(lat[0]) < _LIBRATION_BOUND_DEG
