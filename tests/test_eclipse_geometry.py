
import numpy as np
from astropy.time import Time



def test_2024_solar_eclipse_track_reaches_central_texas():
    """NASA/GSFC's 2024-04-08 totality path crosses central Texas near 18:39 UTC."""
    from ssatk.plots.eclipse_space_view_plotly import (
        RE_KM,
        _eci_surface_to_latlon,
        _great_circle_distance_km,
        _real_solar_eclipse_geometry,
        _shadow_ground_point,
        _time_grid_utc,
    )

    travis_county_lat = 30.3630
    travis_county_lon = -97.9790
    times = _time_grid_utc("2024-04-08T18:00:00", "2024-04-08T19:20:00", 81)
    moon_pos, sun_hat = _real_solar_eclipse_geometry(times)

    hits = []
    for time, moon_i, sun_i in zip(times, moon_pos, sun_hat):
        hit = _shadow_ground_point(moon_i, sun_i, np.zeros(3), earth_r_real=RE_KM)
        assert hit is not None
        lat, lon = _eci_surface_to_latlon(hit, time)
        distance_km = _great_circle_distance_km(lat, lon, travis_county_lat, travis_county_lon)
        hits.append((distance_km, time, lat, lon))

    latlon_path = [(lat, lon) for _, _, lat, lon in hits]
    assert any(21.0 < lat < 26.0 and -109.0 < lon < -104.0 for lat, lon in latlon_path)
    assert any(35.0 < lat < 44.0 and -93.0 < lon < -78.0 for lat, lon in latlon_path)

    distance_km, closest_time, closest_lat, closest_lon = min(hits, key=lambda row: row[0])
    assert abs((closest_time - Time("2024-04-08T18:39:00", scale="utc")).sec) <= 90.0
    assert distance_km < 150.0
    assert 30.0 < closest_lat < 32.0
    assert -100.0 < closest_lon < -97.0


def test_eclipse_surface_latlon_round_trips_with_earth_rotation():
    from ssatk.plots.eclipse_space_view_plotly import (
        _eci_surface_to_latlon,
        _latlon_to_eci_surface,
    )

    sample_time = Time("2024-04-08T18:39:00", scale="utc")
    for lat, lon in [(-45.0, -170.0), (0.0, 0.0), (30.363, -97.979), (71.0, 145.0)]:
        point = _latlon_to_eci_surface(lat, lon, sample_time)
        got_lat, got_lon = _eci_surface_to_latlon(point, sample_time)
        np.testing.assert_allclose(got_lat, lat, atol=1e-9)
        wrapped_diff = ((got_lon - lon + 180.0) % 360.0) - 180.0
        np.testing.assert_allclose(wrapped_diff, 0.0, atol=1e-9)
