"""R1/R2 orbit and camera checks, plus C export contracts for the GEO movie."""

from pathlib import Path

from astropy import units as u
from astropy.coordinates import CartesianRepresentation, GCRS, ITRS
import numpy as np
import pytest

from demos.six_dof import demo_spacecraft_view as demo
from ssatk.constants import AU
from ssatk.plots.spacecraft_view import SpacecraftView


def test_geo_states_match_circular_solution_and_astropy_ground_position():
    """R1/R2: mean-sidereal circular motion and independently transformed subpoint.

    Radius and speed literals follow mu=3.986004418e14 m^3/s^2 and
    T=86164.0905 s. Allow 1 mm and 1 micrometre/s. A trigonometric solution
    seeded by the initial state is a 1-mm oracle for SSAPy's Kepler propagation.
    Astropy's IAU frame chain must keep Greenwich latitude/longitude within
    0.001 degree: includes frame-model differences and one day of equator drift.
    """
    times, r, v, _ = demo.geo_states(49)
    assert times.utc[0].isot == "2026-10-11T12:00:00.000"
    assert times.utc[-1].isot == "2026-10-12T12:00:00.000"
    omega = 2 * np.pi / 86164.0905
    dt = times.gps - times.gps[0]
    phase = omega * dt
    reference_r = np.cos(phase)[:, None] * r[0] + np.sin(phase)[:, None] * v[0] / omega
    reference_v = -omega * np.sin(phase)[:, None] * r[0] + np.cos(phase)[:, None] * v[0]
    np.testing.assert_allclose(np.linalg.norm(r, axis=1), 42164169.624, rtol=0, atol=1e-3)
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 3074.6600995, rtol=0, atol=1e-6)
    np.testing.assert_allclose(r, reference_r, rtol=0, atol=1e-3)
    np.testing.assert_allclose(v, reference_v, rtol=0, atol=1e-6)
    earth_fixed = GCRS(CartesianRepresentation(r.T * u.m), obstime=times).transform_to(ITRS(obstime=times))
    np.testing.assert_allclose(earth_fixed.spherical.lon.wrap_at(180 * u.deg).deg,
                               0, rtol=0, atol=0.001)
    np.testing.assert_allclose(earth_fixed.spherical.lat.deg, 0, rtol=0, atol=0.001)


def test_geo_attitude_points_nadir_and_camera_travels_with_spacecraft():
    """R1: +Z nadir, +X tangent, and a camera aimed at the mesh at five mesh radii.

    Direction tolerance is 1e-12; eye placement allows 10 nm for subtracting
    42,164-km coordinates. The half-day scene must travel to the far side of
    Earth, catching an accidentally fixed camera. Specified Sun/Moon vectors
    avoid loading catalogue data; this checks supplied scene geometry only.
    """
    times, r, v, q = demo.geo_states(49)
    view = SpacecraftView(r, times, q, camera="limb", stars=False, textures=False, environment="earth",
                          sun_position=[AU, 0, 0], moon_position=[0, 384400000., 0])
    for i in [0, 12, 24, 36, 48]:
        axes = view.geometry(i)["body_axes_gcrf"]
        np.testing.assert_allclose(axes[:, 2], -r[i] / np.linalg.norm(r[i]), rtol=0, atol=1e-12)
        np.testing.assert_allclose(axes[:, 0], v[i] / np.linalg.norm(v[i]), rtol=0, atol=1e-12)
        eye, _, basis = view._camera(i)
        np.testing.assert_allclose(basis @ (r[i] - eye), [0, 0, 5 * view.mesh_radius], rtol=0, atol=1e-8)
    assert np.dot(view._camera(0)[0], view._camera(24)[0]) < -0.99 * np.linalg.norm(r[0])**2


@pytest.mark.parametrize("fast,fps,frames,size", [(True, 12, 540, (960, 640)), (False, 24, 1080, (1280, 720))])
def test_demo_exports_full_length_movie_with_one_state_per_frame(monkeypatch, tmp_path, fast, fps, frames, size):
    """C: fast mode may reduce pixels/fps, but never the 45-s playback or frame coverage.

    A save spy checks the demo's export request without rendering hundreds of
    frames in a unit test; the gallery run separately encodes and decodes MP4.
    """
    calls = []

    class View:
        def __init__(self, environment, count):
            self.environment, self.count = environment, count

        def save(self, path, **kwargs):
            calls.append((self.environment, self.count, Path(path), kwargs))
            return path

    monkeypatch.setattr(demo, "example_view", lambda environment, count=None: View(environment, count))
    demo.main(fast=fast, output_dir=tmp_path)
    earth_movies = [call for call in calls if call[0] == "earth" and call[2].suffix == ".mp4"]
    assert len(earth_movies) == 1
    _, count, _, options = earth_movies[0]
    assert count == frames and options["fps"] == fps and options["duration_s"] == 45
    assert (options["width"], options["height"]) == size
    # Actual propagated input grid matches save()'s output grid, avoiding orbit chords.
    times, _, _, _ = demo.geo_states(count)
    np.testing.assert_array_equal(times.gps, np.linspace(times.gps[0], times.gps[-1], frames))
    assert times.gps[-1] - times.gps[0] == 86400
