"""Analytic geometry and interpolation references, separate from export contracts."""

import numpy as np
import pytest
from astropy.time import Time
from PIL import Image

from ssatk.constants import AU, EARTH_RADIUS, MOON_RADIUS
from ssatk.plots.spacecraft_view import SpacecraftMesh, SpacecraftView, _camera_basis, _project, _ray_sphere
from ssatk.satellites import SpacecraftBody


EPOCH = Time("2026-10-11", scale="utc").gps
MOON = np.array([0.0, 384_400_000.0, 0.0])
SUN = np.array([AU, 0.0, 0.0])


def scene(**kwargs):
    options = dict(r=[7e6, 0, 0], t=[EPOCH], q=[1, 0, 0, 0],
                   moon_position=MOON, sun_position=SUN, stars=False, textures=False,
                   body=SpacecraftBody.box(mass=1.0, size=[2.0, 1.0, 0.5]))
    options.update(kwargs)
    return SpacecraftView(**options)


@pytest.mark.parametrize("sign", [1, -1])
def test_body_axes_and_vertices_match_exact_quarter_turn(sign):
    """R1: +90 deg about GCRF z maps body x to y and body y to -x; 1e-12.

    Vertex tolerance is 2 nm because absolute positions contain a 7e6 m offset.
    The antipodal quaternion must describe the same physical orientation.
    """
    view = scene(q=sign * np.array([1, 0, 0, 1]))
    geometry = view.geometry()
    expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(geometry["body_axes_gcrf"], expected, rtol=0, atol=1e-12)
    vertices = view.mesh.vertices
    reference = np.column_stack((-vertices[:, 1], vertices[:, 0], vertices[:, 2])) + view.r[0]
    np.testing.assert_allclose(geometry["vertices_gcrf_m"], reference, rtol=0, atol=2e-9)


def test_sun_vector_is_spacecraft_relative_and_does_not_follow_attitude():
    """R1: right triangle (3,4,0) fixes the sunward unit vector to (0.6,0.8,0)."""
    view = scene(sun_position=[7e6 + 3e7, 4e7, 0], q=[0, 1, 0, 0])
    np.testing.assert_allclose(view.geometry()["sun_direction_gcrf"], [0.6, 0.8, 0], rtol=0, atol=1e-15)
    assert view.geometry()["sun_center_visible"]


@pytest.mark.parametrize("position, expected", [([-7e6, 0, 0], False), ([7e6, 0, 0], True)])
def test_point_sun_earth_occultation(position, expected):
    """R1: on-axis observer behind/in front of Earth, for a +x Sun."""
    assert scene(r=position).geometry()["sun_center_visible"] is expected


def test_moon_occultation_and_environment_selection():
    """R1: lunar nightside blocks a Sun on +x; nearest surface is the Moon."""
    view = scene(r=MOON + [-MOON_RADIUS - 100e3, 0, 0])
    assert not view.geometry()["sun_center_visible"]
    assert view._bodies(0)[0][0] == "moon"


def test_lunar_origin_translation_and_kilometre_conversion():
    """R1: translation by the literal Moon origin, with explicit m/km scaling."""
    r_relative = np.array([0, MOON_RADIUS + 100e3, 0])
    lunar = scene(r=r_relative / 1000, units="km", frame="moon_centered_gcrf",
                  moon_position=MOON / 1000, sun_position=SUN / 1000)
    geocentric = scene(r=MOON + r_relative)
    np.testing.assert_allclose(lunar.r, [MOON + r_relative], rtol=0, atol=1e-7)
    np.testing.assert_allclose(lunar.geometry()["sun_direction_gcrf"],
                               geocentric.geometry()["sun_direction_gcrf"], rtol=0, atol=1e-15)


def test_perspective_projection_matches_pinhole_geometry():
    """R1: 90-deg vertical FOV, aspect 2: x/(2z), y/z, within 1e-15."""
    basis = _camera_basis([0, 0, 1], [0, 1, 0])
    # Looking toward +z with +y up gives image right along -x.
    uv, depth = _project([[2, 1, 2], [0, 0, -1]], basis, 2, 1)
    np.testing.assert_allclose(uv[0], [-0.5, 0.5], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(depth, [2, -1])


def test_sphere_angular_limb_and_ray_distance():
    """R1: sphere radius 1 at distance 2 subtends asin(1/2)=30 degrees."""
    np.testing.assert_allclose(_ray_sphere(np.array([0, 0, 1.0]), np.array([0, 0, 2.0]), 1), 1, rtol=0, atol=1e-15)
    for angle, hit in [(29.99, True), (30.01, False), (180, False)]:
        ray = np.array([np.sin(np.deg2rad(angle)), 0, np.cos(np.deg2rad(angle))])
        assert np.isfinite(_ray_sphere(ray, np.array([0, 0, 2.0]), 1)) == hit


def test_nonuniform_time_interpolation_has_constant_linear_and_angular_rate():
    """R1: x=7e6+10t m, rotation=170+5t deg, at t=0,1,4 s.

    t=2 must be x=7e6+20 and a half-turn, including across q's sign ambiguity.
    A frame-index interpolation would incorrectly choose t=1 instead.
    """
    dt = np.array([0, 1, 4.0])
    theta = np.deg2rad(170 + 5 * dt)
    q = np.column_stack((np.cos(theta / 2), np.zeros((3, 2)), np.sin(theta / 2)))
    q[1] *= -1
    r = np.column_stack((7e6 + 10 * dt, np.zeros((3, 2))))
    sampled = scene(r=r, t=EPOCH + dt, q=q).resample([EPOCH + 2])
    np.testing.assert_allclose(sampled.r, [[7e6 + 20, 0, 0]], rtol=0, atol=1e-9)
    np.testing.assert_allclose(sampled.geometry()["body_axes_gcrf"], np.diag([-1, -1, 1]), rtol=0, atol=1e-14)


def test_obj_scale_and_negative_indices_preserve_geometry(tmp_path):
    """C/R1: a literal 2x3 triangle has area 3, then 0.03 m² at scale 0.1."""
    obj = tmp_path / "triangle.obj"
    obj.write_text("v 0 0 0\nv 2 0 0\nv 0 3 0\nf -3/1 -2/2 -1/3\n")
    mesh = SpacecraftMesh.from_obj(obj, scale=0.1)
    np.testing.assert_allclose(mesh.vertices, [[0, 0, 0], [0.2, 0, 0], [0, 0.3, 0]], rtol=0, atol=1e-15)
    a, b, c = mesh.vertices[mesh.triangles[0]]
    assert np.linalg.norm(np.cross(b - a, c - a)) / 2 == pytest.approx(0.03, abs=1e-15)


def test_body_mesh_is_relative_to_the_propagated_mass_center():
    """R1: literal center-of-mass offset translates vertices without changing size."""
    body = SpacecraftBody.box(mass=1.0, size=[2, 1, 0.5]).with_updates(center_of_mass=[0.2, 0.1, 0])
    mesh = SpacecraftMesh.from_body(body)
    reference = np.vstack([f.vertices_body for f in body.facets]) - [0.2, 0.1, 0]
    np.testing.assert_allclose(mesh.vertices, reference, rtol=0, atol=1e-15)


@pytest.mark.parametrize("options", [
    {"q": [0, 0, 0, 0]}, {"q": [np.nan, 0, 0, 1]},
    {"r": [EARTH_RADIUS - 1, 0, 0]}, {"frame": "lunar_fixed"},
    {"t": [EPOCH, EPOCH]}, {"camera_offset": [0, 0, 0]},
])
def test_invalid_physical_inputs_are_rejected(options):
    """C: invalid orientation, epoch order, origin, or camera cannot imply a valid scene."""
    with pytest.raises(ValueError):
        scene(**options)


@pytest.mark.parametrize("suffix", ["png", "jpg", "gif"])
def test_export_dimensions_timestamps_and_frames(tmp_path, suffix):
    """C: decoding verifies format, dimensions and GIF playback; not physical validation."""
    view = scene(t=[EPOCH, EPOCH + 1], q=[[1, 0, 0, 0], [0.7071067811865476, 0, 0, 0.7071067811865476]])
    path = view.save(tmp_path / f"view.{suffix}", width=320, height=240, fps=5, duration_s=0.6)
    with Image.open(path) as image:
        assert image.size == (320, 240)
        if suffix == "gif":
            assert image.n_frames == 3
            frames = []
            for i in range(image.n_frames):
                image.seek(i)
                assert image.info["duration"] == 200
                frames.append(np.asarray(image.convert("RGB")))
            assert np.any(frames[0] != frames[-1])
        else:
            assert image.format == ("PNG" if suffix == "png" else "JPEG")
