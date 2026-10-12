"""Workflow contracts for scene exports and catalog selections, not physical validation."""
import numpy as np
import pytest
from PIL import Image
from astropy.time import Time
from ssatk.constants import AU
from ssatk.satellites import SpacecraftBody
from ssatk.plots.spacecraft_view import SpacecraftMesh, SpacecraftView, _rasterize_mesh

EPOCH = Time("2026-10-11", scale="utc").gps
MOON = np.array([0.0, 384_400_000.0, 0.0])
SUN = np.array([AU, 0.0, 0.0])


def scene(**kwargs):
    options = dict(r=[7e6, 0, 0], t=[EPOCH], q=[1, 0, 0, 0],
                   moon_position=MOON, sun_position=SUN, stars=False, textures=False,
                   body=SpacecraftBody.box(mass=1.0, size=[2.0, 1.0, 0.5]))
    options.update(kwargs)
    return SpacecraftView(**options)


def test_default_preset_custom_mesh_and_animation_selection():
    """C: no mesh file needed; custom input and resampling retain their geometry."""
    assert SpacecraftMesh.presets(group="generic") == ("starlink", "cubesat", "boxwing")
    default = scene(body=None)
    np.testing.assert_allclose(default.mesh.vertices.max(axis=0), [1.4, 4.7, 0.09], rtol=0, atol=1e-14)
    custom = SpacecraftMesh([[0, 0, 0], [1, 0, 0], [0, 2, 0]], [[0, 1, 2]])
    view = scene(body=None, mesh=custom, t=[EPOCH, EPOCH + 2])
    assert view.mesh is custom
    assert view.resample([EPOCH + 1]).mesh is custom
    # Each preset call owns its data, so editing one view cannot alter the next.
    default.mesh.vertices[:] = 0
    np.testing.assert_allclose(scene(body=None).mesh.vertices.max(axis=0), [1.4, 4.7, 0.09], rtol=0, atol=1e-14)


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


@pytest.mark.parametrize("name", SpacecraftMesh.presets())
def test_catalog_build_and_provenance_contract(name):
    """C: build every selection offline; finite geometry does NOT validate flight dimensions."""
    m = SpacecraftMesh.from_preset(name)
    info = SpacecraftMesh.preset_info(name)
    assert info["name"] == name and info["notes"] and info["axes"] and info["origin"]
    if info["group"] != "generic":
        assert info["sources"] and all(s.startswith("https://") for s in info["sources"])
    assert np.isfinite(m.vertices).all() and len(m.face_parts) == len(m.faces)
    # Convexity is required by fan triangulation. Sign changes reveal a malformed face.
    for face in m.faces:
        points = m.vertices[face]
        edges = np.roll(points, -1, axis=0) - points
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        turns = np.cross(edges, np.roll(edges, -1, axis=0)) @ normal
        assert np.all(turns >= -1e-12 * np.linalg.norm(normal)**2)


def test_aliases_scaling_and_metadata_are_independent():
    """C: common alternate names, punctuation, and caller-owned metadata."""
    for alias, name in (("HST", "hubble"), ("James Webb", "jwst"), ("K2", "kepler"), ("NEOWISE", "wise")):
        assert SpacecraftMesh.preset_info(alias)["name"] == name
        reference = SpacecraftMesh.from_preset(name)
        enlarged = SpacecraftMesh.from_preset(alias, scale=2)
        np.testing.assert_array_equal(enlarged.vertices, reference.vertices * 2)
    info = SpacecraftMesh.preset_info("webb")
    info["notes"] = "changed"
    assert SpacecraftMesh.preset_info("webb")["notes"] != "changed"


def test_polygon_fan_diagonal_is_not_drawn():
    """C: a quad's center is uniform fill; no artificial diagonal from its triangulation."""
    m = SpacecraftMesh([[-.8,-.8,2],[.8,-.8,2],[.8,.8,2],[-.8,.8,2]], [[0,1,2,3]], [1,1,1])
    image, _ = _rasterize_mesh(m.vertices[:,:2], m.vertices[:,2], m.triangles,
                             m.triangle_colors, 100, 100, m.triangle_boundaries)
    np.testing.assert_array_equal(image[40:60,40:60,:3], np.ones((20,20,3)))


@pytest.mark.parametrize("fps", [15, 24, 30])
def test_gif_playback_duration_does_not_accumulate_rounding_error(tmp_path, monkeypatch, fps):
    """C: decoded total delay must be 1 second within one GIF 10-ms tick."""
    def frame(self, index, **kwargs):
        # Distinct exact RGB frames isolate encoder timing from rendering.
        return np.full((16, 16, 3), [index * 7, 0, 0], dtype=np.uint8)
    monkeypatch.setattr(SpacecraftView, "render", frame)
    view = scene(t=[EPOCH, EPOCH+1])
    path = view.save(tmp_path / "timing.gif", fps=fps, duration_s=1)
    with Image.open(path) as image:
        assert image.n_frames == fps
        delays = []
        for i in range(image.n_frames):
            image.seek(i)
            delays.append(image.info["duration"])
        assert min(delays) >= 10
        assert abs(sum(delays) - 1000) <= 10


@pytest.mark.parametrize("body", ["earth", "moon"])
def test_foreground_body_occludes_mesh_in_export(body):
    """C/R1: collinear sphere blocks every box ray; nearer box remains visible."""
    from ssatk.constants import EARTH_RADIUS, MOON_RADIUS
    center, radius = (np.zeros(3), EARTH_RADIUS) if body == "earth" else (MOON, MOON_RADIUS)
    distance = radius * 1.2
    view = scene(r=center+[distance,0,0], camera_offset=[-2*distance,0,0],
                 environment=body, body=SpacecraftBody.box(mass=1, size=[radius/5]*3))
    figure = view.figure(width=160, height=160)
    assert np.count_nonzero(figure.axes[0].images[1].get_array()[...,3]) == 0
    view.camera_offset[:] = [distance/2,0,0]
    figure = view.figure(width=160, height=160)
    assert np.count_nonzero(figure.axes[0].images[1].get_array()[...,3]) > 0
