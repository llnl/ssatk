import json

import numpy as np
import pytest

from ssatk.geometry import SpacecraftMesh, coerce_mesh, convert_mesh
from ssatk.satellites import SpacecraftBody


def write_obj(path, scale=1.0):
    path.write_text(
        "v 0 0 0\n"
        f"v {scale} 0 0\n"
        f"v {scale} {scale} 0\n"
        f"v 0 {scale} 0\n"
        "f -4 -3 -2 -1\n",
        encoding="utf-8",
    )


def test_obj_scale_negative_indices_area_and_export(tmp_path):
    """OBJ negative indices and centimetre scale preserve area to 2e-10 m²."""
    source = tmp_path / "plate.obj"
    write_obj(source, 2.0)
    mesh = SpacecraftMesh.from_file(source, units="cm")
    assert mesh.vertices.max() == pytest.approx(0.02, rel=0, abs=1e-12)
    assert sum(np.linalg.norm(np.cross(mesh.vertices[t[1]] - mesh.vertices[t[0]], mesh.vertices[t[2]] - mesh.vertices[t[0]])) / 2 for t in mesh.triangles) == pytest.approx(0.0004, rel=0, abs=2e-10)
    exported = tmp_path / "roundtrip.obj"
    convert_mesh(source, exported, units="cm")
    assert SpacecraftMesh.from_file(exported).vertices.max() == pytest.approx(0.02, rel=0, abs=1e-12)


def test_body_com_offset_and_payload_dimensions():
    """Body conversion preserves 2x1x0.5 m dimensions and COM offset to 1e-12 m."""
    body = SpacecraftBody.box(mass=10.0, size=(2.0, 1.0, 0.5)).with_updates(center_of_mass=(0.25, -0.1, 0.0))
    mesh = SpacecraftMesh.from_body(body)
    assert np.ptp(mesh.vertices, axis=0).tolist() == pytest.approx([2.0, 1.0, 0.5], rel=0, abs=1e-12)
    assert np.mean(mesh.vertices, axis=0) == pytest.approx(-np.array([0.25, -0.1, 0.0]), rel=0, abs=1e-12)
    assert SpacecraftMesh.from_dict(json.loads(json.dumps(mesh.to_dict()))).to_dict() == mesh.to_dict()
    assert coerce_mesh(body).to_dict() == mesh.to_dict()


def test_preset_is_lazy_and_preserves_payload_size():
    """CubeSat preset scaling preserves its measured envelope to 1e-6 m."""
    mesh = SpacecraftMesh.from_preset("cubesat", scale=2.0)
    assert np.ptp(mesh.vertices, axis=0) == pytest.approx([0.2, 0.2, 0.681], rel=0, abs=1e-6)
    assert "cubesat" in SpacecraftMesh.presets()


def test_trimesh_scene_transform_and_color(tmp_path):
    """Scene-node translation and face color survive GLB import to 1e-6 m."""
    trimesh = pytest.importorskip("trimesh")
    source = tmp_path / "scene.glb"
    first = trimesh.creation.box(extents=(1, 2, 3))
    second = trimesh.creation.box(extents=(1, 1, 1))
    second.visual.face_colors = [255, 0, 0, 255]
    scene = trimesh.Scene()
    scene.add_geometry(first, node_name="first")
    scene.add_geometry(second, node_name="second", transform=trimesh.transformations.translation_matrix([10, 0, 0]))
    scene.export(source)
    mesh = SpacecraftMesh.from_file(source)
    np.testing.assert_allclose(mesh.vertices.min(axis=0), [-0.5, -1, -1.5], rtol=0, atol=1e-6)
    np.testing.assert_allclose(mesh.vertices.max(axis=0), [10.5, 1, 1.5], rtol=0, atol=1e-6)
    assert any(np.allclose(color, [1, 0, 0]) for color in mesh.colors)


@pytest.mark.parametrize("extension", ["stl", "ply", "off", "glb", "gltf", "3mf"])
def test_trimesh_standard_formats_preserve_extent(extension, tmp_path):
    """Standard mesh imports preserve metre extents to 1e-6 m and 3MF mm to 1e-9 m."""
    trimesh = pytest.importorskip("trimesh")
    source = tmp_path / f"box.{extension}"
    fixture = trimesh.creation.box(extents=(2.0, 3.0, 4.0))
    fixture.units = "m"
    SpacecraftMesh(fixture.vertices, fixture.faces).export(source)
    mesh = SpacecraftMesh.from_file(source)
    assert np.ptp(mesh.vertices, axis=0) == pytest.approx([2.0, 3.0, 4.0], rel=0, abs=1e-6)
    if extension == "3mf":
        native_mm = tmp_path / "native-mm.3mf"
        trimesh.creation.box(extents=(2.0, 3.0, 4.0)).export(native_mm)
        assert np.ptp(SpacecraftMesh.from_file(native_mm).vertices, axis=0) == pytest.approx([0.002, 0.003, 0.004], rel=0, abs=1e-9)


@pytest.mark.parametrize("extension", ["step", "iges"])
def test_opencascade_generated_box_import(extension, tmp_path):
    """R1: 100x200x300 mm box extents, 0.22 m² area, 0.006 m³ signed volume.

    Absolute tolerances are 1e-7 m, 1e-10 m² and 1e-12 m³ respectively.
    """
    pytest.importorskip("OCP")
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.IGESControl import IGESControl_Writer
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    shape = BRepPrimAPI_MakeBox(100.0, 200.0, 300.0).Shape()
    source = tmp_path / f"box.{extension}"
    if extension == "step":
        writer = STEPControl_Writer()
        writer.Transfer(shape, STEPControl_AsIs)
        writer.Write(str(source))
    else:
        writer = IGESControl_Writer()
        writer.AddShape(shape)
        writer.Write(str(source))
    mesh = SpacecraftMesh.from_file(source)
    assert np.ptp(mesh.vertices, axis=0) == pytest.approx([0.1, 0.2, 0.3], rel=0, abs=1e-7)
    points = mesh.vertices[mesh.triangles]
    area = np.linalg.norm(np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1).sum() / 2
    volume = np.einsum("ij,ij->i", points[:, 0], np.cross(points[:, 1], points[:, 2])).sum() / 6
    assert area == pytest.approx(0.22, rel=0, abs=1e-10)
    assert volume == pytest.approx(0.006, rel=0, abs=1e-12)
