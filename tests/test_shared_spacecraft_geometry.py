"""Closed-form geometry references across physical and plotting adapters."""

import numpy as np
import pytest

from ssatk.geometry import SpacecraftMesh
from ssatk.plots.build_satellite_viewer import build
from ssatk.plots.spacecraft_view import SpacecraftMesh as PlotMesh
from ssatk.satellites import SpacecraftBody


def test_imported_box_facets_and_inertia_match_rectangular_prism(tmp_path):
    """R1: 2x3x4 m box, area 52 m² and I=mass/12*(side²+side²), 1e-12."""
    reference = SpacecraftBody.box(mass=12.0, size=[2, 3, 4])
    mesh = SpacecraftMesh.from_body(reference)
    path = tmp_path / "box.obj"
    mesh.export(path)
    body = SpacecraftBody.from_file(path, mass=12.0)
    np.testing.assert_allclose(body.inertia, np.diag([25, 20, 13]), rtol=0, atol=1e-12)
    assert sum(facet.area for facet in body.facets) == pytest.approx(52, abs=1e-12)
    np.testing.assert_allclose(
        np.ptp(SpacecraftMesh.from_body(body).vertices, axis=0), [2, 3, 4], rtol=0, atol=1e-12,
    )
    assert PlotMesh is SpacecraftMesh


def test_browser_payload_preserves_body_mass_center_and_metre_geometry(tmp_path):
    """R1: translated 2x3x4 m box; exported body offsets equal vertex-COM, 1e-15 m."""
    import json
    import re

    body = SpacecraftBody.box(mass=12.0, size=[2, 3, 4]).with_updates(
        center_of_mass=[0.25, -0.5, 0.125],
    )
    path = tmp_path / "viewer.html"
    build(
        out_path=str(path), verbose=False,
        state_vectors=[{"name": "box", "r": [[7000, 0, 0]], "v": [[0, 7, 0]], "t": [0]}],
        mesh=body, mesh_scale="physical",
    )
    payload = json.loads(re.search(r"const STATE_VECTOR_TRACKS = (.*);", path.read_text()).group(1))[0]["mesh"]
    actual = SpacecraftMesh.from_dict(payload)
    expected = np.vstack([facet.vertices_body for facet in body.facets]) - [0.25, -0.5, 0.125]
    np.testing.assert_allclose(actual.vertices, expected, rtol=0, atol=1e-15)
    np.testing.assert_allclose(np.ptp(actual.vertices, axis=0), [2, 3, 4], rtol=0, atol=1e-15)


def test_webgl_body_axes_and_scale_match_independent_scipy_rotation():
    """R2: SciPy 90-degree Z rotation and TEME->scene (x,z,-y), 2e-7 km.

    Node evaluates the actual vendored Three.js and viewer geometry functions.
    Float32 GPU position attributes set the absolute tolerance.
    """
    import json
    from pathlib import Path
    import shutil
    import subprocess

    from scipy.spatial.transform import Rotation

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the independent WebGL geometry check")
    folder = Path(__file__).resolve().parents[1] / "ssatk" / "plots"
    source = (folder / "satellite_viewer_scene.js").read_text()
    helpers = []
    for name in ("eciToScene", "applyQuaternionOrientation", "buildSpacecraftMesh"):
        start = source.index(f"function {name}(")
        end = source.index("\n}", start) + 2
        helpers.append(source[start:end])
    mesh = SpacecraftMesh([[1, 2, 3], [2, 2, 3], [1, 4, 3]], [[0, 1, 2]])
    program = "\n".join([
        "const THREE = require(" + json.dumps(str(folder / "three.min.js")) + ");",
        *helpers,
        "const mesh = buildSpacecraftMesh(" + json.dumps(mesh.to_dict()) + ", 100, true);",
        "applyQuaternionOrientation(mesh, new THREE.Vector3(0,0,0), new THREE.Quaternion(0,0,Math.SQRT1_2,Math.SQRT1_2));",
        "mesh.updateMatrixWorld(true);",
        "const points = [];",
        "const attribute = mesh.geometry.getAttribute('position');",
        "for (let index=0; index<attribute.count; index++) { points.push(new THREE.Vector3().fromBufferAttribute(attribute,index).applyMatrix4(mesh.matrixWorld).toArray()); }",
        "console.log(JSON.stringify(points));",
    ])
    actual = json.loads(subprocess.check_output([node, "-e", program], text=True))
    rotated = Rotation.from_euler("z", 90, degrees=True).apply(mesh.vertices) * 0.001
    expected = rotated[:, [0, 2, 1]] * [1, 1, -1]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-7)
