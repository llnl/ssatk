"""Published geometry checks (R3) and analytic visibility checks (R1).

Catalog/build/alias checks are contracts (C), not validation of mission CAD.
Primary references and which dimensions are schematic are in preset_info().
"""
import numpy as np
import pytest

from ssatk.plots.spacecraft_view import SpacecraftMesh, _camera_basis, _project, _rasterize_mesh


def part_faces(mesh, part):
    return [mesh.vertices[f] for f, p in zip(mesh.faces, mesh.face_parts) if p == part]


def part_vertices(mesh, part):
    return np.vstack(part_faces(mesh, part))


@pytest.mark.parametrize("scale", [1, 2])
def test_cubesat_envelope_rails_and_recessed_cells(scale):
    """R3: NASA SmallSat SOA Table 6-1: 100 x 100 x 340.5 mm, 1e-14 m.

    https://www.nasa.gov/smallsat-institute/sst-soa/structures-materials-and-mechanisms/
    Recessed cells and continuous corner rails are design contracts, not flight measurements.
    """
    mesh = SpacecraftMesh.from_preset("cubesat", scale=scale)
    expected = np.array([.100, .100, .3405]) * scale
    np.testing.assert_allclose(np.ptp(mesh.vertices, axis=0), expected, rtol=0, atol=1e-14)
    rails = part_vertices(mesh, "rail")
    assert len(part_faces(mesh, "rail")) == 4 * 6
    np.testing.assert_allclose(np.ptp(rails, axis=0), expected, rtol=0, atol=1e-14)
    for part in ("body_cells_x", "body_cells_y"):
        cells = np.abs(part_vertices(mesh, part))
        assert np.all(cells < expected / 2)


def test_hubble_published_body_and_array_dimensions():
    """R3: NASA Hubble Design and Electrical Power pages; metre tolerances 1e-12."""
    m = SpacecraftMesh.from_preset("hubble")
    hull = np.vstack([part_vertices(m, p) for p in ("aft_shroud", "optical_tube")])
    np.testing.assert_allclose(np.ptp(hull, axis=0), [4.2, 4.2, 13.2], rtol=0, atol=1e-12)
    frames = part_faces(m, "solar_array_frame")
    assert len(frames) == 2
    for panel in frames:
        np.testing.assert_allclose(np.ptp(panel, axis=0), [7.1, 0, 2.6], rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.ptp(part_vertices(m, "aperture"), axis=0)[:2], [2.4, 2.4], rtol=0, atol=1e-12)


def test_webb_segment_count_size_and_five_layer_shield():
    """R3: NASA Webb Telescope Overview: 18, 1.32 m, 21.2 x 14.2 m; 1e-12 m.

    Tests selected published features, not total collecting area or surface curvature.
    """
    m = SpacecraftMesh.from_preset("webb")
    segments = part_faces(m, "primary_segment")
    assert len(segments) == 18
    for polygon in segments:
        assert len(polygon) == 6
        # Across opposite flat sides is sqrt(3) times a regular hexagon's edge.
        edge = np.linalg.norm(np.roll(polygon, -1, axis=0) - polygon, axis=1)
        np.testing.assert_allclose(edge * np.sqrt(3), 1.32, rtol=0, atol=1e-12)
    for layer in range(1, 6):
        assert len(part_faces(m, f"sunshield_{layer}")) == 1
    np.testing.assert_allclose(np.ptp(part_vertices(m, "sunshield_1"), axis=0), [21.2, 14.2, 0], rtol=0, atol=1e-12)


def test_roman_six_panels_and_primary_diameter():
    """R3: NASA Building Roman: six panels; telescope page: 2.4 m primary."""
    m = SpacecraftMesh.from_preset("roman")
    assert sum(p.startswith("sunshield_panel_") and p.endswith("_frame") for p in m.face_parts) == 6
    np.testing.assert_allclose(np.ptp(part_vertices(m, "aperture"), axis=0)[:2], [2.4, 2.4], rtol=0, atol=1e-12)


def test_chandra_deployed_envelope():
    """R3: Chandra specifications: 13.8 m long, 19.5 m wide, tolerance 1e-12 m."""
    m = SpacecraftMesh.from_preset("chandra")
    hull = np.vstack([part_vertices(m, p) for p in ("bus", "optical_bench", "mirror_assembly")])
    assert np.ptp(hull[:, 2]) == pytest.approx(13.8, abs=1e-12)
    assert np.ptp(part_vertices(m, "solar_array_frame")[:, 0]) == pytest.approx(19.5, abs=1e-12)


def test_tess_four_camera_pupils_and_antenna():
    """R3: HEASARC TESS Telescope Information: 4 x 10.5 cm pupils, 0.7 m dish."""
    m = SpacecraftMesh.from_preset("tess")
    pupils = part_faces(m, "camera_pupil")
    assert len(pupils) == 4
    for points in pupils:
        radii = np.linalg.norm(points - points.mean(axis=0), axis=1)
        np.testing.assert_allclose(radii, .0525, atol=1e-14, rtol=0)
    points = part_vertices(m, "antenna")
    np.testing.assert_allclose(np.linalg.norm(points - points.mean(axis=0), axis=1), .35, atol=1e-14, rtol=0)


@pytest.mark.parametrize("name,count,distance", [("nustar", 2, 10.14), ("ixpe", 3, 4.0)])
def test_focal_reference_planes(name, count, distance):
    """R3: NuSTAR observatory guide and IXPE user guide focal distances; 1e-12 m.

    These are explicit geometry reference planes, not a ray-traced optical model.
    """
    m = SpacecraftMesh.from_preset(name)
    optics = part_faces(m, "optics_reference")
    detectors = part_faces(m, "detector_reference")
    assert len(optics) == len(detectors) == count
    for optical, detector in zip(optics, detectors):
        difference = optical.mean(axis=0) - detector.mean(axis=0)
        np.testing.assert_allclose(difference, [0, 0, distance], rtol=0, atol=1e-12)


def test_spherex_three_photon_shields_and_outer_dimensions():
    """R3: JPL SPHEREx press kit: three shields, outer 3.2 m x 1.7 m."""
    m = SpacecraftMesh.from_preset("spherex")
    assert {p for p in m.face_parts if p.startswith("photon_shield_")} == {
        "photon_shield_1", "photon_shield_2", "photon_shield_3"}
    np.testing.assert_allclose(np.ptp(part_vertices(m, "photon_shield_1"), axis=0), [3.2, 3.2, 1.7], rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.ptp(part_vertices(m, "aperture"), axis=0)[:2], [.20, .20], rtol=0, atol=1e-12)


def test_nicer_concentrators_and_compton_corner_modules():
    """R3: NICER mission guide: 56 concentrators; BATSE page: 8 corner modules."""
    nicer = SpacecraftMesh.from_preset("nicer")
    assert len(part_faces(nicer, "pupil")) == 56
    compton = SpacecraftMesh.from_preset("compton")
    # Each convex cylinder has two cap faces. Group by octant independently of tessellation.
    centers = np.array([p.mean(axis=0) for p in part_faces(compton, "BATSE")])
    assert len(np.unique(np.sign(centers), axis=0)) == 8





def test_depth_buffer_against_analytic_ray_box_intersection():
    """R1: ray/slab intersection independently gives the front surface at every pixel.

    Regression: a 0.1 x 0.1 x 0.3 m box at this camera angle was visibly broken
    by mean-triangle sorting. Perspective depth tolerance 2e-14 m, no edge drawing.
    """
    lo, hi = np.array([-.05,-.05,-.15]), np.array([.05,.05,.15])
    vertices = np.array([[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
                         [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]]) * hi
    # Order: -X,+X,-Y,+Y,-Z,+Z. Colors allow independent nearest-face verification.
    faces = [[0,3,7,4], [1,5,6,2], [0,4,5,1], [3,2,6,7], [0,1,2,3], [4,7,6,5]]
    colors = np.array([[.5,0,0], [1,0,0], [0,.5,0], [0,1,0], [0,0,.5], [0,0,1]])
    m = SpacecraftMesh(vertices, faces, colors)
    eye = np.array([.4,.2,-.45])
    basis = _camera_basis(-eye, [0,0,1])
    uv, depths = _project(vertices-eye, basis, .4, .3)
    width, height = 240, 180
    x, y = np.meshgrid((np.arange(width)+.5)*2/width-1, 1-(np.arange(height)+.5)*2/height)
    rays = np.stack([x*.4, y*.3, np.ones_like(x)], axis=-1) @ basis
    near_axis = np.minimum((lo-eye)/rays, (hi-eye)/rays)
    far_axis = np.maximum((lo-eye)/rays, (hi-eye)/rays)
    near, far = near_axis.max(axis=-1), far_axis.min(axis=-1)
    hit = (near > 0) & (near < far)
    axis = near_axis.argmax(axis=-1)
    sign = np.take_along_axis(rays, axis[..., None], axis=-1)[..., 0] < 0
    face_index = 2*axis + sign
    for order in (np.arange(len(m.triangles)), np.arange(len(m.triangles))[::-1]):
        image, z = _rasterize_mesh(uv, depths, m.triangles[order], m.triangle_colors[order], width, height)
        np.testing.assert_array_equal(image[..., 3] == 1, hit)
        np.testing.assert_allclose(z[hit], near[hit], rtol=0, atol=2e-14)
        np.testing.assert_array_equal(image[hit, :3], colors[face_index[hit]])


def test_crossing_triangles_use_perspective_depth_in_both_orders():
    """R1: intersecting planes Z=2+X and Z=2-X give depths 2/(1-x), 2/(1+x)."""
    footprint = np.array([[-.8,-.8], [.8,-.8], [0,.8]])
    uv = np.vstack([footprint, footprint])
    depth = np.r_[2/(1-footprint[:,0]), 2/(1+footprint[:,0])]
    triangles = np.array([[0,1,2], [3,4,5]])
    colors = np.array([[1,0,0], [0,0,1]])
    width = height = 101
    for order in ([0,1], [1,0]):
        image, z = _rasterize_mesh(uv, depth, triangles[order], colors[order], width, height)
        for col, expected in ((35, [1,0,0]), (65, [0,0,1])):
            x = (col+.5)*2/width-1
            np.testing.assert_array_equal(image[50,col,:3], expected)
            assert z[50,col] == pytest.approx(2/(1+abs(x)), abs=1e-14)
