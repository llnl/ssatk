"""Unit-normalized spacecraft mesh geometry and file conversion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import warnings

import numpy as np


__all__ = ["SpacecraftMesh", "coerce_mesh", "convert_mesh"]

_SCHEMA = "ssatk.spacecraft_mesh.v1"
_UNIT_SCALE = {"m": 1.0, "cm": 0.01, "mm": 0.001, "in": 0.0254, "ft": 0.3048}
_MESH_EXTENSIONS = {".stl", ".ply", ".off", ".glb", ".gltf", ".3mf"}
_CAD_EXTENSIONS = {".step", ".stp", ".iges", ".igs"}
_NATIVE_CAD_APPLICATIONS = {
    ".sldprt": "SolidWorks", ".sldasm": "SolidWorks",
    ".ipt": "Autodesk Inventor", ".iam": "Autodesk Inventor",
    ".catpart": "CATIA", ".catproduct": "CATIA",
    ".prt": "the originating CAD application (for example Creo or Siemens NX)",
    ".asm": "the originating CAD application (for example Creo)",
    ".x_t": "a CAD application or converter that supports Parasolid",
    ".x_b": "a CAD application or converter that supports Parasolid",
    ".sat": "a CAD application or converter that supports ACIS",
    ".sab": "a CAD application or converter that supports ACIS",
    ".f3d": "Autodesk Fusion", ".f3z": "Autodesk Fusion",
    ".fcstd": "FreeCAD",
}


def _track(values, count, columns, name):
    values = np.asarray(values, dtype=float)
    if values.shape == (columns,):
        values = np.broadcast_to(values, (count, columns))
    if values.shape != (count, columns) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite with shape ({columns},) or ({count}, {columns})")
    return values.copy()


def _scale_value(scale, units, default):
    if scale is not None and units is not None:
        raise ValueError("scale and units are mutually exclusive")
    if scale is not None:
        scale = float(scale)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be positive")
        return scale
    if units is not None:
        try:
            return _UNIT_SCALE[units.lower()]
        except (AttributeError, KeyError):
            raise ValueError("units must be one of m, cm, mm, in, or ft") from None
    return default


def _extension(path, *, exporting=False):
    suffix = Path(path).suffix.lower()
    if suffix == ".obj" or suffix in _MESH_EXTENSIONS or suffix in _CAD_EXTENSIONS:
        return suffix
    supported = "Supported mesh formats are OBJ, STL, PLY, OFF, GLB, GLTF, and 3MF."
    if exporting:
        instructions = (
            "Choose a supported mesh output, for example mesh.export('satellite.glb') "
            "or mesh.export('satellite.obj'). SSATK cannot write native CAD or STEP/IGES files."
        )
    elif suffix in _NATIVE_CAD_APPLICATIONS:
        instructions = (
            f"Open the model in {_NATIVE_CAD_APPLICATIONS[suffix]} and export a 3D STEP (.step/.stp) "
            "file (preferred) or IGES (.iges/.igs). For an assembly, include all required components "
            "and preserve their placements. Install the CAD backend with "
            "python -m pip install 'ssatk[cad]', then load the exported file with "
            "SpacecraftMesh.from_file('satellite.step')."
        )
    elif suffix in {".fbx", ".blend", ".dae", ".3ds", ".usd", ".usda", ".usdc", ".usdz"}:
        instructions = (
            "Open the model in its authoring application or a converter that supports this format "
            "(for example Blender, if supported), apply the required object transforms, and export "
            "GLB/glTF or triangulated OBJ. Install mesh backends with "
            "python -m pip install 'ssatk[mesh]', then load the exported file with "
            "SpacecraftMesh.from_file('satellite.glb'). Textures/PBR are not preserved by SSATK."
        )
    elif suffix in {".dwg", ".dxf", ".slddrw", ".idw", ".catdrawing"}:
        instructions = (
            "A drawing is not necessarily a 3D body. Open the underlying 3D part or assembly "
            "in its CAD application and export STEP, or export an actual 3D mesh as OBJ/STL. "
            "SSATK does not reconstruct 3D satellites from 2D drawings."
        )
    else:
        instructions = (
            "Use the originating application or a converter that supports this format to export "
            "3D STEP/IGES or a supported mesh. Files without an extension must first be identified "
            "and exported with the correct format and extension. STEP/IGES requires "
            "python -m pip install 'ssatk[cad]'."
        )
    raise ValueError(
        f"Unsupported geometry extension {suffix or '<none>'!r}. {instructions} {supported} "
        "Renaming the file extension does not convert its contents. Check units and body-frame "
        "origin/axes after conversion; for unitless mesh files use units='mm' or scale=... as appropriate."
    )


def _obj(path, scale):
    vertices = []
    faces = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.strip().split()
            if not fields or fields[0].startswith("#"):
                continue
            if fields[0] == "v":
                if len(fields) < 4:
                    raise ValueError(f"invalid OBJ vertex on line {line_number}")
                vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
            elif fields[0] == "f":
                if len(fields) < 4:
                    raise ValueError(f"OBJ face on line {line_number} has fewer than three vertices")
                face = []
                for token in fields[1:]:
                    index = int(token.split("/")[0])
                    if index == 0:
                        raise ValueError(f"OBJ index zero on line {line_number}")
                    index = index - 1 if index > 0 else len(vertices) + index
                    face.append(index)
                faces.append(face)
    return SpacecraftMesh(np.asarray(vertices) * scale, faces)


def _embedded_scale(units, default):
    if units is None:
        return default
    normalized = str(units).strip().lower()
    aliases = {"meter": "m", "meters": "m", "metre": "m", "metres": "m", "millimeter": "mm", "millimeters": "mm", "millimetre": "mm", "millimetres": "mm", "centimeter": "cm", "centimeters": "cm", "inch": "in", "inches": "in", "foot": "ft", "feet": "ft"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in _UNIT_SCALE:
        raise ValueError(f"unsupported embedded mesh units {units!r}; pass units explicitly")
    return _UNIT_SCALE[normalized]


def _trimesh(path, scale, default):
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError("trimesh is required for STL, PLY, OFF, GLB, GLTF, and 3MF files; install with ssatk[mesh]") from exc
    loaded = trimesh.load(str(path), force="scene", process=False)
    scene = loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)
    scale = _embedded_scale(getattr(scene, "units", None), default) if scale is None else scale
    vertices = []
    faces = []
    colors = []
    parts = []
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph.get(node_name)
        geometry = scene.geometry[geometry_name].copy()
        transformed = trimesh.transform_points(np.asarray(geometry.vertices), transform)
        offset = len(vertices)
        vertices.extend(transformed)
        indices = np.asarray(geometry.faces, dtype=int)
        if np.linalg.det(transform[:3, :3]) < 0:
            indices = indices[:, ::-1]
        faces.extend(face + offset for face in indices)
        face_colors = getattr(geometry.visual, "face_colors", None)
        if face_colors is None:
            color = (0.64, 0.72, 0.79)
            colors.extend([color] * len(geometry.faces))
        else:
            colors.extend(np.asarray(face_colors)[:, :3] / 255.0)
        parts.extend([str(node_name)] * len(geometry.faces))
    if not vertices:
        raise ValueError("mesh file contains no geometry")
    return SpacecraftMesh(np.asarray(vertices) * scale, faces, colors, parts)


def _cad(path, scale, linear_deflection, angular_deflection):
    try:
        from OCP.BRep import BRep_Tool
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.IGESControl import IGESControl_Reader
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.STEPControl import STEPControl_Reader
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
        from OCP.TopLoc import TopLoc_Location
        from OCP.TopoDS import TopoDS
    except ImportError as exc:
        raise ImportError("STEP/STP/IGES/IGS import requires cadquery-ocp (OpenCascade); install with ssatk[cad]") from exc
    reader = IGESControl_Reader() if path.suffix.lower() in {".iges", ".igs"} else STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise ValueError(f"OpenCascade could not read {path}")
    if not reader.TransferRoots():
        raise ValueError(f"OpenCascade could not transfer {path}")
    shape = reader.OneShape()
    BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
    vertices = []
    faces = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation is None:
            explorer.Next()
            continue
        offset = len(vertices)
        for index in range(1, triangulation.NbNodes() + 1):
            point = triangulation.Node(index).Transformed(location.Transformation())
            vertices.append([point.X(), point.Y(), point.Z()])
        for index in range(1, triangulation.NbTriangles() + 1):
            triangle = triangulation.Triangle(index)
            first, second, third = triangle.Get()
            if face.Orientation() == TopAbs_REVERSED:
                faces.append([offset + first - 1, offset + third - 1, offset + second - 1])
            else:
                faces.append([offset + first - 1, offset + second - 1, offset + third - 1])
        explorer.Next()
    if not vertices or not faces:
        raise ValueError(f"OpenCascade produced no tessellated geometry for {path}")
    return SpacecraftMesh(np.asarray(vertices) * scale, faces)


@dataclass
class SpacecraftMesh:
    """Body-frame vertices in metres and zero-based polygonal faces."""

    vertices: np.ndarray
    faces: object
    colors: object = (0.64, 0.72, 0.79)
    face_parts: object = None

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=float).copy()
        if (self.vertices.ndim != 2 or self.vertices.shape[1] != 3 or len(self.vertices) < 3
                or not np.all(np.isfinite(self.vertices))):
            raise ValueError("vertices must be a finite (V, 3) array in metres")
        faces = []
        for face in self.faces:
            raw = np.asarray(face)
            if raw.ndim != 1 or len(raw) < 3 or raw.dtype.kind not in "iu" or np.any(raw < 0) or np.any(raw >= len(self.vertices)):
                raise ValueError("faces must contain at least three valid integer vertex indices")
            faces.append(raw.astype(int))
        if not faces:
            raise ValueError("mesh must have faces")
        self.faces = tuple(faces)
        self.face_parts = ("surface",) * len(faces) if self.face_parts is None else tuple(self.face_parts)
        if len(self.face_parts) != len(faces) or not all(isinstance(part, str) for part in self.face_parts):
            raise ValueError("face_parts must contain one string per face")
        self.colors = _track(self.colors, len(faces), 3, "colors")
        if np.any((self.colors < 0) | (self.colors > 1)):
            raise ValueError("RGB colors must lie in [0, 1]")
        triangles, triangle_colors, boundaries, triangle_faces = [], [], [], []
        for face_index, (face, color) in enumerate(zip(self.faces, self.colors)):
            for index in range(1, len(face) - 1):
                triangle = [face[0], face[index], face[index + 1]]
                first, second, third = self.vertices[triangle]
                if np.linalg.norm(np.cross(second - first, third - first)) == 0:
                    raise ValueError("mesh contains a degenerate triangle")
                triangles.append(triangle)
                triangle_colors.append(color)
                boundaries.append((True, index == len(face) - 2, index == 1))
                triangle_faces.append(face_index)
        self.triangles = np.asarray(triangles, dtype=int)
        self.triangle_colors = np.asarray(triangle_colors)
        self.triangle_boundaries = np.asarray(boundaries, dtype=bool)
        self.triangle_faces = np.asarray(triangle_faces, dtype=int)

    @staticmethod
    def presets(group=None):
        """Return the built-in visual mesh names; no files are required."""
        from .plots.spacecraft_presets import preset_names
        return preset_names(group)

    @staticmethod
    def preset_info(name):
        """Return source links, fidelity, visual axes, and approximation notes."""
        from .plots.spacecraft_presets import preset_info
        return preset_info(name)

    @classmethod
    def from_preset(cls, name="starlink", *, scale=1.0):
        """Build an illustrative mesh in metres, optionally scaled uniformly.

        These rigid visual templates are not engineering, optical, or force
        models. ``preset_info(name)`` documents sources, approximation limits,
        and visual body axes for each mission. No assets are downloaded.
        """
        from .satellites import SpacecraftBody
        from .plots.spacecraft_presets import build_preset
        name = cls.preset_info(name)["name"]
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be a positive dimensionless factor")
        if name not in ("starlink", "boxwing"):
            model = build_preset(name)
            return cls(np.asarray(model.vertices) * scale, model.faces, model.colors, model.parts)
        size = (2.8, 1.4, 0.18) if name == "starlink" else (1.2, 1.0, 0.8)
        bus = cls.from_body(SpacecraftBody.box(mass=1.0, size=size))
        vertices, faces, colors = list(bus.vertices), list(bus.faces), list(bus.colors)
        if name == "starlink":
            colors = [(0.76, 0.80, 0.84)] * len(faces)
        axis = 1 if name == "starlink" else 0
        length, width = (4.0, 2.8) if name == "starlink" else (2.0, 1.4)
        for sign in (-1, 1):
            for row in range(8):
                for col in range(4):
                    lo, hi = sorted(sign * (size[axis] / 2 + np.array([row, row + 1]) * length / 8))
                    left, right = -width / 2 + np.array([col, col + 1]) * width / 4
                    points = np.array([[lo, left, 0], [hi, left, 0], [hi, right, 0], [lo, right, 0]])
                    if axis == 1:
                        points = points[::-1, [1, 0, 2]]
                    faces.append(np.arange(len(vertices), len(vertices) + 4))
                    vertices.extend(points)
                    colors.append((0.12, 0.27, 0.48))
        return cls(np.asarray(vertices) * scale, faces, colors)

    @classmethod
    def from_body(cls, body):
        """Read ``SpacecraftBody.facets`` geometry without synthesis.

        The body's current center of mass is subtracted so a propagated
        center-of-mass position can be used directly. Facets without
        ``vertices_body`` are rejected because an area and normal do not
        uniquely specify a physical surface.
        """
        vertices, faces, colors = [], [], []
        origin = np.asarray(getattr(body, "current_center_of_mass", (0, 0, 0)), dtype=float)
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ValueError("body center of mass must be a finite three-vector")
        for facet in body.facets:
            if facet.vertices_body is None:
                raise ValueError(f"facet {facet.name!r} has no vertices_body")
            points = np.asarray(facet.vertices_body, dtype=float) - origin
            faces.append(np.arange(len(vertices), len(vertices) + len(points)))
            vertices.extend(points)
            panel = any(word in facet.name.lower() for word in ("solar", "array", "panel", "wing"))
            colors.append((0.16, 0.37, 0.66) if panel else (0.74, 0.65, 0.40))
        return cls(vertices, faces, colors)

    @classmethod
    def from_obj(cls, path, *, scale=1.0):
        """Read Wavefront OBJ geometry with scale in metres per OBJ unit.

        Face indices may include texture or normal suffixes and may be
        negative. OBJ material and texture files are not interpreted.
        """
        return _obj(path, _scale_value(scale, None, 1.0))

    @classmethod
    def from_file(cls, path, *, scale=None, units=None, linear_deflection=0.1, angular_deflection=0.5):
        """Import a mesh in metres, using OpenCascade millimetres for deflection.

        CAD files default to millimetres and therefore use a ``0.001`` scale;
        ``linear_deflection`` is in OpenCascade's internal millimetres and
        ``angular_deflection`` is in radians. Standard mesh formats use embedded
        units when available, with GLTF in metres and 3MF defaulting to
        millimetres. Unitless OBJ/STL/PLY/OFF default to metres; override with
        ``units`` or ``scale`` when the authoring tool uses different units.
        """
        path = Path(path)
        suffix = _extension(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if not np.isfinite(linear_deflection) or linear_deflection <= 0:
            raise ValueError("linear_deflection must be positive")
        if not np.isfinite(angular_deflection) or angular_deflection <= 0:
            raise ValueError("angular_deflection must be positive")
        default = 0.001 if suffix in _CAD_EXTENSIONS or suffix == ".3mf" else 1.0
        factor = _scale_value(scale, units, default) if scale is not None or units is not None else None
        if suffix == ".obj":
            return cls.from_obj(path, scale=1.0 if factor is None else factor)
        if suffix in _CAD_EXTENSIONS:
            return _cad(path, default if factor is None else factor, linear_deflection, angular_deflection)
        return _trimesh(path, factor, default)

    def to_dict(self):
        return {"schema": _SCHEMA, "units": "m", "frame": "body", "vertices": self.vertices.tolist(),
                "faces": [face.tolist() for face in self.faces], "colors": self.colors.tolist(),
                "face_parts": list(self.face_parts)}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise TypeError("mesh schema must be a dictionary")
        required = {"schema", "units", "frame", "vertices", "colors", "face_parts"}
        if "faces" not in value and "triangles" not in value:
            raise ValueError("mesh schema requires faces or triangles")
        allowed = required | {"faces", "triangles"}
        if set(value) - allowed or not required <= set(value) or ("faces" in value and "triangles" in value):
            raise ValueError("invalid spacecraft mesh schema keys")
        if value["schema"] != _SCHEMA or value["units"] != "m" or value["frame"] != "body":
            raise ValueError("unsupported spacecraft mesh schema, units, or frame")
        return cls(value["vertices"], value.get("faces", value.get("triangles")), value["colors"], value["face_parts"])

    def export(self, path):
        path = Path(path)
        suffix = _extension(path, exporting=True)
        if suffix == ".obj":
            with open(path, "w", encoding="utf-8") as handle:
                for vertex in self.vertices:
                    handle.write("v " + " ".join(f"{value:.17g}" for value in vertex) + "\n")
                for face in self.faces:
                    handle.write("f " + " ".join(str(int(index) + 1) for index in face) + "\n")
            warnings.warn("OBJ export writes geometry only; face colors are not represented", UserWarning, stacklevel=2)
            return path
        if suffix in _CAD_EXTENSIONS:
            raise ValueError(
                "CAD export is not supported. Use mesh.export('satellite.glb'), "
                "mesh.export('satellite.stl'), or mesh.export('satellite.obj'). "
                "STEP/IGES import tessellates surfaces; it does not retain an editable CAD solid. "
                "To obtain CAD output, use the original CAD model and its authoring application."
            )
        try:
            import trimesh
        except ImportError as exc:
            raise ImportError("trimesh is required to export non-OBJ mesh files; install with ssatk[mesh]") from exc
        mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.triangles, process=False)
        mesh.visual.face_colors = np.column_stack((self.triangle_colors * 255, np.full(len(self.triangles), 255))).astype(np.uint8)
        if suffix == ".3mf":
            mesh.vertices *= 1000.0
        mesh.export(str(path))
        return path


def coerce_mesh(value):
    if isinstance(value, SpacecraftMesh):
        return value
    if isinstance(value, dict):
        return SpacecraftMesh.from_dict(value)
    if isinstance(value, str):
        path = Path(value)
        if path.is_file() or path.suffix:
            return SpacecraftMesh.from_file(path)
        return SpacecraftMesh.from_preset(value)
    if isinstance(value, Path):
        return SpacecraftMesh.from_file(value)
    if hasattr(value, "facets") and hasattr(value, "current_center_of_mass"):
        return SpacecraftMesh.from_body(value)
    raise TypeError("value must be a SpacecraftMesh, SpacecraftBody, preset name, path, or mesh dictionary")


def convert_mesh(source, destination, **import_options):
    if import_options:
        if not isinstance(source, (str, Path)) or not Path(source).is_file():
            raise ValueError("import options apply only when source is a geometry file")
        mesh = SpacecraftMesh.from_file(source, **import_options)
    else:
        mesh = coerce_mesh(source)
    return mesh.export(destination)
