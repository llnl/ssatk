"""Resolve eclipse assets from the LLNL SSAPy-Data package or checkout.

V22.2 treats image and ephemeris resources as named scientific inputs instead of
hard-coded files inside the plotting package.  The preferred source is the
installed ``ssa-data-*`` distributions through :mod:`ssatk.data`. A checkout
may also be selected explicitly through ``SSAPY_DATA_DIR`` or a per-asset
override.

Resolution order is deterministic and recorded in every public product:

* explicit caller path or per-asset environment override;
* explicitly configured SSAPy-Data checkout data directories;
* SSAPy's installed-data lookup through ``ssapy.utils.find_file``;
* installed packages through ``ssatk.data`` (including zipped resources);
* packaged offline fallbacks, only when the policy permits them.

Git LFS pointer text is rejected for binary kernels.  Package resources that
must be extracted from a zipped wheel are materialized in a stable cache so
libraries requiring a filesystem path can safely retain the resolved path.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from hashlib import sha256
from importlib import import_module, metadata
from pathlib import Path
from typing import Iterable, Literal
import json
import os

AssetPolicy = Literal["data-first", "strict-data", "package-first", "packaged-only"]

PACKAGE_ASSET_DIR = Path(__file__).resolve().parents[1] / "plots"
CACHE_DIR = Path(
    os.environ.get(
        "SSAPY_ECLIPSE_ASSET_CACHE",
        str(Path.home() / ".cache" / "ssapy-eclipse" / "ssapy-data"),
    )
).expanduser()


@dataclass(frozen=True)
class AssetSpec:
    logical_name: str
    aliases: tuple[str, ...]
    packaged_aliases: tuple[str, ...] = ()
    environment_override: str | None = None
    kind: str = "data"
    required_for_rendering: bool = False
    required_for_strict_ssapy: bool = False


@dataclass(frozen=True)
class ResolvedAsset:
    logical_name: str
    path: str
    source: str
    data_root: str | None
    sha256: str
    bytes: int
    git_lfs_pointer: bool

    def to_dict(self, *, portable: bool = False) -> dict[str, object]:
        payload = asdict(self)
        if not portable:
            return payload
        filename = Path(self.path).name
        if self.source == "packaged-fallback":
            asset_ref = f"package://ssatk.plots/{filename}"
            root_ref = "package://ssatk.plots"
        elif self.source.startswith("ssapy_data") or self.source in {"ssatk.data", "SSAPy-Data", "ssapy.utils.find_file"}:
            relative = None
            if self.data_root:
                try:
                    relative = Path(self.path).resolve().relative_to(Path(self.data_root).resolve()).as_posix()
                except Exception:
                    relative = None
            asset_ref = f"ssapy-data://{relative or filename}"
            root_ref = "ssapy-data://"
        elif self.source.startswith("environment:"):
            asset_ref = f"environment://{self.logical_name}/{filename}"
            root_ref = self.source
        elif self.source == "explicit":
            asset_ref = f"explicit://{self.logical_name}/{filename}"
            root_ref = "explicit"
        else:
            asset_ref = f"asset://{self.logical_name}/{filename}"
            root_ref = self.source
        payload["path"] = asset_ref
        payload["data_root"] = root_ref
        payload["portable_reference"] = True
        return payload


ASSET_SPECS: dict[str, AssetSpec] = {
    "earth_albedo": AssetSpec(
        logical_name="earth_albedo",
        aliases=(
            "eclipse_earth_albedo.png", "eclipse_earth_albedo.jpg", "eclipse_earth_albedo.jpeg",
            "earth.png", "earth.jpg", "earth.jpeg",
            "earth_day_2048.jpg",
            "Earth_graphics/earth.png",
        ),
        packaged_aliases=("earth_ssapy_full.png", "earth_ssapy_2048.png"),
        environment_override="SSAPY_EARTH_TEXTURE",
        kind="image",
        required_for_rendering=True,
    ),
    "moon_albedo": AssetSpec(
        logical_name="moon_albedo",
        aliases=(
            "eclipse_moon_albedo.png", "eclipse_moon_albedo.jpg", "eclipse_moon_albedo.jpeg",
            "moon.png", "moon.jpg", "moon.jpeg",
        ),
        packaged_aliases=("moon_ssapy.png",),
        environment_override="SSAPY_MOON_TEXTURE",
        kind="image",
        required_for_rendering=True,
    ),
    "quadratic_visible_delta_lut": AssetSpec(
        logical_name="quadratic_visible_delta_lut",
        aliases=("eclipse_quadratic_visible_delta_lut.bin",),
        environment_override="SSAPY_ECLIPSE_PHOTOMETRY_LUT",
        kind="binary-lut",
        required_for_rendering=False,
    ),
    "quadratic_visible_delta_lut_metadata": AssetSpec(
        logical_name="quadratic_visible_delta_lut_metadata",
        aliases=("eclipse_quadratic_visible_delta_lut.json",),
        environment_override="SSAPY_ECLIPSE_PHOTOMETRY_LUT_METADATA",
        kind="json",
        required_for_rendering=False,
    ),
    "de430_kernel": AssetSpec(
        logical_name="de430_kernel",
        aliases=(
            "de430_1900_2150.bsp", "de430.bsp",
            "kernels/de430_1900_2150.bsp", "kernels/de430.bsp",
            "ephemerides/de430_1900_2150.bsp", "ephemerides/de430.bsp",
        ),
        environment_override="SSAPY_DE430_KERNEL",
        kind="spk",
        required_for_strict_ssapy=True,
    ),
    "moon_pa_kernel": AssetSpec(
        logical_name="moon_pa_kernel",
        aliases=(
            "moon_pa_de440_200625.bpc", "kernels/moon_pa_de440_200625.bpc",
            "orientation/moon_pa_de440_200625.bpc",
        ),
        environment_override="SSAPY_MOON_PA_KERNEL",
        kind="pck",
        required_for_strict_ssapy=True,
    ),
    "earth_clouds": AssetSpec(
        logical_name="earth_clouds",
        aliases=("eclipse/earth_clouds.png", "eclipse/earth_clouds.jpg", "eclipse/earth_clouds.jpeg"),
        environment_override="SSAPY_EARTH_CLOUDS",
        kind="image",
    ),
    "earth_normal": AssetSpec(
        logical_name="earth_normal",
        aliases=("eclipse/earth_normal.png", "eclipse/earth_normal.jpg", "eclipse/earth_normal.jpeg"),
        environment_override="SSAPY_EARTH_NORMAL",
        kind="image",
    ),
    "moon_normal": AssetSpec(
        logical_name="moon_normal",
        aliases=("eclipse/moon_normal.png", "eclipse/moon_normal.jpg", "eclipse/moon_normal.jpeg"),
        environment_override="SSAPY_MOON_NORMAL",
        kind="image",
    ),
    "solar_2024_totality_duration_grid": AssetSpec(
        logical_name="solar_2024_totality_duration_grid",
        aliases=("eclipse_solar_2024_totality_duration_grid.npz",),
        packaged_aliases=("solar_2024_totality_duration_grid.npz",),
        environment_override="SSAPY_SOLAR_TOTALITY_GRID",
        kind="npz",
    ),
    "solar_2024_partial_visibility_grid": AssetSpec(
        logical_name="solar_2024_partial_visibility_grid",
        aliases=("eclipse_solar_2024_partial_visibility_grid.npz",),
        packaged_aliases=("solar_2024_partial_visibility_grid.npz",),
        environment_override="SSAPY_SOLAR_PARTIAL_GRID",
        kind="npz",
    ),
    "lunar_2025_geographic_visibility_grid": AssetSpec(
        logical_name="lunar_2025_geographic_visibility_grid",
        aliases=("eclipse_lunar_2025_geographic_visibility_grid.npz",),
        packaged_aliases=("lunar_2025_geographic_visibility_grid.npz",),
        environment_override="SSAPY_LUNAR_VISIBILITY_GRID",
        kind="npz",
    ),
}


class AssetNotFoundError(FileNotFoundError):
    """Raised when an asset required by the selected policy is unavailable."""


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def file_sha256(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_git_lfs_pointer(path: str | Path) -> bool:
    p = Path(path)
    try:
        with p.open("rb") as handle:
            header = handle.read(160)
    except OSError:
        return False
    return header.lower().startswith(b"version https://git-lfs.github.com/spec/v1")


def _looks_like_image(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            head = handle.read(16)
    except OSError:
        return False
    return (
        head.startswith(b"\x89PNG\r\n\x1a\n")
        or head.startswith(b"\xff\xd8\xff")
        or head.startswith(b"RIFF") and head[8:12] == b"WEBP"
    )


def _normalize_root(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _data_dir_variants(value: str | Path) -> tuple[Path, ...]:
    """Normalize a repo root, package root, or data directory to data dirs."""
    root = _normalize_root(value)
    nested = [
        root / "src" / "ssapy_data" / "data",
        root / "ssapy_data" / "data",
        root / "data",
    ]
    root_looks_like_data = any(
        (root / name).exists()
        for name in ("earth.png", "moon.png", "de430.bsp", "moon_pa_de440_200625.bpc", "egm96.egm")
    ) or root.name == "data"
    candidates = ([root] + nested) if root_looks_like_data else (nested + [root])
    # If a caller points at ``.../src/ssapy_data`` or ``.../eclipse``, add
    # the obvious parent data directory as well.
    if root.name == "ssapy_data":
        candidates.insert(0, root / "data")
    if root.name in {"eclipse", "eclipses"}:
        candidates.insert(0, root.parent)
    result: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        key = str(resolved)
        if key not in seen and resolved.is_dir():
            seen.add(key)
            result.append(resolved)
    return tuple(result)


def _path_list(value: str | None) -> Iterable[Path]:
    if not value:
        return ()
    result: list[Path] = []
    for item in value.split(os.pathsep):
        if item.strip():
            result.extend(_data_dir_variants(item))
    return tuple(result)


@lru_cache(maxsize=1)
def discover_data_roots() -> tuple[Path, ...]:
    """Return actual SSAPy-Data *data directories* in priority order."""
    values: list[Path] = []
    for name in ("SSAPY_DATA_DIR", "SSAPY_DATA_ROOT", "SSAPY_DATA_REPO", "SSAPY_DATA_PATH"):
        values.extend(_path_list(os.environ.get(name)))

    seen: set[str] = set()
    result: list[Path] = []
    for value in values:
        key = str(value)
        if key not in seen and value.is_dir():
            seen.add(key)
            result.append(value)
    return tuple(result)


def clear_asset_caches() -> None:
    discover_data_roots.cache_clear()


def _materialize_bytes(data: bytes, filename: str) -> Path:
    digest = sha256(data).hexdigest()
    destination = CACHE_DIR / digest[:16] / Path(filename).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file() or destination.stat().st_size != len(data):
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(destination)
    return destination.resolve()


def _package_api_candidates(spec: AssetSpec) -> Iterable[tuple[Path, str, Path | None]]:
    """Resolve through SSATK's split-data helper instead of naming one wheel."""
    try:
        data_resource = import_module("ssatk.data").data_resource
    except Exception:
        return ()
    result: list[tuple[Path, str, Path | None]] = []
    for alias in spec.aliases:
        try:
            resource = data_resource(alias)
            if not resource.is_file():
                continue
            stable = (resource.resolve() if isinstance(resource, Path)
                      else _materialize_bytes(resource.read_bytes(), resource.name))
            result.append((stable, "ssatk.data", stable.parent))
        except (FileNotFoundError, ModuleNotFoundError):
            continue
    return tuple(result)


def _legacy_find_file_candidates(spec: AssetSpec) -> Iterable[tuple[Path, str, Path | None]]:
    try:
        find_file = import_module("ssapy.utils").find_file
    except Exception:
        return ()
    result = []
    for alias in spec.aliases:
        try:
            value = Path(find_file(alias)).expanduser().resolve()
            result.append((value, "ssapy.utils.find_file", value.parent))
        except Exception:
            continue
    return tuple(result)


def asset_candidates(
    logical_name: str,
    *,
    explicit: str | Path | None = None,
    policy: AssetPolicy = "data-first",
) -> tuple[tuple[Path, str, Path | None], ...]:
    if logical_name not in ASSET_SPECS:
        raise KeyError(f"Unknown logical asset: {logical_name}")
    spec = ASSET_SPECS[logical_name]
    external: list[tuple[Path, str, Path | None]] = []
    packaged: list[tuple[Path, str, Path | None]] = []

    if explicit is not None:
        p = Path(explicit).expanduser().resolve()
        external.append((p, "explicit", p.parent))
    if spec.environment_override and os.environ.get(spec.environment_override):
        p = Path(os.environ[spec.environment_override]).expanduser().resolve()
        external.append((p, f"environment:{spec.environment_override}", p.parent))

    for data_root in discover_data_roots():
        for alias in spec.aliases:
            external.append((data_root / alias, "SSAPy-Data", data_root))
    external.extend(_legacy_find_file_candidates(spec))
    external.extend(_package_api_candidates(spec))

    for alias in spec.packaged_aliases:
        packaged.append((PACKAGE_ASSET_DIR / alias, "packaged-fallback", PACKAGE_ASSET_DIR))

    ordered = {
        "data-first": external + packaged,
        "strict-data": external,
        "package-first": packaged + external,
        "packaged-only": packaged,
    }.get(policy)
    if ordered is None:
        raise ValueError(f"Unsupported asset policy: {policy}")

    seen: set[str] = set()
    result: list[tuple[Path, str, Path | None]] = []
    for path, source, root in ordered:
        try:
            resolved = path.expanduser().resolve()
        except Exception:
            resolved = path
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        result.append((resolved, source, root))
    return tuple(result)


def resolve_asset(
    logical_name: str,
    *,
    explicit: str | Path | None = None,
    policy: AssetPolicy = "data-first",
    required: bool = True,
    reject_lfs_pointer: bool = True,
) -> ResolvedAsset | None:
    """Resolve one logical asset and return immutable provenance."""
    spec = ASSET_SPECS[logical_name]
    override = explicit
    if override is None and spec.environment_override:
        override = os.environ.get(spec.environment_override)
    if override is not None:
        path = Path(override).expanduser().resolve()
        if not path.is_file() or is_git_lfs_pointer(path):
            raise AssetNotFoundError(f"Invalid explicit {logical_name} asset: {path}")
        if spec.kind == "image" and not _looks_like_image(path):
            raise ValueError(f"Invalid {logical_name} image: {path}")
    attempted: list[str] = []
    for path, source, root in asset_candidates(logical_name, explicit=explicit, policy=policy):
        attempted.append(str(path))
        if not path.is_file():
            continue
        pointer = is_git_lfs_pointer(path)
        if pointer and reject_lfs_pointer:
            continue
        if ASSET_SPECS[logical_name].kind == "image" and not _looks_like_image(path):
            continue
        stat = path.stat()
        return ResolvedAsset(
            logical_name=logical_name,
            path=str(path),
            source=source,
            data_root=str(root) if root is not None else None,
            sha256=file_sha256(path),
            bytes=int(stat.st_size),
            git_lfs_pointer=pointer,
        )
    if required:
        raise AssetNotFoundError(
            f"Could not resolve {logical_name!r} with policy={policy!r}. "
            "Install the required ssa-data-* distributions or set SSAPY_DATA_DIR to an SSAPy-Data "
            f"checkout. Attempted: {attempted}"
        )
    return None


def resolve_image(
    logical_name: str,
    *,
    explicit: str | Path | None = None,
    policy: AssetPolicy = "data-first",
    required: bool = True,
) -> ResolvedAsset | None:
    result = resolve_asset(logical_name, explicit=explicit, policy=policy, required=required)
    if result is not None and Path(result.path).suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError(f"Resolved {logical_name} is not a supported image: {result.path}")
    return result


def audit_assets(*, policy: AssetPolicy = "data-first", portable: bool = False) -> dict[str, object]:
    assets: dict[str, object] = {}
    for name, spec in ASSET_SPECS.items():
        resolved = resolve_asset(name, policy=policy, required=False)
        assets[name] = {
            "required_for_rendering": spec.required_for_rendering,
            "required_for_strict_ssapy": spec.required_for_strict_ssapy,
            "resolved": resolved.to_dict(portable=portable) if resolved else None,
        }
    rendering_ready = all(
        assets[name]["resolved"] is not None
        for name, spec in ASSET_SPECS.items() if spec.required_for_rendering
    )
    strict_data_ready = all(
        assets[name]["resolved"] is not None
        for name, spec in ASSET_SPECS.items() if spec.required_for_strict_ssapy
    )
    return {
        "$schema": "ssapy-toolkit.eclipse.asset-audit/2.0",
        "policy": policy,
        "llnl_ssapy_data_version": _distribution_version("llnl-ssapy-data"),
        "split_data_versions": {
            name: _distribution_version(name) for name in
            ("ssa-data-core", "ssa-data-lunar", "ssa-data-gravity", "ssa-data-lunar-gravity")
        },
        "data_roots": (
            [f"ssapy-data-root://{index+1}" for index, _ in enumerate(discover_data_roots())]
            if portable else [str(path) for path in discover_data_roots()]
        ),
        "portable_provenance": bool(portable),
        "rendering_ready": rendering_ready,
        "strict_ssapy_data_ready": strict_data_ready,
        "assets": assets,
    }


def write_asset_audit(output: str | Path, *, policy: AssetPolicy = "data-first", portable: bool = True) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit_assets(policy=policy, portable=portable), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def activate_ssapy_data_root(root: str | Path | None = None) -> Path:
    """Point an installed LLNL SSAPy runtime at one SSAPy-Data data dir."""
    if root is None:
        roots = discover_data_roots()
        if not roots:
            raise AssetNotFoundError("No SSAPy-Data data directory was discovered")
        selected = roots[0]
    else:
        variants = _data_dir_variants(root)
        if not variants:
            raise FileNotFoundError(root)
        selected = variants[0]
    os.environ["SSAPY_DATA_DIR"] = str(selected)
    clear_asset_caches()
    try:
        ssapy = import_module("ssapy")
        setattr(ssapy, "datadir", str(selected))
        utils = import_module("ssapy.utils")
        setattr(utils, "datadir", str(selected))
    except Exception:
        # The root can be selected before LLNL packages are installed.
        pass
    return selected


def canonical_ssapy_data_manifest() -> dict[str, object]:
    """Return the manifest consumed by the SSAPy-Data eclipse overlay."""
    logical_assets = {}
    for name, spec in ASSET_SPECS.items():
        logical_assets[name] = {
            "aliases": list(spec.aliases),
            "kind": spec.kind,
            "required_for_rendering": spec.required_for_rendering,
            "required_for_strict_ssapy": spec.required_for_strict_ssapy,
        }
    return {
        "$schema": "ssapy-data.eclipse-assets/2.0",
        "target_layout": "src/ssapy_data/data",
        "logical_assets": logical_assets,
        "notes": [
            "Earth and Moon albedo images may be PNG or JPEG.",
            "The renderer resolves logical names and never assumes a checkout location.",
            "SPK/PCK kernels must be real binary files; Git LFS pointer text is rejected.",
            "Run the SSAPy-Data manifest updater after installing or changing files.",
        ],
    }
