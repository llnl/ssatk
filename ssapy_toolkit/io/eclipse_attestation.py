"""Unambiguous release attestation for SSAPy-Toolkit eclipse artifacts.

V18's final acceptance document accidentally recorded two different hashes for
identically named wheel and overlay artifacts.  V19 replaces that structure
with one stage-aware, schema-validated record.  An artifact path may occur only
once; pre-seal, sealed, and external artifacts have distinct identifiers and
explicit stages.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
import hashlib
import json
import re

SCHEMA_ID = "ssapy-toolkit.eclipse.final-attestation/2.0"
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_STAGES = {"source-tree", "preseal", "sealed", "external"}


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    stage: str
    role: str
    path: str
    bytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "stage": self.stage,
            "role": self.role,
            "path": self.path,
            "bytes": int(self.bytes),
            "sha256": self.sha256,
        }


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(
    path: str | Path,
    *,
    artifact_id: str,
    stage: str,
    role: str,
    display_path: str | None = None,
) -> ArtifactRecord:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    stage = str(stage).strip().lower()
    if stage not in _ALLOWED_STAGES:
        raise ValueError(f"stage must be one of {sorted(_ALLOWED_STAGES)}")
    return ArtifactRecord(
        artifact_id=str(artifact_id),
        stage=stage,
        role=str(role),
        path=str(display_path if display_path is not None else source.name),
        bytes=source.stat().st_size,
        sha256=sha256_file(source),
    )


def tree_manifest(
    root: str | Path,
    *,
    exclude_names: Iterable[str] = (),
) -> dict[str, dict[str, Any]]:
    """Return deterministic path/size/hash records for a release tree."""
    root_path = Path(root).expanduser().resolve()
    excluded = set(exclude_names)
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in root_path.rglob("*") if item.is_file()):
        rel = path.relative_to(root_path).as_posix()
        if rel in excluded or path.name in excluded:
            continue
        result[rel] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return result


def manifest_digest(manifest: Mapping[str, Mapping[str, Any]]) -> str:
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_attestation(
    *,
    release: str,
    package_version: str,
    release_root: str | Path,
    artifacts: Iterable[ArtifactRecord],
    authoritative_distribution: str,
    validation: Mapping[str, Any] | None = None,
    strict_ssapy: Mapping[str, Any] | None = None,
    exclude_names: Iterable[str] = ("FINAL_ATTESTATION.json",),
) -> dict[str, Any]:
    manifest = tree_manifest(release_root, exclude_names=exclude_names)
    records = [record.to_dict() for record in artifacts]
    payload = {
        "$schema": SCHEMA_ID,
        "release": str(release),
        "package_version": str(package_version),
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "authoritative_distribution": str(authoritative_distribution),
        "manifest_scope": {
            "release_root": Path(release_root).name,
            "release_files": len(manifest),
            "manifest_sha256": manifest_digest(manifest),
            "exclusions": sorted(set(exclude_names)),
        },
        "artifacts": records,
        "validation": dict(validation or {}),
        "strict_ssapy": dict(strict_ssapy or {}),
    }
    validate_attestation(payload)
    return payload


def validate_attestation(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate schema invariants and return a compact result record."""
    errors: list[str] = []
    if payload.get("$schema") != SCHEMA_ID:
        errors.append(f"$schema must be {SCHEMA_ID!r}")
    for key in ("release", "package_version", "created_utc", "authoritative_distribution"):
        if not payload.get(key):
            errors.append(f"missing non-empty {key}")

    records = payload.get("artifacts")
    if not isinstance(records, list) or not records:
        errors.append("artifacts must be a non-empty list")
        records = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            errors.append(f"artifacts[{index}] is not an object")
            continue
        artifact_id = str(record.get("artifact_id", ""))
        path = str(record.get("path", ""))
        stage = str(record.get("stage", ""))
        if not artifact_id:
            errors.append(f"artifacts[{index}] missing artifact_id")
        elif artifact_id in seen_ids:
            errors.append(f"duplicate artifact_id {artifact_id!r}")
        seen_ids.add(artifact_id)
        if not path:
            errors.append(f"artifacts[{index}] missing path")
        elif path in seen_paths:
            errors.append(
                f"artifact path {path!r} appears more than once; use distinct stage-qualified paths"
            )
        seen_paths.add(path)
        if stage not in _ALLOWED_STAGES:
            errors.append(f"artifact {artifact_id!r} has invalid stage {stage!r}")
        size = record.get("bytes")
        if not isinstance(size, int) or size < 0:
            errors.append(f"artifact {artifact_id!r} has invalid byte count")
        digest = str(record.get("sha256", ""))
        if not _HASH_RE.fullmatch(digest):
            errors.append(f"artifact {artifact_id!r} has invalid SHA-256")

    authoritative = str(payload.get("authoritative_distribution", ""))
    if authoritative and authoritative not in seen_ids:
        errors.append("authoritative_distribution must name exactly one artifact_id")

    scope = payload.get("manifest_scope")
    if not isinstance(scope, Mapping):
        errors.append("manifest_scope must be an object")
    else:
        count = scope.get("release_files")
        if not isinstance(count, int) or count < 0:
            errors.append("manifest_scope.release_files must be a non-negative integer")
        digest = str(scope.get("manifest_sha256", ""))
        if not _HASH_RE.fullmatch(digest):
            errors.append("manifest_scope.manifest_sha256 must be a SHA-256")

    if errors:
        raise ValueError("invalid final attestation: " + "; ".join(errors))
    return {
        "schema": SCHEMA_ID,
        "valid": True,
        "artifact_count": len(records),
        "authoritative_distribution": authoritative,
    }


def verify_attestation_files(
    payload: Mapping[str, Any],
    *,
    artifact_root: str | Path,
) -> dict[str, Any]:
    """Verify attested artifacts present under ``artifact_root``."""
    summary = validate_attestation(payload)
    root = Path(artifact_root).expanduser().resolve()
    checks: dict[str, dict[str, Any]] = {}
    all_passed = True
    for record in payload["artifacts"]:
        path = root / str(record["path"])
        exists = path.is_file()
        size_ok = exists and path.stat().st_size == int(record["bytes"])
        hash_ok = size_ok and sha256_file(path) == str(record["sha256"])
        passed = bool(exists and size_ok and hash_ok)
        all_passed &= passed
        checks[str(record["artifact_id"])] = {
            "path": str(path),
            "exists": exists,
            "size_ok": size_ok,
            "sha256_ok": hash_ok,
            "passed": passed,
        }
    return {**summary, "files_passed": all_passed, "checks": checks}


def write_attestation(payload: Mapping[str, Any], output: str | Path) -> str:
    validate_attestation(payload)
    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(target)


def read_attestation(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_attestation(payload)
    return payload
