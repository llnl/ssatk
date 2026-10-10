#!/usr/bin/env python3
"""Check repository layout and artifact policy for SSATK."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import PurePosixPath


ALLOWED_TOP_LEVEL = {
    ".flake8",
    ".github",
    ".gitignore",
    ".gitlab-ci.yml",
    ".readthedocs.yaml",
    "CHANGELOG.md",
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "codemeta.json",
    "LICENSE",
    "MANIFEST.in",
    "NOTICE",
    "README.md",
    "demos",
    "docs",
    "pyproject.toml",
    "requirements-dev.txt",
    "scripts",
    "ssatk",
    "packaging",   # packaging/ssapy-toolkit-compat: transitional ssapy-toolkit -> ssatk distribution
    "tests",
}

FORBIDDEN_EXTENSIONS = {
    ".7z",
    ".avi",
    ".bin",
    ".bpc",
    ".bz2",
    ".cof",
    ".csv",
    ".dat",
    ".db",
    ".egm",
    ".feather",
    ".fit",
    ".fits",
    ".gif",
    ".gz",
    ".h5",
    ".hdf5",
    ".heic",
    ".ipynb",
    ".jpeg",
    ".jpg",
    ".mov",
    ".mp4",
    ".npy",
    ".npz",
    ".parquet",
    ".pdf",
    ".pickle",
    ".pkl",
    ".png",
    ".sqlite",
    ".svg",
    ".tar",
    ".tgz",
    ".webm",
    ".xz",
    ".zip",
}

FORBIDDEN_FILENAMES = {".DS_Store", "Thumbs.db"}
MAX_FILE_SIZE_BYTES = 1_000_000


@dataclass(frozen=True)
class ChangedPath:
    status: str
    path: str
    source: str

    @property
    def is_deleted(self) -> bool:
        return self.status == "D"

    @property
    def is_added_like(self) -> bool:
        return self.status in {"A", "C", "R", "?"}


def run_git(args: list[str], *, text: bool = True) -> str | bytes:
    return subprocess.check_output(["git", *args], text=text)


def normalize_path(path: str) -> str:
    return path.replace(os.sep, "/")


def top_level_name(path: str) -> str:
    return PurePosixPath(path).parts[0]


def suffixes(path: str) -> set[str]:
    pure_path = PurePosixPath(path)
    suffix_set = {suffix.lower() for suffix in pure_path.suffixes}
    name = pure_path.name.lower()
    for multi_suffix in (".tar.gz", ".tar.bz2", ".tar.xz"):
        if name.endswith(multi_suffix):
            suffix_set.add(multi_suffix)
    return suffix_set


def parse_name_status(output: str, source: str) -> list[ChangedPath]:
    fields = output.split("\0")
    if fields and fields[-1] == "":
        fields.pop()

    changed: list[ChangedPath] = []
    index = 0
    while index < len(fields):
        status_token = fields[index]
        index += 1
        status = status_token[0]

        if status in {"R", "C"}:
            index += 1
            path = fields[index]
            index += 1
        else:
            path = fields[index]
            index += 1

        changed.append(ChangedPath(status, normalize_path(path), source))

    return changed


def diff_paths(base_ref: str, head_ref: str) -> list[ChangedPath]:
    output = run_git(["diff", "--name-status", "-z", base_ref, head_ref])
    return parse_name_status(output, f"{base_ref}..{head_ref}")


def local_status_paths() -> list[ChangedPath]:
    output = run_git(["status", "--porcelain=v1", "-z"])
    fields = output.split("\0")
    if fields and fields[-1] == "":
        fields.pop()

    changed: list[ChangedPath] = []
    index = 0
    while index < len(fields):
        entry = fields[index]
        index += 1
        status_token = entry[:2]
        path = normalize_path(entry[3:])

        if status_token == "??":
            status = "?"
        elif "D" in status_token:
            status = "D"
        elif "R" in status_token or "C" in status_token:
            status = "R" if "R" in status_token else "C"
            if index < len(fields):
                path = normalize_path(fields[index])
                index += 1
        elif "A" in status_token:
            status = "A"
        else:
            status = "M"

        changed.append(ChangedPath(status, path, "working tree"))

    return changed


def default_base_ref() -> str | None:
    for ref in ("origin/main", "main"):
        try:
            run_git(["rev-parse", "--verify", ref])
            return run_git(["merge-base", ref, "HEAD"]).strip()
        except subprocess.CalledProcessError:
            continue
    return None


def collect_changed_paths(base_ref: str | None, head_ref: str) -> list[ChangedPath]:
    if base_ref:
        return diff_paths(base_ref, head_ref)

    changed: list[ChangedPath] = []
    inferred_base = default_base_ref()
    if inferred_base:
        changed.extend(diff_paths(inferred_base, "HEAD"))
    changed.extend(local_status_paths())
    return deduplicate_changed_paths(changed)


def deduplicate_changed_paths(changed: list[ChangedPath]) -> list[ChangedPath]:
    deduped: dict[str, ChangedPath] = {}
    for item in changed:
        deduped[item.path] = item
    return sorted(deduped.values(), key=lambda item: item.path)


def source_changed(path: str) -> bool:
    return (path.startswith(("ssatk/", "packaging/ssapy-toolkit-compat/ssapy_toolkit/")) and path.endswith(".py")) or path == "pyproject.toml"


def test_changed(path: str) -> bool:
    return path.startswith("tests/")


def blob_bytes(path: str, head_ref: str) -> bytes | None:
    try:
        return run_git(["show", f"{head_ref}:{path}"], text=False)
    except subprocess.CalledProcessError:
        return None


def file_bytes(path: str) -> bytes | None:
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as file_handle:
        return file_handle.read()


def content_for_path(item: ChangedPath, head_ref: str) -> bytes | None:
    if item.is_deleted:
        return None
    if item.source == "working tree":
        return file_bytes(item.path)
    return blob_bytes(item.path, head_ref)


def file_size(item: ChangedPath, head_ref: str) -> int | None:
    if item.is_deleted:
        return None
    if item.source == "working tree":
        if os.path.isfile(item.path):
            return os.path.getsize(item.path)
        return None
    try:
        return int(run_git(["cat-file", "-s", f"{head_ref}:{item.path}"]).strip())
    except subprocess.CalledProcessError:
        return None


def is_binary(content: bytes | None) -> bool:
    if content is None:
        return False
    return b"\0" in content[:8192]


def validate(changed: list[ChangedPath], head_ref: str) -> list[str]:
    errors: list[str] = []
    active_paths = [item for item in changed if not item.is_deleted]

    for item in active_paths:
        if item.is_added_like and top_level_name(item.path) not in ALLOWED_TOP_LEVEL:
            errors.append(
                f"{item.path}: new top-level path '{top_level_name(item.path)}' is not allowed. "
                "Use the existing repository layout or update the policy with justification."
            )

        if PurePosixPath(item.path).name in FORBIDDEN_FILENAMES:
            errors.append(f"{item.path}: generated OS metadata files are not allowed.")

        forbidden_matches = sorted(suffixes(item.path) & FORBIDDEN_EXTENSIONS)
        if forbidden_matches:
            errors.append(
                f"{item.path}: file type {', '.join(forbidden_matches)} is not allowed in this source repo. "
                "Keep generated media/data out of git or move persistent data to the split data repositories."
            )

        size = file_size(item, head_ref)
        if size is not None and size > MAX_FILE_SIZE_BYTES:
            errors.append(
                f"{item.path}: file is {size:,} bytes, above the {MAX_FILE_SIZE_BYTES:,}-byte policy limit."
            )

        if item.is_added_like and is_binary(content_for_path(item, head_ref)):
            errors.append(
                f"{item.path}: binary files are not allowed in this source repo. "
                "Store reusable data in the split data repositories or generate artifacts during CI."
            )

    # Test coverage is validated by the CI test job.  The repository policy
    # checks layout and artifacts only; it does not require every change to
    # add or modify a test file.  Existing tests remain the release gate.

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", help="Base git ref/SHA for policy diff.")
    parser.add_argument("--head-ref", default="HEAD", help="Head git ref/SHA for policy diff.")
    args = parser.parse_args()

    changed = collect_changed_paths(args.base_ref, args.head_ref)
    if not changed:
        print("Repository policy check: no changed paths found.")
        return 0

    errors = validate(changed, args.head_ref)
    if errors:
        print("Repository policy check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"Repository policy check passed for {len(changed)} changed path(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
