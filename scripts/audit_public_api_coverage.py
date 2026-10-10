#!/usr/bin/env python3
"""Audit SSATK function bodies and branch coverage from a coverage.py JSON report."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FunctionRecord:
    path: Path
    line: int
    qualname: str
    kind: str
    executable_lines: frozenset[int]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check that SSATK functions have at least one executable body line "
            "covered, with optional private/nested-function and branch audits."
        )
    )
    parser.add_argument(
        "--coverage-json",
        type=Path,
        default=Path("/tmp/ssatk_coverage.json"),
        help="coverage.py JSON report produced by `coverage json`.",
    )
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=Path("ssatk"),
        help="Package directory to audit.",
    )
    parser.add_argument(
        "--min-hit-pct",
        type=float,
        default=0.0,
        help=(
            "Fail if function body-hit percentage is below this value. The default, 0, "
            "only reports: coverage is a diagnostic, not a gate (docs/testing_policy.md)."
        ),
    )
    parser.add_argument(
        "--include-private",
        action="store_true",
        help="Include private functions/methods whose names start with `_`.",
    )
    parser.add_argument(
        "--include-nested",
        action="store_true",
        help="Include nested function definitions and closures.",
    )
    parser.add_argument(
        "--min-branch-pct",
        type=float,
        default=None,
        help="Fail if branch coverage is below this value. Requires branch coverage JSON.",
    )
    parser.add_argument(
        "--require-branch-data",
        action="store_true",
        help="Fail if the coverage JSON was not generated with branch coverage enabled.",
    )
    parser.add_argument(
        "--write-unhit",
        type=Path,
        default=None,
        help="Optional TSV path for unhit function bodies.",
    )
    parser.add_argument(
        "--write-missing-branches",
        type=Path,
        default=None,
        help="Optional TSV path for missing branch arcs.",
    )
    return parser.parse_args(argv)


def include_name(name: str, *, include_private: bool) -> bool:
    return include_private or not name.startswith("_")


def function_body_lines(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[int]:
    lines: set[int] = set()
    for child in node.body:
        if (
            isinstance(child, ast.Expr)
            and isinstance(child.value, ast.Constant)
            and isinstance(child.value.value, str)
        ):
            continue
        for desc in ast.walk(child):
            lineno = getattr(desc, "lineno", None)
            end_lineno = getattr(desc, "end_lineno", lineno)
            if lineno is None:
                continue
            lines.update(range(lineno, end_lineno + 1))
    return {line for line in lines if line >= node.lineno}


class FunctionCollector(ast.NodeVisitor):
    def __init__(
        self,
        path: Path,
        executed_lines: set[int],
        executed_or_missing: set[int],
        *,
        include_private: bool,
        include_nested: bool,
    ):
        self.path = path
        self.executed_lines = executed_lines
        self.executed_or_missing = executed_or_missing
        self.include_private = include_private
        self.include_nested = include_nested
        self.class_scope: list[str] = []
        self.function_scope: list[str] = []
        self.records: list[FunctionRecord] = []

    def visit_If(self, node: ast.If) -> None:
        if not self.class_scope and not self.function_scope:
            if self._is_main_guard(node):
                for item in node.orelse:
                    self.visit(item)
                return
            if node.lineno in self.executed_lines:
                for branch in (node.body, node.orelse):
                    if self._branch_was_defined(branch):
                        for item in branch:
                            self.visit(item)
                return
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if not include_name(node.name, include_private=self.include_private):
            return
        if self.function_scope and not self.include_nested:
            return
        self.class_scope.append(node.name)
        for item in node.body:
            self.visit(item)
        self.class_scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        is_nested = bool(self.function_scope)
        if not self.include_nested and is_nested:
            return
        if not include_name(node.name, include_private=self.include_private):
            return

        qualname = ".".join([*self.class_scope, *self.function_scope, node.name])
        kind = "method" if self.class_scope and not self.function_scope else "function"
        if is_nested:
            kind = "nested"

        executable = function_body_lines(node) & self.executed_or_missing
        if executable:
            self.records.append(
                FunctionRecord(self.path, node.lineno, qualname, kind, frozenset(executable))
            )

        if self.include_nested:
            self.function_scope.append(node.name)
            for item in node.body:
                self.visit(item)
            self.function_scope.pop()

    def _branch_was_defined(self, branch: list[ast.stmt]) -> bool:
        return any(
            getattr(stmt, "lineno", None) in self.executed_lines
            for stmt in branch
        )

    @staticmethod
    def _is_main_guard(node: ast.If) -> bool:
        test = node.test
        if not isinstance(test, ast.Compare):
            return False
        if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
            return False
        if len(test.comparators) != 1:
            return False
        left = test.left
        right = test.comparators[0]
        return (
            isinstance(left, ast.Name)
            and left.id == "__name__"
            and isinstance(right, ast.Constant)
            and right.value == "__main__"
        ) or (
            isinstance(right, ast.Name)
            and right.id == "__name__"
            and isinstance(left, ast.Constant)
            and left.value == "__main__"
        )


def iter_functions(
    path: Path,
    executed_lines: set[int],
    executed_or_missing: set[int],
    *,
    include_private: bool,
    include_nested: bool,
) -> list[FunctionRecord]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    collector = FunctionCollector(
        path,
        executed_lines,
        executed_or_missing,
        include_private=include_private,
        include_nested=include_nested,
    )
    collector.visit(tree)
    return collector.records


def coverage_key_candidates(path: Path, repo_root: Path) -> list[str]:
    candidates = [
        str(path),
        str(path.as_posix()),
        str((repo_root / path).resolve()),
        str((repo_root / path).resolve().as_posix()),
    ]
    return list(dict.fromkeys(candidates))


def coverage_lines(files: dict, path: Path, repo_root: Path) -> tuple[set[int], set[int]]:
    candidates = coverage_key_candidates(path, repo_root)
    for key in candidates:
        if key in files:
            entry = files[key]
            return set(entry.get("executed_lines", [])), set(entry.get("missing_lines", []))
    return set(), set()


def coverage_entry(files: dict, path: Path, repo_root: Path) -> dict:
    candidates = coverage_key_candidates(path, repo_root)
    for key in candidates:
        if key in files:
            return files[key]
    return {}


def package_branch_summary(
    files: dict,
    package_dir: Path,
    repo_root: Path,
) -> tuple[int, int, float | None]:
    covered = 0
    missing = 0
    for path in sorted(package_dir.rglob("*.py")):
        entry = coverage_entry(files, path, repo_root)
        summary = entry.get("summary", {})
        if "covered_branches" in summary or "missing_branches" in summary:
            covered += int(summary.get("covered_branches", 0) or 0)
            missing += int(summary.get("missing_branches", 0) or 0)
            continue
        covered += len(entry.get("executed_branches", []) or [])
        missing += len(entry.get("missing_branches", []) or [])
    total = covered + missing
    if total <= 0:
        return covered, total, None
    return covered, total, 100.0 * covered / total


def missing_branch_rows(files: dict, package_dir: Path, repo_root: Path) -> list[tuple[Path, int, int]]:
    rows: list[tuple[Path, int, int]] = []
    for path in sorted(package_dir.rglob("*.py")):
        entry = coverage_entry(files, path, repo_root)
        for branch in entry.get("missing_branches", []) or []:
            try:
                start, end = branch
            except Exception:
                continue
            rows.append((path, int(start), int(end)))
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path.cwd().resolve()
    coverage = json.loads(args.coverage_json.read_text(encoding="utf-8"))
    files = coverage.get("files", {})
    branch_enabled = bool(coverage.get("meta", {}).get("branch_coverage", False))

    if args.require_branch_data and not branch_enabled:
        print(
            "ERROR: coverage JSON does not include branch data. "
            "Run `python3 -m coverage run --branch -m pytest -q` first.",
            file=sys.stderr,
        )
        return 1

    records: list[FunctionRecord] = []
    hit_records: list[FunctionRecord] = []
    unhit_records: list[FunctionRecord] = []

    for path in sorted(args.package_dir.rglob("*.py")):
        executed, missing = coverage_lines(files, path, repo_root)
        executed_or_missing = executed | missing
        for record in iter_functions(
            path,
            executed,
            executed_or_missing,
            include_private=args.include_private,
            include_nested=args.include_nested,
        ):
            records.append(record)
            if record.executable_lines & executed:
                hit_records.append(record)
            else:
                unhit_records.append(record)

    hit_pct = (100.0 * len(hit_records) / len(records)) if records else 100.0
    scope = "all_functions" if args.include_private else "public_functions"
    if args.include_nested:
        scope += "_including_nested"
    print(f"scope={scope}")
    print(f"functions={len(records)}")
    print(f"body_hit={len(hit_records)}")
    print(f"body_unhit={len(unhit_records)}")
    print(f"body_hit_pct={hit_pct:.1f}")

    covered_branches, total_branches, branch_pct = package_branch_summary(
        files, args.package_dir, repo_root
    )
    if branch_enabled:
        print(f"branches={total_branches}")
        print(f"branches_hit={covered_branches}")
        print(f"branch_hit_pct={branch_pct:.1f}" if branch_pct is not None else "branch_hit_pct=n/a")
    else:
        print("branch_hit_pct=not-collected")

    if unhit_records:
        print("unhit_functions:")
        for record in unhit_records[:80]:
            print(f"  {record.path}:{record.line} {record.qualname} ({record.kind})")
        if len(unhit_records) > 80:
            print(f"  ... {len(unhit_records) - 80} more")

    if args.write_unhit:
        args.write_unhit.parent.mkdir(parents=True, exist_ok=True)
        args.write_unhit.write_text(
            "\n".join(
                f"{record.path}\t{record.line}\t{record.qualname}\t{record.kind}\t"
                f"body_lines={len(record.executable_lines)}"
                for record in unhit_records
            )
            + ("\n" if unhit_records else ""),
            encoding="utf-8",
        )

    missing_branches = missing_branch_rows(files, args.package_dir, repo_root) if branch_enabled else []
    if args.write_missing_branches:
        args.write_missing_branches.parent.mkdir(parents=True, exist_ok=True)
        args.write_missing_branches.write_text(
            "\n".join(f"{path}\t{start}\t{end}" for path, start, end in missing_branches)
            + ("\n" if missing_branches else ""),
            encoding="utf-8",
        )

    if hit_pct < args.min_hit_pct:
        print(
            f"ERROR: function body-hit percentage {hit_pct:.1f}% "
            f"is below required {args.min_hit_pct:.1f}%",
            file=sys.stderr,
        )
        return 1
    if args.min_branch_pct is not None:
        if not branch_enabled:
            print(
                "ERROR: branch threshold requested, but coverage JSON has no branch data.",
                file=sys.stderr,
            )
            return 1
        if branch_pct is not None and branch_pct < args.min_branch_pct:
            print(
                f"ERROR: branch coverage {branch_pct:.1f}% "
                f"is below required {args.min_branch_pct:.1f}%",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
