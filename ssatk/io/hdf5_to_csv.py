#!/usr/bin/env python3
"""
hdf5_to_csv.py

Library + script:

- Import and call: hdf5_to_csv_per_key("input.h5")
- Or run directly: python hdf5_to_csv.py input.h5
"""

from __future__ import annotations

import csv
import argparse
import re
import warnings
from pathlib import Path
from typing import Any, List, Tuple

import h5py


def _stringify(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode("utf-8")
        except UnicodeDecodeError:
            return repr(x)
    return str(x)


def iter_datasets(h5: h5py.File) -> List[Tuple[str, h5py.Dataset]]:
    found: List[Tuple[str, h5py.Dataset]] = []

    def visitor(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            key = ("/" + name).replace("//", "/")
            found.append((key, obj))

    h5.visititems(visitor)
    found.sort(key=lambda t: t[0])
    return found


def dataset_to_python(ds: h5py.Dataset) -> Any:
    return ds[()]


def place_cell(grid: List[List[str]], r: int, c: int, val: str) -> None:
    while len(grid) <= r:
        grid.append([])
    row = grid[r]
    if len(row) <= c:
        row.extend([""] * (c + 1 - len(row)))
    row[c] = val


def normalize_grid(grid: List[List[str]]) -> List[List[str]]:
    width = max((len(r) for r in grid), default=0)
    for r in grid:
        if len(r) < width:
            r.extend([""] * (width - len(r)))
    return grid


def write_column(grid: List[List[str]], top: int, col: int, key: str, values: List[Any]) -> int:
    place_cell(grid, top, col, key)
    for i, v in enumerate(values):
        place_cell(grid, top + 1 + i, col, _stringify(v))
    return 1 + len(values)


def write_block(grid: List[List[str]], top: int, left: int, key: str, arr2d: Any) -> Tuple[int, int]:
    m = int(arr2d.shape[0])
    n = int(arr2d.shape[1])
    place_cell(grid, top, left, key)
    for i in range(m):
        for j in range(n):
            place_cell(grid, top + 1 + i, left + j, _stringify(arr2d[i, j]))
    return (1 + m, max(1, n))


def key_to_filename(key: str, max_len: int = 180) -> str:
    s = key.strip()
    if s.startswith("/"):
        s = s[1:]
    if not s:
        s = "root"

    s = re.sub(r"[\/\\:\*\?\"<>\|\s]+", "_", s)
    s = re.sub(r"_{2,}", "_", s).strip("_")

    if len(s) > max_len:
        s = s[:max_len].rstrip("_")
    return s or "dataset"


def write_grid_csv(path: Path, grid: List[List[str]], encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding=encoding) as f:
        w = csv.writer(f)
        w.writerows(grid)


def write_dataset_csv(out_path: Path, key: str, data: Any, encoding: str = "utf-8") -> Tuple[bool, str | None]:
    """
    Returns (written, warning). written=False if skipped due to ndim>2.
    """
    shape = getattr(data, "shape", ())
    ndim = len(shape) if shape is not None else 0

    grid: List[List[str]] = []

    if ndim == 0:
        write_column(grid, 0, 0, key, [data])
    elif ndim == 1:
        write_column(grid, 0, 0, key, list(data))
    elif ndim == 2:
        write_block(grid, 0, 0, key, data)
    else:
        return False, f"Ignoring {key}: ndim={ndim} shape={shape}"

    write_grid_csv(out_path, normalize_grid(grid), encoding=encoding)
    return True, None


def hdf5_to_csv_per_key(h5_filename: str | Path, *, encoding: str = "utf-8") -> Path:
    """
    Callable API (for importing).

    Always:
    - creates directory <input_stem>/ next to the input file
    - writes one CSV per dataset key path inside that directory

    Returns:
      Path to the output directory.
    """
    h5_path = Path(h5_filename)

    out_dir = h5_path.parent / h5_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as h5:
        for key, ds in iter_datasets(h5):
            data = dataset_to_python(ds)
            out_csv = out_dir / (key_to_filename(key) + ".csv")

            written, warning = write_dataset_csv(out_csv, key, data, encoding=encoding)
            if warning:
                warnings.warn(warning, UserWarning, stacklevel=2)

    return out_dir


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Write one CSV file per dataset in an HDF5 file.")
    parser.add_argument("h5_path", type=Path, help="Input HDF5 file.")
    args = parser.parse_args(argv)

    h5_path = args.h5_path

    if not h5_path.exists():
        raise SystemExit(f"HDF5 file not found: {h5_path}")

    out_dir = hdf5_to_csv_per_key(h5_path)
    print(f"Wrote: {out_dir}")


if __name__ == "__main__":
    main()
