"""Local user data output paths for SSATK workflows."""

from pathlib import Path

from ssatk._paths import SSATK_OUTPUT_ENV, output_root, safe_relative_parts

__all__ = ["datapath"]

# Common data extensions (case-insensitive). We check only the final suffix.
_KNOWN_DATA_EXTS = {
    ".csv", ".tsv", ".txt", ".log",
    ".json", ".jsonl", ".ndjson",
    ".yaml", ".yml",
    ".parquet", ".feather",
    ".h5", ".hdf5", ".hdf",
    ".npz", ".npy",
    ".pkl", ".pickle",
    ".xls", ".xlsx",
    ".zip", ".gz", ".bz2", ".xz", ".zst", ".tar"
}


def datapath(filename="data", dirs=None):
    """
    Construct a safe local data path under an SSATK output directory.

    Defaults to ``~/ssatk_output``. Set ``SSATK_OUTPUT_DIR`` or pass ``dirs``
    to choose an explicit alternate root.

    Returns a string path. Creates parent directories if needed.
    """
    if not isinstance(filename, (str, Path)):
        raise TypeError("datapath(filename): filename must be str or pathlib.Path")

    # Normalize to safe relative parts
    relative_parts = safe_relative_parts(filename)
    if not relative_parts:
        relative_parts = ["data"]  # default base name if only dirs/empties were given

    # Determine final name and extension policy
    base_name = relative_parts[-1]

    # Subdirectory tree under the selected data directory.
    subdir = Path(*relative_parts[:-1]) if len(relative_parts) > 1 else Path()

    if dirs is None:
        base_dirs = [output_root()]
    else:
        base_dirs = [Path(base).expanduser() for base in dirs]
    for base_dir in base_dirs:
        try:
            target_dir = base_dir / subdir
            target_dir.mkdir(parents=True, exist_ok=True)
            return str(target_dir / base_name)
        except (OSError, PermissionError):
            continue

    raise RuntimeError(
        f"Could not create or access {base_dirs[0]}. Set {SSATK_OUTPUT_ENV} "
        "or pass dirs=[...] to an explicit writable data directory."
    )
