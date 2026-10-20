from pathlib import Path

from ssatk._paths import SSATK_OUTPUT_ENV, output_root, safe_relative_parts

__all__ = ["ssatk_path", "figpath", "document_path"]

# You can keep this around if you like, but it's no longer used for extension logic.
_KNOWN_EXTS = {
    ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".gif",
    ".svg", ".svgz", ".pdf", ".ps", ".eps",
    ".mp4", ".mov", ".avi", ".mpeg", ".mpg", ".webm",
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


def ssatk_path(filename="figure"):
    """
    Build a path under the shared SSATK figure output directory.

    Rules:
      - The path is rooted under ~/ssatk_output/figures by default.
      - Set SSATK_OUTPUT_DIR to choose an explicit alternate output root.
      - Subfolders in `filename` are preserved and created as needed.
      - The basename is used exactly as given (no automatic extension added).
      - Absolute paths and '..' are normalized to stay under the output root.

    Examples
    --------
    ssatk_path("plot")                          -> ~/ssatk_output/figures/plot
    ssatk_path("burn_to_dv")                  -> ~/ssatk_output/figures/burn_to_dv
    ssatk_path("burn_to_dv.png")              -> ~/ssatk_output/figures/burn_to_dv.png
    ssatk_path("/abs/path/ignored/name.svg")    -> ~/ssatk_output/figures/abs/path/ignored/name.svg
    ssatk_path("weird/name.foo")                -> ~/ssatk_output/figures/weird/name.foo
    """
    if not isinstance(filename, (str, Path)):
        raise TypeError("ssatk_path(filename): filename must be str or pathlib.Path")

    # Normalize to a safe relative path (no drive, no leading slash, no traversal)
    rel_parts = safe_relative_parts(filename)
    if not rel_parts:
        rel_parts = ["figure"]

    # Use the basename exactly as provided (no auto extension)
    basename = rel_parts[-1]
    final_name = basename

    base = _figure_root()
    try:
        # Construct full subdir path and ensure it exists
        subdir = Path(*rel_parts[:-1]) if len(rel_parts) > 1 else Path()
        target_dir = base / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        return str(target_dir / final_name)
    except (OSError, PermissionError) as exc:
        raise RuntimeError(
            f"Could not create or access {base}. Set {SSATK_OUTPUT_ENV} "
            "to an explicit writable output directory."
        ) from exc


def _figure_root():
    return output_root() / "figures"


figpath = ssatk_path


def document_path(filename="document"):
    """Build a path under the shared SSATK document output directory."""
    if not isinstance(filename, (str, Path)):
        raise TypeError("document_path(filename): filename must be str or pathlib.Path")
    rel_parts = safe_relative_parts(filename)
    if not rel_parts:
        rel_parts = ["document"]
    base = output_root() / "documents"
    target_dir = base / Path(*rel_parts[:-1])
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as exc:
        raise RuntimeError(
            f"Could not create or access {base}. Set {SSATK_OUTPUT_ENV} "
            "to an explicit writable output directory."
        ) from exc
    return str(target_dir / rel_parts[-1])
