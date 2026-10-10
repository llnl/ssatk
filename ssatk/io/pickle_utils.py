import pickle
from pathlib import Path

from ssatk._paths import ensure_file_parent


def save_pickle(data, path):
    """Writes a dictionary to a pickle file."""
    output_path = ensure_file_parent(path)
    with output_path.open('wb') as f:
        pickle.dump(data, f)


def read_pickle(path):
    """Reads a dictionary from a pickle file."""
    with Path(path).open('rb') as f:
        return pickle.load(f)
