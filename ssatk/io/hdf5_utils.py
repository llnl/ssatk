import os
import h5py
import numpy as np
from ssatk._paths import ensure_file_parent


def _ensure_parent(h5, key: str) -> h5py.Group:
    """Ensure parent groups for a full path like 'a/b/c' exist; return the parent group."""
    parts = key.strip("/").split("/")
    if len(parts) == 1:
        return h5  # parent is root
    parent_path = "/".join(parts[:-1])
    return h5.require_group(parent_path)


def h5_key_exists(filename: str, key: str) -> bool:
    """
    True if `key` exists anywhere in the file (supports nested paths like 'a/b/c').
    """
    try:
        with h5py.File(filename, "r") as f:
            try:
                _ = f[key]  # will raise KeyError if not present
                return True
            except KeyError:
                return False
    except OSError:
        return False


def save_h5(filename: str, key: str, data) -> None:
    """
    Create a dataset at `key`. Creates parent groups if needed. Fails if dataset exists.
    """
    try:
        ensure_file_parent(filename)
        with h5py.File(filename, "a") as f:
            parent = _ensure_parent(f, key)
            name = key.strip("/").split("/")[-1]
            parent.create_dataset(name, data=data, maxshape=None)
            f.flush()
    except ValueError as err:
        # Typically "name already exists"
        print(f"Did not save, key: {key} exists in file: {filename}. {err}")
    except (BlockingIOError, OSError) as err:
        print(f"\n{err}\nPath: {key}\nFile: {filename}\n")


def overwrite_h5(filename: str, key: str, new_data) -> None:
    """
    Overwrite (or create) dataset at `key`.
    """
    ensure_file_parent(filename)
    with h5py.File(filename, "a") as f:
        parent = _ensure_parent(f, key)
        name = key.strip("/").split("/")[-1]
        if name in parent:
            del parent[name]
        parent.create_dataset(name, data=new_data, maxshape=None)


def append_h5(filename: str, key: str, append_data) -> None:
    """
    Append rows along axis 0. If dataset doesn't exist, create it.
    Note: `append_data` must be broadcastable to the dataset shape except on axis 0.
    """
    arr = np.asarray(append_data)
    ensure_file_parent(filename)
    with h5py.File(filename, "a") as f:
        parent = _ensure_parent(f, key)
        name = key.strip("/").split("/")[-1]

        if name in parent:
            dset = parent[name]
            if dset.shape == ():
                # Scalar in file; replace with 1D array of scalars then append
                data0 = dset[()]
                del parent[name]
                dset = parent.create_dataset(name, data=np.asarray([data0]), maxshape=(None,), chunks=True)

            # Ensure first dimension is the append axis
            if dset.ndim == 0:
                raise ValueError(f"Cannot append to scalar dataset at {key}")

            # Prepare append with correct shape
            arr2 = np.asarray(arr)
            if arr2.ndim < dset.ndim:
                # Try to expand dims to match (prepend batch dimension if needed)
                arr2 = np.expand_dims(arr2, axis=0)

            # Check compatibility (all dims except axis 0)
            if dset.ndim != arr2.ndim or any(
                (s is not None) and (s != a)
                for s, a in zip(dset.shape[1:], arr2.shape[1:])
            ):
                raise ValueError(f"Incompatible shapes: existing {dset.shape} vs append {arr2.shape}")

            new_len = dset.shape[0] + arr2.shape[0]
            dset.resize((new_len, *dset.shape[1:]))
            dset[-arr2.shape[0]:] = arr2
        else:
            # Create a resizable dataset to allow future appends
            maxshape = (None,) + arr.shape[1:] if arr.ndim >= 1 else (None,)
            chunks = True
            parent.create_dataset(name, data=arr, maxshape=maxshape, chunks=chunks)


def read_h5(filename: str, key: str):
    """
    Load data from an HDF5 file. Returns np.ndarray (or scalar) or None if missing.
    """
    try:
        with h5py.File(filename, "r") as f:
            try:
                data = f[key]
            except KeyError:
                return None
            return np.array(data) if isinstance(data, h5py.Dataset) else None
    except FileNotFoundError:
        print(f'File not found. {filename}')
        raise
    except (BlockingIOError, OSError) as err:
        print(f"\n{err}\nPath: {key}\nFile: {filename}\n")
        raise
    except (ValueError, TypeError):
        return None


def read_h5_to_dict(file_path: str) -> dict:
    def recursively_load(h5obj):
        out = {}
        for k in h5obj.keys():
            item = h5obj[k]
            if isinstance(item, h5py.Group):
                out[k] = recursively_load(item)
            else:
                out[k] = item[()]
        return out

    with h5py.File(file_path, 'r') as h5file:
        return recursively_load(h5file)


def read_h5_all(file_path: str) -> dict:
    """
    Flatten all datasets into a dict keyed by their full HDF5 paths.
    """
    data_dict: dict = {}

    with h5py.File(file_path, 'r') as file:
        def traverse(group, path=''):
            for key, item in group.items():
                new_path = f"{path}/{key}" if path else key
                if isinstance(item, h5py.Group):
                    traverse(item, path=new_path)
                else:
                    data_dict[new_path] = item[()]
        traverse(file)
    return data_dict


def h5_keys(file_path: str) -> list:
    """
    List full dataset paths in an HDF5 file.
    """
    out: list = []
    with h5py.File(file_path, 'r') as file:
        def traverse(group, path=''):
            for key, item in group.items():
                new_path = f"{path}/{key}" if path else key
                if isinstance(item, h5py.Group):
                    traverse(item, path=new_path)
                else:
                    out.append(new_path)
        traverse(file)
    return out


def h5_root_keys(file_path: str) -> list:
    """
    List top-level members.
    """
    with h5py.File(file_path, 'r') as file:
        return list(file.keys())


def combine_h5(filename: str, files: list, verbose: bool = False, overwrite: bool = False) -> None:
    """
    Merge datasets from multiple HDF5 files into `filename` without clobbering existing keys.
    """
    if overwrite and os.path.exists(filename):
        os.remove(filename)

    for idx, src in enumerate(files):
        try:
            if not os.path.exists(src):
                if verbose:
                    print(f"[{idx}] Skipping missing file: {src}")
                continue
            for key in h5_keys(src):
                if not h5_key_exists(filename, key):
                    data = read_h5(src, key)
                    if data is None:
                        if verbose:
                            print(f"[{idx}] Skipping empty/missing key {key} in {src}")
                        continue
                    save_h5(filename, key, data)
                elif verbose:
                    print(f"[{idx}] Exists, skip: {key}")
        except Exception as e:
            print(f"Error processing file {src}: {e}")


def verify_h5_file(filename: str, mode: str = "structure", verbose: bool = False) -> bool:
    """
    Verify an HDF5 file with selectable depth of checking.

    Modes
    -----
    open
        Only verify the file exists and can be opened.
        Fastest, but weakest check.

    structure
        Verify the file opens and every object in the hierarchy can be accessed.
        Fast and usually sufficient for structural sanity checking.

    full
        Verify the file opens and every dataset can be fully read.
        Slowest, but strongest check.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file.
    mode : str
        One of {"open", "structure", "full"}.
    verbose : bool
        If True, print any errors encountered.

    Returns
    -------
    bool
        True if the file passes the selected verification mode, otherwise False.
    """
    if mode not in {"open", "structure", "full"}:
        raise ValueError(f"Invalid mode '{mode}'. Expected one of: 'open', 'structure', 'full'.")

    if not os.path.exists(filename):
        if verbose:
            print(f"File does not exist: {filename}")
        return False

    try:
        with h5py.File(filename, "r") as f:
            if mode == "open":
                return True

            bad = False

            if mode == "structure":
                def _check(name):
                    nonlocal bad
                    if bad:
                        return
                    try:
                        _ = f[name]
                    except Exception as e:
                        bad = True
                        if verbose:
                            print(f"{name}: {type(e).__name__}: {e}")

                f.visit(_check)
                return not bad

            if mode == "full":
                def _check(name, obj):
                    nonlocal bad
                    if bad:
                        return
                    try:
                        if isinstance(obj, h5py.Dataset):
                            _ = obj[()]
                    except Exception as e:
                        bad = True
                        if verbose:
                            print(f"{name}: {type(e).__name__}: {e}")

                f.visititems(_check)
                return not bad

    except Exception as e:
        if verbose:
            print(f"Failed to open/read file: {type(e).__name__}: {e}")
        return False
