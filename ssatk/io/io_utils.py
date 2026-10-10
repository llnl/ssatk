# flake8: noqa: E501
import numpy as np
import os
import re
import shutil
from glob import glob
from ..utils import sortbynum


def file_exists(filename: str) -> bool:
    """
    Check if a file with the given name exists (including its extensions).
    """
    name, _ = os.path.splitext(filename)
    return bool(glob(f"{name}.*"))


def exists(pathname: str) -> bool:
    """
    Check if a given path is either a file or a directory.
    """
    return os.path.isdir(pathname) or os.path.isfile(pathname)


def mkdir(pathname: str) -> None:
    """
    Create a directory if it does not already exist.
    """
    if not exists(pathname):
        try:
            os.makedirs(pathname, exist_ok=True)
            print(f"Directory '{pathname}' created.")
        except OSError as e:
            print(f"Error creating directory {pathname}: {e}")


def mvdir(source_: str, destination_: str) -> None:
    """
    Move a directory from the source path to the destination path.
    """
    if not exists(destination_):
        print(f"Moving {source_} to {destination_}")
        shutil.move(source_, destination_)
    else:
        print(f"Destination path {destination_} already exists.")


def rmdir(source_: str) -> None:
    """
    Remove a directory and its contents.
    """
    if not exists(source_):
        print(f'{source_} does not exist, no delete.')
    else:
        try:
            shutil.rmtree(source_)
            print(f'Deleted {source_}')
        except OSError as e:
            print(f"Error deleting {source_}: {e}")


def rmfile(pathname: str) -> None:
    """
    Remove a file if it exists.
    """
    if exists(pathname):
        os.remove(pathname)
        print(f"File: '{pathname}' deleted.")


def listdir(
    dir_path: str = '*',
    files_only: bool = False,
    exclude: str = None,
    do_sort: bool = True,
) -> list:
    """
    List files in a directory, with options to filter, exclude, and sort.
    """
    if '*' not in dir_path:
        dir_path = os.path.join(dir_path, '*')
    expanded_paths = glob(dir_path)

    if files_only:
        files = [f for f in expanded_paths if os.path.isfile(f)]
        print(f'{len(files)} files in {dir_path}')
    else:
        files = expanded_paths
        print(f'{len(files)} files in {dir_path}')

    if exclude:
        files = [file for file in files if exclude not in os.path.basename(file)]

    if do_sort:
        return sortbynum(files)
    return files


def get_image_paths(folder_path: str, sort_by_number: bool = True) -> list:
    """
    Returns a list of full paths for all image files in the specified folder.
    """
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    image_paths: list = []

    if not os.path.exists(folder_path):
        raise ValueError(f"The folder path '{folder_path}' does not exist")

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_paths.append(os.path.join(root, file))

    if sort_by_number:
        def extract_last_number(path: str):
            filename = os.path.basename(path)
            parts = re.split(r'(\d+)', filename)
            if len(parts) == 1:
                return (1, filename)
            natural_parts = tuple((1, int(part)) if part.isdigit() else (0, part) for part in parts)
            return (0, natural_parts)

        image_paths.sort(key=extract_last_number)

    return image_paths


def pd_flatten(data: list, factor: float = 1) -> list:
    """
    Flatten a list of data by splitting string elements like '[1,2,3]'
    and converting them into floats. Divides each float by `factor`.
    """
    tmp: list = []
    for x in data:
        try:
            tmp.extend(x[1:-1].split(','))
        except TypeError:
            tmp.append(x)
    return [float(x) / factor for x in tmp]


def str_to_array(s: str) -> np.ndarray:
    """
    Convert a string representation of a list (e.g., '[1, 2, 3]') into a NumPy array of floats.
    """
    s = s.replace('[', '').replace(']', '')
    parts = [p.strip() for p in s.split(',') if p.strip() != '']
    return np.array([float(x) for x in parts])


def pdstr_to_arrays(df) -> np.ndarray:
    """
    Apply `str_to_array` to each element of a DataFrame and convert it to a NumPy array of arrays.
    """
    if hasattr(df, "map"):
        return df.map(str_to_array).to_numpy()
    return df.applymap(str_to_array).to_numpy()


def allfiles(dirName: str = os.getcwd()) -> list:
    """
    Get a list of all files in the directory tree starting at the given path.
    """
    listOfFiles: list = []
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
    return listOfFiles
