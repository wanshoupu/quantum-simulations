import numpy as np
import os

from numpy.typing import NDArray


def load_ndarray(file_path, fmt=None) -> NDArray:
    """
    Loads an array from a file into a numpy array.
    :param file_path: A path to file containing a numpy array in one of the supported formats: .npy, .csv, .txt.
    :param fmt:
    :return:
    """
    if not fmt:
        ext = os.path.splitext(file_path)[-1].lower()
        fmt = {"npy": "npy", "npz": "npz", "csv": "csv", "txt": "txt"}.get(ext.strip("."), None)
        if fmt is None:
            raise ValueError(f"Unknown format for file: {file_path}")

    if fmt == "npy":
        return np.load(file_path)
    elif fmt in {"txt", "csv"}:
        delimiter = "," if fmt == "csv" else None
        return np.loadtxt(file_path, delimiter=delimiter)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def save_ndarray(file_path, array, fmt=None):
    """
    Dump a NumPy array to a file, supporting multiple formats.

    Parameters:
    - path (str): Path to save the array.
    - array (np.ndarray): The NumPy array to save.
    - fmt (str): Format override (npy, csv, txt).
    """
    if not fmt:
        ext = os.path.splitext(file_path)[-1].lower()
        fmt = {
            ".npy": "npy",
            ".npz": "npz",
            ".csv": "csv",
            ".txt": "txt"
        }.get(ext, None)
        if fmt is None:
            raise ValueError(f"Unsupported file extension for path: {file_path}")

    if fmt == "npy":
        np.save(file_path, array)
    elif fmt in {"csv", "txt"}:
        delimiter = "," if fmt == "csv" else None
        np.savetxt(file_path, array, delimiter=delimiter)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
