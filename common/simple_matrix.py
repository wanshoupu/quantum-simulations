from typing import Any

import numpy as np
from numpy import ndarray, dtype


def nformat(x: complex) -> object:
    """Convert to real if possible, otherwise to integer if exact."""
    if np.isclose(x.imag, 0):  # Check if imaginary part is negligible
        real_part = np.real(x)
        if np.isclose(real_part % 1, 0):  # Check if it's effectively an integer
            return int(round(real_part))
        return round(real_part, 5)  # Round to 5 decimal places if real
    return x  # Return complex if it cannot be simplified


def mformat(x: np.ndarray[np.complex64]) -> ndarray[object]:
    """
    Convert elements in a NumPy array to non-negative integer, integer, positive decimal real, decimal real, or leave as complex as is possible.
    """
    s = x.shape
    r = np.zeros(s).astype(object)
    for idx in np.ndindex(s):
        r[idx] = nformat(x[idx])
    return r


if __name__ == '__main__':
    matrix = np.array([[1.0, 2.345678], [3 + 0j, 4 + 5j], [-3 - 0j, -4 - 5j], [-2.0, -5.5], [+2.0, -5.5]], dtype=np.complex64)
    print(mformat(matrix))
    shape = (4, 4)  # Example shape, change as needed

    # Generate random binary matrix (0s and 1s)
    binary_matrix = np.random.randint(0, 2, shape)

    # Convert to complex type
    complex_binary_matrix = binary_matrix.astype(complex)

    # Print the result
    print(complex_binary_matrix)
    print(mformat(complex_binary_matrix))
