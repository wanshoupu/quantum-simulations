import numpy as np
from numpy import ndarray


class MatrixFormatter:

    def __init__(self, precision=None):
        self.precision = precision
        if precision:
            np.set_printoptions(precision=precision)

    def nformat(self, x: np.complex64) -> object:
        """Convert to real if possible, otherwise to integer if exact."""
        if np.isclose(x.imag, 0):  # Check if imaginary part is negligible
            real_part = 0 if np.isclose(x.real, 0) else np.real(x)
            if np.isclose(real_part % 1, 0):  # Check if it's effectively an integer
                return int(real_part)
            if self.precision:
                return round(real_part, self.precision)  # Round to n decimal places if real
            return real_part
        # Return complex if it cannot be simplified
        if self.precision:
            return np.round(x, self.precision)
        return x

    def mformat(self, x: np.ndarray[np.complex64]) -> ndarray[object]:
        """
        Convert elements in a NumPy array to non-negative integer, integer, positive decimal real, decimal real, or leave as complex as is possible.
        """
        s = x.shape
        r = np.zeros(s).astype(object)
        for idx in np.ndindex(s):
            r[idx] = self.nformat(x[idx])
        return r

    def tostr(self, x: np.ndarray[np.complex64]) -> str:
        m = self.mformat(x)
        s = m.shape
        f = max(len(repr(m[i])) for i in np.ndindex(s))
        rows = []
        for row in m:
            r = [str(e).rjust(f) for e in row]
            rows.append(' '.join(r))
        return '[[' + ']\n ['.join(rows) + ']]'


if __name__ == '__main__':
    matrix = np.array([[1.0, 2.345678], [3 + 0j, 4 + 5j], [-3 - 0j, -4 - 5j], [-2.0, -5.5], [+2.0, -5.5]], dtype=np.complex64)
    formatter = MatrixFormatter()
    print(formatter.mformat(matrix))
    print(formatter.tostr(matrix))
    shape = (4, 4)  # Example shape, change as needed

    # Generate random binary matrix (0s and 1s)
    binary_matrix = np.random.randint(0, 2, shape)

    # Convert to complex type
    complex_binary_matrix = binary_matrix.astype(complex)

    # Print the result
    print(complex_binary_matrix)
    print(formatter.mformat(complex_binary_matrix))
    print(formatter.tostr(complex_binary_matrix))
