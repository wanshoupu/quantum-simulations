import numpy as np
from numpy import ndarray
from numpy.typing import NDArray


class MatrixFormatter:

    def __init__(self, precision=None):
        self.precision = precision
        if precision:
            np.set_printoptions(precision=precision)

    def nformat(self, x: np.complexfloating) -> object:
        """Convert to real if possible, otherwise to integer if exact."""
        if self.precision:
            x = np.round(x, self.precision)

        if np.isclose(x.imag, 0):  # Check if imaginary part is negligible
            real_part = 0 if np.isclose(x.real, 0) else np.real(x)
            if np.isclose(real_part % 1, 0):  # Check if it's effectively an integer
                return int(real_part)
            return real_part
        # Return complex if it cannot be simplified
        return x

    def mformat(self, x: NDArray[np.complexfloating]) -> ndarray[object]:
        """
        Convert elements in a NumPy array to non-negative integer, integer, positive decimal real, decimal real, or leave as complex as is possible.
        """
        s = x.shape
        r = np.zeros(s).astype(object)
        for idx in np.ndindex(s):
            r[idx] = self.nformat(x[idx])
        return r

    def tostr(self, x: NDArray[np.complexfloating], indent=0) -> str:
        m = self.mformat(x)
        s = m.shape
        f = max(len(repr(m[i])) for i in np.ndindex(s))
        rows = []
        for row in m:
            r = [str(e).rjust(f) for e in row]
            rows.append(' '.join(r))
        pad = ' ' * indent
        sep = ']\n' + pad + ' ['
        return pad + '[[' + sep.join(rows) + ']]'
