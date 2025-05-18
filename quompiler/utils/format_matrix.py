from typing import Callable

import numpy as np
from numpy import ndarray
from numpy.typing import NDArray


class MatrixFormatter:

    def __init__(self, precision=None, fun: Callable = None):
        self.precision = precision
        self.fun = fun
        if precision:
            np.set_printoptions(precision=precision)

    def nformat(self, x: np.complexfloating) -> object:
        """
        Convert to real if possible, otherwise to integer if exact.
        """
        if self.fun is not None:
            x = self.fun(x)
        if self.precision:
            x = np.round(x, self.precision)
        if np.isclose(x, 0):
            return 0
        if np.isclose(x.imag, 0):  # Check if imaginary part is negligible
            return self.intify(np.real(x))
        if np.isclose(x.real, 0):  # Check if real part is negligible
            return self.intify(np.imag(x)) * 1j
        return x

    def intify(self, x: float):
        if np.isclose(x % 1, 0):  # Check if it's effectively an integer
            return round(x)
        return x

    def mformat(self, x: NDArray[np.complexfloating]) -> ndarray[object]:
        """
        Convert elements in a NumPy array to non-negative integer, integer, positive decimal real, decimal real, or leave as complex as is possible.
        """
        return np.vectorize(self.nformat, otypes=[object])(x)

    def tostr(self, x: NDArray[np.complexfloating], indent=0) -> str:
        m = self.mformat(x)
        s = m.shape
        f = max(len(repr(m[i])) for i in np.ndindex(s))
        stray = np.vectorize(lambda e: str(e).rjust(f))(m)
        pad = ' ' * indent
        if len(s) == 1:
            return pad + '[' + ' '.join(stray) + ']'
        rows = []
        for row in stray:
            rows.append(' '.join(row))
        sep = ']\n' + pad + ' ['
        return pad + '[[' + sep.join(rows) + ']]'
