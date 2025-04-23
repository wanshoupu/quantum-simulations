from itertools import product
from typing import Tuple, Sequence

import numpy as np
import sympy
from sympy import symbols, Matrix as NDArray

from quompiler.construct.cmat import QubitClass
from quompiler.construct.controller import Controller


def control2core(controls, qubit_class):
    length = len(controls)
    return [sum(t) for t in product(*[(1 << (length - 1 - i), 0) for i, c in enumerate(controls) if c == qubit_class])]


class CUnitary:
    def __init__(self, m: NDArray, controls: Sequence[QubitClass]):
        """
        Instantiate a controlled single-qubit unitary matrix.
        :param m: the core matrix.
        :param controls: the control qubit together with the 0(False) and 1 (True) state to actuate the control. There should be exactly one None state which is the target qubit.
        Dimension of the matrix is given by len(controls).
        """
        self.mat = m
        self.controls = controls
        self.controller = Controller(controls)
        self.core = sorted(control2core(controls, QubitClass.TARGET))
        self.multiplier = control2core(controls, QubitClass.IDLER)

    def __repr__(self):
        result = super().__repr__()
        return result + f',controls={repr(self.controls)}'

    def inflate(self) -> NDArray:
        length = len(self.controls)
        result = sympy.eye(1 << length)
        for i, j in np.ndindex(self.mat.shape):
            for m in self.multiplier:
                result[self.controller.mask(m + self.core[i]), self.controller.mask(m + self.core[j])] = self.mat[i, j]
        return result
