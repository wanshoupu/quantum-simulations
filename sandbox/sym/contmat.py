from itertools import product
from typing import Sequence

import numpy as np
import sympy
from sympy import Matrix as NDArray

from quompiler.construct.types import QType
from quompiler.construct.controller import Controller


class CUnitary:
    def __init__(self, m: NDArray, controls: Sequence[QType]):
        """
        Instantiate a controlled single-qubit unitary matrix.
        :param m: the core matrix.
        :param controls: the control qubit together with the 0(False) and 1 (True) state to actuate the control. There should be exactly one None state which is the target qubit.
        Dimension of the matrix is given by len(controls).
        """
        self.mat = m
        self.controls = controls
        self.controller = Controller(controls)
        self.core = sorted(self.controller.subindexes(QType.TARGET))
        self.multiplier = self.controller.subindexes(QType.IDLER)

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
