"""
This module contains sparse matrices, 'UnitaryM', as arbitrary unitary operator, consisting of the total dimension, the core submatrix, and the identity indexes.
It also contains the controlled mat (cmat) which is represented by a core unitary matrix and a list of control qubits.
"""

import itertools
from typing import Tuple, Optional

import numpy as np
from attr import dataclass
from numpy.typing import NDArray

from common.utils.format_matrix import MatrixFormatter
from common.utils.mgen import random_unitary


@dataclass
class UnitaryM:
    """
    Instantiate a unitary matrix.
    :param dimension: dimension of the matrix.
    :param matrix: the core matrix.
    :param indexes: the indexes occupied by the core submatrix.
    """
    dimension: int
    matrix: NDArray
    indexes: Tuple[int]

    def __post_init__(self):
        s = self.matrix.shape
        assert len(s) == 2, f'Matrix must be square {s}.'
        assert np.allclose(self.matrix @ self.matrix.conj().T, np.eye(s[0])), f'Matrix is not unitary {self.matrix}'
        assert self.dimension >= s[0], f'Dimension must be greater than or equal to the dimension of the core matrix.'
        assert len(self.indexes) == s[0], f'The number of indexes must match the dimension of the core matrix.'

    def inflate(self) -> NDArray:
        s = self.matrix.shape
        if self.dimension == s[0]:
            return self.matrix
        result = np.eye(self.dimension, dtype=np.complexfloating)
        for i, j in itertools.product(range(s[0]), range(s[0])):
            i1, j1 = self.indexes[i], self.indexes[j]
            result[i1, j1] = self.matrix[i, j]
        return result


class CUnitary(UnitaryM):
    def __init__(self, m: NDArray, controls: Tuple[Optional[bool]]):
        """
        Instantiate a controlled single-qubit unitary matrix.
        :param m: the 2 x 2 core matrix.
        :param controls: the control qubit together with the 0(False) and 1 (True) state to actuate the control. There should be exactly one None state which is the target qubit.
        Dimension of the matrix is given by len(controls).
        """
        super().__init__(1 << len(controls), m, CUnitary.target_indexes(controls))
        self.controls = controls

    @staticmethod
    def target_indexes(controls: Tuple[bool]) -> Tuple[int, int]:
        i = controls.index(None)
        bits = list(controls)
        num = lambda: int(''.join('1' if bits[i] else '0' for v in bits), 2)
        bits[i] = False
        i = num()
        bits[i] = True
        j = num()
        return i, j


if __name__ == '__main__':
    m = random_unitary(2)
    formatter = MatrixFormatter()
    cu = CUnitary(m, [True, True, None])
    print(formatter.tostr(cu.inflate()))
