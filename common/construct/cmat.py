"""
This module contains sparse matrices, 'UnitaryM', as arbitrary unitary operator, consisting of the total dimension, the core submatrix, and the identity indexes.
It also contains the controlled mat (cmat) which is represented by a core unitary matrix and a list of control qubits.
"""

import itertools
from typing import Tuple, Optional

import numpy as np
from attr import dataclass
from numba.core.typing.npydecl import NdIndex
from numpy.typing import NDArray


def coreindexes(m) -> Tuple[int, ...]:
    s = m.shape
    assert len(s) == 2, f'Matrix must be 2D array but got {s}.'
    assert s[0] == s[1], f'Matrix must be square but got {s}.'
    dimension = s[0]
    identity = np.eye(dimension)
    # indexes of rows / columns that's not that of an identity
    return tuple(i for i in range(dimension) if not (np.allclose(m[i, :], identity[i]) and np.allclose(m[:, i], identity[i])))


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
    indexes: Tuple[int, ...]

    def __post_init__(self):
        s = self.matrix.shape
        assert len(s) == 2, f'Matrix must be 2D array but got {s}.'
        assert s[0] == s[1], f'Matrix must be square but got {s}.'
        assert np.allclose(self.matrix @ self.matrix.conj().T, np.eye(s[0])), f'Matrix is not unitary {self.matrix}'
        assert self.dimension >= s[0], f'Dimension must be greater than or equal to the dimension of the core matrix.'
        assert len(self.indexes) == s[0], f'The number of indexes must match the dimension of the core matrix.'

    def inflate(self) -> NDArray:
        """
        Create a full-blown NDArray represented by UnitaryM. It is a readonly method.
        :return: The full-blown NDArray represented by UnitaryM.
        """
        s = self.matrix.shape
        if self.dimension == s[0]:
            return self.matrix.copy()
        result = np.eye(self.dimension, dtype=np.complexfloating)
        for i, j in itertools.product(range(s[0]), range(s[0])):
            i1, j1 = self.indexes[i], self.indexes[j]
            result[i1, j1] = self.matrix[i, j]
        return result

    def __getitem__(self, index: NdIndex):
        return self.matrix[index]

    def __setitem__(self, index: NdIndex, value):
        self.matrix[index] = value

    def __matmul__(self, other: 'UnitaryM') -> 'UnitaryM':
        if self.dimension != other.dimension:
            raise ValueError('matmul: Input operands have dimension mismatch.')
        if self.indexes == other.indexes:
            return UnitaryM(self.dimension, self.matrix @ other.matrix, self.indexes)
        return UnitaryM(self.dimension, self.inflate() @ other.inflate(), indexes=tuple(range(self.dimension)))

    @classmethod
    def deflate(cls, m: NDArray) -> 'UnitaryM':
        s = m.shape
        assert len(s) == 2, f'Matrix must be 2D array but got {s}.'
        assert s[0] == s[1], f'Matrix must be square but got {s}.'
        dimension = s[0]
        indexes = coreindexes(m)
        core = m[np.ix_(indexes)]
        return UnitaryM(dimension, core, tuple(indexes))


class CUnitary(UnitaryM):
    def __init__(self, m: NDArray, controls: Tuple[Optional[bool]]):
        """
        Instantiate a controlled single-qubit unitary matrix.
        :param m: the 2 x 2 core matrix.
        :param controls: the control qubit together with the 0(False) and 1 (True) state to actuate the control. There should be exactly one None state which is the target qubit.
        Dimension of the matrix is given by len(controls).
        """
        super().__init__(1 << len(controls), m, CUnitary.core_indexes(controls))
        self.controls = controls

    @staticmethod
    def core_indexes(controls: Tuple[bool]) -> Tuple[int, int]:
        t = controls.index(None)
        bits = list(controls)
        num = lambda: int(''.join('1' if b else '0' for b in bits), 2)
        bits[t] = False
        i = num()
        bits[t] = True
        j = num()
        return i, j


if __name__ == '__main__':
    from common.utils.format_matrix import MatrixFormatter
    from common.utils.mgen import random_unitary
    import random
    import numpy as np

    random.seed(42)
    np.random.seed(42)
    formatter = MatrixFormatter()


    def _test_UnitaryM_init():
        m = random_unitary(2)
        cu = UnitaryM(3, m, indexes=(1, 2))
        inflate = cu.inflate()
        print(formatter.tostr(inflate))
        assert inflate[0, :].tolist() == inflate[:, 0].tolist() == [1, 0, 0]


    def _test_create():
        cu = UnitaryM(3, random_unitary(2), indexes=(1, 2))
        m = cu.inflate()
        u = UnitaryM.deflate(m)
        assert u.indexes == (1, 2), f'Core indexes is unexpected {u.indexes}'


    def _test_CUnitary_init():
        m = random_unitary(2)
        cu = CUnitary(m, [True, True, None])
        print(formatter.tostr(cu.inflate()))
        assert cu.indexes == (6, 7), f'Core indexes is unexpected {cu.indexes}'


    _test_UnitaryM_init()
    _test_create()
    _test_CUnitary_init()
