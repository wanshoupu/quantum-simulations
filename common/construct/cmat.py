"""
This module contains sparse matrices, 'UnitaryM', as arbitrary unitary operator, consisting of the total dimension, the core submatrix, and the identity indexes.
It also contains the controlled mat (cmat) which is represented by a core unitary matrix and a list of control qubits.
"""

from itertools import product
from typing import Tuple, Optional, Union

import numpy as np
from attr import dataclass
from numba.core.typing.npydecl import NdIndex
from numpy.typing import NDArray

from common.utils.gray import control_bits

X = np.eye(2)[[1, 0]]


def coreindexes(m) -> Tuple[int, ...]:
    validm(m)
    dimension = m.shape[0]
    identity = np.eye(dimension)
    # indexes of rows / columns that's not that of an identity
    return tuple(i for i in range(dimension) if not (np.allclose(m[i, :], identity[i]) and np.allclose(m[:, i], identity[i])))


def validm(m):
    s = m.shape
    if len(s) != 2:
        raise ValueError(f'Matrix must be 2D array but got {s}.')
    if s[0] != s[1]:
        raise ValueError(f'Matrix must be square but got {s}.')


def validm2l(m: NDArray):
    """
    Validate if m is a 2-level unitary matrix.
    :param m: input matrix.
    :return: bool True if m is a 2-level unitary matrix; otherwise False.
    """
    count = len(coreindexes(m))
    return count <= 2


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
        for i, j in product(range(s[0]), range(s[0])):
            i1, j1 = self.indexes[i], self.indexes[j]
            result[i1, j1] = self.matrix[i, j]
        return result

    def __getitem__(self, index: NdIndex):
        return self.matrix[index]

    def __setitem__(self, index: NdIndex, value):
        self.matrix[index] = value

    def __matmul__(self, other: Union['UnitaryM', np.ndarray]) -> Union['UnitaryM', np.ndarray]:
        if isinstance(other, np.ndarray):
            return self.inflate() @ other
        if self.dimension != other.dimension:
            raise ValueError('matmul: Input operands have dimension mismatch.')
        if self.indexes == other.indexes:
            return UnitaryM(self.dimension, self.matrix @ other.matrix, self.indexes)
        return UnitaryM.deflate(self.inflate() @ other.inflate())

    @classmethod
    def deflate(cls, m: NDArray) -> 'UnitaryM':
        validm(m)
        indexes = coreindexes(m)
        core = m[np.ix_(indexes)]
        return UnitaryM(m.shape[0], core, tuple(indexes))

    def isid(self) -> bool:
        return np.allclose(self.matrix, np.eye(2))

    def is2l(self) -> bool:
        return self.matrix.shape[0] <= 2


class CUnitary(UnitaryM):
    def __init__(self, m: NDArray, controls: Tuple[Optional[bool], ...]):
        """
        Instantiate a controlled single-qubit unitary matrix.
        :param m: the core matrix.
        :param controls: the control qubit together with the 0(False) and 1 (True) state to actuate the control. There should be exactly one None state which is the target qubit.
        Dimension of the matrix is given by len(controls).
        """
        super().__init__(1 << len(controls), m, CUnitary.control2core(controls))
        self.controls = controls

    @staticmethod
    def control2core(controls: Tuple[Optional[bool], ...]) -> Tuple[int, ...]:
        t = [i for i, b in enumerate(controls) if b is None]
        bits = ['1' if b else '0' for b in controls]
        result = []
        for keys in product(*[['0', '1']] * len(t)):
            for i, k in enumerate(keys):
                bits[t[i]] = k
            result.append(int(''.join(bits), 2))
        return tuple(result)

    @classmethod
    def deflate(cls, m: NDArray) -> 'CUnitary':
        validm(m)
        dimension = m.shape[0]
        if dimension & (dimension - 1):
            raise ValueError(f'The dimension of the unitary matrix is not power of 2: {dimension}')
        indexes = coreindexes(m)
        core = m[np.ix_(indexes)]
        return CUnitary(core, control_bits(dimension, indexes))

    def __repr__(self):
        result = super().__repr__()
        return result + f',controls={repr(self.controls)}'


if __name__ == '__main__':
    from common.utils.format_matrix import MatrixFormatter
    from common.utils.mgen import random_unitary
    import random

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
        cu = CUnitary(m, (True, True, None))
        print(formatter.tostr(cu.inflate()))
        assert cu.indexes == (6, 7), f'Core indexes is unexpected {cu.indexes}'


    def _test_X():
        assert np.all(np.equal(X[::-1], np.eye(2)))


    _test_UnitaryM_init()
    _test_create()
    _test_CUnitary_init()
    _test_X()
