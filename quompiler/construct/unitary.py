from typing import Sequence

import numpy as np
from numpy._typing import NDArray

from quompiler.utils.mat_utils import coreindexes, validm


def ispow2(n):
    assert n >= 0
    return n & (n - 1) == 0

class UnitaryM:
    """
    Represent a sparse unitary matrix of order dimension.
    The core indexes representing the mapping of the matrix elements to the inflated matrix.
    """

    def __init__(self, dimension: int, core: Sequence[int], matrix: NDArray):
        """
        Instantiate a unitary matrix. The inflate method creates the extended matrix. See mesh_product for the requirements on the core, eyes, and factors.
        :param dimension: dimension of the matrix. TODO remove dimension which is unnecessary.
        :param core: the row indexes occupied by the core submatrix. The total length of core must correspond to the shape of extended matrix.
        :param matrix: the core matrix.
        """
        s = matrix.shape
        assert len(s) == 2, f'Matrix must be 2D array but got {s}.'
        assert s[0] == s[1], f'Matrix must be square but got {s}.'
        assert np.allclose(matrix @ matrix.conj().T, np.eye(s[0])), f'Matrix is not unitary {matrix}'
        assert dimension >= max(s[0], s[1]), f'Dimension must be greater than or equal to the dimension of the core matrix.'
        assert len(core) == s[0], f'The number of indexes must match the size of the expansion matrix.'
        assert len(set(core)) == len(core), f'The indexes in core must be unique.'
        self.dimension = dimension
        self.core = core
        self.matrix = matrix

    def order(self):
        return self.dimension

    def __getitem__(self, index: np.ndindex):
        return self.matrix[index]

    def __setitem__(self, index: np.ndindex, value):
        self.matrix[index] = value

    def __matmul__(self, other: 'UnitaryM') -> 'UnitaryM':
        if self.dimension != other.dimension:
            raise ValueError('matmul: Input operands have dimension mismatch.')
        if self.core == other.core:
            return UnitaryM(self.dimension, self.core, self.matrix @ other.matrix)
        # TODO this is a quick but slow implementation. May be improved by finding the union/intersection of indices
        return UnitaryM.deflate(self.inflate() @ other.inflate())

    def __repr__(self):
        return f'{{dimension={self.dimension}, core={self.core}, matrix={self.matrix}}}'

    def inflate(self) -> NDArray:
        """
        Create a full-blown NDArray represented by UnitaryM. It is a readonly method.
        :return: The full-blown NDArray represented by UnitaryM.
        """
        result = np.eye(self.dimension, dtype=np.complexfloating)
        result[np.ix_(self.core, self.core)] = self.matrix
        return result

    @classmethod
    def deflate(cls, m: NDArray) -> 'UnitaryM':
        validm(m)
        indxs = coreindexes(m)
        if not indxs:
            indxs = (0, 1)
        matrix = m[np.ix_(indxs, indxs)]
        return UnitaryM(m.shape[0], indxs, matrix)

    def isid(self) -> bool:
        return np.allclose(self.matrix, np.eye(self.matrix.shape[0]))

    def is2l(self) -> bool:
        return len(self.core) <= 2

    def issinglet(self) -> bool:
        """
        Check if the UnitaryM is a matrix
        :return:
        """
        if not ispow2(self.dimension):
            return False
        if len(self.core) != 2:
            return False
        i, j = self.core
        n = i ^ j
        return n & (n - 1) == 0
