"""
This module contains sparse matrices, 'UnitaryM', as arbitrary unitary operator, consisting of the total dimension, the core submatrix, and the identity indexes.
It also contains the controlled mat (cmat) which is represented by a core unitary matrix and a list of control qubits.
This module differs from scipy.sparse in that we provide convenience specifically for quantum computer controlled unitary matrices.
"""
from dataclasses import dataclass
from itertools import product
from typing import Tuple, Optional, Union

import numpy as np
from numba.core.typing.npydecl import NdIndex
from numpy.typing import NDArray
from enum import Enum


class UnivGate(Enum):
    I = ('I', np.eye(2))
    X = ('X', np.eye(2)[[1, 0]])
    Y = ('Y', np.array([[0j, 1j], [-1j, 0j]]))
    Z = ('Z', np.array([[1, 0j], [0j, -1]]))
    H = ('H', np.array([[1, 1], [1, -1]]) / np.sqrt(2))
    S = ('S', np.array([[1, 0j], [0j, 1j]]))
    T = ('T', np.array([[1, 0j], [0j, np.exp(1j * np.pi / 4)]]))

    def __init__(self, label, mat: NDArray):
        self.label = label
        self.mat = mat

    @staticmethod
    def get(m: NDArray) -> Union['UnivGate', None]:
        for g in UnivGate:
            if np.allclose(m, g.mat):
                return g
        return None


def immutable(m: NDArray):
    return tuple(map(tuple, m))


def idindexes(m: NDArray) -> Tuple[int, ...]:
    """
    Identity indexes are defined as a list of indexes [i...]
    where both the ith row and the ith column are identical to that of an identity matrix of same dimension.
    :param m: an input square matrix.
    :return: a tuple of identity indexes.
    """
    validm(m)
    dimension = m.shape[0]
    identity = np.eye(dimension)
    idindx = [i for i in range(dimension) if np.allclose(m[:, i], identity[i]) and np.allclose(m[i, :], identity[i])]
    return tuple(idindx)


def coreindexes(m: NDArray) -> Tuple[int, ...]:
    """
    Core indexes are the complementary indexes to the identity indexes. See 'idindexes'.
    :param m: an input square matrix.
    :return: a tuple of core indexes.
    """
    validm(m)
    dimension = m.shape[0]
    identity = np.eye(dimension)
    idindx = [i for i in range(dimension) if not np.allclose(m[:, i], identity[i]) or not np.allclose(m[i, :], identity[i])]
    return tuple(idindx)


def validm(m: NDArray):
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
    indxs = coreindexes(m)
    return len(indxs) <= 2


@dataclass
class UnitaryM:
    """
    Instantiate a unitary matrix.
    :param dimension: dimension of the matrix.
    :param matrix: the core matrix.
    :param indexes: the row indexes occupied by the core submatrix.
    """
    dimension: int
    matrix: NDArray
    indexes: Tuple[int, ...]

    def __post_init__(self):
        s = self.matrix.shape
        assert len(s) == 2, f'Matrix must be 2D array but got {s}.'
        assert s[0] == s[1], f'Matrix must be square but got {s}.'
        assert np.allclose(self.matrix @ self.matrix.conj().T, np.eye(s[0])), f'Matrix is not unitary {self.matrix}'
        assert self.dimension >= max(s[0], s[1]), f'Dimension must be greater than or equal to the dimension of the core matrix.'
        assert len(self.indexes) == s[0], f'The number of row_indexes must match the row dimension of the core matrix.'

    def inflate(self) -> NDArray:
        """
        Create a full-blown NDArray represented by UnitaryM. It is a readonly method.
        :return: The full-blown NDArray represented by UnitaryM.
        """
        matd = self.matrix.shape[0]
        if self.dimension == matd:
            return self.matrix.copy()
        result = np.eye(self.dimension, dtype=np.complexfloating)
        for i, j in product(range(matd), range(matd)):
            result[self.indexes[i], self.indexes[j]] = self.matrix[i, j]
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
        # TODO this is a quick but slow implementation. May be improved by finding the union/intersection of indices
        return UnitaryM.deflate(self.inflate() @ other.inflate())

    @classmethod
    def deflate(cls, m: NDArray) -> 'UnitaryM':
        validm(m)
        indxs = coreindexes(m)
        core = m[np.ix_(indxs, indxs)]
        return UnitaryM(m.shape[0], core, indxs)

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

    def __repr__(self):
        result = super().__repr__()
        return result + f',controls={repr(self.controls)}'
