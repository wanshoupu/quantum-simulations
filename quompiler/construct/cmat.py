"""
This module contains sparse matrices, 'UnitaryM', as arbitrary unitary operator, consisting of the total dimension, the core submatrix, and the identity indexes.
It also contains the controlled mat (cmat) which is represented by a core unitary matrix and a list of control qubits.
This module differs from scipy.sparse in that we provide convenience specifically for quantum computer controlled unitary matrices.
"""
from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import Tuple, Optional, Union, Sequence

import numpy as np
from numba.core.typing.npydecl import NdIndex
from numpy.typing import NDArray


class UnivGate(Enum):
    I = ('I', np.eye(2))
    X = ('X', np.eye(2)[[1, 0]])
    Y = ('Y', np.array([[0j, -1j], [1j, 0j]]))
    Z = ('Z', np.array([[1, 0j], [0j, -1]]))
    H = ('H', np.array([[1, 1], [1, -1]]) / np.sqrt(2))
    S = ('S', np.array([[1, 0j], [0j, 1j]]))
    T = ('T', np.array([[1, 0j], [0j, np.exp(1j * np.pi / 4)]]))

    def __init__(self, label, mat: NDArray):
        self.label = label
        self.mat = mat

    @staticmethod
    def get(m: NDArray) -> Optional['UnivGate']:
        for g in UnivGate:
            if m.shape == g.mat.shape and np.allclose(m, g.mat):
                return g


class QubitClass(Enum):
    TARGET = (0, (0, 1))  # target
    IDLER = (1, (0, 1))  # neither target nor control, non-interactive
    CONTROL0 = (2, (0,))  # control with value 0
    CONTROL1 = (3, (1,))  # control with value 1

    def __init__(self, id, base: tuple[int, ...]):
        self.id = id
        self.base = base

    def __repr__(self):
        return self.name

    @staticmethod
    def get(id, default=IDLER) -> 'QubitClass':
        for q in QubitClass:
            if q.id == id:
                return q
        return default


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


def target2core(n: int, target: Tuple[int, ...]) -> Tuple[int, ...]:
    pass


def control2core(controls: Tuple[Optional[bool], ...]) -> Tuple[int, ...]:
    """
    Calculate the core indexes based on controls. Core indexes = the indexes spanned by the target bits encoded in controls.
    Target bits are encoded in controls as those positions with value None.
    :param controls: A tuple of bool or None on position of target bit
    :return: a tuple of indexes spanned by the target bits in the matrix of dimension len(control)
    """
    t = [i for i, b in enumerate(controls) if b is None]
    bits = ['1' if b else '0' for b in controls]
    result = []
    for keys in product(*[['0', '1']] * len(t)):
        for i, k in enumerate(keys):
            bits[t[i]] = k
        result.append(int(''.join(bits), 2))
    return tuple(result)


def core2control(bitlength: int, core: Sequence) -> Tuple[Optional[bool], ...]:
    """
    Generate the control bits of a bundle of indexes given by core.
    The control bits are those bits shared by all the indexes in the core. The rest are target bits.
    The control bits are set to the corresponding common bits in the core (0->False, 1->True) whereas the target bit set to None.
    Big endian is used, namely, most significant bits on the left most end of the array.
    :param bitlength: total length of the control bits
    :param core: the core indexes, i.e., the indexes of the target bits
    :return: Tuple[bool] corresponding to the control bits
    """
    assert core, f'Core cannot be empty'
    dim = 1 << bitlength
    assert max(core) < dim, f'Invalid core: some index in core are bigger than numbers representable by {dim} bits.'
    idiff = []
    for i in range(bitlength):
        mask = 1 << i
        if len({(a & mask) for a in core}) == 2:
            idiff.append(i)
    bits = [bool(core[0] & (1 << j)) for j in range(bitlength)]
    for i in idiff:
        bits[i] = None
    return tuple(bits[::-1])


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
    :param core: the row indexes occupied by the core submatrix.
    """
    dimension: int
    matrix: NDArray
    core: Tuple[int, ...]

    # TODO: add interleaving kronecker product representation. I need a language to describe it.

    def __post_init__(self):
        s = self.matrix.shape
        assert len(s) == 2, f'Matrix must be 2D array but got {s}.'
        assert s[0] == s[1], f'Matrix must be square but got {s}.'
        assert np.allclose(self.matrix @ self.matrix.conj().T, np.eye(s[0])), f'Matrix is not unitary {self.matrix}'
        assert self.dimension >= max(s[0], s[1]), f'Dimension must be greater than or equal to the dimension of the core matrix.'
        assert len(self.core) == s[0], f'The number of indexes must match the size of the core matrix.'

    def __getitem__(self, index: NdIndex):
        return self.matrix[index]

    def __setitem__(self, index: NdIndex, value):
        self.matrix[index] = value

    def __matmul__(self, other: Union['UnitaryM', np.ndarray]) -> Union['UnitaryM', np.ndarray]:
        if isinstance(other, np.ndarray):
            return self.inflate() @ other
        if self.dimension != other.dimension:
            raise ValueError('matmul: Input operands have dimension mismatch.')
        if self.core == other.core:
            return UnitaryM(self.dimension, self.matrix @ other.matrix, self.core)
        # TODO this is a quick but slow implementation. May be improved by finding the union/intersection of indices
        return UnitaryM.deflate(self.inflate() @ other.inflate())

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
            result[self.core[i], self.core[j]] = self.matrix[i, j]
        return result

    @classmethod
    def deflate(cls, m: NDArray) -> 'UnitaryM':
        validm(m)
        indxs = coreindexes(m)
        if not indxs:
            indxs = (0, 1)
        core = m[np.ix_(indxs, indxs)]
        return UnitaryM(m.shape[0], core, indxs)

    def isid(self) -> bool:
        return np.allclose(self.matrix, np.eye(self.matrix.shape[0]))

    def is2l(self) -> bool:
        return self.matrix.shape[0] <= 2

    def issinglet(self) -> bool:
        if self.dimension & (self.dimension - 1) != 0:
            return False
        n = self.dimension.bit_length() - 1
        control = core2control(n, self.core)
        return control.count(None) == 1


class CUnitary:
    def __init__(self, m: NDArray, controls: Tuple[QubitClass, ...]):
        """
        Instantiate a controlled single-qubit unitary matrix.
        :param m: the core matrix.
        :param controls: the control qubit together with the 0(False) and 1 (True) state to actuate the control. There should be exactly one None state which is the target qubit.
        Dimension of the matrix is given by len(controls).
        """
        super().__init__(1 << len(controls), m, control2core(controls))
        self.controls = controls

    def __repr__(self):
        result = super().__repr__()
        return result + f',controls={repr(self.controls)}'

    @classmethod
    def convert(cls, u: UnitaryM) -> 'CUnitary':
        assert u.dimension & (u.dimension - 1) == 0
        n = u.dimension.bit_length() - 1
        controls = core2control(n, u.core)
        core = control2core(controls)
        lookup = {idx: i for i, idx in enumerate(u.core)}

        dim = len(core)
        m = np.eye(dim, dtype=np.complexfloating)
        for i, j in product(range(dim), range(dim)):
            if core[i] not in lookup or core[j] not in lookup:
                continue
            idx = lookup[core[i]], lookup[core[j]]
            m[i, j] = u.matrix[idx]
        return CUnitary(m, controls)


class KronUnitaryM(UnitaryM):

    def __init__(self, register_size: int, m: NDArray, target: Tuple[int, ...]):
        """
        Instantiate a Kronecker unitary matrix.
        This matrix represents an uncontrolled unitary transformation on the sub-Hilbert space of the target qubits.
        :param qubits: the total number of qubits where this matrix possibly acts on
        :param target: the target qubits to apply the core submatrix on.
        """
        super().__init__(1 << register_size, m, target2core(register_size, target))
        self.register_size: int = register_size
        self.target: Tuple[int, ...] = target
