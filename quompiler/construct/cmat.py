"""
This module contains sparse matrices, 'UnitaryM', as arbitrary unitary operator, consisting of the total dimension, the core submatrix, and the identity indexes.
It also contains the controlled mat (cmat) which is represented by a core unitary matrix and a list of control qubits.
This module differs from scipy.sparse in that we provide convenience specifically for quantum computer controlled unitary matrices.
"""
import copy
from itertools import product
from typing import Tuple, Optional, Union, Sequence

import numpy as np
from numpy.typing import NDArray
from qiskit.circuit.add_control import control

from quompiler.construct.qontroller import Qontroller, QSpace
from quompiler.construct.types import QType


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
    dimension = m.shape[0]
    return tuple(sorted(set(range(dimension)) - set(idindexes(m))))


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


def ispow2(n):
    assert n >= 0
    return n & (n - 1) == 0


def pow2cover(n):
    assert n >= 0
    return (n - 1).bit_length()


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
        if self.core == other.core and self.matrix.shape == other.matrix.shape:
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


def shuffle(mat: NDArray, indexes: Sequence[int]) -> NDArray:
    assert mat.shape[0] == mat.shape[1]
    assert set(indexes) == set(range(mat.shape[0]))
    return mat[np.ix_(indexes, indexes)]


class CUnitary:
    """
    TODO rename to CtrlM or ControlledM
    Represent a controlled unitary operation with a control sequence and an n-qubit unitary matrix.
    Optionally a qubit space may be specified for the total control + target qubits. If not specified, assuming the range [0, 1, ...].
    """

    def __init__(self, m: NDArray, control: Union[Sequence[QType], Qontroller], qspace: Union[Sequence[int], QSpace] = None, aspace: Sequence[int] = None):
        """
        Instantiate a controlled n-qubit unitary matrix.
        :param m: the core matrix.
        :param control: the control sequence or a Qontroller.
        Dimension of the matrix is given by len(controls).
        :param qspace: the qubits to be operated on; provided in a list of integer ids. If not provided, will assume the id in the range(n).
        :param aspace: the ancilla qubits to be used for side computation; provided in a list of integer ids. If not provided, will assume the id in the range(n) in the ancilla space.
               The usage of aspace is TODO.
        """
        if aspace:
            assert len(set(aspace)) == len(aspace)  # uniqueness
        self.controller = control if isinstance(control, Qontroller) else Qontroller(control)
        core = self.controller.core()
        assert m.shape[0] == len(core)
        dim = 1 << self.controller.length
        self.unitary = UnitaryM(dim, core, m)
        if not qspace:
            self.qspace = QSpace(list(range(self.controller.length)))
        elif isinstance(qspace, QSpace):
            self.qspace = qspace
        else:
            # qspace is a sequence
            assert len(qspace) == len(set(qspace))  # uniqueness
            assert len(qspace) == len(control)  # consistency
            self.qspace = QSpace(qspace)
        self.aspace = aspace or []

    def __repr__(self):
        return f'{repr(self.unitary)},controls={repr(self.controller.controls)},qspace={self.qspace}'

    def inflate(self) -> NDArray:
        res = self.unitary.inflate()
        indexes = [self.qspace.map(i) for i in range(res.shape[0])]
        return res[np.ix_(indexes, indexes)]

    def isid(self) -> bool:
        return self.unitary.isid()

    def is2l(self) -> bool:
        return self.unitary.is2l()

    def issinglet(self) -> bool:
        """
        Check if the UnitaryM is a matrix
        :return:
        """
        return self.unitary.issinglet()

    def __matmul__(self, other: 'CUnitary') -> 'CUnitary':
        if self.qspace == other.qspace:
            res = self.unitary @ other.unitary
            return CUnitary.convert(res)
        univ = sorted(set(self.qspace.qids + other.qspace.qids))
        controls = [QType.]
        qspace = QSpace(univ)
        return self.expand(qspace) @ other.expand(qspace)

    def __copy__(self):
        return CUnitary(self.unitary.matrix, self.controller.controls, self.qspace, self.aspace)

    def __deepcopy__(self, memodict={}):
        if self in memodict:
            return memodict[self]
        new = CUnitary(
            copy.deepcopy(self.unitary.matrix, memodict),
            copy.deepcopy(self.controller.controls, memodict),
            copy.deepcopy(self.qspace, memodict),
            copy.deepcopy(self.aspace, memodict),
        )
        memodict[self] = new
        return new

    @classmethod
    def convert(cls, u: UnitaryM, qspace: Sequence[int] = None, aspace: Sequence[int] = None) -> 'CUnitary':
        """
        Convert a UnitaryM to CUnitary based on organically grown control sequence.
        This can potentially expand the order of matrix to a number that is power of 2.
        :param u:
        :param qspace:
        :param aspace:
        :return:
        """
        if qspace:
            n = len(qspace)
            assert u.order() == 1 << n
            assert u.core in set(qspace)
        else:
            assert u.order() & (u.order() - 1) == 0
            n = u.order().bit_length() - 1

        controller = Qontroller.create(n, u.core)
        core = controller.core()
        # assert set(u.core) <= set(core)
        lookup = {idx: i for i, idx in enumerate(core)}
        indxs = [lookup[c] for c in u.core]
        m = np.eye(len(core), dtype=np.complexfloating)
        m[np.ix_(indxs, indxs)] = u.matrix
        return CUnitary(m, controller, qspace=qspace, aspace=aspace)

    def expand(self, qspace: QSpace) -> 'CUnitary':
        """
        Expand into the super qspace by adding necessary dimensions.
        :param qspace: the super qspace that must be a cover of self.qspace
        :return: a CUnitary
        """
        assert set(self.qspace.qids) <= set(qspace.qids)
        controls = [QType.T]
        return [control[i] for i in indexes]
        controls = qspace.align(self.controller.controls)
        core = qspace.map(self.unitary.core)
        mat = shuffle(self.unitary.matrix, core)
        return CUnitary(mat, controls, qspace)
