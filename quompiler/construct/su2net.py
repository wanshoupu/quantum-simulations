from functools import reduce
from itertools import product
from logging import warning

import numpy as np
from numpy.typing import NDArray

from quompiler.construct.bytecode import Bytecode
from quompiler.construct.types import UnivGate
from quompiler.utils.mfun import herm, dist
from scipy.spatial import KDTree


class SU2Net:

    def __init__(self, error=.4):
        """
        A SU2 ε-net is a point cloud with distance between adjacent points no greater than `error`.
        It's organized in nary-tree similar to Geohash: the longer the sequence, the more precise it is.
        :param error: optional, if provided, will be used as the error tolerance parameter.
        """
        self.error = error
        self.depth = int(1 / self.error)
        self._kdtree = None
        self._seqs = None
        self.constructed = False

    def lookup(self, mat: NDArray) -> Bytecode:
        """
        This is the nearest neighbor lookup function for the input matrix.
        :param mat: input 2x2 unitary matrix.
        :return: the nearest neighbor of the input matrix.
        """
        if not self.constructed:
            self.constructed = True
            seqs = cliffordt_seqs(self.depth)
            self._seqs = seqs
            self._kdtree = KDTree(np.array([u2key(u) for u, _ in self._seqs]))

        # only assert these when debugging and skip when running in optimized mode (python -O ...)
        assert mat.shape == (2, 2), f'Mat must be a single-qubit operator: mat.shape = (2, 2)'
        assert np.allclose(mat @ herm(mat), np.eye(2)), f'mat must be unitary.'
        assert np.isclose(np.linalg.det(mat), 1), f'Mat must have unit determinant.'
        key = u2key(mat)
        _, index = self._kdtree.query([key], k=1)
        approx_U, approx_seq = self._seqs[index[0]]
        return Bytecode(approx_U, [Bytecode(g) for g in approx_seq])


def cliffordt_seqs(depth: int) -> list[tuple]:
    """
    Grow the ε-bound tree rooted at `node` until the minimum distance between parent and child is less than `error`.
    We grow the subtree by gc_decompose the node into its commutators
    :param node: the root to begin with.
    """
    pairs = [(UnivGate.I.matrix, UnivGate.I)]
    cliffordt = UnivGate.cliffordt()
    cliffordt.remove(UnivGate.I)
    for length in range(1, depth + 1):
        for seq in product(cliffordt, repeat=length):
            u = reduce(lambda a, b: a @ b, seq)
            pairs.append((np.array(u), seq))
    return pairs


def u2key(u):
    return u[0, 0].real, u[0, 0].imag, u[0, 1].real, u[0, 1].imag, u[1, 1].real, u[1, 1].imag
