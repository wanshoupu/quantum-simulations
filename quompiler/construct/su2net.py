from logging import warning

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors

from quompiler.construct.bytecode import Bytecode
from quompiler.construct.types import UnivGate
from quompiler.utils.group_su2 import dist, vec


class SU2Net:

    def __init__(self, error=.2):
        """
        A SU2 ε-net is a point cloud with distance between adjacent points no greater than `error`.
        This works for U2 just as well because the distance function is phase agnostic.
        It's organized in nary-tree similar to Geohash: the longer the sequence, the more precise it is.
        Implementation uses KDTree with `group_su2.vec` as the vectorization method.
        :param error: optional, if provided, will be used as the error tolerance parameter.
        """
        self.error = error
        self.depth = int(1 / self.error) + 1
        self._root = None
        self._seqs = None
        self.constructed = False

    def lookup(self, mat: NDArray) -> tuple[Bytecode, float]:
        """
        This is the nearest neighbor lookup function for the input matrix.
        :param mat: input 2x2 unitary matrix.
        :return: the nearest neighbor of the input matrix.
        """
        if not self.constructed:
            self.constructed = True
            seqs = cliffordt_seqs(self.depth)
            self._seqs = seqs
            self._root = KDTree(np.array([vec(u) for u, _ in self._seqs]))

        # only assert these when debugging and skip when running in optimized mode (python -O ...)
        assert mat.shape == (2, 2), f'Mat must be a single-qubit operator: mat.shape = (2, 2)'
        # assert np.allclose(mat @ herm(mat), np.eye(2)), f'mat must be unitary.'
        # assert np.isclose(np.linalg.det(mat), 1), f'Mat must have unit determinant.'
        key = vec(mat)
        distances, indices = self._root.query([key], k=1)
        error, index = distances[0], indices[0]
        if self.error < error:
            warning(f'Search for {mat} did not converge to within the error range: {self.error}.')
        approx_U, approx_seq = self._seqs[index]
        return Bytecode(approx_U, [Bytecode(g) for g in approx_seq]), error


def cliffordt_seqs(depth: int) -> list[tuple]:
    """
    Grow the ε-bound tree rooted at `node` until the minimum distance between parent and child is less than `error`.
    We grow the subtree by gc_decompose the node into its commutators
    :param node: the root to begin with.
    """
    pairs = [(UnivGate.I.matrix, (UnivGate.I,))]  # start with identity
    cliffordt = UnivGate.cliffordt()
    cliffordt.remove(UnivGate.I)

    stack = [(np.eye(2), list())]
    while stack:
        mat, seq = stack.pop(0)
        if len(seq) == depth:
            continue
        for c in cliffordt:
            if seq and seq[-1] == c:  # avoid consecutive repeat
                continue
            new_seq = seq + [c]
            pairs.append((mat @ c, tuple(new_seq)))
            stack.append((mat @ c, new_seq))
    return pairs
