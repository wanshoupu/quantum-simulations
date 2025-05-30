from logging import warning

import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors

from quompiler.construct.bytecode import Bytecode
from quompiler.construct.types import UnivGate
from quompiler.utils.group_su2 import dist


class SU2Net:

    def __init__(self, error=.2):
        """
        A SU2 ε-net is a point cloud with distance between adjacent points no greater than `error`.
        This works for U2 just as well because the distance function is phase agnostic.
        It's organized in nary-tree similar to Geohash: the longer the sequence, the more precise it is.
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
            self._root = NearestNeighbors(n_neighbors=1, algorithm='brute', metric=lambda k1, k2: dist(key2u(k1), key2u(k2)))
            self._root.fit(np.array([u2key(u) for u, _ in self._seqs]))

        # only assert these when debugging and skip when running in optimized mode (python -O ...)
        assert mat.shape == (2, 2), f'Mat must be a single-qubit operator: mat.shape = (2, 2)'
        # assert np.allclose(mat @ herm(mat), np.eye(2)), f'mat must be unitary.'
        # assert np.isclose(np.linalg.det(mat), 1), f'Mat must have unit determinant.'
        key = u2key(mat)
        distances, indices = self._root.kneighbors([key], n_neighbors=1)
        error, index = distances[0][0], indices[0][0]
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


def u2key(u):
    return u[0, 0].real, u[0, 0].imag, u[0, 1].real, u[0, 1].imag, u[1, 0].real, u[1, 0].imag, u[1, 1].real, u[1, 1].imag


def key2u(key):
    return np.array([[complex(key[0], key[1]), complex(key[2], key[3])],
                     [complex(key[4], key[5]), complex(key[6], key[7])]])
