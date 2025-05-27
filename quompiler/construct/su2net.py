import numpy as np
from numpy.typing import NDArray

from quompiler.construct.bytecode import Bytecode
from quompiler.construct.types import UnivGate
from quompiler.utils.mfun import herm, dist


class SU2Net:

    def __init__(self, error=1e-1):
        """
        A SU2 ε-net is a point cloud with distance between adjacent points no greater than `error`.
        It's organized in nary-tree similar to Geohash: the longer the sequence, the more precise it is.
        :param error: optional, if provided, will be used as the error tolerance parameter.
        """
        self.error = error
        self.su2net = Bytecode(UnivGate.I.matrix)
        _grow_tree(self.su2net, self.error)

    def lookup(self, mat: NDArray) -> Bytecode:
        """
        This is the nearest neighbor lookup function for the input matrix.
        :param mat: input 2x2 unitary matrix.
        :return: the nearest neighbor of the input matrix.
        """
        # only assert these when debugging and skip when running in optimized mode (python -O ...)
        assert mat.shape == (2, 2), f'Mat must be a single-qubit operator: mat.shape = (2, 2)'
        assert np.allclose(mat @ herm(mat), np.eye(2)), f'mat must be unitary.'
        assert np.isclose(np.linalg.det(mat), 1), f'Mat must have unit determinant.'
        return self._lookup(mat, self.su2net)

    def _lookup(self, mat: NDArray, root: Bytecode) -> Bytecode:
        """
        This is the nearest neighbor lookup function for the input matrix.
        :param mat: input 2x2 unitary matrix.
        :return: the nearest neighbor of the input matrix.
        """
        if dist(mat, root.data) < self.error:
            return root
        assert not root.is_leaf()
        i = np.argmin([dist(mat, c.data) for c in root.children])
        return self._lookup(mat, root.children[i])


def _grow_tree(node: Bytecode, error: float):
    """
    Grow the ε-bound tree rooted at `node` until the minimum distance between parent and child is less than `error`.
    We grow the subtree in a trie data structure, e.g., extend each sequence by the clifford + T set
    :param node: the root to begin with.
    """
    # Add children to the tree: nodes from the Clifford+T set excluding the parent and identity.
    node.children = [Bytecode(g) for g in UnivGate.cliffordt() if g != node.data and g != UnivGate.I]
    for child in node.children:
        if dist(child.data, node.data) > error:
            _grow_tree(child, error)
