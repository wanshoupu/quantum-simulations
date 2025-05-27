from logging import warning

import numpy as np
from numpy.typing import NDArray

from quompiler.construct.bytecode import Bytecode
from quompiler.construct.types import UnivGate
from quompiler.utils.mfun import herm, dist


class SU2Net:

    def __init__(self, error=.4):
        """
        A SU2 ε-net is a point cloud with distance between adjacent points no greater than `error`.
        It's organized in nary-tree similar to Geohash: the longer the sequence, the more precise it is.
        :param error: optional, if provided, will be used as the error tolerance parameter.
        """
        self.error = error
        self.depth = int(1 / self.error)
        self._root = Bytecode(UnivGate.I)
        self.constructed = False

    def lookup(self, mat: NDArray) -> tuple[Bytecode, float]:
        """
        This is the nearest neighbor lookup function for the input matrix.
        :param mat: input 2x2 unitary matrix.
        :return: the nearest neighbor of the input matrix.
        """
        if not self.constructed:
            self.constructed = True
            _grow_tree(self._root, self.depth)

        # only assert these when debugging and skip when running in optimized mode (python -O ...)
        assert mat.shape == (2, 2), f'Mat must be a single-qubit operator: mat.shape = (2, 2)'
        assert np.allclose(mat @ herm(mat), np.eye(2)), f'mat must be unitary.'
        assert np.isclose(np.linalg.det(mat), 1), f'Mat must have unit determinant.'
        stack = [self._root]
        product = np.eye(2)
        error = dist(mat, product)
        while error > self.error and not stack[-1].is_leaf():
            candidates = [product @ np.array(c.data) for c in stack[-1].children]
            i = np.argmin([dist(mat, c) for c in candidates])
            stack.append(stack[-1].children[i])
            product = candidates[i]
            error = dist(mat, product)
        return Bytecode(product, children=stack[1:]), error


def _grow_tree(node: Bytecode, n: int):
    """
    Grow the ε-bound tree rooted at `node` until the minimum distance between parent and child is less than `error`.
    We grow the subtree by gc_decompose the node into its commutators
    :param node: the root to begin with.
    """
    queue = [node]
    for i in range(n):
        new = []
        for node in queue:
            # Add children to the tree: nodes from the Clifford+T set excluding the parent and identity.
            node.children = [Bytecode(g) for g in UnivGate.cliffordt() if g != node.data and g != UnivGate.I]
            new.extend(node.children)
        queue = new
