from math import log

import numpy as np
from numpy.typing import NDArray

from quompiler.construct.bytecode import Bytecode, BytecodeIter
from quompiler.construct.su2net import SU2Net
from quompiler.construct.types import UnivGate
from quompiler.utils.group_su2 import gc_decompose
from quompiler.utils.mfun import herm

MAX_ERROR = .15e-1


class SKDecomposer:

    def __init__(self, rtol=1.e-5, atol=1.e-8):
        """
        heuristic curve: based on the tolerance requirement, estimate the needed length of Solovay-Kitaev decomposition.
        :param rtol: optional, if provided, will be used as the relative tolerance parameter.
        :param atol: optional, if provided, will be used as the absolute tolerance parameter.
        """
        self.depth = int(max([0, -log(rtol, 2), -log(atol, 2)]))
        self.su2net = SU2Net(MAX_ERROR)

    def approx(self, mat: NDArray) -> list[UnivGate]:
        """
        This is an implementation of an algorithm based on the Solovay-Kitaev theorem (SK).

        Approximate a 2x2 unitary matrix with the product of UnivGate matrice, particularly H and X.
        :param mat: input 2x2 unitary matrix.
        :return: a list of NDArrays whose product is an approximation to the input within the specified tolerance.
        """
        assert mat.shape == (2, 2), f'Mat must be a single-qubit operator: mat.shape = (2, 2)'
        assert np.allclose(mat @ herm(mat), np.eye(2)), f'mat must be unitary.'
        assert np.isclose(np.linalg.det(mat), 1), f'Mat must have unit determinant.'

        sk_node = self._sk_decompose(mat, self.depth)
        return [node.data for node in BytecodeIter(sk_node) if node.is_leaf()]

    def _sk_decompose(self, U: NDArray, n: int) -> Bytecode:
        """
        This implements the main Solovay-Kitaev decomposition algorithm.
        :param U: input 2x2 unitary matrix.
        :param n: recursion depth.
        :return: a tuple of the approximation of the input matrix and the decomposed component universal gates.
        """
        if n == 0:
            return self.su2net.lookup(U)
        node = self._sk_decompose(U, n - 1)
        V, W, sign = gc_decompose(U @ herm(node.data))
        vnode = self._sk_decompose(V, n - 1)
        wnode = self._sk_decompose(W, n - 1)
        data = sign * vnode.data @ wnode.data @ herm(vnode.data) @ herm(wnode.data) @ node.data
        children = [vnode, wnode, vnode.herm(), wnode.herm(), node]
        if sign == -1:
            children.append(Bytecode(-UnivGate.I.matrix))

        return Bytecode(data, children=children)
