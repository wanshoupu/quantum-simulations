from math import log

import numpy as np
from numpy.typing import NDArray

from quompiler.construct.bytecode import Bytecode, BytecodeIter
from quompiler.construct.su2net import SU2Net
from quompiler.construct.types import UnivGate
from quompiler.utils.group_su2 import gc_decompose, gphase
from quompiler.utils.mfun import herm

MAX_ERROR = .5


class SKDecomposer:

    def __init__(self, rtol=1.e-3, atol=1.e-5):
        """
        heuristic curve: based on the tolerance requirement, estimate the needed length of Solovay-Kitaev decomposition.
        :param rtol: optional, if provided, will be used as the relative tolerance parameter.
        :param atol: optional, if provided, will be used as the absolute tolerance parameter.
        """
        self.depth = int(max([0, -log(rtol), -log(atol)]))
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
        phase = gphase(mat)
        sk_node = self._sk_decompose(mat / phase, self.depth)
        if not np.isclose(phase, 1):
            phase_node = phase * UnivGate.I.matrix
        return [node.data for node in BytecodeIter(sk_node) if node.is_leaf()]

    def _sk_decompose(self, U: NDArray, n: int) -> tuple[Bytecode, complex]:
        """
        This implements the main Solovay-Kitaev decomposition algorithm.
        :param U: input 2x2 unitary matrix.
        :param n: recursion depth.
        :return: a Bytecode tree whose leaf nodes are universal gates.
        """
        if n == 0:
            node, _ = self.su2net.lookup(U)
            return node, 1
        node, _ = self._sk_decompose(U, n - 1)
        V, W, gc_phase = gc_decompose(U @ herm(node.data))
        vnode, vphase = self._sk_decompose(V, n - 1)
        wnode, wphase = self._sk_decompose(W, n - 1)
        phase = gc_phase * vphase * wphase
        data = phase * vnode.data @ wnode.data @ herm(vnode.data) @ herm(wnode.data) @ node.data
        children = [vnode, wnode, vnode.herm(), wnode.herm(), node]
        return Bytecode(data, children=children), phase
