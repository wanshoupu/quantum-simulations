import numpy as np
from numpy.typing import NDArray

from quompiler.construct.bytecode import Bytecode, BytecodeIter
from quompiler.construct.su2net import SU2Net
from quompiler.construct.types import UnivGate
from quompiler.utils.group_su2 import gc_decompose
from quompiler.utils.mfun import herm


class SKDecomposer:

    def __init__(self, rtol=1, atol=1, lookup_error=.16):
        """
        Solovay-Kitaev decomposition using a heuristic curve based on the specified tolerance.

        Adjusts the recursion depth according to the relative and absolute error tolerances to ensure
        that the decomposition scales its precision appropriately with the desired accuracy.

        :param rtol: optional, if provided, will be used as the relative tolerance parameter.
        :param atol: optional, if provided, will be used as the absolute tolerance parameter.
        """
        # controls how deep recursion goes
        self.offset = 0
        self.rtol = rtol
        self.atol = atol
        self.rtol_coef = 2
        self.atol_coef = 2
        self.depth = int(max([0, -np.log(rtol) * self.rtol_coef, -np.log(atol) * self.atol_coef])) + self.offset

        # controls the initial lookup error margin needed by the SK algorithm.
        self.lookup_error = lookup_error
        self.su2net = SU2Net(self.lookup_error)

    def approx(self, mat: NDArray) -> list[UnivGate]:
        """
        Approximate a 2×2 unitary matrix using a sequence of universal quantum gates,
        based on the Solovay-Kitaev (SK) theorem.

        The algorithm constructs a product of UnivGate matrices (e.g., H, X, T) that approximates
        the input matrix to within a specified tolerance. The approximation is up to a global phase,
        which is physically unobservable and thus ignored.

        Parameters:
            mat (NDArray): A 2×2 unitary matrix to approximate.

        Returns:
            list[UnivGate]: A list of gates whose product approximates the input unitary matrix.
        """
        assert mat.shape == (2, 2), f'Mat must be a single-qubit operator: mat.shape = (2, 2)'
        assert np.allclose(mat @ herm(mat), np.eye(2)), f'mat must be unitary.'
        sk_node = self._sk_decompose(mat, self.depth)
        return [node.data for node in BytecodeIter(sk_node) if node.is_leaf()]

    def _sk_decompose(self, U: NDArray, n: int) -> Bytecode:
        """
        This implements the main Solovay-Kitaev decomposition algorithm.
        :param U: input 2x2 unitary matrix.
        :param n: recursion depth.
        :return: a Bytecode tree whose leaf nodes are universal gates.
        """
        node, error = self.su2net.lookup(U)
        if n == 0 or np.isclose(error, 0, atol=self.rtol, rtol=self.atol):
            return node
        node = self._sk_decompose(U, n - 1)
        V, W = gc_decompose(U @ herm(node.data))
        vnode = self._sk_decompose(V, n - 1)
        wnode = self._sk_decompose(W, n - 1)
        data = vnode.data @ wnode.data @ herm(vnode.data) @ herm(wnode.data) @ node.data
        children = [vnode, wnode, vnode.herm(), wnode.herm(), node]
        return Bytecode(data, children=children)
