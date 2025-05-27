from math import log

import numpy as np
from numpy.typing import NDArray

from quompiler.construct.bytecode import Bytecode, BytecodeIter
from quompiler.construct.types import UnivGate
from quompiler.utils.group_su2 import gc_decompose
from quompiler.utils.mfun import herm


class SKDecomposer:

    def __init__(self, rtol=1.e-5, atol=1.e-8):
        """
        heuristic curve: based on the tolerance requirement, estimate the needed length of Solovay-Kitaev decomposition.
        :param rtol: optional, if provided, will be used as the relative tolerance parameter.
        :param atol: optional, if provided, will be used as the absolute tolerance parameter.
        """
        self.depth = int(max([0, -log(rtol, 2), -log(atol, 2)]))

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

        approx = solovay_kitaev(mat, self.depth)
        return [node.data for node in BytecodeIter(approx) if node.is_leaf()]


def _basic_lookup(mat: NDArray) -> Bytecode:
    """
    This is the base-case lookup function for zeroth-order approximation of the input matrix.
    :param mat: input 2x2 unitary matrix.
    :return: an approximation of the input matrix with all leaf nodes composed of universal gates.
    """
    pass


def solovay_kitaev(U: NDArray, n: int) -> Bytecode:
    """
    This implements the main Solovay-Kitaev decomposition algorithm.
    :param U: input 2x2 unitary matrix.
    :param n: recursion depth.
    :return: a tuple of the approximation of the input matrix and the decomposed component universal gates.
    """
    if n == 0:
        return _basic_lookup(U)
    node = solovay_kitaev(U, n - 1)
    V, W, sign = gc_decompose(U @ herm(node.data))
    vnode = solovay_kitaev(V, n - 1)
    wnode = solovay_kitaev(W, n - 1)
    data = sign * vnode.data @ wnode.data @ herm(vnode.data) @ herm(wnode.data) @ node.data
    children = [vnode, wnode, vnode.herm(), wnode.herm(), node]
    if sign == -1:
        children.append(Bytecode(-UnivGate.I.matrix))

    return Bytecode(data, children=children)
