import numpy as np
from numpy.typing import NDArray

from quompiler.construct.types import UnivGate
from quompiler.utils.mfun import herm, herms


def sk_approx(mat: NDArray, rtol=1.e-5, atol=1.e-8) -> list[UnivGate]:
    """
    This is an implementation of an algorithm based on the Solovay-Kitaev theorem (SK).

    Approximate a 2x2 unitary matrix with the product of UnivGate matrice, particularly H and X.
    :param mat: input 2x2 unitary matrix.
    :param rtol: optional, if provided, will be used as the relative tolerance parameter.
    :param atol: optional, if provided, will be used as the absolute tolerance parameter.
    :return: a list of NDArrays whose product is an approximation to the input within the specified tolerance.
    """
    assert mat.shape == (2, 2), f'Mat must be a single-qubit operator: mat.shape = (2, 2)'
    n = 0
    while True:
        approx = sk(mat, n)
        if np.allclose(mat @ herm(approx), np.eye(2), atol=atol, rtol=rtol):
            return components(approx.tobytes())
        n += 1


def components(key: bytes) -> list[UnivGate]:
    pass


def gc_decompose(param) -> tuple[NDArray, NDArray]:
    """
    Group commutator decomposition. This is exact decomposition with no approximation.
    :param param:
    :return:
    """
    pass


def _basic_lookup(mat: NDArray) -> tuple[NDArray, list[UnivGate]]:
    """
    This is the base-case lookup function for zeroth-order approximation of input matrix.
    :param mat: input 2x2 unitary matrix.
    :return: a tuple of the approximation of the input matrix and the decomposed component universal gates.
    """
    pass


def sk(mat: NDArray, n: int) -> tuple[NDArray, list[UnivGate]]:
    """
    This implements the main Solovay-Kitaev decomposition algorithm.
    :param mat: input 2x2 unitary matrix.
    :param n: recursion depth.
    :return: a tuple of the approximation of the input matrix and the decomposed component universal gates.
    """
    if n == 0:
        return _basic_lookup(mat)
    mat1, mats = sk(mat, n - 1)
    v, w = gc_decompose(mat @ np.conj(mat1))
    v1, vs = sk(v, n - 1)
    w1, ws = sk(w, n - 1)
    approx = v1 @ w1 @ herm(v1) @ herm(w1) @ mat1
    coms = vs + ws + herms(vs) + herms(ws) + mats
    return approx, coms
