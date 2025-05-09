from numpy.typing import NDArray

from quompiler.construct.types import UnivGate


def sk_approx(mat: NDArray, rtol=1.e-5, atol=1.e-8) -> list[UnivGate]:
    """
    This is an implementation of an algorithm based on the Solovay-Kitaev theorem (SK).

    Approximate a 2x2 unitary matrix with the product of UnivGate matrice, particularly H and X.
    :param mat: input 2x2 unitary matrix.
    :param rtol: optional, if provided, will be used as the relative tolerance parameter.
    :param atol: optional, if provided, will be used as the absolute tolerance parameter.
    :return: a list of NDArrays whose product is an approximation to the input within the specified tolerance.
    """
    pass
