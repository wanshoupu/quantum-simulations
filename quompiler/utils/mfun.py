import numpy as np
from numpy.typing import NDArray


def allprop(a: NDArray, b: NDArray, rtol=1.e-5, atol=1.e-8, equal_nan=False) -> tuple[bool, any]:
    """
    This function tests if whether a is proportional to b such that a = λb with λ as a scalar.
    If the test results in False, ignore the ratio.
    :param a: NDArray.
    :param b: NDArray of same shape.
    :param rtol: optional, if provided, will be passed on to np.allclose
    :param atol: optional, if provided, will be passed on to np.allclose
    :param equal_nan: optional, if provided, will be passed on to np.allclose
    :return: return (True, λ) if a is proportional to b such that a = λb; otherwise return (False, _).
    """
    if a.shape != b.shape:
        return False, 0
    # Avoid division by zero
    mask = (b != 0)

    if not np.any(mask):
        return np.allclose(a, 0, rtol=rtol, atol=atol, equal_nan=equal_nan), 0

    ratios = a[mask] / b[mask]
    r = ratios[0]

    return np.allclose(ratios, r, rtol=rtol, atol=atol, equal_nan=equal_nan), r
