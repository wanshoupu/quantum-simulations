import numpy as np
from numpy.typing import NDArray


# Helper function: Hermite transformation of matrix
def herms(ms):
    return [herm(g) for g in ms[::-1]]


def herm(m: NDArray) -> NDArray:
    """
    The Hermite of a unitary matrix, e.g., m†, (read as 'm dagger').
    :param m:
    :return:
    """
    return np.conj(m.T)


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
    ratio = (a[mask] / b[mask])[0]
    return np.allclose(a, b * ratio, rtol=rtol, atol=atol, equal_nan=equal_nan), ratio


def id_prop(a: NDArray, rtol=1.e-5, atol=1.e-8, equal_nan=False) -> tuple[bool, any]:
    """
    This function tests if a is proportional to an identity matrix such that a = λI with λ as a scalar.
    If the test results in False, ignore the ratio.
    :param a: NDArray.
    :param rtol: optional, if provided, will be passed on to np.allclose
    :param atol: optional, if provided, will be passed on to np.allclose
    :param equal_nan: optional, if provided, will be passed on to np.allclose
    :return: return (True, λ) if a is proportional to an identity matrix such that a = λI; otherwise return (False, _).
    """
    assert len(a.shape) == 2
    if a.shape[0] != a.shape[1]:
        return False, 1
    return allprop(a, np.eye(a.shape[0]), rtol=rtol, atol=atol, equal_nan=equal_nan)


def unitary_prop(a: NDArray, rtol=1.e-5, atol=1.e-8, equal_nan=False) -> tuple[bool, any]:
    """
    This function tests if whether a is proportional to a unitary matrix U, such that a = λU with λ as a scalar.
    If the test results in False, ignore the ratio.
    :param a: NDArray.
    :param rtol: optional, if provided, will be passed on to np.allclose
    :param atol: optional, if provided, will be passed on to np.allclose
    :param equal_nan: optional, if provided, will be passed on to np.allclose
    :return: return (True, λ) if a is proportional to a unitary matrix such that a = λb; otherwise return (False, _).
    """
    assert len(a.shape) == 2
    if a.shape[0] != a.shape[1]:
        return False, 0
    zeros = np.zeros(a.shape)
    if np.allclose(a, zeros, rtol=rtol, atol=atol, equal_nan=equal_nan):
        return False, 0
    mat = a.conj() @ a.T
    prop, r = id_prop(mat, rtol=rtol, atol=atol, equal_nan=equal_nan)
    return prop, np.sqrt(r)
