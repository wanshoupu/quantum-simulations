import numpy as np
from numpy.typing import NDArray


# Helper function: Hermite transformation of matrix
def herms(ms):
    return [herm(g) for g in ms]


def dist(u: NDArray, v: NDArray) -> float:
    """
    Compute the trace distance between two unitary matrices of shape (2,2), e.g., for a single qubit.
    Distance of two unitary matrices is defined as
    1. calculate the product Δ = u @ v†;
    2. Model Δ as a rotation around certain axis and calculate the rotation angle θ ∈ [-π, π];
    3. D(u,v) = 2 abs(sin(θ/4)).
    :param u: unitary matrix.
    :param v: another unitary matrix.
    :return: trace distance as defined.
    """
    assert u.shape == v.shape == (2, 2), "operators must have shape (2, 2)"
    delta = u @ herm(v)
    assert np.allclose(delta @ herm(delta), np.eye(2)), "matmul product must be unitary"
    det = np.linalg.det(delta)
    # we don't use operator `/=` to avoid `error cast from dtype('complex128') to dtype('float64')`
    delta = delta / np.sqrt(np.complex128(det))
    ct = delta[0, 0] + delta[1, 1]
    assert np.isclose(ct.imag, 0)

    # sometimes the argument is slightly negative, so we cast as complex type before sqrt to prevent NAN.
    result = 2 * np.sqrt(np.complex128(.5 - ct / 4))

    # sometimes the result is slightly complex. We take the abs
    return np.abs(result)


def herm(m: NDArray) -> NDArray:
    """
    The Hermite of a unitary matrix, e.g., m†.
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
