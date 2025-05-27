from logging import warning

import numpy as np
from numpy._typing import NDArray

from quompiler.construct.types import UnivGate
from quompiler.utils.mfun import herm
from scipy.linalg import eig, inv


def rangle(U: NDArray) -> float:
    """
    Rotation angle for the input SU(2) operator around some unit vector n.
    :param U: SU(2) operator
    :return: the rotation angle (alpha)
    """
    trace = np.trace(U)
    if not np.isclose(np.imag(trace), 0):
        warning("Warning: Trace has non-negligible imaginary part due to numerical error.")
    trace_real = np.real(trace)
    theta = 2 * np.arccos(np.clip(trace_real / 2, -1.0, 1.0))  # prevent domain error
    return theta


def gc_decompose(U: NDArray) -> tuple[NDArray, NDArray]:
    """
    Group commutator decomposition. This is exact decomposition with no approximation.
    Given a SU(2) operator, representing rotation R(n, φ), we decompose it into V @ W @ V† @ W†,
    with V and W rotations of angle θ around x-axis and y-axis, respectively.
    :param U: input 2x2 unitary matrix.
    :return: tuple(V, W) such that U = V @ W @ V† @ W†
    """
    alpha = rangle(U)
    beta = 2 * np.arccos(np.sqrt(1 - np.cos(alpha / 4)))
    assert np.isclose(np.sin(alpha / 2), 2 * np.sin(beta / 2) ** 2 * np.sqrt(1 - np.sin(beta / 2) ** 4))
    W = UnivGate.X.rotation(beta)
    V = UnivGate.Y.rotation(beta)
    U2 = -V @ W @ herm(V) @ herm(W)
    assert np.isclose(rangle(U2), alpha)
    eigU, simU = eig(U)
    eigU2, simU2 = eig(U2)
    assert np.allclose(eigU, eigU2)
    P = (simU) @ herm(simU2)
    WA = P @ W @ herm(P)
    VA = P @ V @ herm(P)
    U3 = -VA @ WA @ herm(VA) @ herm(WA)
    assert np.allclose(U3, U)
    return VA, WA
