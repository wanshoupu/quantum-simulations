import random

import numpy as np
from numpy.typing import NDArray
from scipy.stats import unitary_group

from quompiler.construct.cmat import ControlledM
from quompiler.construct.types import QType
from quompiler.construct.unitary import UnitaryM


def random_unitary(n) -> NDArray:
    """Generate a random n x n unitary matrix."""
    # Step 1: Generate a random complex matrix
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    # Step 2: Compute QR decomposition
    Q, R = np.linalg.qr(A)
    # Step 3: Ensure Q is unitary (QR decomposition sometimes returns non-unitary Q due to signs)
    # Adjust phases to make Q truly unitary
    D = np.diag(R) / np.abs(np.diag(R))
    Q = Q @ np.diag(D)
    return Q


def random_UnitaryM_2l(n, r1, r2) -> UnitaryM:
    u = unitary_group.rvs(2)
    r1, r2 = min(r1, r2), max(r1, r2)
    return UnitaryM(n, (r1, r2), u)


def random_UnitaryM(dim, indexes) -> UnitaryM:
    cdim = len(indexes)
    u = unitary_group.rvs(cdim)
    return UnitaryM(dim, indexes, u)


def random_ControlledM(controls: tuple[QType]) -> UnitaryM:
    core = controls.count(QType.TARGET)
    u = unitary_group.rvs(core)
    return ControlledM(u, controls)


def random_indexes(n, k):
    indexes = list(range(n))
    return tuple(random.sample(indexes, k=k))


def random_control(k, t=None) -> tuple[QType, ...]:
    """
    Generate a random control sequence with total n qubits, k target qubits, (n-k) control qubits
    :param k: positive integer
    :param t: optional, it specifies the number of TARGET. If k is specified, it must satisfy 0 < t <= k. Otherwise, the number is left uncontrolled.
    :return: Control sequence
    """
    if t is None:
        return random.choices([QType.CONTROL1, QType.CONTROL0, QType.TARGET], k=k)

    assert 0 < t <= k
    result = random.choices([QType.CONTROL1, QType.CONTROL0], k=k)
    for i in random.sample(range(k), t):
        result[i] = QType.TARGET
    return tuple(result)


def random_matrix_2l(n, r1, r2):
    u = unitary_group.rvs(2)
    m = np.diag([1 + 0j] * n)
    r1, r2 = min(r1, r2), max(r1, r2)
    m[r1, r1] = u[0, 0]
    m[r2, r1] = u[1, 0]
    m[r1, r2] = u[0, 1]
    m[r2, r2] = u[1, 1]
    return m


def permeye(indexes):
    """
    Create a square identity matrix n x n, with the permuted indexes
    :param indexes: a permutation of indexes of list(range(len(indexes)))
    :return: the resultant matrix
    """
    return np.diag([1] * len(indexes))[indexes]


def xindexes(n, i, j):
    """
    Generate indexes list(range(n)) with the ith and jth swapped
    :param n: length of indexes
    :param i: ith index
    :param j: jth index
    :return: indexes list(range(n)) with the ith and jth swapped
    """
    indexes = list(range(n))
    indexes[i], indexes[j] = indexes[j], indexes[i]
    return indexes


def cyclic_matrix(n, i=0, j=None, c=1):
    """
    create a cyclic permuted matrix from identity
    :param n: dimension
    :param i: starting index of the cyclic permutation (inclusive). default 0
    :param j: ending index of the cyclic permutation (exclusive). default n
    :param c: shift cycles, default 1
    :return:
    """
    if j is None:
        j = n
    indexes = list(range(n))
    xs = indexes[:i] + np.roll(indexes[i:j], c).tolist() + indexes[j:]
    return permeye(xs)
