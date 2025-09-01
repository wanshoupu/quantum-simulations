import random
from functools import reduce
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.stats import unitary_group

from quompiler.construct.bytecode import Bytecode
from quompiler.construct.cgate import CtrlGate
from quompiler.construct.qspace import Qubit
from quompiler.construct.su2gate import RGate
from quompiler.construct.types import QType, UnivGate
from quompiler.construct.unitary import UnitaryM


def random_su2() -> NDArray:
    """
    generate random SU(2) operator, i.e., det(u) = 1.
    """
    u = random_unitary(2)
    det = np.linalg.det(u)
    phase = np.sqrt(det, dtype=np.complex128)
    return u / phase


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


def random_UnitaryM_2l(n, r1, r2, phase=1) -> UnitaryM:
    u = unitary_group.rvs(2)
    r1, r2 = min(r1, r2), max(r1, r2)
    return UnitaryM(n, (r1, r2), u, phase=phase)


def random_UnitaryM(dim, indexes, phase=1) -> UnitaryM:
    cdim = len(indexes)
    u = unitary_group.rvs(cdim)
    return UnitaryM(dim, indexes, u, phase=phase)


def random_CtrlGate(controls: Sequence[QType], qubits: Sequence[Qubit] = None) -> CtrlGate:
    n = controls.count(QType.TARGET)
    u = unitary_group.rvs(1 << n)
    return CtrlGate(u, controls, qubits)


def random_indexes(n, k):
    return tuple(random.sample(range(n), k=k))


def random_control(k, t=None) -> tuple[QType, ...]:
    """
    Generate a random control sequence with total n qubits, k target qubits, (n-k) control qubits
    :param k: positive integer
    :param t: optional, it specifies the number of TARGET. If k is specified, it must satisfy 0 < t <= k. Otherwise, the number is left uncontrolled.
    :return: Control sequence
    """
    if t is None:
        return random.choices([QType.CONTROL1, QType.CONTROL0, QType.TARGET], k=k)

    assert 0 <= t <= k
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


def qft_matrix(n):
    """
    create a cyclic permuted matrix from identity
    :param n: number of qubits
    :return:
    """
    N = 1 << n
    omega = np.exp(2j * np.pi / N)
    j, k = np.meshgrid(np.arange(N), np.arange(N))
    return (omega ** (j * k)) / np.sqrt(N)


def random_ctrlgate(ctrnum, targetnum, qnum=None) -> CtrlGate:
    controls = random_control(ctrnum, targetnum)
    if qnum is None:
        return random_CtrlGate(controls)
    qubits = [Qubit(q) for q in random.sample(range(qnum), ctrnum)]
    return random_CtrlGate(controls, qubits)


def random_state(dimension) -> NDArray:
    """
    Generate a random state vector e.g., normalized quantum state of dimension `dimension`.
    :param dimension:
    :return: NDArray representing the state vector.
    """
    arr = np.random.randn(dimension) + 1j * np.random.randn(dimension)
    return arr / np.linalg.norm(arr)  # normalize to unit length


def random_phase():
    return np.exp(1j * np.random.uniform(0, 2 * np.pi))


def random_gate_seq(length, pop: Sequence[UnivGate] = None):
    if not pop:
        pop = list(UnivGate)
    else:
        assert 1 < len(pop)
    result = [random.choice(pop)]
    for i in range(length):
        c = random.choice(sorted(set(pop) - {result[-1]}))
        result.append(c)
    return result


def random_rgate():
    angle = random.uniform(0, 2 * np.pi)
    theta = random.uniform(0, np.pi)
    phi = random.uniform(0, 2 * np.pi)
    gate = RGate(angle, [theta, phi])
    return gate


def create_bytecode(gate_seq: str, num_ctrl: int) -> Bytecode:
    """
    Create a bytecode from gate_seq and num_ctrl.
    For example, create_bytecode("S,T,SD,TD", 1) will create a Bytecode tree with root and 4 children, each of control size 1.
    """
    controls = random_control(num_ctrl, 1)
    children = [CtrlGate(UnivGate[gate.strip()], controls) for gate in gate_seq.split(',')]
    product = reduce(lambda x, y: x @ y, children)
    return Bytecode(product, [Bytecode(g) for g in children])
