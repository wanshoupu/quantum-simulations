"""
This module breaks down the input, U, a two-level unitary matrix of shape(n, n) into product of CNOT's and a controlled single-qubit unitary matrix such that
    U = [CN1, CN2, CN3, V, CN3, CN2, CN1]
The input, U, must be of a form similar to
     000 001 010 011 100 101 110 111
    =================================
000 | 1     |       #       |       |
001 |     1 |       #       |       |
    |-------|-------#-------|-------|
010 |       | 1     #       |       |
011 |       |     a # c     |       |
    |=======|=======#=======|=======|
100 |       |     b # d     |       |
101 |       |       #     1 |       |
    |-------|-------#-------|-------|
110 |       |       #       | 1     |
111 |       |       #       |     1 |
    =================================

The output will be a sequence of CNOT's and unitary matrix V:
    [CN1, CN2, CN3, V]

where V is a controlled single-qubit unitary matrix. For example:
V is a controlled unitary matrix on the 1st qubit by the last two qubits on the condition (x01, y01)
     000 001 010 011 100 101 110 111
    =================================
000 | 1     |       #       |       |
001 |     a |       #     c |       |
    |-------|-------#-------|-------|
010 |       | 1     #       |       |
011 |       |     1 #       |       |
    |=======|=======#=======|=======|
100 |       |       # 1     |       |
101 |     b |       #     d |       |
    |-------|-------#-------|-------|
110 |       |       #       | 1     |
111 |       |       #       |     1 |
    =================================

V is a controlled unitary matrix on the 2nd qubit by the first and third qubit on the condition (0x1, 0y1)
     000 001 010 011 100 101 110 111
    =================================
000 | 1     |       #       |       |
001 |     a |     c #       |       |
    |-------|-------#-------|-------|
010 |       |       #       |       |
011 |     b |     d #       |       |
    |=======|=======#=======|=======|
100 |       |       # 1     |       |
101 |       |       #     1 |       |
    |-------|-------#-------|-------|
110 |       |       #       | 1     |
111 |       |       #       |     1 |
    =================================

V is a controlled unitary matrix on the 2nd qubit by the first and third qubit on the condition (01x, 01y)
     000 001 010 011 100 101 110 111
    =================================
000 | 1     |       #       |       |
001 |     1 |       #       |       |
    |-------|-------#-------|-------|
010 |       | a   c #       |       |
011 |       | b   d #       |       |
    |=======|=======#=======|=======|
100 |       |       # 1     |       |
101 |       |       #     1 |       |
    |-------|-------#-------|-------|
110 |       |       #       | 1     |
111 |       |       #       |     1 |
    =================================
"""
from typing import Tuple, Sequence

import numpy as np

from quompiler.construct.types import UnivGate, QType

from quompiler.construct.cmat import UnitaryM, CUnitary
from quompiler.construct.qontroller import core2control
from quompiler.utils.gray import gray_code
from numpy.typing import NDArray


def euler_decompose(u: NDArray) -> tuple[complex, float, float, float]:
    """
    Given a U(2) matrix, decompose it into Euler angles + an overall scalar factor.
    :param u: U(2) matrix as input
    :return: scalar factor + Euler angles (b, c, d), such that u = a * Rz(b) @ Ry(c) @ Rz(d)
    """
    assert len(u.shape) == 2
    assert u.shape[0] == 2 == u.shape[1]
    assert np.allclose(u.conj() @ u.T, np.eye(2))
    det = np.linalg.det(u)
    c2 = u[0, 0] * u[1, 1] / det
    s2 = -u[1, 0] * u[0, 1] / det
    # assert np.isclose(1, c2 + s2)
    plus = c2 if np.isclose(c2, 0) else np.angle(u[1, 1] / u[0, 0])
    minus = s2 if np.isclose(s2, 0) else np.angle(-u[1, 0] / u[0, 1])
    b = (plus + minus) / 2
    d = (plus - minus) / 2
    x = c2 - s2
    y = 2 * u[1, 0] * u[1, 1] / det / np.exp(1j * b)
    # assert np.isclose(x.imag, 0) and np.isclose(y.imag, 0)
    c = np.arctan2(y.real, x.real)
    a = (u[1, 1] / (np.cos(c / 2) * np.exp(.5j * (b + d)))) if c2 > s2 else (u[1, 0] / (np.sin(c / 2) * np.exp(.5j * (b - d))))
    return a, b, c, d


def control_decompose(cu: CUnitary) -> list[CUnitary]:
    assert cu.issinglet()
    u = cu.matrix
    a, b, c, d = euler_decompose(u)
    phase = a * np.eye(2)
    A = UnivGate.Z.rmat(b) @ UnivGate.Y.rmat(c / 2)
    B = UnivGate.Y.rmat(-c / 2) @ UnivGate.Z.rmat(-(d + b) / 2)
    C = UnivGate.Z.rmat((d - b) / 2)
    target = cu.qspace[cu.controller.controls.index(QType.TARGET)]

    return [CUnitary(phase, [QType.TARGET], [target]),
            CUnitary(A, [QType.TARGET], [target]),
            CUnitary(UnivGate.X.mat, cu.controller.controls, cu.qspace),
            CUnitary(B, [QType.TARGET], [target]),
            CUnitary(UnivGate.X.mat, cu.controller.controls, cu.qspace),
            CUnitary(C, [QType.TARGET], [target])]


def cnot_decompose(m: UnitaryM, qspace: Sequence[int] = None, aspace: Sequence[int] = None) -> Tuple[CUnitary, ...]:
    """
    Decompose an arbitrary unitary matrix into single-qubit operations in universal gates.
    :param m: UnitaryM to be decomposed
    :param qspace: the qubits to be operated on; provided in a list of integer ids. If not provided, will assume the id in the range(n).
    :param aspace: the ancilla qubits to be used for side computation; provided in a list of integer ids. If not provided, will assume the id in the range(n) in the ancilla space.
    :return: a tuple of CUnitary.
    """
    if m.dimension & (m.dimension - 1):
        raise ValueError(f'The dimension of the unitary matrix is not power of 2: {m.dimension}')
    n = m.dimension.bit_length() - 1
    if m.isid():
        return tuple()
    if not m.is2l():
        raise ValueError(f'The unitary matrix is not 2 level: {m}')
    code = gray_code(*m.core)
    xmat = UnivGate.X.mat
    components = [CUnitary(xmat, core2control(n, core), qspace, aspace) for core in zip(code, code[1:-1])]
    if code[-2] < code[-1]:
        #  the final swap preserves the original ordering of the core matrix
        v = m.matrix
    else:
        #  the final swap altered the original ordering of the core matrix
        v = xmat @ m.matrix @ xmat
    cu = CUnitary(v, core2control(n, code[-2:]), qspace, aspace)
    return tuple(components + control_decompose(cu) + components[::-1])
