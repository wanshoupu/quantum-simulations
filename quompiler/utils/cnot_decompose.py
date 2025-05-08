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
from numpy.typing import NDArray

from quompiler.construct.cgate import ControlledGate
from quompiler.construct.unitary import UnitaryM
from quompiler.construct.qontroller import core2control
from quompiler.construct.types import UnivGate, QType
from quompiler.utils.gray import gray_code


def euler_decompose(u: NDArray) -> tuple[complex, float, float, float]:
    """
    Given a U(2) matrix, decompose it into Euler angles + an overall scalar factor.
    :param u: U(2) matrix as input
    :return: scalar factor + Euler angles (b, c, d), such that u = a * Rz(b) @ Ry(c) @ Rz(d)
    """
    assert u.shape == (2, 2) and np.allclose(u.conj().T @ u, np.eye(2)), "u must be unitary"
    det = np.linalg.det(u)
    c2 = u[0, 0] * u[1, 1] / det
    s2 = -u[1, 0] * u[0, 1] / det
    # assert np.isclose(1, c2 + s2)
    plus = c2 if np.isclose(c2, 0) else np.angle(u[1, 1] / u[0, 0])
    minus = s2 if np.isclose(s2, 0) else np.angle(-u[1, 0] / u[0, 1])

    b = (plus + minus) / 2  # angle before Y rotation
    d = (plus - minus) / 2  # angle after Y rotation

    # Determine rotation angle around Y (theta)
    x = np.real_if_close(c2 - s2)
    y = np.real_if_close(2 * u[1, 0] * u[1, 1] / det / np.exp(1j * b))
    # assert np.isclose(x.imag, 0) and np.isclose(y.imag, 0)
    c = np.arctan2(y, x)
    # Global phase
    a = (u[1, 1] / (np.cos(c / 2) * np.exp(.5j * (b + d)))) if c2 > s2 else (u[1, 0] / (np.sin(c / 2) * np.exp(.5j * (b - d))))
    return a, b, c, d  # Global phase (a), and Euler angles (b, c, d)


def std_decompose(cu: ControlledGate) -> list[ControlledGate]:
    """
    Given a controlled unitary matrix, decompose it into controlled standard gate operations `ControlledGate`.
    :param cu: controlled unitary matrix as input
    :return: a list of controlled standard gate operations
    """
    assert cu.issinglet()
    a, b, c, d = euler_decompose(cu.unitary.matrix)
    phase = a * np.eye(2)
    A = UnivGate.Z.rotationM(b) @ UnivGate.Y.rotationM(c / 2)
    B = UnivGate.Y.rotationM(-c / 2) @ UnivGate.Z.rotationM(-(d + b) / 2)
    C = UnivGate.Z.rotationM((d - b) / 2)
    target = cu.target_qids()

    result = [ControlledGate(phase, cu.controller, cu.qspace),
              ControlledGate(A, [QType.TARGET], target),
              ControlledGate(UnivGate.X.matrix, cu.controller, cu.qspace),
              ControlledGate(B, [QType.TARGET], target),
              ControlledGate(UnivGate.X.matrix, cu.controller, cu.qspace),
              ControlledGate(C, [QType.TARGET], target)]
    return result


def cnot_decompose(m: UnitaryM, qspace: Sequence[int] = None, aspace: Sequence[int] = None) -> Tuple[ControlledGate, ...]:
    """
    Decompose an arbitrary unitary matrix into single-qubit operations in universal gates.
    :param m: UnitaryM to be decomposed
    :param qspace: the qubits to be operated on; provided in a list of integer ids. If not provided, will assume the id in the range(n).
    :param aspace: the ancilla qubits to be used for side computation; provided in a list of integer ids. If not provided, will assume the id in the range(n) in the ancilla space.
    :return: a tuple of ControlledGate objects.
    """
    if m.dimension & (m.dimension - 1):
        raise ValueError(f'The dimension of the unitary matrix is not power of 2: {m.dimension}')
    n = m.dimension.bit_length() - 1
    if m.isid():
        return tuple()
    if not m.is2l():
        raise ValueError(f'The unitary matrix is not 2 level: {m}')
    code = gray_code(*m.core)
    xmat = UnivGate.X.matrix
    components = [ControlledGate(xmat, core2control(n, core), qspace, aspace) for core in zip(code, code[1:-1])]
    if code[-2] < code[-1]:
        #  the final swap preserves the original ordering of the core matrix
        v = m.matrix
    else:
        #  the final swap altered the original ordering of the core matrix
        v = xmat @ m.matrix @ xmat
    cu = ControlledGate(v, core2control(n, code[-2:]), qspace, aspace)
    return tuple(components + [cu] + components[::-1])
