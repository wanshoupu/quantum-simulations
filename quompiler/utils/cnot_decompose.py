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

from quompiler.construct.cgate import CtrlGate
from quompiler.construct.unitary import UnitaryM
from quompiler.construct.qontroller import core2control
from quompiler.construct.types import UnivGate
from quompiler.utils.gray import gray_code


def cnot_decompose(m: UnitaryM, qspace: Sequence[int] = None, aspace: Sequence[int] = None) -> Tuple[CtrlGate, ...]:
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
    components = [CtrlGate(xmat, core2control(n, core), qspace) for core in zip(code, code[1:-1])]
    if code[-2] < code[-1]:
        #  the final swap preserves the original ordering of the core matrix
        v = m.matrix
    else:
        #  the final swap altered the original ordering of the core matrix
        v = xmat @ m.matrix @ xmat
    cu = CtrlGate(v, core2control(n, code[-2:]), qspace)
    return tuple(components + [cu] + components[::-1])
