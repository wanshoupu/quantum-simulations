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
from typing import Tuple

from common.construct.cmat import UnitaryM, CUnitary, X
from common.utils.gray import gray_code, control_bits, cogray_code


def cnot_decompose(m: UnitaryM) -> Tuple[CUnitary, ...]:
    u = m.matrix.copy()
    if m.dimension & (m.dimension - 1):
        raise ValueError(f'The dimension of the unitary matrix is not power of 2: {m.dimension}')
    n = m.dimension.bit_length() - 1
    if m.isid():
        control = [None] + [True] * (n - 1)
        return (CUnitary(u, controls=tuple(control)),)
    if not m.is2l():
        raise ValueError(f'The unitary matrix is not 2 level: {m}')
    ri, rj = m.row_indexes
    ci, cj = m.col_indexes
    (lcode1, lcode2), (rcode1, rcode2) = cogray_code((ri, rj), (ci, cj))
    left_coms = [CUnitary(X, control_bits(n, core)) for core in zip(lcode1, lcode1[1:-1])] + [
        CUnitary(X, control_bits(n, core)) for core in zip(lcode2, lcode2[1:-1])
    ]
    if (ri - rj) * (lcode1[-1] - lcode2[-1]) < 0:
        u = X @ u
    right_coms = [CUnitary(X, control_bits(n, core)) for core in zip(rcode1, rcode1[1:-1])] + [
        CUnitary(X, control_bits(n, core)) for core in zip(rcode2, rcode2[1:-1])
    ]
    if (ci - cj) * (rcode1[-1] - rcode2[-1]) < 0:
        u = u @ X
    return tuple(left_coms + [CUnitary(u, control_bits(n, (lcode1[-1], lcode2[-1])))] + right_coms[::-1])
