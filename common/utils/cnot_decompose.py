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
import functools
from typing import Tuple, Optional, Iterable

import numpy as np

from common.construct.cmat import UnitaryM, CUnitary, X
from common.utils.gray import gray_code, control_bits


def cnot_decompose(m: UnitaryM) -> Tuple[CUnitary, ...]:
    if m.dimension & (m.dimension - 1):
        raise ValueError(f'The dimension of the unitary matrix is not power of 2: {m.dimension}')
    n = m.dimension.bit_length() - 1
    if m.isid():
        control = [None] + [True] * (n - 1)
        return (CUnitary(m.matrix, controls=tuple(control)),)
    if not m.is2l():
        raise ValueError(f'The unitary matrix is not 2 level: {m}')
    code = gray_code(*m.row_indexes)
    components = [CUnitary(X, control_bits(n, core)) for core in zip(code, code[1:-1])]
    if code[-2] < code[-1]:
        #  the final swap preserves the original ordering of the core matrix
        v = m.matrix
    else:
        #  the final swap altered the original ordering of the core matrix
        v = X @ m.matrix @ X
    return tuple(components + [CUnitary(v, control_bits(n, code[-2:]))])


if __name__ == '__main__':
    from common.utils.format_matrix import MatrixFormatter
    import random
    from common.utils.mgen import random_matrix_2l, random_UnitaryM_2l

    random.seed(5)
    formatter = MatrixFormatter()


    def _test_cnot_decompose8():
        r1, r2 = 3, 4
        m = random_matrix_2l(8, r1, r2)
        print(f'test = \n{formatter.tostr(m)}')
        ms = cnot_decompose(m)
        print(f'decompose =')
        for x in ms:
            print(formatter.tostr(x), ',')
        print()
        m3 = functools.reduce(lambda x, y: x @ y, ms + ms[:-1][::-1])
        assert np.all(m3 == m), f'm3 != m: \n{formatter.tostr(m3)},\n\n{formatter.tostr(m)}'


    def _test_cnot_decompose4():
        m = random_UnitaryM_2l(4, 1, 2)
        print(f'test = \n{formatter.tostr(m.inflate())}')
        ms = cnot_decompose(m)
        print(f'decompose =')
        for x in ms:
            print(formatter.tostr(x.inflate()), ',')
        print()
        s, v = ms[:-1], ms[-1]
        palindrome = s + (v,) + s[::-1]
        recovered = functools.reduce(lambda x, y: x @ y, palindrome)
        assert np.allclose(recovered.inflate(), m.inflate()), f'recovered != expected: \n{formatter.tostr(recovered.inflate())},\n\n{formatter.tostr(m.inflate())}'


    def _test_cnot_decompose_random():
        for _ in range(10):
            nqubit = random.randint(2, 5)
            n = 1 << nqubit
            r2 = random.randrange(n)
            while True:
                r1 = random.randrange(n)
                if r1 != r2:
                    break
            m = random_UnitaryM_2l(n, r1, r2)
            print(f'test = \n{formatter.tostr(m.inflate())}')
            ms = cnot_decompose(m)
            print(f'decompose =')
            for x in ms:
                print(formatter.tostr(x.inflate()), ',')
            print()
            s, v = ms[:-1], ms[-1]
            palindrome = s + (v,) + s[::-1]
            recovered = functools.reduce(lambda x, y: x @ y, palindrome)
            assert np.allclose(recovered.inflate(), m.inflate()), f'recovered != expected: \n{formatter.tostr(recovered.inflate())},\n\n{formatter.tostr(m.inflate())}'


    _test_cnot_decompose4()
    _test_cnot_decompose_random()
