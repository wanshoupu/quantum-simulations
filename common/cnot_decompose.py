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

import numpy as np
from sympy.codegen.ast import complex64

from common.format_matrix import MatrixFormatter


def gray(n1, n2):
    result = [n1]
    for i in range(max(n1.bit_length(), n2.bit_length())):
        mask = 1 << i
        bit = n2 & mask
        if result[-1] & mask != bit:
            result.append((result[-1] ^ mask) | bit)
    return result


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


def permeye(indexes):
    """
    Create a square identity matrix n x n, with the permuted indexes
    :param indexes: a permutation of indexes of list(range(len(indexes)))
    :return: the resultant matrix
    """
    return np.diag([1] * len(indexes))[indexes]


def _test_gray_code():
    import random
    for _ in range(10):
        a = random.randint(10, 100)
        b = random.randint(10, 100)
        print(f'{a}, {b}')
        blength = max(a.bit_length(), b.bit_length())
        mybin = lambda x: bin(x)[2:].zfill(blength)
        gcs = gray(a, b)
        print(gcs)
        for m, n in zip(gcs, gcs[1:]):
            print(mybin(m), mybin(n))
            assert mybin(m ^ n).count('1') == 1


def _test_xindexes():
    import random
    random.seed(3)
    for _ in range(10):
        n = random.randint(10, 100)
        a = random.randrange(n)
        b = random.randrange(n)
        xs = xindexes(n, a, b)
        assert xs[a] == b and xs[b] == a


def _test_permeye():
    import random
    random.seed(3)
    for _ in range(10):
        n = random.randint(10, 16)
        a = random.randrange(n)
        b = random.randrange(n)
        xs = xindexes(n, a, b)
        pi = permeye(xs)
        if a == b:
            assert pi[a, a] == 1 == pi[b, b], f'diagonal {a},{b}\n{pi}'
        else:
            assert pi[a, b] == 1 == pi[b, a], f'off diagonal {a},{b}\n{pi}'
            assert pi[a, a] == 0 == pi[b, b], f'diagonal {a},{b}\n{pi}'


def cnot_decompose(m: np.ndarray):
    n = m.shape[0]
    if np.array_equal(m, np.eye(n)):
        return m
    indexes = [i for i in range(n) if m[i, i] != 1]
    if len(indexes) > 2:
        raise ValueError(f'Two-level matrix is expected but got multilevel matrix {m}')
    if len(indexes) < 2:
        indexes.append(indexes[-1] + 1)
    r1, r2 = indexes
    gcs = gray(r1, r2)
    components = [permeye(xindexes(n, a, b)) for a, b in zip(gcs, gcs[1:-1])]
    palindrome = components[::-1] + [m] + components
    v = functools.reduce(lambda a, b: a @ b, palindrome)
    return components + [v]


def _test_cnot_decompose8():
    r1, r2 = 3, 4
    m = _test_matrix_2l(8, r1, r2)
    formatter = MatrixFormatter()
    print(f'test = \n{formatter.tostr(m)}')
    ms = cnot_decompose(m)
    print(f'decompose =')
    for x in ms:
        print(formatter.tostr(x), ',')
    print()
    m3 = functools.reduce(lambda x, y: x @ y, ms + ms[:-1][::-1])
    assert np.all(m3 == m), f'm3 != m: \n{formatter.tostr(m3)},\n\n{formatter.tostr(m)}'


def _test_cnot_decompose4():
    r1, r2 = 1, 2
    m = _test_matrix_2l(4, r1, r2)
    formatter = MatrixFormatter()
    print(f'test = \n{formatter.tostr(m)}')
    ms = cnot_decompose(m)
    print(f'decompose =')
    for x in ms:
        print(formatter.tostr(x), ',')
    print()
    s, v = ms[:-1], ms[-1]
    palindrome = s + [v] + s[::-1]
    m3 = functools.reduce(lambda x, y: x @ y, palindrome)
    assert np.all(m3 == m), f'm != m3: \n{formatter.tostr(m)},\n\n{formatter.tostr(m3)}'


def _test_cnot_decompose_random():
    import random
    random.seed(5)
    for _ in range(10):
        nqubit = random.randint(2, 5)
        n = 1 << nqubit
        r2 = random.randrange(n)
        while True:
            r1 = random.randrange(n)
            if r1 != r2:
                break
        m = _test_matrix_2l(n, r1, r2)
        formatter = MatrixFormatter()
        print(f'test = \n{formatter.tostr(m)}')
        ms = cnot_decompose(m)
        print(f'decompose =')
        for x in ms:
            print(formatter.tostr(x), ',')
        print()
        s, v = ms[:-1], ms[-1]
        palindrome = s + [v] + s[::-1]
        m3 = functools.reduce(lambda x, y: x @ y, palindrome)
        assert np.all(m3 == m), f'm != m3: \n{formatter.tostr(m)},\n\n{formatter.tostr(m3)}'


def _test_matrix_2l(n, r1, r2):
    import random
    random.seed(3)
    rr = lambda: random.randint(0,10)
    m = np.diag([1 + 0j] * n)
    r1, r2 = min(r1, r2), max(r1, r2)
    m[r1, r1] = complex(rr(), rr())
    m[r2, r1] = complex(rr(), rr())
    m[r1, r2] = complex(rr(), rr())
    m[r2, r2] = complex(rr(), rr())
    return m


if __name__ == '__main__':
    # _test_gray_code()
    # _test_xindexes()
    # _test_permeye()
    # _test_cnot_decompose4()
    _test_cnot_decompose_random()
