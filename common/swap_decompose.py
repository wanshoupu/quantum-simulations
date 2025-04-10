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

from common.format_matrix import MatrixFormatter


def gray(n1, n2):
    result = [n1]
    for i in range(max(n1.bit_length(), n2.bit_length())):
        mask = 1 << i
        bit = n2 & mask
        if result[-1] & mask != bit:
            result.append((result[-1] ^ mask) | bit)
    return result


def swap_perm(n, i, j):
    indexes = list(range(n))
    indexes[i], indexes[j] = indexes[j], indexes[i]
    return indexes


def swap_decompose(m: np.ndarray):
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
    components = [permeye(swap_perm(n, a, b)) for a, b in zip(gcs, gcs[1:])]
    v = functools.reduce(lambda a, b: a @ b, components + [m] + components[:-1][::-1])
    return components + [v]


def swapindex(n, i, j):
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


def _test_swap_decompose8():
    r1, r2 = 3, 4
    m = np.diag([1 + 0j] * 8)
    m[r1, r1] = 1j
    m[r1, r2] = 2j
    m[r2, r1] = 3j
    m[r2, r2] = 4j
    formatter = MatrixFormatter()
    print(f'test = \n{formatter.mformat(m)}')
    ms = swap_decompose(m)
    print(f'decompose =')
    for x in ms:
        print(formatter.mformat(x), ',')
    print()
    m3 = functools.reduce(lambda x, y: x @ y, ms + ms[:-1][::-1])
    assert np.all(m3 == m), f'm3 != m: \n{formatter.tostr(m3)},\n\n{formatter.tostr(m)}'


def _test_swap_decompose4():
    r1, r2 = 1, 2
    m = np.diag([1 + 0j] * 4)
    m[r1, r1] = 1j
    m[r2, r1] = 2j
    m[r1, r2] = 3j
    m[r2, r2] = 4j
    formatter = MatrixFormatter()
    print(f'test = \n{formatter.mformat(m)}')
    ms = swap_decompose(m)
    print(f'decompose =')
    for x in ms:
        print(formatter.mformat(x), ',')
    print()
    m3 = functools.reduce(lambda x, y: x @ y, ms + ms[:-1][::-1])
    assert np.all(m3 == m), f'm != m3: \n{formatter.tostr(m)},\n\n{formatter.tostr(m3)}'


if __name__ == '__main__':
    # _test_gray_code()
    _test_swap_decompose4()
