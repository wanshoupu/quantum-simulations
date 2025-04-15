import random
from functools import reduce

import numpy as np

from common.utils.cnot_decompose import cnot_decompose
from common.utils.format_matrix import MatrixFormatter
from common.utils.mgen import random_matrix_2l, random_UnitaryM_2l

random.seed(5)
formatter = MatrixFormatter()


def test_cnot_decompose8():
    r1, r2 = 3, 4
    m = random_matrix_2l(8, r1, r2)
    print(f'test = \n{formatter.tostr(m)}')
    ms = cnot_decompose(m)
    print(f'decompose =')
    for x in ms:
        print(formatter.tostr(x), ',')
    print()
    m3 = reduce(lambda x, y: x @ y, ms + ms[:-1][::-1])
    assert np.all(m3 == m), f'm3 != m: \n{formatter.tostr(m3)},\n\n{formatter.tostr(m)}'


def test_cnot_decompose4():
    m = random_UnitaryM_2l(4, 1, 2)
    print(f'test = \n{formatter.tostr(m.inflate())}')
    ms = cnot_decompose(m)
    print(f'decompose =')
    for x in ms:
        print(formatter.tostr(x.inflate()), ',')
    print()
    s, v = ms[:-1], ms[-1]
    palindrome = s + (v,) + s[::-1]
    recovered = reduce(lambda x, y: x @ y, palindrome)
    assert np.allclose(recovered.inflate(), m.inflate()), f'recovered != expected: \n{formatter.tostr(recovered.inflate())},\n\n{formatter.tostr(m.inflate())}'


def test_cnot_decompose_random():
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
        recovered = reduce(lambda x, y: x @ y, palindrome)
        assert np.allclose(recovered.inflate(), m.inflate()), f'recovered != expected: \n{formatter.tostr(recovered.inflate())},\n\n{formatter.tostr(m.inflate())}'
