import random
from functools import reduce

import numpy as np

from common.construct.cmat import UnitaryM
from common.utils.cnot_decompose import cnot_decompose
from common.utils.format_matrix import MatrixFormatter
from common.utils.mgen import random_UnitaryM_2l

random.seed(5)
np.random.seed(5)
formatter = MatrixFormatter(precision=2)


def test_decompose_identity_matrix():
    n = 3
    dim = 1 << n
    id = UnitaryM(dim, np.eye(2), (0, 1))
    bc = cnot_decompose(id)
    assert bc == tuple()


def test_cnot_decompose8():
    m = random_UnitaryM_2l(8, 3, 4)
    # print(f'test = \n{formatter.tostr(m.inflate())}')
    ms = cnot_decompose(m)
    # print(f'decompose =')
    # for x in ms:
    #     print(formatter.tostr(x.inflate()), ',')
    # print()
    recovered = reduce(lambda x, y: x @ y, ms)
    assert np.allclose(recovered.inflate(), m.inflate()), f'recovered != expected: \n{formatter.tostr(recovered.inflate())},\n\n{formatter.tostr(m.inflate())}'


def test_cnot_decompose4():
    m = random_UnitaryM_2l(4, 1, 2)
    # print(f'test = \n{formatter.tostr(m.inflate())}')
    ms = cnot_decompose(m)
    # print(f'decompose =')
    # for x in ms:
    #     print(formatter.tostr(x.inflate()), ',')
    # print()
    recovered = reduce(lambda x, y: x @ y, ms)
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
        # print(f'test = \n{formatter.tostr(m.inflate())}')
        ms = cnot_decompose(m)
        # print(f'decompose =')
        # for x in ms:
        #     print(formatter.tostr(x.inflate()), ',')
        # print()
        recovered = reduce(lambda x, y: x @ y, ms)
        assert np.allclose(recovered.inflate(), m.inflate()), f'recovered != expected: \n{formatter.tostr(recovered.inflate())},\n\n{formatter.tostr(m.inflate())}'
