from functools import reduce
import numpy as np
from quompiler.construct.cmat import UnitaryM
from quompiler.utils.mat2l_decompose import mat2l_decompose
from quompiler.utils.mgen import cyclic_matrix, random_matrix_2l, random_unitary
from quompiler.utils.format_matrix import MatrixFormatter
import random

random.seed(3)
np.random.seed(3)
formatter = MatrixFormatter(precision=5)


def test_decompose_identity_matrix():
    n = 3
    dim = 1 << n
    id = UnitaryM(dim, np.eye(2), (0, 1))
    bc = mat2l_decompose(id)
    print(bc)


def test_decompose_sing_qubit_circuit():
    n = 1
    dim = 1 << n
    u = UnitaryM(dim, random_unitary(dim), (0, 1))
    coms = mat2l_decompose(u)
    # print(coms)
    assert len(coms) == 1


def test_mat2l_cyclic():
    m = cyclic_matrix(8, 1)
    # print(formatter.tostr(m))
    tlms = mat2l_decompose(UnitaryM(8, m, tuple(range(8))))
    recovered = reduce(lambda a, b: a @ b, tlms)
    assert np.allclose(recovered.inflate(), m), f'original\n{m}\n, recovered\n{recovered}'


def test_mat2l_2x2_noop():
    m = random_matrix_2l(2, 0, 1)
    # print(formatter.tostr(m))
    tlms = mat2l_decompose(UnitaryM(2, m, (0, 1)))
    # print(f'decompose =')
    # for x in tlms:
    #     print(formatter.tostr(x.inflate()), ',')
    recovered = reduce(lambda a, b: a @ b, tlms)
    assert np.allclose(recovered.inflate(), m), f'original\n{formatter.tostr(m)}\n, recovered\n{formatter.tostr(recovered.inflate())}'


def test_mat2l_3x3_2l():
    mp = UnitaryM(3, random_matrix_2l(2, 0, 1), (0, 1)).inflate()
    # print(formatter.tostr(mp))
    tlms = mat2l_decompose(UnitaryM(3, mp, tuple(range(3))))
    # print(f'decompose =')
    # for x in tlms:
    #     print(formatter.tostr(x.inflate()), ',')
    assert len(tlms) == 1, f'Input is already a 2l matrix but got decomposed into multiple components {len(tlms)}'
    recovered = reduce(lambda a, b: a @ b, tlms)
    assert np.allclose(recovered.inflate(), mp), f'original\n{formatter.tostr(mp)}\n, recovered\n{formatter.tostr(recovered.inflate())}'


def test_mat2l_noop():
    m = random_matrix_2l(3, 0, 1)
    # print(formatter.tostr(m))
    tlms = mat2l_decompose(UnitaryM(3, m, tuple(range(3))))
    # print(f'decompose =')
    # for x in tlms:
    #     print(formatter.tostr(x.inflate()), ',')
    recovered = reduce(lambda a, b: a @ b, tlms)
    assert np.allclose(recovered.inflate(), m), f'original\n{formatter.tostr(m)}\n, recovered\n{formatter.tostr(recovered.inflate())}'


def test_mat2l_3x3():
    m = random_unitary(3)
    # print(f'original =')
    # print(formatter.tostr(m))
    tlms = mat2l_decompose(UnitaryM(3, m, tuple(range(3))))
    # print(f'decompose =')
    # for x in tlms:
    #     print(formatter.tostr(x.inflate()), ',')
    recovered = reduce(lambda a, b: a @ b, tlms)
    assert np.allclose(recovered.inflate(), m), f'original\n{formatter.tostr(m)}\n, recovered\n{formatter.tostr(recovered.inflate())}'


def test_mat2l_random():
    for _ in range(10):
        n = random.randint(2, 8)
        m = random_unitary(n)
        # print(f'original =')
        # print(formatter.tostr(m))
        tlms = mat2l_decompose(UnitaryM(n, m, tuple(range(n))))
        # print(f'decompose =')
        # for x in tlms:
        #     print(formatter.tostr(x.inflate()), ',\n')
        recovered = reduce(lambda a, b: a @ b, tlms)
        assert np.allclose(recovered.inflate(), m), f'original\n{formatter.tostr(m)}\n, recovered\n{formatter.tostr(recovered.inflate())}'
