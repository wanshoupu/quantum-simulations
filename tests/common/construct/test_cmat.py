import random

import numpy as np
import pytest
from scipy.stats import unitary_group

from common.construct.cmat import UnitaryM, CUnitary, X
from common.utils.format_matrix import MatrixFormatter
from common.utils.mgen import random_unitary, random_indexes

random.seed(42)
np.random.seed(42)
formatter = MatrixFormatter(precision=2)


def test_UnitaryM_init_invalid_dim_smaller_than_mat():
    with pytest.raises(AssertionError, match="Dimension must be greater than or equal to the dimension of the core matrix."):
        UnitaryM(1, random_unitary(2), (1, 2))


def test_UnitaryM_init_invalid_higher_dimensional_mat():
    with pytest.raises(AssertionError) as exc:
        UnitaryM(3, np.array([[[1]]]), (1, 2))


def test_UnitaryM_init():
    cu = UnitaryM(3, random_unitary(2), (1, 2))
    inflate = cu.inflate()
    print(formatter.tostr(inflate))
    assert inflate[0, :].tolist() == inflate[:, 0].tolist() == [1, 0, 0]


def test_inflate_deflate():
    cu = UnitaryM(3, random_unitary(2), (1, 2))
    m = cu.inflate()
    u = UnitaryM.deflate(m)
    assert u.row_indexes == (1, 2) == u.col_indexes, f'Core indexes is unexpected {u.row_indexes, u.col_indexes}'


def test_CUnitary_init():
    m = random_unitary(2)
    cu = CUnitary(m, (True, True, None))
    print(formatter.tostr(cu.inflate()))
    assert cu.row_indexes == (6, 7) == cu.col_indexes, f'Core indexes is unexpected {cu.row_indexes, cu.col_indexes}'


def test_X():
    assert np.all(np.equal(X[::-1], np.eye(2)))


def test_UnitaryM_create_asymmetric():
    m = unitary_group.rvs(2)
    assert np.allclose(np.conj(m).T @ m, np.eye(2))
    # print(f'm = \n{formatter.tostr(m)}')
    # print()
    row_indxs = 0, 5
    col_indxs = 5, 7
    u = UnitaryM(8, m, row_indxs, col_indxs)
    # print(f'test = \n{formatter.tostr(u.inflate())}')
    inflate = u.inflate()
    assert np.allclose(np.conj(inflate).T @ inflate, np.eye(8)), f'Not unitary'


def test_UnitaryM_mult_asymmetric():
    A = UnitaryM(8, unitary_group.rvs(2), (0, 5), (5, 7))
    B = UnitaryM(8, unitary_group.rvs(2), (1, 0), (3, 4))
    C = A @ B
    print(f'A = \n{formatter.tostr(A.inflate())}')
    print(f'B = \n{formatter.tostr(B.inflate())}')
    print(f'C = \n{formatter.tostr(C.inflate())}')
    print(f'C.core = \n{formatter.tostr(C.matrix)}')

def test_UnitaryM_deflate_asymmetric_random():
    for _ in range(20):
        print(f'Test # {_}')
        nqubit = random.randint(1, 5)
        n = 1 << nqubit
        k = random.randint(2, n)
        row, col = random_indexes(n, k), random_indexes(n, k)
        original = UnitaryM(n, unitary_group.rvs(k), row, col)
        m = original.inflate()
        print(f'original = \n{formatter.tostr(m)}')
        u = UnitaryM.deflate(m)
        assert np.allclose(u.inflate(), m), f'/Deflate/inflate has problem'
