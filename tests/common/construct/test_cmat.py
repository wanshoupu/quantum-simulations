import random

import numpy as np
import pytest

from common.construct.cmat import UnitaryM, CUnitary, X, coreindexes, idindexes
from common.utils.format_matrix import MatrixFormatter
from common.utils.mgen import random_unitary, cyclic_matrix

random.seed(42)
np.random.seed(42)
formatter = MatrixFormatter(precision=2)


def test_coreindexes():
    m = cyclic_matrix(8, 2)
    indxs = coreindexes(m)
    assert indxs == tuple(range(2, 8))


def test_idindexes():
    m = cyclic_matrix(8, 2)
    indxs = idindexes(m)
    assert indxs == (0, 1)


def test_complementary_indexes():
    m = cyclic_matrix(8, 2)
    indxs = sorted(idindexes(m) + coreindexes(m))
    assert indxs == list(range(8))


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


def test_inflate():
    cu = UnitaryM(3, random_unitary(2), (1, 2))
    m = cu.inflate()
    indxs = coreindexes(m)
    assert indxs == (1, 2), f'Core indexes is unexpected {indxs}'


def test_deflate():
    m = cyclic_matrix(8, 2)
    u = UnitaryM.deflate(m)
    expected = cyclic_matrix(6)
    assert np.array_equal(u.matrix, expected), f'Core matrix is unexpected: {u.matrix}'


def test_inflate_deflate():
    cu = UnitaryM(3, random_unitary(2), (1, 2))
    m = cu.inflate()
    u = UnitaryM.deflate(m)
    assert u.indexes == (1, 2), f'Core indexes is unexpected {u.indexes}'


def test_CUnitary_init():
    m = random_unitary(2)
    cu = CUnitary(m, (True, True, None))
    # print(formatter.tostr(cu.inflate()))
    assert cu.indexes == (6, 7), f'Core indexes is unexpected {cu.indexes}'


def test_X():
    assert np.all(np.equal(X[::-1], np.eye(2)))
