import random

import numpy as np
import pytest

from common.construct.cmat import UnitaryM, CUnitary, X
from common.utils.format_matrix import MatrixFormatter
from common.utils.mgen import random_unitary

random.seed(42)
np.random.seed(42)
formatter = MatrixFormatter()


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
