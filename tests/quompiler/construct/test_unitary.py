import numpy as np
import pytest

from quompiler.construct.unitary import UnitaryM
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mat_utils import coreindexes
from quompiler.utils.mgen import random_unitary, cyclic_matrix

formatter = MatrixFormatter(precision=2)


def test_init_invalid_dim_smaller_than_mat():
    with pytest.raises(AssertionError, match="Dimension must be greater than or equal to the dimension of the core matrix."):
        UnitaryM(1, (1, 2), random_unitary(2))


def test_init_invalid_higher_dimensional_mat():
    with pytest.raises(AssertionError) as exc:
        UnitaryM(3, (1, 2), np.array([[[1]]]))


def test_init():
    cu = UnitaryM(3, (1, 2), random_unitary(2))
    inflate = cu.inflate()
    # print(formatter.tostr(inflate))
    assert inflate[0, :].tolist() == inflate[:, 0].tolist() == [1, 0, 0]


def test_inflate():
    cu = UnitaryM(3, (1, 2), random_unitary(2))
    m = cu.inflate()
    indxs = coreindexes(m)
    assert indxs == (1, 2), f'Core indexes is unexpected {indxs}'


def test_inflate_shuffled_core():
    unitary = random_unitary(2)
    dim = 2
    a = UnitaryM(dim, (1, 0), unitary)
    expected = np.eye(dim, dtype=np.complexfloating)
    idxs = np.ix_(a.core, a.core)
    expected[idxs] = unitary
    assert np.allclose(a.inflate(), expected), f'Expected:\n{formatter.tostr(expected)},\nActual:\n{formatter.tostr(a.inflate())}'


def test_deflate():
    m = cyclic_matrix(8, 2)
    u = UnitaryM.deflate(m)
    expected = cyclic_matrix(6)
    assert np.array_equal(u.matrix, expected), f'Core matrix is unexpected: {u.matrix}'


def test_inflate_deflate():
    cu = UnitaryM(3, (1, 2), random_unitary(2))
    m = cu.inflate()
    u = UnitaryM.deflate(m)
    assert u.core == (1, 2), f'Core indexes is unexpected {u.core}'


def test_matmul_identical_cores():
    core = (1, 2)
    a = UnitaryM(3, core, random_unitary(2))
    b = UnitaryM(3, core, random_unitary(2))
    c = a @ b
    assert c.core == core, f'Core indexes is unexpected {c.core}'
    assert np.allclose(c.matrix, a.matrix @ b.matrix)


def test_matmul_diff_cores():
    dim = 3
    a = UnitaryM(dim, (1, 2), random_unitary(2))
    b = UnitaryM(dim, (0, 2), random_unitary(2))
    c = a @ b
    assert c.core == (0, 1, 2), f'Core indexes is unexpected {c.core}'
    a_expanded = np.eye(dim, dtype=np.complexfloating)
    idxs = np.ix_(a.core, a.core)
    a_expanded[idxs] = a.matrix
    b_expanded = np.eye(dim, dtype=np.complexfloating)
    idxs = np.ix_(b.core, b.core)
    b_expanded[idxs] = b.matrix
    expected = a_expanded @ b_expanded
    assert np.allclose(c.matrix, expected)


@pytest.mark.parametrize("dim,core,size,expected", [
    [8, (3, 2), 2, True],
    [4, (2, 3), 2, True],
    [8, (3, 4), 2, False],  # core spans more than one qubit
    [3, (3, 2), 2, False],  # dimension is not power of 2
    [8, (0, 1, 2), 3, False],  # matrix takes more than one qubit
])
def test_is_singlet(dim, core, size, expected):
    u = UnitaryM(dim, core, random_unitary(size))
    assert u.issinglet() == expected, f'Unexpected {u}'
