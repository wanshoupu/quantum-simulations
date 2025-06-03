import numpy as np
import pytest

from quompiler.construct.types import UnivGate
from quompiler.construct.unitary import UnitaryM
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mat_utils import coreindexes
from quompiler.utils.mgen import random_unitary, cyclic_matrix, random_phase

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


def test_init_with_phase():
    phase = random_phase()
    unitary = random_unitary(2)
    cu = UnitaryM(3, (1, 2), unitary, phase=phase)
    actual = cu.inflate()
    # print(formatter.tostr(actual))
    expected = np.eye(3, dtype=np.complexfloating)
    expected[np.ix_(cu.core, cu.core)] = unitary * phase
    assert np.allclose(actual, expected)


def test_inflate():
    unitary = random_unitary(2)
    cu = UnitaryM(3, (1, 2), unitary)
    actual = cu.inflate()
    indxs = coreindexes(actual)
    assert indxs == (1, 2), f'Core indexes is unexpected {indxs}'
    expected = np.eye(3, dtype=np.complexfloating)
    expected[np.ix_(cu.core, cu.core)] = unitary
    assert np.allclose(actual, expected)


def test_inflate_with_phase():
    phase = random_phase()
    unitary = random_unitary(2)
    cu = UnitaryM(3, (1, 2), unitary, phase=phase)
    # execute
    actual = cu.inflate()
    # verify
    expected = np.eye(3, dtype=np.complexfloating)
    expected[np.ix_(cu.core, cu.core)] = unitary * phase
    assert np.allclose(actual, expected)


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
    assert np.allclose(u.matrix, cu.matrix), f'Core matrix is unexpected: {u.matrix}'


def test_inflate_deflate_with_phase():
    phase = random_phase()
    um = UnitaryM(3, (1, 2), random_unitary(2), phase=phase)
    mat = um.inflate()
    actual = UnitaryM.deflate(mat)
    assert actual.core == (1, 2), f'Core indexes is unexpected {actual.core}'
    expected = um.matrix * phase
    assert np.allclose(actual.matrix, expected), f'Core matrix is unexpected: {actual.matrix}'


def test_matmul_identical_cores():
    core = (1, 2)
    a = UnitaryM(3, core, random_unitary(2), 1.0)
    b = UnitaryM(3, core, random_unitary(2), 1.0)
    c = a @ b
    assert c.core == core, f'Core indexes is unexpected {c.core}'
    assert np.allclose(c.matrix, a.matrix @ b.matrix)


def test_deflate_deficit_core():
    dim = 4
    for n in range(4):
        m = np.eye(dim, dtype=np.complexfloating)
        indxs = np.random.choice(dim, size=n, replace=False)
        m[np.ix_(indxs, indxs)] = random_unitary(n)
        u = UnitaryM.deflate(m)
        assert u is not None
        assert len(u.core) >= 2, f'Core has < 2 indexes: {u.core}'
        assert sorted(u.core) == list(u.core), f'Core indexes is not sorted {u.core}'


def test_matmul_eye_phase():
    core = (1, 3)
    a = UnitaryM(4, core, np.array(UnivGate.I), 1.0)
    core2 = (0, 2)
    b = UnitaryM(4, core2, np.array(UnivGate.S), 1.0)
    c = a @ b
    assert c is not None
    assert len(c.core) == 2, f'Core indexes is unexpected {c.core}'


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


def test_matmul_with_phase():
    phase1 = random_phase()
    phase2 = random_phase()
    dim = 3
    a = UnitaryM(dim, (1, 2), random_unitary(2), phase=phase1)
    b = UnitaryM(dim, (0, 2), random_unitary(2), phase=phase2)
    c = a @ b
    assert c.core == (0, 1, 2), f'Core indexes is unexpected {c.core}'
    actual = c.matrix
    # print(f'Actual:\n{formatter.tostr(actual)}')
    a_expanded = np.eye(dim, dtype=np.complexfloating)
    a_expanded[np.ix_(a.core, a.core)] = a.matrix * phase1
    b_expanded = np.eye(dim, dtype=np.complexfloating)
    b_expanded[np.ix_(b.core, b.core)] = b.matrix * phase2
    expected = a_expanded @ b_expanded
    # print(f'Expected:\n{formatter.tostr(expected)}')
    assert np.allclose(actual, expected)


def test_matmul_with_phase_verify_identity_rows_cols():
    phase1 = random_phase()
    phase2 = random_phase()
    dim = 8
    a = UnitaryM(dim, (1, 2), random_unitary(2), phase=phase1)
    b = UnitaryM(dim, (0, 2), random_unitary(2), phase=phase2)
    c = a @ b
    actual = c.inflate()

    # print(f'Actual:\n{formatter.tostr(c.inflate())}')
    a_expanded = np.eye(3, dtype=np.complexfloating)
    a_expanded[np.ix_(a.core, a.core)] = a.matrix * phase1
    b_expanded = np.eye(3, dtype=np.complexfloating)
    b_expanded[np.ix_(b.core, b.core)] = b.matrix * phase2
    expected = a_expanded @ b_expanded
    assert np.allclose(actual[:3, :3], expected)
    assert np.allclose(actual[3:, 3:], np.eye(5))
    assert np.allclose(actual[:3, 3:], np.zeros((3, 5)))
    assert np.allclose(actual[3:, :3], np.zeros((5, 3)))


@pytest.mark.parametrize("dim,core,size,expected", [
    [8, (3, 2), 2, True],
    [4, (2, 3), 2, True],
    [8, (3, 4), 2, False],  # core spans more than one qubit
    [3, (3, 2), 2, False],  # dimension is not power of 2
    [8, (0, 1, 2), 3, False],  # matrix takes more than one qubit
])
def test_is_singlet(dim, core, size, expected):
    u = UnitaryM(dim, core, random_unitary(size), 1.0)
    assert u.issinglet() == expected, f'Unexpected {u}'
