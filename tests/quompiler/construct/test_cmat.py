import random

import numpy as np
import pytest

from quompiler.construct.cmat import UnitaryM, CUnitary, coreindexes, idindexes
from quompiler.construct.qontroller import Qontroller, core2control
from quompiler.construct.types import UnivGate, QType
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mgen import random_unitary, cyclic_matrix, random_indexes, random_UnitaryM, random_control

formatter = MatrixFormatter(precision=2)


def test_coreindexes():
    m = cyclic_matrix(8, 2)
    indxs = coreindexes(m)
    assert indxs == tuple(range(2, 8))


def test_core2control():
    for _ in range(10):
        core = [random.randint(10, 100) for _ in range(random.randint(2, 3))]
        # print(core)
        blength = max(i.bit_length() for i in core)
        gcb = core2control(blength, core)
        bitmatrix = np.array([list(bin(i)[2:].zfill(blength)) for i in core])
        # print(bitmatrix)
        expected = [(QType.CONTROL1 if int(bitmatrix[0, i]) else QType.CONTROL0) if len(set(bitmatrix[:, i])) == 1 else QType.TARGET for i in range(blength)]
        assert gcb == tuple(expected), f'gcb {gcb} != expected {expected}'


def test_control2core_empty_core():
    with pytest.raises(AssertionError):
        # core cannot be empty
        core2control(5, [])


def test_control2core_big_endian():
    n = 3
    core = [2, 3]
    control = core2control(n, core)
    assert control == (QType.CONTROL0, QType.CONTROL1, QType.TARGET)


def test_control2core_single_index():
    for _ in range(10):
        print(f'Test round {_}...')
        n = random.randint(1, 5)
        dim = 1 << n
        core = random_indexes(dim, 1)
        assert len(core) == 1
        index = core[0]
        control = core2control(n, core)
        print(control)
        expected = tuple(QType.CONTROL1 if bool(index & 1 << i) else QType.CONTROL0 for i in range(n))[::-1]
        assert control == expected


@pytest.mark.parametrize('n,core,expected', [
    [3, list(range(3)), (QType.CONTROL0, QType.TARGET, QType.TARGET)],
    [4, list(range(3)), (QType.CONTROL0, QType.CONTROL0, QType.TARGET, QType.TARGET)],
    [4, list(range(3, 6)), (QType.CONTROL0, QType.TARGET, QType.TARGET, QType.TARGET)],
    [4, [2, 4], (QType.CONTROL0, QType.TARGET, QType.TARGET, QType.CONTROL0)],
    [4, [3, 5], (QType.CONTROL0, QType.TARGET, QType.TARGET, QType.CONTROL1)],
])
def test_control2core_unsaturated_core(n, core, expected):
    """
    this test is to verify that unsaturated core (m indexes where 2^(n-1) < m < 2^n for some n) creates the correct control sequence.
    """
    control = core2control(n, core)
    assert control == expected


def test_control2core_random():
    for _ in range(10):
        n = random.randint(1, 5)
        control = [random.choice(list(QType)) for _ in range(n)]
        k = control.count(QType.TARGET) + control.count(QType.IDLER)
        core = Qontroller(control).core()
        assert len(core) == 1 << k
        for i in range(len(control)):
            if control[i] == QType.TARGET or control[i] == QType.IDLER:
                control[i] = QType.TARGET
        expected = core2control(n, core)
        assert tuple(control) == expected


def test_CUnitary_convert_invalid_dimension():
    cu = UnitaryM(3, (1, 2), random_unitary(2))
    with pytest.raises(AssertionError) as exc:
        CUnitary.convert(cu)


def test_CUnitary_convert_no_inflation():
    for _ in range(10):
        n = random.randint(1, 5)
        dim = 1 << n
        k = random.randint(1, n)
        control = random_control(n, k)
        core = Qontroller(control).core()
        m = random_unitary(1 << k)
        assert m.shape[0] == 1 << k
        assert len(core) == 1 << k
        u = UnitaryM(dim, core, m)

        # execute
        c = CUnitary.convert(u)
        assert np.array_equal(c.matrix, u.matrix)


def test_CUnitary_convert_verify_dimension():
    dimension = 4
    m = np.array([[0, 1], [1, 0]])
    u = UnitaryM(dimension, (0, 1), m)
    c = CUnitary.convert(u)
    assert c.dimension == 4


def test_CUnitary_convert():
    for _ in range(20):
        print(f'test round {_}')
        n = random.randint(1, 5)
        dim = 1 << n
        core = random.randint(2, dim)
        indexes = random.sample(range(dim), core)
        u = random_UnitaryM(dim, indexes)
        c = CUnitary.convert(u)
        assert c
        # print()
        # print(formatter.tostr(c.matrix))


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
        UnitaryM(1, (1, 2), random_unitary(2))


def test_UnitaryM_init_invalid_higher_dimensional_mat():
    with pytest.raises(AssertionError) as exc:
        UnitaryM(3, (1, 2), np.array([[[1]]]))


def test_UnitaryM_init():
    cu = UnitaryM(3, (1, 2), random_unitary(2))
    inflate = cu.inflate()
    # print(formatter.tostr(inflate))
    assert inflate[0, :].tolist() == inflate[:, 0].tolist() == [1, 0, 0]


def test_inflate():
    cu = UnitaryM(3, (1, 2), random_unitary(2))
    m = cu.inflate()
    indxs = coreindexes(m)
    assert indxs == (1, 2), f'Core indexes is unexpected {indxs}'


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


def test_CUnitary_init():
    m = random_unitary(2)
    controls = (QType.CONTROL1, QType.CONTROL1, QType.TARGET)
    cu = CUnitary(m, controls)
    # print(formatter.tostr(cu.inflate()))
    assert tuple(cu.core) == (6, 7), f'Core indexes is unexpected {cu.core}'


def test_univ_Y():
    gate = UnivGate.Y
    control = (QType.CONTROL0, QType.CONTROL0, QType.TARGET)
    cu = CUnitary(gate.mat, control)
    expected = np.eye(8, dtype=np.complexfloating)
    expected[:2, :2] = gate.mat
    # print()
    # print(formatter.tostr(expected))
    u = cu.inflate()
    assert np.allclose(u, expected), f'Expected:\n{formatter.tostr(expected)},\nActual:\n{formatter.tostr(u)}'


def test_standard_cunitary():
    gate = UnivGate.Z
    control = (QType.CONTROL1, QType.CONTROL0, QType.CONTROL0, QType.TARGET)
    cu = CUnitary(gate.mat, control)
    expected = np.eye(16, dtype=np.complexfloating)
    expected[8:10, 8:10] = gate.mat
    # print()
    # print(formatter.tostr(expected))
    u = cu.inflate()
    assert np.allclose(u, expected), f'Expected:\n{formatter.tostr(expected)},\nActual:\n{formatter.tostr(u)}'
