import random

import numpy as np
import pytest

from quompiler.construct.cmat import UnitaryM, CUnitary, coreindexes, idindexes, UnivGate, control2core, core2control
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mgen import random_unitary, cyclic_matrix, random_control, random_UnitaryM, random_indexes

random.seed(42)
np.random.seed(42)
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
        expected = [bool(int(bitmatrix[0, i])) if len(set(bitmatrix[:, i])) == 1 else None for i in range(blength)]
        assert gcb == tuple(expected), f'gcb {gcb} != expected {expected}'


def test_control2core_empty_core():
    with pytest.raises(AssertionError):
        # core cannot be empty
        core2control(5, [])


def test_control2core_big_endian():
    n = 3
    core = [2, 3]
    control = core2control(n, core)
    assert control == (False, True, None)


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
        expected = tuple(bool(index & 1 << i) for i in range(n))[::-1]
        assert control == expected


def test_control2core():
    for _ in range(10):
        n = random.randint(1, 5)
        k = random.randint(1, n)
        control = random_control(n, k)
        core = control2core(control)
        assert len(core) == 1 << k
        recovered_control = core2control(n, core)
        assert control == recovered_control


def test_CUnitary_convert_invalid_dimension():
    cu = UnitaryM(3, random_unitary(2), (1, 2))
    with pytest.raises(AssertionError) as exc:
        CUnitary.convert(cu)


def test_CUnitary_convert_no_inflation():
    for _ in range(10):
        n = random.randint(1, 5)
        dim = 1 << n
        k = random.randint(1, n)
        control = random_control(n, k)
        core = control2core(control)
        m = random_unitary(1 << k)
        assert m.shape[0] == 1 << k
        assert len(core) == 1 << k
        u = UnitaryM(dim, m, core)
        c = CUnitary.convert(u)
        assert np.array_equal(c.matrix, u.matrix)


def test_CUnitary_convert_verify_dimension():
    dimension = 4
    m = np.array([[0, 1], [1, 0]])
    u = UnitaryM(dimension, m, (0, 1))
    c = CUnitary.convert(u)
    assert c.dimension == 4


def test_CUnitary_convert():
    for _ in range(20):
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
        UnitaryM(1, random_unitary(2), (1, 2))


def test_UnitaryM_init_invalid_higher_dimensional_mat():
    with pytest.raises(AssertionError) as exc:
        UnitaryM(3, np.array([[[1]]]), (1, 2))


def test_UnitaryM_init():
    cu = UnitaryM(3, random_unitary(2), (1, 2))
    inflate = cu.inflate()
    # print(formatter.tostr(inflate))
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


def test_univ_Y():
    gate = UnivGate.Y
    control = (False, False, None)
    cu = CUnitary(gate.mat, control)
    expected = np.eye(8, dtype=np.complexfloating)
    expected[:2, :2] = gate.mat
    # print()
    # print(formatter.tostr(expected))
    u = cu.inflate()
    assert np.allclose(u, expected), f'Expected:\n{formatter.tostr(expected)},\nActual:\n{formatter.tostr(u)}'


def test_standard_cunitary():
    gate = UnivGate.Z
    control = (True, False, False, None)
    cu = CUnitary(gate.mat, control)
    expected = np.eye(16, dtype=np.complexfloating)
    expected[8:10, 8:10] = gate.mat
    # print()
    # print(formatter.tostr(expected))
    u = cu.inflate()
    assert np.allclose(u, expected), f'Expected:\n{formatter.tostr(expected)},\nActual:\n{formatter.tostr(u)}'


def test_univ_gate_X():
    assert np.all(np.equal(UnivGate.X.mat[::-1], np.eye(2)))


def test_univ_gate_Y():
    mat = UnivGate.Y.mat
    assert np.array_equal(mat, UnivGate.Z.mat[[1, 0]] * 1j)


def test_univ_gate_get_none():
    m = random_unitary(2)
    a = UnivGate.get(m)
    assert a is None


def test_univ_gate_get_T():
    mat = np.sqrt(np.array([[1, 0], [0, 1j]]))
    gate = UnivGate.get(mat)
    assert gate == UnivGate.T


def test_univ_gate_get_Z():
    mat = UnivGate.H.mat @ UnivGate.X.mat @ UnivGate.H.mat
    gate = UnivGate.get(mat)
    assert gate == UnivGate.Z


def test_univ_gate_get_H():
    mat = (UnivGate.Z.mat + UnivGate.X.mat) / np.sqrt(2)
    gate = UnivGate.get(mat)
    assert gate == UnivGate.H


def test_univ_gate_commutator():
    mat = UnivGate.Z.mat @ UnivGate.Y.mat @ UnivGate.Z.mat
    assert np.array_equal(mat, -UnivGate.Y.mat), f'mat unexpected {mat}'
