import random

import numpy as np
import pytest
from numpy import kron

from quompiler.construct.cmat import UnitaryM, CUnitary, coreindexes, idindexes
from quompiler.construct.qontroller import Qontroller, core2control, QSpace
from quompiler.construct.types import UnivGate, QType
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.inter_product import mesh_product, inter_product
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
        assert np.array_equal(c.unitary.matrix, u.matrix)


def test_CUnitary_convert_verify_dimension():
    for _ in range(10):
        n = random.randint(1, 5)
        dim = 1 << n
        core = random.randint(2, dim)
        indexes = random.sample(range(dim), core)
        u = random_UnitaryM(dim, indexes)
        c = CUnitary.convert(u)
        assert c.unitary.dimension == dim


def test_CUnitary_convert_expansion():
    for _ in range(1):
        print(f'Test round {_}')
        n = random.randint(1, 5)
        dim = 1 << n
        core = random.randint(2, dim)
        indexes = random.sample(range(dim), core)
        unitM = random_UnitaryM(dim, indexes)
        ctrlM = CUnitary.convert(unitM)
        assert np.allclose(ctrlM.inflate(), unitM.inflate()), f'ctrlM=\n{formatter.tostr(ctrlM.inflate())}\nexpected=\n{formatter.tostr(unitM.inflate())}'


def test_CUnitary_convert_verify_inflation_invariance():
    dimension = 4
    k = random.randint(2, dimension)
    core = sorted(random_indexes(dimension, k))
    u = random_UnitaryM(dimension, core)
    c = CUnitary.convert(u)
    assert np.allclose(c.inflate(), u.inflate()), f'actual=\n{formatter.tostr(c.inflate())}\nexpected=\n{formatter.tostr(u.inflate())}'


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


def test_UnitaryM_inflate():
    cu = UnitaryM(3, (1, 2), random_unitary(2))
    m = cu.inflate()
    indxs = coreindexes(m)
    assert indxs == (1, 2), f'Core indexes is unexpected {indxs}'


def test_UnitaryM_inflate_shuffled_core():
    unitary = random_unitary(2)
    dim = 2
    a = UnitaryM(dim, (1, 0), unitary)
    expected = np.eye(dim, dtype=np.complexfloating)
    idxs = np.ix_(a.core, a.core)
    expected[idxs] = unitary
    assert np.allclose(a.inflate(), expected), f'Expected:\n{formatter.tostr(expected)},\nActual:\n{formatter.tostr(a.inflate())}'


def test_UnitaryM_deflate():
    m = cyclic_matrix(8, 2)
    u = UnitaryM.deflate(m)
    expected = cyclic_matrix(6)
    assert np.array_equal(u.matrix, expected), f'Core matrix is unexpected: {u.matrix}'


def test_UnitaryM_inflate_deflate():
    cu = UnitaryM(3, (1, 2), random_unitary(2))
    m = cu.inflate()
    u = UnitaryM.deflate(m)
    assert u.core == (1, 2), f'Core indexes is unexpected {u.core}'


def test_UnitaryM_matmul_identical_cores():
    core = (1, 2)
    a = UnitaryM(3, core, random_unitary(2))
    b = UnitaryM(3, core, random_unitary(2))
    c = a @ b
    assert c.core == core, f'Core indexes is unexpected {c.core}'
    assert np.allclose(c.matrix, a.matrix @ b.matrix)


def test_UnitaryM_matmul_diff_cores():
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
def test_UnitaryM_is_singlet(dim, core, size, expected):
    u = UnitaryM(dim, core, random_unitary(size))
    assert u.issinglet() == expected, f'Unexpected {u}'


def test_CUnitary_init():
    m = random_unitary(2)
    controls = (QType.CONTROL1, QType.CONTROL1, QType.TARGET)
    cu = CUnitary(m, controls)
    # print(formatter.tostr(cu.inflate()))
    assert tuple(cu.unitary.core) == (6, 7), f'Core indexes is unexpected {cu.unitary.core}'


def test_CUnitary_init_qontroller():
    m = random_unitary(2)
    controller = Qontroller((QType.CONTROL1, QType.CONTROL1, QType.TARGET))
    cu = CUnitary(m, controller)
    assert cu.controller == controller


def test_CUnitary_init_qspace_seq():
    m = random_unitary(2)
    controls = (QType.CONTROL1, QType.CONTROL1, QType.TARGET)
    qspace = list(range(3))
    random.shuffle(qspace)
    cu = CUnitary(m, controls, qspace)
    assert cu.qspace.qids == qspace


def test_CUnitary_init_qspace_numpy_array():
    k = random.randint(2, 5)
    t = random.randint(1, k)
    controls = random_control(k, t)
    qids = np.random.choice(1 << k, size=k, replace=False)
    cu = CUnitary(random_unitary(2), controls, qspace=qids)
    assert cu.qspace.qids == qids.tolist()


def test_CUnitary_init_qspace_obj():
    m = random_unitary(2)
    controls = (QType.CONTROL1, QType.CONTROL1, QType.TARGET)
    qspace = QSpace(list(range(3)))
    cu = CUnitary(m, controls, qspace)
    assert cu.qspace.qids == list(range(3))


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


def test_CUnitary_UnivGate_Z():
    gate = UnivGate.Z
    control = (QType.CONTROL1, QType.CONTROL0, QType.CONTROL0, QType.TARGET)
    cu = CUnitary(gate.mat, control)
    expected = np.eye(16, dtype=np.complexfloating)
    expected[8:10, 8:10] = gate.mat
    # print()
    # print(formatter.tostr(expected))
    u = cu.inflate()
    assert np.allclose(u, expected), f'Expected:\n{formatter.tostr(expected)},\nActual:\n{formatter.tostr(u)}'


def test_CUnitary_sorted_4x4():
    m = random_unitary(2)
    controls = (QType.TARGET, QType.CONTROL1)
    qids = [3, 0]
    cu = CUnitary(m, controls, qids)
    # print()
    # print(formatter.tostr(cu.inflate()))
    assert tuple(cu.unitary.core) == (1, 3), f'Core indexes is unexpected {cu.unitary.core}'
    sorted_cu = cu.sorted()
    assert tuple(sorted_cu.unitary.core) == (2, 3), f'Core indexes is unexpected {sorted_cu.unitary.core}'
    assert np.allclose(sorted_cu.inflate(), cu.inflate())


def test_CUnitary_sorted_noop():
    m = random_unitary(2)
    controls = (QType.CONTROL1, QType.CONTROL1, QType.TARGET)
    qids = list(range(3))
    expected = CUnitary(m, controls, qids)
    # print()
    # print(formatter.tostr(expected.inflate()))
    actual = expected.sorted()
    assert actual == expected


def test_CUnitary_sorted_8x8():
    m = random_unitary(2)
    controls = (QType.CONTROL1, QType.CONTROL1, QType.TARGET)
    qids = [2, 0, 1]
    cu = CUnitary(m, controls, qids)
    expected = cu.inflate()
    # print()
    # print(formatter.tostr(expected))

    # execute
    sorted_cu = cu.sorted()

    assert tuple(sorted_cu.unitary.core) == (5, 7), f'Core indexes is unexpected {sorted_cu.unitary.core}'
    actual = sorted_cu.inflate()
    assert np.allclose(actual, expected)


def test_CUnitary_expand():
    m = random_unitary(2)
    controls = (QType.TARGET, QType.CONTROL1)
    qids = [1, 0]
    cu = CUnitary(m, controls, qids)
    univ = list(range(3))
    ex = cu.expand(univ)
    # print()
    # print(formatter.tostr(ex.inflate()))
    expected = np.block([[np.eye(4), np.zeros((4, 4))], [np.zeros((4, 4)), kron(cu.unitary.matrix, np.eye(2))]])
    assert np.allclose(ex.inflate(), expected)


def test_CUnitary_expand_random():
    for _ in range(10):
        # print(f'Test {_}th round')
        n = random.randint(2, 5)
        k = random.randint(2, n)
        t = random.randint(1, k)
        m = random_unitary(1 << t)
        controls = random_control(k, t)
        qids = random.sample(range(n), k)
        cu = CUnitary(m, controls, qids)
        univ = list(range(n))

        # execute
        ex = cu.expand(univ)
        assert ex
        # print()
        # print(formatter.tostr(ex.inflate()))


def test_CUnitary_matmul_identical_controls():
    k = random.randint(2, 5)
    t = random.randint(1, k)
    controls = random_control(k, t)
    a = CUnitary(random_unitary(1 << t), controls)
    b = CUnitary(random_unitary(1 << t), controls)
    c = a @ b
    assert np.allclose(c.unitary.matrix, a.unitary.matrix @ b.unitary.matrix)
    assert c.qspace.qids == a.qspace.qids == b.qspace.qids


def test_CUnitary_matmul_identical_qspace_diff_controls():
    k = random.randint(2, 5)
    t = random.randint(1, k)
    a = CUnitary(random_unitary(2), random_control(k, t))
    b = CUnitary(random_unitary(2), random_control(k, t))

    # execute
    c = a @ b
    # print()
    # print(formatter.tostr(c.inflate()))
    expected = a.inflate() @ b.inflate()
    assert np.allclose(c.inflate(), expected)
    assert c.qspace.qids == a.qspace.qids == b.qspace.qids


def test_CUnitary_matmul_verify_qspace():
    controls = [QType.TARGET, QType.TARGET]  # all targets, no control
    qid1 = [3, 0]
    a = CUnitary(random_unitary(4), controls, qspace=qid1)
    qid2 = [1, 0]
    b = CUnitary(random_unitary(4), controls, qspace=qid2)

    # execute
    c = a @ b

    assert a.qspace.qids == qid1
    assert b.qspace.qids == qid2
    assert c.qspace.qids == sorted(set(qid1 + qid2))


def test_CUnitary_expand_eqiv_left_kron():
    controls = [QType.TARGET, QType.TARGET]  # all targets, no control
    qid1 = [1, 3]
    unitary1 = random_unitary(4)
    print('unitary1')
    print(formatter.tostr(unitary1))
    a = CUnitary(unitary1, controls, qspace=qid1)
    print('a')
    print(formatter.tostr(a.inflate()))

    univ = [0, 1, 3]
    actual = a.expand(univ)
    # print('actual')
    # print(formatter.tostr(actual.inflate()))
    expected = kron(np.eye(2), a.inflate())
    assert np.allclose(actual.inflate(), expected)


def test_CUnitary_expand_eqiv_inter_product():
    controls = [QType.TARGET, QType.TARGET]  # all targets, no control
    qid1 = [3, 0]
    unitary1 = random_unitary(4)
    print('unitary1')
    print(formatter.tostr(unitary1))
    a = CUnitary(unitary1, controls, qspace=qid1)
    print('a')
    print(formatter.tostr(a.inflate()))

    univ = [0, 1, 3]
    actual = a.expand(univ)
    print('actual')
    print(formatter.tostr(actual.inflate()))
    expected = inter_product(a.sorted().inflate(), np.eye(2), 2)
    print('expected')
    print(formatter.tostr(expected))
    assert np.allclose(actual.inflate(), expected)


def test_CUnitary_matmul_uncontrolled_diff_qspace():
    controls = [QType.TARGET, QType.TARGET]  # all targets, no control
    a = CUnitary(random_unitary(4), controls, qspace=[1, 3])
    # print('a')
    # print(formatter.tostr(a.inflate()))

    b = CUnitary(random_unitary(4), controls, qspace=[3, 0])
    # print('b')
    # print(formatter.tostr(b.inflate()))

    # execute
    c = a @ b
    # print('c')
    # print(formatter.tostr(c.inflate()))

    univ = sorted(set(a.qspace.qids + b.qspace.qids))
    assert c.qspace.qids == univ

    ai = kron(np.eye(2), a.sorted().inflate())
    # print('ai')
    # print(formatter.tostr(ai))
    bi = inter_product(b.sorted().inflate(), np.eye(2), 2)
    # print('bi')
    # print(formatter.tostr(bi))
    expected = ai @ bi
    # print('expected')
    # print(formatter.tostr(expected))
    assert np.allclose(c.inflate(), expected)


def test_CUnitary_matmul_random():
    for _ in range(10):
        # print(f'Test {_}th round')
        k = random.randint(2, 5)
        t = random.randint(1, k)
        rqids = lambda: np.random.choice(1 << k, size=k, replace=False)
        a = CUnitary(random_unitary(1 << t), random_control(k, t), qspace=rqids())
        b = CUnitary(random_unitary(1 << t), random_control(k, t), qspace=rqids())

        # execute
        c = a @ b
        assert c is not None
        # print()
        # print(formatter.tostr(c.inflate()))


def test_CUnitary_matmul_diff_cores():
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
