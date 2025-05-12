import random

import numpy as np
import pytest
from numpy import kron

from quompiler.construct.cgate import CtrlGate
from quompiler.construct.qontroller import Qontroller
from quompiler.construct.qspace import QSpace
from quompiler.construct.types import UnivGate, QType
from quompiler.construct.unitary import UnitaryM
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.inter_product import inter_product
from quompiler.utils.mgen import random_unitary, random_indexes, random_UnitaryM, random_control

formatter = MatrixFormatter(precision=2)


def test_convert_invalid_dimension():
    cu = UnitaryM(3, (1, 2), random_unitary(2))
    with pytest.raises(AssertionError) as exc:
        CtrlGate.convert(cu)


def test_convert_no_inflation():
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
        c = CtrlGate.convert(u)
        assert np.array_equal(c.unitary.matrix, u.matrix)


def test_convert_verify_dimension():
    for _ in range(10):
        n = random.randint(1, 5)
        dim = 1 << n
        core = random.randint(2, dim)
        indexes = random.sample(range(dim), core)
        u = random_UnitaryM(dim, indexes)
        c = CtrlGate.convert(u)
        assert c.unitary.dimension == dim


def test_convert_expansion():
    for _ in range(1):
        print(f'Test round {_}')
        n = random.randint(1, 5)
        dim = 1 << n
        core = random.randint(2, dim)
        indexes = random.sample(range(dim), core)
        unitM = random_UnitaryM(dim, indexes)
        ctrlM = CtrlGate.convert(unitM)
        assert np.allclose(ctrlM.inflate(), unitM.inflate()), f'ctrlM=\n{formatter.tostr(ctrlM.inflate())}\nexpected=\n{formatter.tostr(unitM.inflate())}'


def test_convert_verify_inflation_invariance():
    dimension = 4
    k = random.randint(2, dimension)
    core = sorted(random_indexes(dimension, k))
    u = random_UnitaryM(dimension, core)
    c = CtrlGate.convert(u)
    assert np.allclose(c.inflate(), u.inflate()), f'actual=\n{formatter.tostr(c.inflate())}\nexpected=\n{formatter.tostr(u.inflate())}'


def test_convert():
    for _ in range(20):
        print(f'test round {_}')
        n = random.randint(1, 5)
        dim = 1 << n
        core = random.randint(2, dim)
        indexes = random.sample(range(dim), core)
        u = random_UnitaryM(dim, indexes)
        c = CtrlGate.convert(u)
        assert c
        # print()
        # print(formatter.tostr(c.matrix))


def test_init():
    m = random_unitary(2)
    controls = (QType.CONTROL1, QType.CONTROL1, QType.TARGET)
    cu = CtrlGate(m, controls)
    # print(formatter.tostr(cu.inflate()))
    assert tuple(cu.unitary.core) == (6, 7), f'Core indexes is unexpected {cu.unitary.core}'


def test_init_qontroller():
    m = random_unitary(2)
    controller = Qontroller((QType.CONTROL1, QType.CONTROL1, QType.TARGET))
    cu = CtrlGate(m, controller)
    assert cu.controller == controller


def test_init_qspace_seq():
    m = random_unitary(2)
    controls = (QType.CONTROL1, QType.CONTROL1, QType.TARGET)
    qspace = list(range(3))
    random.shuffle(qspace)
    cu = CtrlGate(m, controls, qspace)
    assert cu.qspace.qids == qspace


def test_init_qspace_numpy_array():
    k = random.randint(2, 5)
    t = random.randint(1, k)
    controls = random_control(k, t)
    qids = np.random.choice(1 << k, size=k, replace=False)
    cu = CtrlGate(random_unitary(2), controls, qspace=qids)
    assert cu.qspace.qids == qids.tolist()


def test_init_qspace_obj():
    m = random_unitary(2)
    controls = (QType.CONTROL1, QType.CONTROL1, QType.TARGET)
    qspace = QSpace(list(range(3)))
    cu = CtrlGate(m, controls, qspace)
    assert cu.qspace.qids == list(range(3))


def test_univ_Y():
    gate = UnivGate.Y
    control = (QType.CONTROL0, QType.CONTROL0, QType.TARGET)
    cu = CtrlGate(gate.matrix, control)
    expected = np.eye(8, dtype=np.complexfloating)
    expected[:2, :2] = gate.matrix
    # print()
    # print(formatter.tostr(expected))
    u = cu.inflate()
    assert np.allclose(u, expected), f'Expected:\n{formatter.tostr(expected)},\nActual:\n{formatter.tostr(u)}'


def test_UnivGate_Z():
    gate = UnivGate.Z
    control = (QType.CONTROL1, QType.CONTROL0, QType.CONTROL0, QType.TARGET)
    cu = CtrlGate(gate.matrix, control)
    expected = np.eye(16, dtype=np.complexfloating)
    expected[8:10, 8:10] = gate.matrix
    # print()
    # print(formatter.tostr(expected))
    u = cu.inflate()
    assert np.allclose(u, expected), f'Expected:\n{formatter.tostr(expected)},\nActual:\n{formatter.tostr(u)}'


def test_sorted_4x4():
    m = random_unitary(2)
    controls = (QType.TARGET, QType.CONTROL1)
    qids = [3, 0]
    cu = CtrlGate(m, controls, qids)
    # print()
    # print(formatter.tostr(cu.inflate()))
    assert tuple(cu.unitary.core) == (1, 3), f'Core indexes is unexpected {cu.unitary.core}'
    sorted_cu = cu.sorted()
    assert tuple(sorted_cu.unitary.core) == (2, 3), f'Core indexes is unexpected {sorted_cu.unitary.core}'
    assert np.allclose(sorted_cu.inflate(), cu.inflate())


def test_sorted_noop():
    m = random_unitary(2)
    controls = (QType.CONTROL1, QType.CONTROL1, QType.TARGET)
    qids = list(range(3))
    expected = CtrlGate(m, controls, qids)
    print('expected\n')
    print(formatter.tostr(expected.inflate()))
    actual = expected.sorted()
    print('actual\n')
    print(formatter.tostr(actual.inflate()))
    assert np.allclose(actual.inflate(), expected.inflate())


def test_sorted_8x8():
    m = random_unitary(2)
    controls = (QType.CONTROL1, QType.CONTROL1, QType.TARGET)
    qids = [2, 0, 1]
    cu = CtrlGate(m, controls, qids)
    expected = cu.inflate()
    # print()
    # print(formatter.tostr(expected))

    # execute
    sorted_cu = cu.sorted()

    assert tuple(sorted_cu.unitary.core) == (5, 7), f'Core indexes is unexpected {sorted_cu.unitary.core}'
    actual = sorted_cu.inflate()
    assert np.allclose(actual, expected)


def test_expand():
    m = random_unitary(2)
    controls = (QType.TARGET, QType.CONTROL1)
    qids = [1, 0]
    cu = CtrlGate(m, controls, qids)
    univ = list(range(3))
    ex = cu.expand(univ)
    # print()
    # print(formatter.tostr(ex.inflate()))
    expected = np.block([[np.eye(4), np.zeros((4, 4))], [np.zeros((4, 4)), kron(cu.unitary.matrix, np.eye(2))]])
    assert np.allclose(ex.inflate(), expected)


def test_expand_random():
    for _ in range(10):
        # print(f'Test {_}th round')
        n = random.randint(2, 5)
        k = random.randint(2, n)
        t = random.randint(1, k)
        m = random_unitary(1 << t)
        controls = random_control(k, t)
        qids = random.sample(range(n), k)
        cu = CtrlGate(m, controls, qids)
        univ = list(range(n))

        # execute
        ex = cu.expand(univ)
        assert ex
        # print()
        # print(formatter.tostr(ex.inflate()))


def test_matmul_identical_controls():
    k = random.randint(2, 5)
    t = random.randint(1, k)
    controls = random_control(k, t)
    a = CtrlGate(random_unitary(1 << t), controls)
    b = CtrlGate(random_unitary(1 << t), controls)
    c = a @ b
    assert np.allclose(c.unitary.matrix, a.unitary.matrix @ b.unitary.matrix)
    assert c.qspace.qids == a.qspace.qids == b.qspace.qids


def test_matmul_identical_qspace_diff_controls():
    k = random.randint(2, 5)
    t = random.randint(1, k)
    a = CtrlGate(random_unitary(2), random_control(k, t))
    b = CtrlGate(random_unitary(2), random_control(k, t))

    # execute
    c = a @ b
    # print()
    # print(formatter.tostr(c.inflate()))
    expected = a.inflate() @ b.inflate()
    assert np.allclose(c.inflate(), expected)
    assert c.qspace.qids == a.qspace.qids == b.qspace.qids


def test_matmul_verify_qspace():
    controls = [QType.TARGET, QType.TARGET]  # all targets, no control
    qid1 = [3, 0]
    a = CtrlGate(random_unitary(4), controls, qspace=qid1)
    qid2 = [1, 0]
    b = CtrlGate(random_unitary(4), controls, qspace=qid2)

    # execute
    c = a @ b

    assert a.qspace.qids == qid1
    assert b.qspace.qids == qid2
    assert c.qspace.qids == sorted(set(qid1 + qid2))


def test_expand_eqiv_left_kron():
    controls = [QType.TARGET, QType.TARGET]  # all targets, no control
    qid1 = [1, 3]
    unitary1 = random_unitary(4)
    print('unitary1')
    print(formatter.tostr(unitary1))
    a = CtrlGate(unitary1, controls, qspace=qid1)
    print('a')
    print(formatter.tostr(a.inflate()))

    univ = [0, 1, 3]
    actual = a.expand(univ)
    # print('actual')
    # print(formatter.tostr(actual.inflate()))
    expected = kron(np.eye(2), a.inflate())
    assert np.allclose(actual.inflate(), expected)


def test_expand_eqiv_inter_product():
    controls = [QType.TARGET, QType.TARGET]  # all targets, no control
    qid1 = [3, 0]
    unitary1 = random_unitary(4)
    print('unitary1')
    print(formatter.tostr(unitary1))
    a = CtrlGate(unitary1, controls, qspace=qid1)
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


def test_matmul_uncontrolled_diff_qspace():
    controls = [QType.TARGET, QType.TARGET]  # all targets, no control
    a = CtrlGate(random_unitary(4), controls, qspace=[1, 3])
    # print('a')
    # print(formatter.tostr(a.inflate()))

    b = CtrlGate(random_unitary(4), controls, qspace=[3, 0])
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


def test_matmul_random():
    for _ in range(10):
        # print(f'Test {_}th round')
        k = random.randint(2, 5)
        t = random.randint(1, k)
        rqids = lambda: np.random.choice(1 << k, size=k, replace=False)
        a = CtrlGate(random_unitary(1 << t), random_control(k, t), qspace=rqids())
        b = CtrlGate(random_unitary(1 << t), random_control(k, t), qspace=rqids())

        # execute
        c = a @ b
        assert c is not None
        # print()
        # print(formatter.tostr(c.inflate()))


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
