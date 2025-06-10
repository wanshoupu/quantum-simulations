import random

import numpy as np
import pytest
from numpy import kron

from quompiler.construct.cgate import CtrlGate
from quompiler.construct.qontroller import ctrl2core
from quompiler.construct.qspace import Qubit
from quompiler.construct.su2gate import RGate
from quompiler.construct.types import UnivGate, QType, PrincipalAxis
from quompiler.construct.unitary import UnitaryM
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.inter_product import mykron, qproject
from quompiler.utils.mgen import random_unitary, random_indexes, random_UnitaryM, random_control, random_ctrlgate, random_CtrlGate, random_state, random_phase, random_rgate
from quompiler.utils.permute import Permuter

formatter = MatrixFormatter(precision=2)


def test_convert_invalid_dimension():
    cu = UnitaryM(3, (1, 2), random_unitary(2))
    with pytest.raises(AssertionError):
        CtrlGate.convert(cu)


def test_convert_no_inflation():
    for _ in range(10):
        n = random.randint(1, 5)
        dim = 1 << n
        k = random.randint(1, n)
        control = random_control(n, k)
        core = ctrl2core(control)
        m = random_unitary(1 << k)
        assert m.shape[0] == 1 << k
        assert len(core) == 1 << k
        u = UnitaryM(dim, core, m)

        # execute
        c = CtrlGate.convert(u)
        assert np.array_equal(c.matrix(), u.matrix)


def test_convert_verify_dimension():
    for _ in range(10):
        n = random.randint(1, 5)
        dim = 1 << n
        core = random.randint(2, dim)
        indexes = random.sample(range(dim), core)
        u = random_UnitaryM(dim, indexes)
        c = CtrlGate.convert(u)
        assert c.order() == dim


def test_convert_expansion():
    for _ in range(1):
        # print(f'Test round {_}')
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
        # print(f'test round {_}')
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
    assert not cu.is_std()
    assert tuple(cu.core()) == (6, 7), f'Core indexes is unexpected {cu.core()}'


def test_init_qspace_seq():
    m = random_unitary(2)
    controls = (QType.CONTROL1, QType.CONTROL1, QType.TARGET)
    qspace = [Qubit(i) for i in range(3)]
    random.shuffle(qspace)
    cu = CtrlGate(m, controls, qspace)
    assert cu.qspace == qspace


def test_init_qspace_numpy_array():
    k = random.randint(2, 5)
    t = random.randint(1, k)
    controls = random_control(k, t)
    qids = np.random.choice(1 << k, size=k, replace=False)
    cu = CtrlGate(random_unitary(2), controls, qspace=[Qubit(q) for q in qids])
    assert cu.qspace == [Qubit(q) for q in qids]


def test_init_qspace_obj():
    m = random_unitary(2)
    controls = (QType.CONTROL1, QType.CONTROL1, QType.TARGET)
    qspace = [Qubit(i) for i in range(3)]
    cu = CtrlGate(m, controls, qspace)
    assert cu.qspace == [Qubit(i) for i in range(3)]


def test_init_rgate():
    rg = random_rgate()
    gate = CtrlGate(rg, [QType.TARGET], [Qubit(10)])
    actual = gate.matrix()
    expected = rg.matrix
    assert np.allclose(actual, expected)


def test_univ_Y():
    gate = UnivGate.Y
    control = (QType.CONTROL0, QType.CONTROL0, QType.TARGET)
    cu = CtrlGate(np.array(gate), control)
    expected = np.eye(8, dtype=np.complexfloating)
    expected[:2, :2] = np.array(gate)
    # print()
    # print(formatter.tostr(expected))
    u = cu.inflate()
    assert np.allclose(u, expected), f'Expected:\n{formatter.tostr(expected)},\nActual:\n{formatter.tostr(u)}'


def test_UnivGate_Z():
    gate = UnivGate.Z
    control = (QType.CONTROL1, QType.CONTROL0, QType.CONTROL0, QType.TARGET)
    cu = CtrlGate(np.array(gate), control)
    expected = np.eye(16, dtype=np.complexfloating)
    expected[8:10, 8:10] = np.array(gate)
    # print()
    # print(formatter.tostr(expected))
    u = cu.inflate()
    assert np.allclose(u, expected), f'Expected:\n{formatter.tostr(expected)},\nActual:\n{formatter.tostr(u)}'


@pytest.mark.parametrize("gate", list(UnivGate))
def test_proportional_univ_gate(gate):
    prop_factor = random_phase()
    actual = CtrlGate(np.array(gate) * prop_factor, random_control(3, 1))
    assert actual.is_std()
    assert np.isclose(actual.phase(), prop_factor)


@pytest.mark.parametrize("axis", list(PrincipalAxis))
def test_proportional_rgate(axis):
    gate = RGate(np.pi, axis)
    actual = CtrlGate(gate, random_control(3, 1))
    assert actual.is_std()
    assert np.isclose(actual.phase(), -1j)


def test_sorted_noop():
    m = random_unitary(2)
    controls = (QType.CONTROL1, QType.CONTROL1, QType.TARGET)
    qids = [Qubit(i) for i in range(3)]
    expected = CtrlGate(m, controls, qids)
    # print('expected\n')
    # print(formatter.tostr(expected.inflate()))
    actual = expected.sorted()
    # print('actual\n')
    # print(formatter.tostr(actual.inflate()))
    assert np.allclose(actual.inflate(), expected.inflate())


def test_sorted_2_targets():
    m = random_unitary(4)
    controls = (QType.TARGET, QType.TARGET)
    qids = [Qubit(3), Qubit(0)]
    cu = CtrlGate(m, controls, qids)
    # print()
    # print(formatter.tostr(cu.inflate()))
    actual = cu.sorted()
    assert tuple(cu.core()) == tuple(range(4)), f'Core indexes is unexpected {cu.core()}'
    assert np.array_equal(cu.inflate(), m)
    # print()
    # print(formatter.tostr(cu.inflate()))

    indexes = [0, 2, 1, 3]
    expected = cu.inflate()[np.ix_(indexes, indexes)]
    assert np.array_equal(actual.inflate(), expected)


def test_sorted_with_ctrl():
    m = random_unitary(2)
    controls = (QType.TARGET, QType.CONTROL1)
    qids = [Qubit(3), Qubit(0)]
    cu = CtrlGate(m, controls, qids)
    # print()
    # print(formatter.tostr(cu.inflate()))
    assert tuple(cu.core()) == (1, 3), f'Core indexes is unexpected {cu.core()}'
    actual = cu.sorted()
    assert tuple(actual.core()) == (2, 3), f'Core indexes is unexpected {actual.core()}'

    sh = Permuter(qids)
    indexes = sh.bitsortall(range(cu.order()))
    expected = cu.inflate()[np.ix_(indexes, indexes)]
    assert np.allclose(actual.inflate(), expected)


def test_sorted_ctrl_2target():
    m = random_unitary(4)
    controls = (QType.CONTROL1, QType.TARGET, QType.TARGET)
    qids = [Qubit(2), Qubit(10), Qubit(1)]
    original = CtrlGate(m, controls, qids)
    # print('original\n')
    # print(formatter.tostr(original.inflate()))

    # execute
    actual = original.sorted()
    # print('actual\n')
    # print(formatter.tostr(actual.inflate()))

    perm = Permuter.from_permute(qids, list(actual.qspace))
    eqiv_original = np.eye(original.order(), dtype=np.complexfloating)
    eqiv_original[np.ix_(original.core(), original.core())] = original.matrix()
    assert np.array_equal(eqiv_original, original.inflate())

    permuted_core = perm.bitpermuteall(original.core())
    expected = np.eye(original.order(), dtype=np.complexfloating)
    expected[np.ix_(permuted_core, permuted_core)] = original.matrix()
    # print('expected\n')
    # print(formatter.tostr(expected))
    assert np.allclose(actual.inflate(), expected)


def test_sorted_by_ctrl():
    n = random.randint(1, 4)
    t = random.randint(1, n)
    m = random_unitary(1 << t)
    controls = random_control(n, t)
    qids = [Qubit(i) for i in np.random.choice(100, size=n, replace=False)]
    cu = CtrlGate(m, controls, qids)

    # execute
    sorting = np.argsort(cu.controls())
    sorted_cu = cu.sorted(sorting=sorting)
    assert sorted_cu.controls()[:t] == [QType.TARGET] * t
    ctrls = sorted_cu.controls()[t:]
    assert all(c in QType(0x110) for c in ctrls)


def test_expand_target_in_order():
    m = random_unitary(2)
    controls = (QType.TARGET, QType.CONTROL1)
    qids = [Qubit(1), Qubit(0)]
    cu = CtrlGate(m, controls, qids)
    extended_qspace = [Qubit(2)]
    actual = cu.expand(extended_qspace).sorted()
    # print()
    # print(formatter.tostr(actual.inflate()))
    expected = np.block([[np.eye(4), np.zeros((4, 4))], [np.zeros((4, 4)), kron(cu.matrix(), np.eye(2))]])
    # print()
    # print(formatter.tostr(expected))
    assert np.allclose(actual.inflate(), expected)


def test_expand_target_out_of_order():
    m = random_unitary(2)
    controls = (QType.TARGET, QType.CONTROL1)
    qids = [Qubit(2), Qubit(0)]
    cu = CtrlGate(m, controls, qids)
    extended_qspace = [Qubit(1)]
    actual = cu.expand(extended_qspace)
    # print()
    # print(formatter.tostr(actual.inflate()))
    expected = CtrlGate(np.kron(cu.matrix(), np.eye(2)), list(controls) + [QType.TARGET], qids + extended_qspace)
    # print()
    # print(formatter.tostr(expected.inflate()))
    assert np.allclose(actual.inflate(), expected.inflate())


def test_expand_ctrl0():
    m = random_unitary(2)
    controls = (QType.TARGET, QType.CONTROL1)
    qids = [Qubit(1), Qubit(0)]
    cu = CtrlGate(m, controls, qids)
    actual = cu.expand([Qubit(2)], [QType.CONTROL0]).inflate()
    # print()
    # print(formatter.tostr(actual))
    expected = extend_helper(cu.inflate(), 0)
    # print()
    # print(formatter.tostr(expected))
    assert np.allclose(actual, expected)


def test_expand_ctrl1():
    m = random_unitary(2)
    controls = (QType.TARGET, QType.CONTROL1)
    qids = [Qubit(1), Qubit(0)]
    cu = CtrlGate(m, controls, qids)
    actual = cu.expand([Qubit(2)], [QType.CONTROL1]).inflate()
    # print()
    # print(formatter.tostr(actual))
    expected = extend_helper(cu.inflate(), 1)
    # print()
    # print(formatter.tostr(expected))
    assert np.allclose(actual, expected)


def test_expand_invalid_qubit():
    k = random.randint(2, 5)
    t = random.randint(1, k)
    cu = random_ctrlgate(k, t)
    qs = cu.qids()
    # execute
    with pytest.raises(AssertionError):
        cu.expand([Qubit(qs[0].qid), Qubit(qs[1].qid)])


def test_expand_default_ctrls():
    k = 3
    t = 1
    cu = random_ctrlgate(k, t)
    # print(cu._controller)
    # print('cu\n')
    # print(formatter.tostr(cu.inflate()))

    # execute
    actual = cu.expand([Qubit(k + 1)])
    # print('actual\n')
    # print(formatter.tostr(actual.inflate()))

    ctrls = actual.controls()[len(cu.controls()):]
    assert ctrls == [QType.TARGET] * len(ctrls)


def extend_helper(mat, offset=1):
    length = mat.shape[0]
    indexes = [(i << 1) + offset for i in range(length)]
    result = np.eye(length << 1, dtype=np.complex128)
    result[np.ix_(indexes, indexes)] = mat
    return result


def test_expand_target_random():
    for _ in range(10):
        # print(f'Test {_}th round')
        n = random.randint(2, 5)
        k = random.randint(2, n)
        t = random.randint(1, k)
        original = random_ctrlgate(k, t)
        # print('original\n')
        # print(formatter.tostr(original.inflate()))
        e = random.randint(1, 3)
        univ = list(set(range(n)) - {q.qid for q in original.qspace}) + list(range(n + 1, n + 1 + original.order()))
        extended_qspace = [Qubit(i) for i in random.sample(univ, e)]
        new_ctrls = random.sample([QType.TARGET, QType.CONTROL1, QType.CONTROL0], e)
        # print(f'extended_qspace={extended_qspace},ctrl={new_ctrls}')

        # execute
        actual = original.expand(extended_qspace, new_ctrls)
        assert actual
        # print('actual\n')
        # print(formatter.tostr(actual.inflate()))


def test_expand_eqiv_left_kron():
    controls = [QType.TARGET, QType.TARGET]  # all targets, no control
    qspace = [Qubit(1), Qubit(3)]
    cu = CtrlGate(random_unitary(4), controls, qspace=qspace)
    # print('cu')
    # print(formatter.tostr(cu.inflate()))

    actual = cu.expand([Qubit(0)])
    # print('actual')
    # print(formatter.tostr(actual.inflate()))
    expected = kron(cu.inflate(), np.eye(2))
    assert np.allclose(actual.inflate(), expected)


def test_expand_upon_multi_target():
    controls = [QType.TARGET, QType.TARGET]  # all targets, no control
    qspace = [Qubit(3), Qubit(0)]
    unitary1 = random_unitary(4)
    # print('unitary1')
    # print(formatter.tostr(unitary1))
    a = CtrlGate(unitary1, controls, qspace=qspace)
    # print('a')
    # print(formatter.tostr(a.inflate()))

    actual = a.expand([Qubit(1)])
    # print('actual')
    # print(formatter.tostr(actual.inflate()))
    expected = kron(a.inflate(), np.eye(2))
    # print('expected')
    # print(formatter.tostr(expected))
    assert np.allclose(actual.inflate(), expected)


def test_expand_by_multi_target():
    controls = [QType.TARGET]  # all targets, no control
    qspace = [Qubit(3)]
    a = CtrlGate(random_unitary(2), controls, qspace=qspace)
    # print('a')
    # print(formatter.tostr(a.inflate()))

    actual = a.expand([Qubit(1), Qubit(0)])
    # print('actual')
    # print(formatter.tostr(actual.inflate()))
    expected = kron(a.inflate(), np.eye(4))
    # print('expected')
    # print(formatter.tostr(expected))
    assert np.allclose(actual.inflate(), expected)


@pytest.mark.parametrize("ctrs,qspace,qubits", [
    [[QType.TARGET, QType.CONTROL1, QType.CONTROL1], [Qubit(1), Qubit(0), Qubit(4)], [Qubit(0), Qubit(4)]],
    [[QType.TARGET, QType.CONTROL1], [Qubit(1), Qubit(0)], [Qubit(0)]],
    [[QType.TARGET, QType.CONTROL0], [Qubit(1), Qubit(0)], [Qubit(0)]],
    [[QType.CONTROL1, QType.TARGET], [Qubit(1), Qubit(0)], [Qubit(1)]],
    [[QType.CONTROL1, QType.TARGET, QType.CONTROL0], [Qubit(1), Qubit(0), Qubit(10)], [Qubit(1)]],
    [[QType.CONTROL1, QType.TARGET, QType.CONTROL0], [Qubit(1), Qubit(0), Qubit(10)], [Qubit(1), Qubit(0)]],
])
def test_promote_target_invariant(ctrs, qspace, qubits):
    expected = CtrlGate(random_unitary(2), ctrs, qspace=qspace)
    # print('expected')
    # print(formatter.tostr(expected.inflate()))

    # execute
    actual = expected.promote(qubits)

    # print('actual')
    # print(formatter.tostr(actual.inflate()))
    assert np.allclose(actual.inflate(), expected.inflate())


def test_matmul_identical_controls():
    k = random.randint(2, 5)
    t = random.randint(1, k)
    controls = random_control(k, t)
    a = CtrlGate(random_unitary(1 << t), controls)
    b = CtrlGate(random_unitary(1 << t), controls)
    c = a @ b
    assert np.allclose(c.matrix(), a.matrix() @ b.matrix())
    assert c.qspace == a.qspace == b.qspace


@pytest.mark.parametrize("ctrl1,ctrl2", [
    [[QType.CONTROL1, QType.TARGET], [QType.TARGET, QType.CONTROL0]],
    [[QType.CONTROL1, QType.TARGET], [QType.TARGET, QType.TARGET]],
    [[QType.CONTROL1, QType.TARGET, QType.CONTROL0], [QType.TARGET, QType.TARGET, QType.TARGET]],
    [[QType.CONTROL1, QType.TARGET, QType.CONTROL0], [QType.TARGET, QType.CONTROL1, QType.TARGET]],
])
def test_matmul_identical_qspace_diff_controls(ctrl1, ctrl2):
    target_count1 = ctrl1.count(QType.TARGET)
    target_count2 = ctrl2.count(QType.TARGET)
    a = CtrlGate(random_unitary(1 << target_count1), ctrl1)
    b = CtrlGate(random_unitary(1 << target_count2), ctrl2)

    # execute
    actual = a @ b
    # print()
    # print(formatter.tostr(actual.inflate()))
    assert actual.qspace == a.qspace == b.qspace
    expected = a.inflate() @ b.inflate()
    assert np.allclose(actual.inflate(), expected)


def test_matmul_rgate_ndarray_same_qspace():
    rg = random_rgate()
    gate1 = CtrlGate(rg, [QType.TARGET], [Qubit(10)])
    mat = random_unitary(2)
    gate2 = CtrlGate(mat, [QType.TARGET], [Qubit(10)])
    actual = gate1 @ gate2
    expected = rg @ mat
    assert np.allclose(actual.matrix(), expected)


def test_matmul_rgate_ndarray_diff_qspace():
    rg = random_rgate()
    gate1 = CtrlGate(rg, [QType.TARGET], [Qubit(10)])
    mat = random_unitary(2)
    gate2 = CtrlGate(mat, [QType.TARGET, QType.CONTROL1], [Qubit(10), Qubit(1)])
    actual = gate1 @ gate2
    expected = mykron(rg, np.eye(2)) @ gate2.inflate()
    assert np.allclose(actual.matrix(), expected)


def test_matmul_qs_1_6_qs_1_7_6():
    for _ in range(1):
        # print(f'Test {_}th round')
        a = random_CtrlGate(random_control(2, 1), [Qubit(1), Qubit(6)])
        # print(f'a ctr = {a.controls}\n')
        # print(formatter.tostr(a.inflate()))
        b = random_CtrlGate(random_control(3, 2), [Qubit(1), Qubit(7), Qubit(6)])

        # execute
        actual = a @ b
        # print('actual\n')
        # print(formatter.tostr(actual.inflate()))
        expected = kron(a.inflate(), np.eye(2)) @ b.sorted().inflate()
        # print('expected\n')
        # print(formatter.tostr(expected))
        assert np.allclose(actual.sorted().inflate(), expected)


def test_matmul_qs_1_0_4_qs_1():
    for _ in range(10):
        # print(f'Test {_}th round')
        a = random_CtrlGate(random_control(3, 1), [Qubit(1), Qubit(0), Qubit(4)])
        # print(f'a ctr = {a.controls}\n')
        # print(formatter.tostr(a.inflate()))
        b = random_CtrlGate(random_control(1, 1), [Qubit(1)])

        # execute
        actual = a @ b
        # print('actual\n')
        # print(formatter.tostr(actual.inflate()))
        expected = a.sorted().inflate() @ mykron(np.eye(2), b.matrix(), np.eye(2))
        # print('expected\n')
        # print(formatter.tostr(expected))
        assert np.allclose(actual.sorted().inflate(), expected)


def test_matmul_qs_1_0_qs_1_7():
    for _ in range(10):
        a = random_CtrlGate(random_control(2, 1), [Qubit(1), Qubit(0)])
        b = random_CtrlGate(random_control(2, 1), [Qubit(1), Qubit(7)])

        # execute
        actual = a @ b
        # print('actual\n')
        # print(formatter.tostr(actual.inflate()))
        expected = kron(a.sorted().inflate(), np.eye(2)) @ kron(np.eye(2), b.sorted().inflate())
        assert np.allclose(actual.inflate(), expected)


def test_matmul_permute_qspace_controls():
    # same qspace and controls but permuted. Sorting should be used.
    ctrl1 = [QType.CONTROL0, QType.TARGET]
    ctrl2 = [QType.TARGET, QType.CONTROL0]
    qspace1 = [Qubit(1), Qubit(0)]
    qspace2 = [Qubit(0), Qubit(1)]
    a = random_CtrlGate(ctrl1, qspace1)
    b = random_CtrlGate(ctrl2, qspace2)

    # execute
    actual = a @ b
    # print('actual\n')
    # print(formatter.tostr(actual.inflate()))
    expected = a.sorted().inflate() @ b.sorted().inflate()
    assert np.allclose(actual.inflate(), expected)


def test_matmul_target_ctrl_mult_target():
    # same qspace and controls but permuted. Sorting should be used.
    ctrl1 = [QType.CONTROL1, QType.TARGET]
    qspace1 = [Qubit(0), Qubit(1)]
    a = random_CtrlGate(ctrl1, qspace1)
    b = random_CtrlGate([QType.TARGET], [Qubit(1)])

    # execute
    actual = a @ b
    # print('actual\n')
    # print(formatter.tostr(actual.inflate()))


def test_matmul_random():
    univ = [Qubit(i) for i in range(10)]
    rcg = lambda k, t: CtrlGate(random_unitary(1 << t), random_control(k, t), qspace=(random.sample(univ, k)))
    for _ in range(10):
        k1, k2 = random.randint(1, 5), random.randint(1, 5)
        t1, t2 = random.randint(1, k1), random.randint(1, k2)
        a = rcg(k1, t1)
        b = rcg(k2, t2)

        # execute
        c = a @ b

        # print('actual\n')
        # print(formatter.tostr(c.inflate()))


def test_matmul_verify_qspace():
    controls = [QType.TARGET, QType.TARGET]  # all targets, no control
    qid1 = [Qubit(3), Qubit(0)]
    a = CtrlGate(random_unitary(4), controls, qspace=qid1)
    qid2 = [Qubit(1), Qubit(0)]
    b = CtrlGate(random_unitary(4), controls, qspace=qid2)

    # execute
    c = a @ b

    assert a.qspace == qid1
    assert b.qspace == qid2
    assert c.qspace == sorted(set(qid1 + qid2))


def test_matmul_uncontrolled_diff_qspace():
    controls = [QType.TARGET, QType.TARGET]  # all targets, no control
    a = CtrlGate(random_unitary(4), controls, qspace=[Qubit(1), Qubit(3)])
    # print('a')
    # print(formatter.tostr(a.inflate()))

    b = CtrlGate(random_unitary(4), controls, qspace=[Qubit(3), Qubit(0)])
    # print('b')
    # print(formatter.tostr(b.inflate()))

    # execute
    actual = a @ b
    # print('actual')
    # print(formatter.tostr(actual.inflate()))

    univ = sorted(set(a.qspace + b.qspace))
    assert actual.qspace == univ

    expected = a.expand([Qubit(0)]).sorted() @ b.expand([Qubit(1)]).sorted()
    # print('expected')
    # print(formatter.tostr(expected))
    assert np.allclose(actual.inflate(), expected.inflate())


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


def test_is_idler_promoted_qubits():
    """
    Promoted qubits is usually not idler.
    """
    ctrls = [QType.CONTROL1, QType.TARGET]
    cg = CtrlGate(random_unitary(2), ctrls)
    qubit = cg.qspace[0]
    cg = cg.promote([qubit])

    # execute
    assert not cg.is_idler(qubit)


def test_is_idler_true_case():
    ctrls = [QType.CONTROL1, QType.TARGET, QType.TARGET]
    u = kron(np.eye(2), random_unitary(2))
    cg = CtrlGate(u, ctrls)
    qubit = cg.qspace[0]

    # execute and verify
    assert not cg.is_idler(qubit)


def test_dela_single_ancilla():
    ctrls = [QType.CONTROL1, QType.TARGET, QType.TARGET]
    qspace = [Qubit(10), Qubit(12, ancilla=True), Qubit(7)]
    unitary = random_unitary(2)
    mat = kron(np.eye(2), unitary)
    cg = CtrlGate(mat, ctrls, qspace=qspace)

    # execute
    actual = cg.dela()
    expected = CtrlGate(unitary, ctrls[:2], [qspace[0], qspace[2]])
    assert actual.qspace == expected.qspace
    assert np.allclose(actual.inflate(), expected.inflate())


def test_dela_double_ancilla():
    ctrls = [QType.CONTROL1, QType.TARGET, QType.TARGET, QType.TARGET]
    qspace = [Qubit(10), Qubit(12, ancilla=True), Qubit(7), Qubit(1, ancilla=True)]
    unitary = random_unitary(2)
    mat = mykron(np.eye(2), unitary, np.eye(2))
    cg = CtrlGate(mat, ctrls, qspace=qspace)

    # execute
    actual = cg.dela()

    expected = CtrlGate(unitary, ctrls[:2], [qspace[0], qspace[2]])

    assert actual.qspace == [qspace[0], qspace[2]]
    assert np.allclose(actual.inflate(), expected.inflate())


def test_dela_del_ctrl():
    ctrls = [QType.CONTROL1, QType.TARGET, QType.TARGET, QType.TARGET]
    qspace = [Qubit(10, ancilla=True), Qubit(12), Qubit(7, ancilla=True), Qubit(1)]
    unitary = random_unitary(2)
    mat = mykron(np.eye(2), unitary, np.eye(2))
    cg = CtrlGate(mat, ctrls, qspace=qspace)

    # execute
    actual = cg.dela()
    assert actual.qspace == [qspace[1], qspace[-1]]
    assert np.allclose(actual.inflate(), np.eye(4))


def test_project_invalid_shape():
    with pytest.raises(AssertionError) as e:
        cg = random_ctrlgate(3, 1, 10)
        qubit = cg.qspace[0]
        state = np.array([[1], [0]])
        cg.project(qubit, state)
    assert str(e.value) == 'state vector must be a 1D array of length 2, but got (2, 1).'


def test_project_unnormalized_state():
    with pytest.raises(AssertionError) as e:
        cg = random_ctrlgate(3, 1, 10)
        qubit = cg.qspace[0]
        state = np.array([1, 1])
        cg.project(qubit, state)
    assert str(e.value) == 'state vector must normalized but got [1 1].'


def test_project_invalid_qubit():
    with pytest.raises(AssertionError) as e:
        cg = random_ctrlgate(3, 1, 3)
        qubit = Qubit(10)
        state = np.array([1, 0])
        cg.project(qubit, state)
    assert str(e.value) == 'Qubit q10 not in qspace.'


@pytest.mark.parametrize("ctr, state", [
    [QType.CONTROL0, [0, 1]],
    [QType.CONTROL1, [1, 0]],
])
def test_project_CONTROL_eye(ctr, state):
    ctrls = [ctr, QType.TARGET]
    cg = random_CtrlGate(ctrls)
    actual = cg.project(Qubit(0), np.array(state))
    assert np.array_equal(actual.inflate(), np.eye(2))


@pytest.mark.parametrize("ctr, state", [
    [QType.CONTROL0, [1, 0]],
    [QType.CONTROL1, [0, 1]],
])
def test_project_CONTROL_non_eye(ctr, state):
    ctrls = [ctr, QType.TARGET]
    u = random_unitary(2)
    cg = CtrlGate(u, ctrls)
    actual = cg.project(Qubit(0), np.array(state))
    assert np.array_equal(actual.inflate(), u)


@pytest.mark.parametrize("ctr", [QType.CONTROL0, QType.CONTROL1])
def test_project_2x2_base_eye(ctr):
    state = random_state(2)
    ctrls = [ctr, QType.TARGET]
    cg = random_CtrlGate(ctrls)
    actual = cg.project(Qubit(1), np.array(state))
    assert np.array_equal(actual.inflate(), np.eye(2))


@pytest.mark.parametrize("ctrl,state,qidx", [
    [QType.CONTROL0, [1, 0], 0],
    [QType.CONTROL0, [0, 1], 0],
    [QType.CONTROL0, [1, 0], 1],
    [QType.CONTROL0, [0, 1], 1],
    [QType.CONTROL1, [1, 0], 0],
    [QType.CONTROL1, [0, 1], 0],
    [QType.CONTROL1, [1, 0], 1],
    [QType.CONTROL1, [0, 1], 1],
])
def test_project_4x4_base_states(ctrl, state, qidx):
    """
    :param ctrl: ctr type
    :param state: the state vector to project
    :param qidx: the qubit index
    :return:
    """
    state = np.array(state)
    unitary = random_unitary(2)
    mat = kron(np.eye(2), unitary) if qidx == 0 else kron(unitary, np.eye(2))

    ctrls = [ctrl, QType.TARGET, QType.TARGET]
    cg = CtrlGate(mat, ctrls)
    actual = cg.project(Qubit(1 + qidx), np.array(state))
    # print()
    # print(formatter.tostr(actual.inflate()))

    expected = qproject(mat, qidx, state)
    # print()
    # print(expected)
    assert actual.order() == 4
    assert np.array_equal(actual.matrix(), expected)


def test_invalid_phase():
    u = random_unitary(2)
    phase = -1j + 1
    with pytest.raises(AssertionError) as e:
        CtrlGate(u, random_control(2, 1), phase=phase)
    assert str(e.value) == 'phase factor must be normalized.'


def test_verify_phase_inflation():
    phase = np.sqrt(-1j)
    cg = CtrlGate(np.eye(2), [QType.TARGET], phase=phase)
    actual = cg.inflate()
    assert np.array_equal(actual, np.eye(2) * phase)


def test_verify_phase_matmul():
    phase1, phase2 = random_phase(), random_phase()
    u = random_unitary(2)
    cg1 = CtrlGate(u, [QType.TARGET], phase=phase1)
    cg2 = CtrlGate(UnivGate.X, [QType.TARGET], phase=phase2)
    # execute
    actual = (cg1 @ cg2).inflate()
    # print('actual')
    # print(formatter.tostr(actual))

    # verify
    expected = u @ np.array(UnivGate.X) * phase1 * phase2
    # print('expected')
    # print(formatter.tostr(expected))
    assert np.allclose(actual, expected)


def test_verify_phase_project():
    phase = random_phase()
    ctrls = [QType.CONTROL1, QType.TARGET, QType.TARGET]
    mat = kron(random_unitary(2), np.eye(2))
    cg_phase = CtrlGate(mat, ctrls, phase=phase)
    cg_no_phase = CtrlGate(mat, ctrls)
    actual = cg_phase.project(Qubit(2), np.array([0, 1]))
    # print(f'actual:\n{formatter.tostr(actual.inflate())}')
    expected = cg_no_phase.project(Qubit(2), np.array([0, 1]))
    # print(f'expected:\n{formatter.tostr(expected.inflate())}')
    # verify that the core matrices differ by phase
    cmp_indexes = np.ix_(actual.core(), actual.core())
    assert np.allclose(actual.inflate()[cmp_indexes], expected.inflate()[cmp_indexes] * phase)


def test_verify_phase_sorted():
    phase = random_phase()
    n = 4
    t = 2
    m = random_unitary(1 << t)
    controls = [QType.TARGET, QType.CONTROL1, QType.CONTROL0, QType.TARGET]
    qids = [Qubit(i) for i in np.random.choice(100, size=n, replace=False)]
    cg_phase = CtrlGate(m, controls, qids, phase=phase)
    cg_no_phase = CtrlGate(m, controls, qids)
    sorting = np.argsort(cg_phase.controls())

    # execute
    actual = cg_phase.sorted(sorting=sorting)
    # print(f'actual:\n{formatter.tostr(actual.inflate())}')
    expected = cg_no_phase.sorted(sorting=sorting)
    # print(f'expected:\n{formatter.tostr(expected.inflate())}')
    # verify that the core matrices differ by phase
    cmp_indexes = np.ix_(actual.core(), actual.core())
    assert np.allclose(actual.inflate()[cmp_indexes], expected.inflate()[cmp_indexes] * phase)


def test_verify_phase_expand():
    phase = random_phase()
    k = 3
    t = 1
    controls = random_control(k, t)
    unitary = random_unitary(2)
    cg_phase = CtrlGate(unitary, controls, phase=phase)
    no_phase = CtrlGate(unitary, controls)

    # execute
    actual = cg_phase.expand([Qubit(k + 1)])
    # print(f'actual:\n{formatter.tostr(actual.inflate())}')
    expected = no_phase.expand([Qubit(k + 1)])
    # print(f'expected:\n{formatter.tostr(expected.inflate())}')
    assert actual.qspace == expected.qspace
    # verify that the core matrices differ by phase
    cmp_indexes = np.ix_(actual.core(), actual.core())
    assert np.allclose(actual.inflate()[cmp_indexes], expected.inflate()[cmp_indexes] * phase)


def test_verify_phase_promote():
    phase = random_phase()
    ctrls = [QType.CONTROL1, QType.TARGET]
    unitary = random_unitary(2)
    cg_phase = CtrlGate(unitary, ctrls, phase=phase)
    qubit = cg_phase.qspace[0]

    # execute
    actual = cg_phase.promote([qubit])
    # print(f'actual:\n{formatter.tostr(actual.inflate())}')
    no_phase = CtrlGate(unitary, ctrls)
    expected = no_phase.promote([qubit])
    # print(f'expected:\n{formatter.tostr(expected.inflate())}')
    assert actual.qspace == expected.qspace
    # verify that the core matrices differ by phase
    assert np.allclose(actual.inflate()[2:, 2:], expected.inflate()[2:, 2:] * phase)
    assert np.allclose(actual.inflate()[:2, :2], np.eye(2))
    assert np.allclose(actual.inflate()[:2, 2:], np.zeros((2, 2)))
    assert np.allclose(actual.inflate()[2:, :2], np.zeros((2, 2)))


def test_verify_phase_dela():
    phase = random_phase()
    ctrls = [QType.CONTROL1, QType.TARGET, QType.TARGET, QType.TARGET]
    qspace = [Qubit(10), Qubit(12, ancilla=True), Qubit(7), Qubit(1, ancilla=True)]
    unitary = random_unitary(2)
    mat = mykron(np.eye(2), unitary, np.eye(2))
    cg_phase = CtrlGate(mat, ctrls, qspace=qspace, phase=phase)
    no_phase = CtrlGate(mat, ctrls, qspace=qspace)

    # execute
    actual = cg_phase.dela()
    expected = no_phase.dela()

    assert actual.qspace == expected.qspace
    cmp_indexes = np.ix_(actual.core(), actual.core())
    assert np.allclose(actual.inflate()[cmp_indexes], expected.inflate()[cmp_indexes] * phase)
