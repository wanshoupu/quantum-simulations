import random

import numpy as np
import pytest
from numpy import kron

from quompiler.construct.cgate import CtrlGate
from quompiler.construct.qontroller import ctrl2core
from quompiler.construct.qspace import Qubit, Ancilla
from quompiler.construct.types import UnivGate, QType
from quompiler.construct.unitary import UnitaryM
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.inter_product import mykron, qproject
from quompiler.utils.mgen import random_unitary, random_indexes, random_UnitaryM, random_control, random_ctrlgate, random_CtrlGate, random_state
from quompiler.utils.permute import Permuter

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
        core = ctrl2core(control)
        m = random_unitary(1 << k)
        assert m.shape[0] == 1 << k
        assert len(core) == 1 << k
        u = UnitaryM(dim, core, m)

        # execute
        c = CtrlGate.convert(u)
        assert np.array_equal(c._unitary.matrix, u.matrix)


def test_convert_verify_dimension():
    for _ in range(10):
        n = random.randint(1, 5)
        dim = 1 << n
        core = random.randint(2, dim)
        indexes = random.sample(range(dim), core)
        u = random_UnitaryM(dim, indexes)
        c = CtrlGate.convert(u)
        assert c._unitary.dimension == dim


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
    assert tuple(cu._unitary.core) == (6, 7), f'Core indexes is unexpected {cu._unitary.core}'


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
    assert tuple(cu._unitary.core) == tuple(range(4)), f'Core indexes is unexpected {cu._unitary.core}'
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
    assert tuple(cu._unitary.core) == (1, 3), f'Core indexes is unexpected {cu._unitary.core}'
    actual = cu.sorted()
    assert tuple(actual._unitary.core) == (2, 3), f'Core indexes is unexpected {actual._unitary.core}'

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
    eqiv_original[np.ix_(original._unitary.core, original._unitary.core)] = original._unitary.matrix
    assert np.array_equal(eqiv_original, original.inflate())

    permuted_core = perm.bitpermuteall(original._unitary.core)
    expected = np.eye(original.order(), dtype=np.complexfloating)
    expected[np.ix_(permuted_core, permuted_core)] = original._unitary.matrix
    # print('expected\n')
    # print(formatter.tostr(expected))
    assert np.allclose(actual.inflate(), expected)


def test_sorted_by_ctrl():
    n = random.randint(1, 4)
    t = random.randint(1, n)
    m = random_unitary(1 << t)
    controls = random_control(n, t)
    qids = np.random.choice(100, size=n, replace=False)
    cu = CtrlGate(m, controls, qids)

    # execute
    sorting = np.argsort(cu.controls)
    sorted_cu = cu.sorted(sorting=sorting)
    assert sorted_cu.controls[:t] == [QType.TARGET] * t
    ctrls = sorted_cu.controls[t:]
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
    expected = np.block([[np.eye(4), np.zeros((4, 4))], [np.zeros((4, 4)), kron(cu._unitary.matrix, np.eye(2))]])
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
    expected = CtrlGate(np.kron(cu._unitary.matrix, np.eye(2)), list(controls) + [QType.TARGET], qids + extended_qspace)
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

    ctrls = actual.controls[len(cu.controls):]
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
    assert np.allclose(c._unitary.matrix, a._unitary.matrix @ b._unitary.matrix)
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
        expected = a.sorted().inflate() @ mykron(np.eye(2), b._unitary.matrix, np.eye(2))
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
    print('actual\n')
    print(formatter.tostr(actual.inflate()))


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

        print('actual\n')
        print(formatter.tostr(c.inflate()))


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


def test_dela_happy_case():
    ctrls = [QType.CONTROL1, QType.TARGET, QType.TARGET]
    qspace = [Qubit(10), Ancilla(12), Qubit(7)]
    unitary = random_unitary(2)
    mat = kron(np.eye(2), unitary)
    cg = CtrlGate(mat, ctrls, qspace=qspace)
    qubit = cg.qspace[1]
    assert isinstance(qubit, Ancilla)

    # execute
    actual = cg.dela(qubit)
    expected = CtrlGate(unitary, ctrls[:2], [qspace[0], qspace[2]])
    assert actual.qspace == expected.qspace
    assert np.allclose(actual.inflate(), expected.inflate())
