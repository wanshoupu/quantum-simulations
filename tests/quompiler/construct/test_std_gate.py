import random

import numpy as np
import pytest
from numpy import kron

from quompiler.construct.cgate import ControlledGate
from quompiler.construct.std_gate import ControlledStdGate
from quompiler.construct.types import UnivGate, QType
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.inter_product import inter_product
from quompiler.utils.mgen import random_unitary, random_control

formatter = MatrixFormatter(precision=2)


def test_convert_invalid_dimension():
    m = random_unitary(4)
    controls = (QType.CONTROL1, QType.CONTROL1, QType.TARGET, QType.TARGET)
    cu = ControlledGate(m, controls)
    with pytest.raises(AssertionError) as exc:
        ControlledStdGate.convert(cu)


def test_convert_convert_standard_gates():
    for gate in UnivGate:
        controls = (QType.CONTROL1, QType.CONTROL1, QType.TARGET)
        cu = ControlledGate(gate.matrix, controls)
        cg = ControlledStdGate.convert(cu)
        assert cg is not None


def test_convert_init_standard_gates():
    for gate in UnivGate:
        controls = (QType.CONTROL1, QType.CONTROL1, QType.TARGET)
        cg = ControlledStdGate(gate, controls)
        assert cg is not None


def test_univ_Y():
    gate = UnivGate.Y
    control = (QType.CONTROL0, QType.CONTROL0, QType.TARGET)
    cu = ControlledStdGate(gate, control)
    expected = np.eye(8, dtype=np.complexfloating)
    expected[:2, :2] = gate.matrix
    # print()
    # print(formatter.tostr(expected))
    u = cu.inflate()
    assert np.allclose(u, expected), f'Expected:\n{formatter.tostr(expected)},\nActual:\n{formatter.tostr(u)}'


def test_UnivGate_Z():
    gate = UnivGate.Z
    control = (QType.CONTROL1, QType.CONTROL0, QType.CONTROL0, QType.TARGET)
    cu = ControlledStdGate(gate, control)
    expected = np.eye(16, dtype=np.complexfloating)
    expected[8:10, 8:10] = gate.matrix
    # print()
    # print(formatter.tostr(expected))
    u = cu.inflate()
    assert np.allclose(u, expected), f'Expected:\n{formatter.tostr(expected)},\nActual:\n{formatter.tostr(u)}'


def test_expand():
    m = UnivGate.Y
    controls = (QType.TARGET, QType.CONTROL1)
    qids = [1, 0]
    cu = ControlledStdGate(m, controls, qids)
    univ = list(range(3))
    ex = cu.expand(univ)
    # print()
    # print(formatter.tostr(ex.inflate()))
    expected = np.block([[np.eye(4), np.zeros((4, 4))], [np.zeros((4, 4)), kron(cu.gate.matrix, np.eye(2))]])
    assert np.allclose(ex.inflate(), expected)


def test_expand_random():
    for _ in range(10):
        # print(f'Test {_}th round')
        n = random.randint(2, 5)
        k = random.randint(2, n)
        m = random.choice(list(UnivGate))
        controls = random_control(k, 1)
        qids = random.sample(range(n), k)
        cu = ControlledStdGate(m, controls, qids)
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
    a = ControlledStdGate(UnivGate.X, controls)
    b = ControlledStdGate(UnivGate.S, controls)
    c = a @ b
    assert np.allclose(c.unitary.matrix, a.gate.matrix @ b.gate.matrix)


def test_matmul_identical_qspace_diff_controls():
    k = random.randint(2, 5)
    t = random.randint(1, k)
    a = ControlledStdGate(UnivGate.T, random_control(k, t))
    b = ControlledStdGate(UnivGate.H, random_control(k, t))

    # execute
    c = a @ b
    # print()
    # print(formatter.tostr(c.inflate()))
    expected = a.inflate() @ b.inflate()
    assert np.allclose(c.inflate(), expected)


def test_expand_eqiv_inter_product():
    controls = [QType.TARGET, QType.CONTROL0]  # all targets, no control
    qid1 = [3, 0]
    a = ControlledStdGate(UnivGate.T, controls, qspace=qid1)
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
    controls = [QType.CONTROL1, QType.TARGET]  # all targets, no control
    a = ControlledStdGate(UnivGate.H, controls, qspace=[1, 3])
    # print('a')
    # print(formatter.tostr(a.inflate()))

    b = ControlledStdGate(UnivGate.S, controls, qspace=[3, 0])
    # print('b')
    # print(formatter.tostr(b.inflate()))

    # execute
    c = a @ b
    # print('c')
    # print(formatter.tostr(c.inflate()))

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
    for n in range(2, 20):
        # print(f'Test {_}th round')
        k = random.randint(2, 5)
        rqids = lambda: np.random.choice(1 << n, size=k, replace=False)
        a = ControlledStdGate(random.choice(list(UnivGate)), random_control(k, 1), qspace=rqids())
        b = ControlledStdGate(random.choice(list(UnivGate)), random_control(k, 1), qspace=rqids())

        # execute
        c = a @ b
        assert c is not None
