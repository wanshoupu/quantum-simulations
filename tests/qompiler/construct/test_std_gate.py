import random

import numpy as np
import pytest
from numpy import kron

from quompiler.construct.cgate import CtrlGate
from quompiler.construct.qspace import Qubit
from quompiler.construct.std_gate import CtrlStdGate
from quompiler.construct.types import UnivGate, QType
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.inter_product import inter_product
from quompiler.utils.mgen import random_unitary, random_control

formatter = MatrixFormatter(precision=2)


def test_convert_invalid_dimension():
    m = random_unitary(4)
    controls = (QType.CONTROL1, QType.CONTROL1, QType.TARGET, QType.TARGET)
    cu = CtrlGate(m, controls)
    with pytest.raises(AssertionError) as exc:
        CtrlStdGate.convert(cu)


def test_convert_standard_gates():
    for gate in UnivGate:
        controls = (QType.CONTROL1, QType.CONTROL1, QType.TARGET)
        cu = CtrlGate(gate.matrix, controls)
        cg = CtrlStdGate.convert(cu)
        assert cg is not None


def test_convert_init_standard_gates():
    for gate in UnivGate:
        controls = (QType.CONTROL1, QType.CONTROL1, QType.TARGET)
        cg = CtrlStdGate(gate, controls)
        assert cg is not None


def test_univ_Y():
    gate = UnivGate.Y
    control = (QType.CONTROL0, QType.CONTROL0, QType.TARGET)
    cu = CtrlStdGate(gate, control)
    expected = np.eye(8, dtype=np.complexfloating)
    expected[:2, :2] = gate.matrix
    # print()
    # print(formatter.tostr(expected))
    u = cu.inflate()
    assert np.allclose(u, expected), f'Expected:\n{formatter.tostr(expected)},\nActual:\n{formatter.tostr(u)}'


def test_UnivGate_Z():
    gate = UnivGate.Z
    control = (QType.CONTROL1, QType.CONTROL0, QType.CONTROL0, QType.TARGET)
    cu = CtrlStdGate(gate, control)
    expected = np.eye(16, dtype=np.complexfloating)
    expected[8:10, 8:10] = gate.matrix
    # print()
    # print(formatter.tostr(expected))
    u = cu.inflate()
    assert np.allclose(u, expected), f'Expected:\n{formatter.tostr(expected)},\nActual:\n{formatter.tostr(u)}'


def test_matmul_identical_controls():
    k = random.randint(2, 5)
    t = random.randint(1, k)
    controls = random_control(k, t)
    a = CtrlStdGate(UnivGate.X, controls)
    b = CtrlStdGate(UnivGate.S, controls)
    c = a @ b
    assert np.allclose(c._unitary.matrix, a.gate.matrix @ b.gate.matrix)


def test_matmul_identical_qspace_diff_controls():
    k = random.randint(2, 5)
    t = random.randint(1, k)
    a = CtrlStdGate(UnivGate.T, random_control(k, t))
    b = CtrlStdGate(UnivGate.H, random_control(k, t))

    # execute
    c = a @ b
    # print()
    # print(formatter.tostr(c.inflate()))
    expected = a.inflate() @ b.inflate()
    assert np.allclose(c.inflate(), expected)


def test_matmul_uncontrolled_diff_qspace():
    controls = [QType.CONTROL1, QType.TARGET]  # all targets, no control
    a = CtrlStdGate(UnivGate.H, controls, qspace=[1, 3])
    # print('a')
    # print(formatter.tostr(a.inflate()))

    b = CtrlStdGate(UnivGate.S, controls, qspace=[3, 0])
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
        a = CtrlStdGate(random.choice(list(UnivGate)), random_control(k, 1), qspace=rqids())
        b = CtrlStdGate(random.choice(list(UnivGate)), random_control(k, 1), qspace=rqids())

        # execute
        c = a @ b
        assert c is not None
