import random
from functools import reduce

import numpy as np
import pytest

from quompiler.construct.cgate import CtrlGate
from quompiler.construct.qspace import Qubit
from quompiler.construct.types import UnivGate, QType
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mgen import random_unitary, random_control
from quompiler.utils.std_decompose import euler_decompose, euler_param

formatter = MatrixFormatter(precision=2)


@pytest.mark.parametrize('gate,expected', [
    [UnivGate.I, (1, 0, 0, 0)],
    [UnivGate.X, (-1j, np.pi / 2, np.pi, -np.pi / 2)],
    [UnivGate.Y, (-1j, 0, -np.pi, 0)],
    [UnivGate.Z, (1j, np.pi / 2, 0, np.pi / 2)],
    [UnivGate.H, (1j, np.pi, -np.pi / 2, 0)],
    [UnivGate.S, (np.sqrt(1j), np.pi / 4, 0, np.pi / 4)],
    [UnivGate.T, (np.power(1j, 1 / 4), np.pi / 8, 0, np.pi / 8)],
])
def test_std_gate(gate: UnivGate, expected: tuple):
    coms = euler_param(gate.matrix)
    a, b, c, d = coms
    actual = a * UnivGate.Z.rotation(b) @ UnivGate.Y.rotation(c) @ UnivGate.Z.rotation(d)
    assert np.allclose(actual, gate.matrix), f'Decomposition altered\n{formatter.tostr(actual)}!=\n{formatter.tostr(gate.matrix)}'
    assert np.allclose(coms, expected), f'for gate={gate.name}, {formatter.tostr(coms)} != {formatter.tostr(expected)}'


def test_identity():
    for _ in range(10):
        # print(f'Test {_}th round')
        n = random.randint(1, 5)
        controls = random_control(n, 1)
        cu = CtrlGate(random_unitary(2), controls)

        # execute
        result = euler_decompose(cu)
        assert len(result) == 6
        assert all(com is not None for com in result)
        actual = reduce(lambda a, b: a @ b, result[1::2]).inflate()
        assert np.allclose(actual, np.eye(2))


def test_uncontrolled_equality():
    expected = random_unitary(2)
    cu = CtrlGate(expected, [QType.TARGET])

    # execute
    result = euler_decompose(cu)
    actual = reduce(lambda a, b: a @ b, result).inflate()
    assert np.allclose(actual, expected)


def test_controlled_equality_1_target():
    u = random_unitary(2)
    controls = [QType.CONTROL1, QType.TARGET]
    # random.shuffle(controls)
    cu = CtrlGate(u, controls)

    # execute
    result = euler_decompose(cu)
    actual = reduce(lambda a, b: a @ b, result)
    # print('actual:')
    # print(formatter.tostr(actual.inflate()))
    # print('expected:')
    # print(formatter.tostr(cu.inflate()))
    assert np.allclose(actual.inflate(), cu.inflate())


def test_controlled_equality_C1T():
    u = random_unitary(2)
    controls = [QType.CONTROL1, QType.TARGET]
    # random.shuffle(controls)
    cu = CtrlGate(u, controls)

    # execute
    result = euler_decompose(cu)
    actual = reduce(lambda a, b: a @ b, result)
    # print('actual:')
    # print(formatter.tostr(actual.inflate()))
    # print('expected:')
    # print(formatter.tostr(cu.inflate()))
    assert np.allclose(actual.inflate(), cu.inflate())


@pytest.mark.parametrize('controls', [
    [QType.CONTROL0, QType.TARGET],
    [QType.TARGET, QType.CONTROL1],
    [QType.CONTROL1, QType.TARGET],
    [QType.TARGET, QType.CONTROL0],
])
def test_controlled_equality_C0T(controls):
    u = random_unitary(2)
    # random.shuffle(controls)
    cu = CtrlGate(u, controls)

    # execute
    result = euler_decompose(cu)
    actual = reduce(lambda a, b: a @ b, result)
    # print('actual:')
    # print(formatter.tostr(actual.inflate()))
    # print('expected:')
    # print(formatter.tostr(cu.inflate()))
    assert np.allclose(actual.inflate(), cu.inflate())


def test_multi_controlled_decompose_equality():
    """
    Question: is euler decomposition applicable to multi-control operation?
    This test shows yes, it is.
    """
    u = random_unitary(2)
    controls = [QType.CONTROL1, QType.CONTROL1, QType.TARGET]
    # random.shuffle(controls)
    cu = CtrlGate(u, controls)
    # execute
    result = euler_decompose(cu)
    actual = reduce(lambda a, b: a @ b, result)
    # print('actual:')
    # print(formatter.tostr(actual.inflate()))
    # print('expected:')
    # print(formatter.tostr(cu.inflate()))
    assert np.allclose(actual.inflate(), cu.inflate())


def test_noop_control():
    # Verify that ABC = I
    u = random_unitary(2)
    controls = random_control(3, 1)
    cu = CtrlGate(u, controls)
    result = euler_decompose(cu)
    assert len(result) == 6
    assert all(com is not None for com in result)
    actual = reduce(lambda a, b: a @ b, result)
    assert np.allclose(actual.inflate(), cu.inflate())


def test_custom_qspace():
    u = random_unitary(2)
    n = 3
    offset = 500
    qspace = [Qubit(i) for i in range(offset, offset + n)]
    controls = [QType.CONTROL0, QType.TARGET, QType.CONTROL1]
    target = controls.index(QType.TARGET)
    cu = CtrlGate(u, controls, qspace)

    # execute
    result = euler_decompose(cu)

    # verify
    for i in [1, 3, 5]:
        r = result[i]
        assert tuple(r.qspace) == (Qubit(offset + target),)


def test_verify_identity_random():
    for _ in range(10):
        # print(f'Test {_}th round')
        expected = random_unitary(2)
        a, b, c, d = euler_param(expected)
        actual = a * UnivGate.Z.rotation(b) @ UnivGate.Y.rotation(c) @ UnivGate.Z.rotation(d)
        assert np.allclose(actual, expected)


def test_verify_qspace_random():
    for _ in range(10):
        u = random_unitary(2)
        n = random.randint(1, 5)
        offset = random.randrange(500)
        qspace = [Qubit(i) for i in range(offset, offset + n)]
        controls = random_control(n, 1)
        target = controls.index(QType.TARGET)
        cu = CtrlGate(u, controls, qspace)

        # execute
        result = euler_decompose(cu)

        # verify
        assert len(result) == 6
        assert all(com is not None for com in result)
        uncontrolled = [1, 3, 5]
        for i in uncontrolled:
            r = result[i]
            assert tuple(r.qspace) == (Qubit(offset + target),)
