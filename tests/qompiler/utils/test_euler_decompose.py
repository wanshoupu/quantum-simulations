import random
from functools import reduce

import numpy as np
import pytest

from quompiler.construct.cgate import CtrlGate
from quompiler.construct.qspace import Qubit
from quompiler.construct.types import QType
from quompiler.utils.euler_decompose import euler_decompose
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mgen import random_unitary, random_control

formatter = MatrixFormatter(precision=2)

identity_coms_indexes = [1, 2, 4, 5, 7]


def test_identity():
    for _ in range(10):
        # print(f'Test {_}th round')
        n = random.randint(1, 5)
        controls = random_control(n, 1)
        cu = CtrlGate(random_unitary(2), controls)

        # execute
        result = euler_decompose(cu)
        assert len(result) == 8
        assert all(com is not None for com in result)
        identity_coms = [result[i] for i in identity_coms_indexes]
        actual = reduce(lambda a, b: a @ b, identity_coms).inflate()
        assert np.allclose(actual, np.eye(2))


def test_uncontrolled_equality():
    expected = random_unitary(2)
    cu = CtrlGate(expected, [QType.TARGET])

    # execute
    result = euler_decompose(cu)
    actual = reduce(lambda a, b: a @ b, result).inflate()
    assert np.allclose(actual, expected)


def test_controlled_equality_1_ctr_1_target():
    u = random_unitary(2)
    controls = [QType.CONTROL1, QType.TARGET]
    # random.shuffle(controls)
    cu = CtrlGate(u, controls)

    # execute
    result = euler_decompose(cu)

    # verify
    phase_gate = result[0]
    assert not phase_gate.control_qids()
    assert phase_gate.qspace[0] == cu.qspace[0]
    assert result[3].is_std() and result[6].is_std()

    actual = reduce(lambda a, b: a @ b, result)
    # print('actual:')
    # print(formatter.tostr(actual.inflate()))
    # print('expected:')
    # print(formatter.tostr(cu.inflate()))
    assert np.allclose(actual.inflate(), cu.inflate())


def test_controlled_equality_2_ctrl_1_target():
    u = random_unitary(2)
    controls = [QType.CONTROL1, QType.CONTROL0, QType.TARGET]
    # random.shuffle(controls)
    cu = CtrlGate(u, controls)

    # execute
    result = euler_decompose(cu)

    # verify
    phase_gate = result[0]
    assert len(phase_gate.control_qids()) > 1

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

    # verify
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
    actual = reduce(lambda a, b: a @ b, result).sorted()
    # print('actual:')
    # print(formatter.tostr(actual.inflate()))
    # print('expected:')
    # print(formatter.tostr(cu.inflate()))
    assert actual.qspace == cu.qspace
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
    assert len(result) == 8
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
    for i in identity_coms_indexes:
        r = result[i]
        assert tuple(r.qspace) == (Qubit(offset + target),)


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
        assert len(result) == 8
        assert all(com is not None for com in result)
        for i in identity_coms_indexes:
            r = result[i]
            assert tuple(r.qspace) == (Qubit(offset + target),)
