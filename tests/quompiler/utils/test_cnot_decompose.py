import random
from functools import reduce

import numpy as np
import pytest

from quompiler.construct.cmat import UnitaryM, CUnitary
from quompiler.construct.types import UnivGate, QType
from quompiler.utils.cnot_decompose import cnot_decompose, euler_decompose, control_decompose
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mgen import random_UnitaryM_2l, random_unitary, random_control

formatter = MatrixFormatter(precision=2)


def test_decompose_identity_matrix():
    n = 3
    dim = 1 << n
    idmat = UnitaryM(dim, (0, 1), np.eye(2))
    bc = cnot_decompose(idmat)
    assert bc == tuple()


def test_decompose_sing_qubit_circuit():
    n = 1
    dim = 1 << n
    u = UnitaryM(dim, (0, 1), random_unitary(dim))
    bc = cnot_decompose(u)
    # print(bc)
    assert len(bc) == 1
    assert all(isinstance(v, CUnitary) for v in bc)


def test_cnot_decompose8():
    m = random_UnitaryM_2l(8, 3, 4)
    # print(f'test = \n{formatter.tostr(m.inflate())}')
    ms = cnot_decompose(m)
    # print(f'decompose =')
    # for x in ms:
    #     print(formatter.tostr(x.inflate()), ',')
    # print()
    recovered = reduce(lambda x, y: x @ y, ms)
    assert np.allclose(recovered.inflate(), m.inflate()), f'recovered != expected: \n{formatter.tostr(recovered.inflate())},\n\n{formatter.tostr(m.inflate())}'
    assert all(isinstance(v, CUnitary) for v in ms)


def test_cnot_decompose4():
    m = random_UnitaryM_2l(4, 1, 2)
    # print(f'test = \n{formatter.tostr(m.inflate())}')
    ms = cnot_decompose(m)
    # print(f'decompose =')
    # for x in ms:
    #     print(formatter.tostr(x.inflate()), ',')
    # print()
    recovered = reduce(lambda x, y: x @ y, ms)
    assert np.allclose(recovered.inflate(), m.inflate()), f'recovered != expected: \n{formatter.tostr(recovered.inflate())},\n\n{formatter.tostr(m.inflate())}'
    assert all(isinstance(v, CUnitary) for v in ms)


def test_cnot_decompose_random():
    for _ in range(10):
        nqubit = random.randint(2, 5)
        n = 1 << nqubit
        r2 = random.randrange(n)
        while True:
            r1 = random.randrange(n)
            if r1 != r2:
                break
        m = random_UnitaryM_2l(n, r1, r2)
        # print(f'test = \n{formatter.tostr(m.inflate())}')
        ms = cnot_decompose(m)
        # print(f'decompose =')
        # for x in ms:
        #     print(formatter.tostr(x.inflate()), ',')
        # print()
        recovered = reduce(lambda x, y: x @ y, ms)
        assert np.allclose(recovered.inflate(), m.inflate()), f'recovered != expected: \n{formatter.tostr(recovered.inflate())},\n\n{formatter.tostr(m.inflate())}'
        assert all(isinstance(v, CUnitary) for v in ms)


@pytest.mark.parametrize('gate,expected', [
    [UnivGate.I, (1, 0, 0, 0)],
    [UnivGate.X, (-1j, np.pi / 2, np.pi, -np.pi / 2)],
    [UnivGate.Y, (-1j, 0, -np.pi, 0)],
    [UnivGate.Z, (1j, np.pi / 2, 0, np.pi / 2)],
    [UnivGate.H, (1j, np.pi, -np.pi / 2, 0)],
    [UnivGate.S, (np.sqrt(1j), np.pi / 4, 0, np.pi / 4)],
    [UnivGate.T, (np.power(1j, 1 / 4), np.pi / 8, 0, np.pi / 8)],
])
def test_euler_decompose(gate: UnivGate, expected: tuple):
    coms = euler_decompose(gate.mat)
    a, b, c, d = coms
    actual = a * UnivGate.Z.rmat(b) @ UnivGate.Y.rmat(c) @ UnivGate.Z.rmat(d)
    assert np.allclose(actual, gate.mat), f'Decomposition altered\n{formatter.tostr(actual)}!=\n{formatter.tostr(gate.mat)}'
    assert np.allclose(coms, expected), f'for gate={gate.name}, {formatter.tostr(coms)} != {formatter.tostr(expected)}'


def test_euler_decompose_random():
    for _ in range(10):
        # print(f'Test {_}th round')
        expected = random_unitary(2)
        a, b, c, d = euler_decompose(expected)
        actual = a * UnivGate.Z.rmat(b) @ UnivGate.Y.rmat(c) @ UnivGate.Z.rmat(d)
        assert np.allclose(actual, expected)


def test_control_decompose_identity():
    for _ in range(10):
        # print(f'Test {_}th round')
        n = random.randint(1, 5)
        controls = random_control(n, 1)
        cu = CUnitary(random_unitary(2), controls)

        # execute
        result = control_decompose(cu)
        assert len(result) == 6
        assert all(com is not None for com in result)
        actual = reduce(lambda a, b: a @ b, result[1::2]).inflate()
        assert np.allclose(actual, np.eye(2))


def test_control_decompose_uncontrolled_equality():
    expected = random_unitary(2)
    cu = CUnitary(expected, [QType.TARGET])

    # execute
    result = control_decompose(cu)
    actual = reduce(lambda a, b: a @ b, result).inflate()
    assert np.allclose(actual, expected)


def test_control_decompose_controlled_equality_1_target():
    u = random_unitary(2)
    controls = [QType.CONTROL1, QType.TARGET]
    # random.shuffle(controls)
    cu = CUnitary(u, controls)

    # execute
    result = control_decompose(cu)
    actual = reduce(lambda a, b: a @ b, result)
    print('actual:')
    print(formatter.tostr(actual.inflate()))
    print('expected:')
    print(formatter.tostr(cu.inflate()))
    assert np.allclose(actual.inflate(), cu.inflate())


def test_control_decompose_controlled_equality_C1T():
    u = random_unitary(2)
    controls = [QType.CONTROL1, QType.TARGET]
    # random.shuffle(controls)
    cu = CUnitary(u, controls)

    # execute
    result = control_decompose(cu)
    actual = reduce(lambda a, b: a @ b, result)
    # print('actual:')
    # print(formatter.tostr(actual.inflate()))
    # print('expected:')
    # print(formatter.tostr(cu.inflate()))
    assert np.allclose(actual.inflate(), cu.inflate())


def test_control_decompose_controlled_equality_C0T():
    u = random_unitary(2)
    controls = [QType.CONTROL0, QType.TARGET]
    # random.shuffle(controls)
    cu = CUnitary(u, controls)

    # execute
    result = control_decompose(cu)
    actual = reduce(lambda a, b: a @ b, result)
    # print('actual:')
    # print(formatter.tostr(actual.inflate()))
    # print('expected:')
    # print(formatter.tostr(cu.inflate()))
    assert np.allclose(actual.inflate(), cu.inflate())


def test_control_decompose_controlled_equality_C0TC1():
    u = random_unitary(2)
    controls = [QType.CONTROL0, QType.TARGET, QType.CONTROL1]
    # random.shuffle(controls)
    cu = CUnitary(u, controls)

    # execute
    result = control_decompose(cu)
    actual = reduce(lambda a, b: a @ b, result)
    # print('actual:')
    # print(formatter.tostr(actual.inflate()))
    # print('expected:')
    # print(formatter.tostr(cu.inflate()))
    assert np.allclose(actual.inflate(), cu.inflate())


def test_control_decompose_noop_control():
    # Verify that ABC = I
    u = random_unitary(2)
    controls = random_control(3, 1)
    cu = CUnitary(u, controls)
    result = control_decompose(cu)
    assert len(result) == 6
    assert all(com is not None for com in result)
    actual = reduce(lambda a, b: a @ b, result)
    assert np.allclose(actual.inflate(), cu.inflate())


def test_control_decompose_custom_qspace():
    u = random_unitary(2)
    n = 3
    offset = 500
    qspace = list(range(offset, offset + n))
    controls = [QType.CONTROL0, QType.TARGET, QType.CONTROL1]
    target = controls.index(QType.TARGET)
    cu = CUnitary(u, controls, qspace)

    # execute
    result = control_decompose(cu)

    # verify
    for i in [1, 3, 5]:
        r = result[i]
        assert tuple(r.qspace) == (offset + target,)


def test_control_decompose_random():
    for _ in range(10):
        u = random_unitary(2)
        n = random.randint(1, 5)
        offset = random.randrange(500)
        qspace = list(range(offset, offset + n))
        controls = random_control(n, 1)
        target = controls.index(QType.TARGET)
        cu = CUnitary(u, controls, qspace)

        # execute
        result = control_decompose(cu)

        # verify
        assert len(result) == 6
        assert all(com is not None for com in result)
        uncontrolled = [1, 3, 5]
        for i in uncontrolled:
            r = result[i]
            assert tuple(r.qspace) == (offset + target,)
