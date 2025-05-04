import random
from functools import reduce

import numpy as np
import pytest

from quompiler.construct.cmat import UnitaryM, CUnitary
from quompiler.construct.types import UnivGate
from quompiler.utils.cnot_decompose import cnot_decompose, euler_decompose
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mgen import random_UnitaryM_2l, random_unitary

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
