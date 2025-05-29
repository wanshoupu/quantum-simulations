import random

import numpy as np
import pytest

from quompiler.construct.types import UnivGate
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.group_su2 import rangle, gc_decompose, tsim, raxis, rot, euler_params, gphase, dist
from quompiler.utils.mfun import herm
from quompiler.utils.mgen import random_unitary, random_su2

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
def test_euler_params_std_gate(gate: UnivGate, expected: tuple):
    coms = euler_params(gate.matrix)
    a, b, c, d = coms
    actual = a * UnivGate.Z.rotation(b) @ UnivGate.Y.rotation(c) @ UnivGate.Z.rotation(d)
    assert np.allclose(actual, gate.matrix)
    assert np.allclose(coms, expected)


def test_euler_params_verify_identity_random():
    for _ in range(100):
        # print(f'Test {_}th round')
        expected = random_unitary(2)
        a, b, c, d = euler_params(expected)
        actual = a * UnivGate.Z.rotation(b) @ UnivGate.Y.rotation(c) @ UnivGate.Z.rotation(d)
        assert np.allclose(actual, expected)


def test_gphase():
    for _ in range(100):
        u = random_unitary(2)
        gp = gphase(u)
        v = np.conj(gp) * u
        assert np.isclose(np.linalg.det(v), 1)
        assert np.trace(v) >= 0


@pytest.mark.parametrize("gate,expected", [
    [UnivGate.I, 0],
    [UnivGate.X, np.pi],
    [UnivGate.Y, np.pi],
    [UnivGate.Z, np.pi],
    [UnivGate.H, np.pi],
    [UnivGate.S, np.pi / 2],
    [UnivGate.T, np.pi / 4],
    [UnivGate.SD, np.pi / 2],
    [UnivGate.TD, np.pi / 4],
])
def test_rangle_std_gates(gate, expected):
    actual = rangle(gate)
    assert np.isclose(actual, expected), f'{actual} != {expected}'


def test_rangle_diff_by_2pi():
    """
    if operators u + v = 0, then their rotation angles sum up to 2π
    """
    for _ in range(100):
        u = random_su2()
        # print()
        # print(formatter.tostr(u))
        actual = rangle(u)
        expected = rangle(-u)
        # print(actual, expected)
        assert np.isclose(actual + expected, 2 * np.pi), f'{actual + expected} != 2π'


def test_rangle_random_su2():
    for _ in range(100):
        gate = random_su2()
        actual = rangle(gate)
        trace = np.trace(gate)
        if trace > 0:
            assert actual <= np.pi
        else:
            assert actual >= -np.pi


def test_rangle_random_unitary():
    for _ in range(100):
        gate = random_unitary(2)
        actual = rangle(gate)
        assert actual <= np.pi


def test_rangle_eq_2pi():
    gate = -UnivGate.I.matrix
    actual = rangle(gate)
    assert np.isclose(actual, 2 * np.pi), f'{actual} != 2π'


@pytest.mark.parametrize("u", [
    np.array([[1, 0]]),
    np.array([[1, 0, 1], [0, 1j, 3]]),
    np.array([[1, 0, 0, 1]]),
])
def test_dist_invalid_shapes(u):
    v = random_unitary(2)
    with pytest.raises(AssertionError) as e:
        dist(u, v)
    assert str(e.value) == "operators must have shape (2, 2)"


#
# @pytest.mark.parametrize("u", [
#     np.array([[1, 0], [1, 1j]]),
#     np.array([[1, 0], [1, 0]]),
#     np.array([[1, 1], [1, 1j]]),
#     np.array([[1, -1], [1, 1]]),
# ])
# def test_dist_non_unitary(u):
#     v = random_unitary(2)
#     with pytest.raises(AssertionError) as e:
#         dist(u, v)
#     assert str(e.value) == "matmul product must be unitary"

def test_dist_zero():
    for _ in range(100):
        # print(f'Test round {_}')
        u = random_unitary(2)
        d = dist(u, u)
        assert np.isclose(d, 0, rtol=1.e-4, atol=1.e-7), f'{d} != 0'


def test_dist_maximum():
    """
    The maximum distance between two unitary matrices is 2 which can only be achieved between u and -u.
    """
    for _ in range(100):
        # print(f'Test round {_}')
        u = random_unitary(2)
        d = dist(u, -u)
        assert np.isclose(d, 2, rtol=1.e-4, atol=1.e-7), f'{d} != 2'


@pytest.mark.parametrize("u,v,angle", [
    [UnivGate.I, UnivGate.X, np.pi],
    [UnivGate.Z, UnivGate.H, np.pi / 2],
    [UnivGate.H, UnivGate.X, np.pi / 2],
    [UnivGate.I, UnivGate.Z, np.pi],
])
def test_dist_real_unitary(u, v, angle):
    actual = dist(u.matrix, v.matrix)
    expected = 2 * np.sin(angle / 4)
    assert np.isclose(actual, expected, rtol=1.e-4, atol=1.e-7), f'{actual} != {expected}'


@pytest.mark.parametrize("u,v,angle", [
    [UnivGate.X, UnivGate.Y, np.pi],
    [UnivGate.I, UnivGate.Z, np.pi],
    [UnivGate.T, UnivGate.Y, np.pi],
    [UnivGate.T, UnivGate.TD, np.pi / 2],
    [UnivGate.S, UnivGate.TD, np.pi * 3 / 4],
    [UnivGate.S, UnivGate.H, np.pi * 2 / 3],
    [UnivGate.SD, UnivGate.Y, np.pi],
])
def test_dist_std_gates(u, v, angle):
    actual = dist(u.matrix, v.matrix)
    expected = 2 * np.sin(angle / 4)
    assert np.isclose(actual, expected, rtol=1.e-4, atol=1.e-7), f'{actual} != {expected}'
