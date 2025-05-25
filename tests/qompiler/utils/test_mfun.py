from functools import reduce

import numpy as np
import pytest

from quompiler.construct.types import UnivGate
from quompiler.utils.mfun import allprop, dist, herm, herms
from quompiler.utils.mgen import random_unitary, random_phase, random_su2


def test_allprop_false():
    a = np.array([[1, 1], [1, 1]]) * np.pi
    p, r = allprop(a, np.eye(a.shape[0]))
    assert not p
    if p:
        assert np.isclose(r, np.pi)


def test_allprop_zeros_prop():
    shape = (2, 3)
    a = np.zeros(shape)
    b = np.zeros(shape)
    p, r = allprop(a, b)
    assert p
    assert np.isclose(r, 0)


def test_allprop_zeros_atol():
    shape = (2, 3)
    a = np.zeros(shape)
    b = np.zeros(shape) + 1e-6
    p, r = allprop(a, b)
    assert p
    assert r == 0


def test_allprop_partial_zeros():
    shape = (2, 3)
    a = np.array([[0, 0], [0, 1]]) * np.pi
    b = np.zeros(shape) + 1e-6
    p, r = allprop(a, b)
    assert not p


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


@pytest.mark.parametrize("u", [
    np.array([[1, 0], [1, 1j]]),
    np.array([[1, 0], [1, 0]]),
    np.array([[1, 1], [1, 1j]]),
    np.array([[1, -1], [1, 1]]),
])
def test_dist_non_unitary(u):
    v = random_unitary(2)
    with pytest.raises(AssertionError) as e:
        dist(u, v)
    assert str(e.value) == "matmul product must be unitary"


def test_dist_phase_factor():
    """
    Distance function can handle non SU(2) unitary operators
    """
    phase = random_phase()
    u = random_unitary(2) * phase
    v = random_unitary(2)
    # assert no error
    dist(u, v)


def test_su2():
    for _ in range(100):
        su2 = random_su2()
        assert np.isclose(np.linalg.det(su2), 1)


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


def test_herms_equality():
    us = [random_unitary(2) for _ in range(3)]
    vs = herms(us)
    u = reduce(lambda x, y: x @ y, us)
    v = reduce(lambda x, y: x @ y, vs)
    assert np.allclose(u @ v, np.eye(2)), f'{u} != {v}'
