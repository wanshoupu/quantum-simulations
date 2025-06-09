from functools import reduce

import numpy as np
import pytest

from quompiler.utils.su2fun import dist
from quompiler.utils.mfun import allprop, herms, get_principal
from quompiler.utils.mgen import random_unitary, random_phase, random_su2


def test_allprop_false():
    a = np.array([[1, 1], [1, 1]]) * np.pi
    p = allprop(a, np.eye(a.shape[0]))
    assert not p
    if p:
        assert np.isclose(p.result, np.pi)


def test_allprop_zeros_prop():
    shape = (2, 3)
    a = np.zeros(shape)
    b = np.zeros(shape)
    p = allprop(a, b)
    assert p
    assert np.isclose(p.result, 0)


def test_allprop_zeros_atol():
    shape = (2, 3)
    a = np.zeros(shape)
    b = np.zeros(shape) + 1e-6
    p = allprop(a, b)
    assert p
    assert p.result == 0


def test_allprop_partial_zeros():
    shape = (2, 3)
    a = np.array([[0, 0], [0, 1]]) * np.pi
    b = np.zeros(shape) + 1e-6
    p = allprop(a, b)
    assert not p


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


def test_herms_equality():
    us = [random_unitary(2) for _ in range(3)]
    vs = herms(us)
    u = reduce(lambda x, y: x @ y, us)
    v = reduce(lambda x, y: x @ y, vs)
    assert np.allclose(u @ v, np.eye(2)), f'{u} != {v}'


@pytest.mark.parametrize("axis,principal,factor", [
    [[1, 0, 0], 'x', 1],
    [[0, 1, 0], 'y', 1],
    [[0, 0, 1], 'z', 1],
    [[-1, 0, 0], 'x', -1],
    [[0, -1, 0], 'y', -1],
    [[0, 0, -1], 'z', -1],
    [[-1.5, 1e-9, 0], 'x', -1.5],
    [[1e-9, -np.pi, 1e-9], 'y', -np.pi],
    [[0, 0, 3.14], 'z', 3.14],
])
def test_get_principal_3d_affirmative(axis, principal, factor):
    chk = get_principal(axis)
    assert chk
    p, f = chk.result
    assert p == principal
    assert np.isclose(f, factor)


@pytest.mark.parametrize("axis,principal,factor", [
    [[0, 0], 'z', 1],
    [[np.pi, 0], 'z', -1],
    [[0, 2 * np.pi], 'z', 1],
    [[0, -2 * np.pi], 'z', 1],
    [[np.pi / 2, 0], 'x', 1],
    [[-np.pi / 2, 0], 'x', -1],
    [[-np.pi / 2, np.pi / 2], 'y', -1],
])
def test_get_principal_2d_affirmative(axis, principal, factor):
    chk = get_principal(axis)
    assert chk
    p, f = chk.result
    assert p == principal
    assert np.isclose(f, factor)


@pytest.mark.parametrize("axis", [
    [-1, -1, 0],
    [0, -1, 1],
    [1, 1, -1],
    [-1.5, 1e-5, 0],
    [1e-5, 1e-5, 1e-9],
    [1, 0],
    [.5 * np.pi, 1e-3],
    [1e-3, .2 * np.pi],
    [-1e-5, -1],
    [-np.pi / 3, 0],
    [-np.pi / 3, np.pi / 4],
])
def test_get_principal_none_result(axis):
    chk = get_principal(axis)
    assert not chk
