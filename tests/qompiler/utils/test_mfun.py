from functools import reduce

import numpy as np
import pytest

from quompiler.construct.types import UnivGate
from quompiler.utils.mfun import allprop, herm, herms
from quompiler.utils.group_su2 import euler_params, dist, gphase
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
