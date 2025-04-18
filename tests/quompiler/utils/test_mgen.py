from random import random

from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mgen import *
import numpy as np

random.seed(3)
np.random.seed(3)
formatter = MatrixFormatter()


def test_2l():
    m2l = random_matrix_2l(10, 1, 6)
    # print(formatter.tostr(m2l))


def test_unitary():
    randu = random_unitary(2)
    # print(formatter.tostr(randu))
    identity = randu.T @ np.conj(randu)
    assert np.all(np.isclose(identity, np.eye(*identity.shape))), print(formatter.tostr(identity))


def test_permeye():
    for _ in range(10):
        n = random.randint(10, 16)
        a = random.randrange(n)
        b = random.randrange(n)
        xs = xindexes(n, a, b)
        pi = permeye(xs)
        if a == b:
            assert pi[a, a] == 1 == pi[b, b], f'diagonal {a},{b}\n{pi}'
        else:
            assert pi[a, b] == 1 == pi[b, a], f'off diagonal {a},{b}\n{pi}'
            assert pi[a, a] == 0 == pi[b, b], f'diagonal {a},{b}\n{pi}'


def test_cyclic():
    cm = cyclic_matrix(8, 2)
    # print(formatter.tostr(cm))


def test_xindexes():
    for _ in range(10):
        n = random.randint(10, 100)
        a = random.randrange(n)
        b = random.randrange(n)
        xs = xindexes(n, a, b)
        assert xs[a] == b and xs[b] == a


def test_random_indexes():
    for _ in range(10):
        n = random.randint(10, 100)
        size = random.randrange(1, n)
        indxs = random_indexes(n, size)
        assert len(indxs) == len(set(indxs)), "Indexes contain duplicates!"
        assert 0 <= min(indxs) and max(indxs) < n, "Indexes are out of boundaries!"


def test_random_control():
    for _ in range(10):
        n = random.randint(10, 100)
        size = random.randrange(n)
        control = random_control(n, size)
        assert len(control) == n, f'Length of control {len(control)} is not {n}'
        assert control.count(None) == size, f'Number of target bits {control.count(None)} is not expected {size}'
