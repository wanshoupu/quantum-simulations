import random
from functools import reduce

import numpy as np
import pytest

from quompiler.construct.su2net import SU2Net, cliffordt_seqs
from quompiler.construct.types import UnivGate
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mfun import dist, gphase, herm
from quompiler.utils.mgen import random_su2, random_unitary

formatter = MatrixFormatter(precision=2)


def test_init_lazy():
    error = .4
    su2net = SU2Net(error)
    assert su2net.error == error
    # tree not constructed until first lookup
    assert not su2net.constructed


def test_lookup_identity():
    gate = UnivGate.I
    su2net = SU2Net(.4)
    gph = gphase(gate.matrix)
    matrix = gate.matrix / gph
    node = su2net.lookup(matrix)
    seq = [n.data for n in node.children]
    v = reduce(lambda a, b: a @ b, seq, np.eye(2))
    assert np.allclose(np.array(v), node.data)  # self-consistent
    error = dist(np.array(v), np.array(matrix))
    assert error <= su2net.error


@pytest.mark.parametrize('gate', list(UnivGate))
def test_lookup_std(gate):
    su2net = SU2Net(.2)
    gph = gphase(gate.matrix)
    matrix = gate.matrix / gph
    node = su2net.lookup(matrix)
    seq = [n.data for n in node.children]
    v = reduce(lambda a, b: a @ b, seq, np.eye(2))
    assert np.allclose(np.array(v), node.data)  # self-consistent
    error = dist(np.array(v), np.array(matrix))
    # print(f'{gate} ~{seq} lookup error: {error}')
    assert error < su2net.error


@pytest.mark.parametrize("seed", [42, 123, 999, 2023])
def test_lookup_terminates(seed):
    su2net = SU2Net(.4)
    random.seed(seed)
    np.random.seed(seed)
    u = random_su2()
    node = su2net.lookup(u)
    v = reduce(lambda a, b: a @ b, [n.data for n in node.children], np.eye(2))
    assert np.allclose(np.array(v), node.data)  # self-consistent
    error = dist(np.array(v), np.array(u))
    # print(f'\n{formatter.tostr(u)}\nlookup error: {error}')
    assert error < 2 * su2net.error


@pytest.mark.parametrize("seed", [42, 123, 999, 2023])
def test_lookup_random_su2(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    su2net = SU2Net()
    u = random_su2()
    node = su2net.lookup(u)
    v = reduce(lambda a, b: a @ b, [n.data for n in node.children], np.eye(2))
    assert np.allclose(np.array(v), node.data)  # self-consistent
    error = dist(np.array(v), np.array(u))
    # print(f'\n{formatter.tostr(u)}\nlookup error: {error}')
    assert error < 2 * su2net.error


@pytest.mark.parametrize("seed", [42, 123, 999, 2023])
def test_lookup_random_u2(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    su2net = SU2Net()
    u = random_unitary(2)
    node = su2net.lookup(u)
    v = reduce(lambda a, b: a @ b, [n.data for n in node.children], np.eye(2))
    assert np.allclose(np.array(v), node.data)  # self-consistent
    error = dist(np.array(v), np.array(u))
    # print(f'\n{formatter.tostr(u)}\nlookup error: {error}')
    gp = gphase(v @ herm(u))
    print(f'gphase error: {gp}')
    assert error < 2 * su2net.error


def test_cliffordt_seqs():
    depth = 3
    set_size = len(set(UnivGate.cliffordt()) - {UnivGate.I})
    seqs = cliffordt_seqs(depth)
    assert len(seqs) == ((set_size - 1) ** depth - 1) * set_size / 5 + 1
    for u, seq in seqs:
        # print(f'u:\n{formatter.tostr(u)}')
        expected = reduce(lambda a, b: a @ b, seq, np.eye(2))
        # print(f'expected:\n{expected}')
        assert np.allclose(u, np.array(expected))
