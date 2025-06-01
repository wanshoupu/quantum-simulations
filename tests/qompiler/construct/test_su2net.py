import random
from functools import reduce

import numpy as np
import pytest

from quompiler.construct.su2net import SU2Net, cliffordt_seqs
from quompiler.construct.types import UnivGate
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mfun import herm
from quompiler.utils.group_su2 import dist, gphase
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
    matrix = gate.matrix
    node, lookup_error = su2net.lookup(matrix)
    seq = [n.data for n in node.children]
    v = reduce(lambda a, b: a @ b, seq, np.eye(2))
    assert np.allclose(np.array(v), node.data)  # self-consistent
    error = dist(np.array(v), np.array(matrix))
    assert error == lookup_error == 0


@pytest.mark.parametrize('gate', list(UnivGate))
def test_lookup_std(gate):
    su2net = SU2Net(.4)
    matrix = gate.matrix
    node, lookup_error = su2net.lookup(matrix)
    seq = [n.data for n in node.children]
    v = reduce(lambda a, b: a @ b, seq, np.eye(2))
    assert np.allclose(np.array(v), node.data)  # self-consistent
    error = dist(np.array(v), np.array(matrix))
    assert np.isclose(error, lookup_error) and np.isclose(error, 0)


@pytest.mark.parametrize("seed", random.sample(range(1 << 20), 10))
def test_lookup_random_su2(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    su2net = SU2Net(.4)
    u = random_su2()
    node, lookup_error = su2net.lookup(u)
    leaves = [n for n in node.children]
    assert all(l.is_leaf() for l in leaves)
    coms = [l.data for l in leaves]
    v = reduce(lambda a, b: a @ b, coms, np.eye(2))
    assert np.allclose(np.array(v), node.data)  # self-consistent
    error = dist(np.array(v), np.array(u))
    # print(f'\n{formatter.tostr(u)}\nlookup error: {lookup_error}, dist: {error}')
    assert error < su2net.error


@pytest.mark.parametrize("seed", random.sample(range(1 << 20), 10))
def test_lookup_random_unitary_verify_error_margin(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    su2net = SU2Net(.4)
    u = random_unitary(2)
    node, lookup_error = su2net.lookup(u)
    coms = [n.data for n in node.children]
    v = reduce(lambda a, b: a @ b, coms, np.eye(2))
    assert np.allclose(np.array(v), node.data)  # self-consistent
    error = dist(np.array(v), np.array(u))
    # print(f'\n{formatter.tostr(u)}\nlookup error: {lookup_error}, dist: {error}')
    # print(f'gphase error: {gphase(v @ herm(u))}')
    assert error < su2net.error


@pytest.mark.parametrize("seed", random.sample(range(1 << 20), 10))
def test_lookup_random_unitary_verify_unitarity(seed: int):
    """
    Verify that the result of lookup is still unitary with unit absolute det
    """
    random.seed(seed)
    np.random.seed(seed)

    su2net = SU2Net(.4)
    u = random_unitary(2)
    # execute
    node, lookup_error = su2net.lookup(u)
    # verify
    det = np.abs(np.linalg.det(node.data))
    # print(f'det: {det}')
    assert np.isclose(det, 1)
    unitarity = np.sqrt(np.sum((node.data @ herm(node.data) - np.eye(2)) ** 2))
    # print(f'unitarity error: {unitarity}')
    assert np.isclose(unitarity, 0)


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
