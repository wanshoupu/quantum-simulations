from functools import reduce

import numpy as np
import pytest

from quompiler.construct.nn.kdtree import KDTreeNN
from quompiler.construct.nn.nn import BruteNN, AutoNN
from quompiler.construct.types import UnivGate
from quompiler.utils.su2fun import dist, vec
from quompiler.utils.std_decompose import cliffordt_seqs


@pytest.mark.parametrize('gate', list(UnivGate))
def test_lookup_std_autonn(gate):
    matrix = np.array(gate)
    nn = AutoNN(cliffordt_seqs(5))
    node, lookup_error = nn.lookup(matrix)
    seq = [n.data for n in node.children]
    v = reduce(lambda a, b: a @ b, seq, np.eye(2))
    assert np.allclose(np.array(v), node.data)  # self-consistent
    error = dist(np.array(v), np.array(matrix))
    mvec = vec(matrix)
    vvec = vec(v)
    print(mvec)
    print(vvec)
    assert np.isclose(error, 0)
    assert np.isclose(error, lookup_error)


@pytest.mark.parametrize('gate', list(UnivGate))
def test_lookup_std_brutenn(gate):
    matrix = np.array(gate)
    nn = BruteNN(cliffordt_seqs(5))
    node, lookup_error = nn.lookup(matrix)
    seq = [n.data for n in node.children]
    v = reduce(lambda a, b: a @ b, seq, np.eye(2))
    mvec = vec(matrix)
    vvec = vec(v)
    print(mvec)
    print(vvec)
    assert np.allclose(np.array(v), node.data)  # self-consistent
    error = dist(np.array(v), np.array(matrix))
    assert np.isclose(error, lookup_error, rtol=1.e-5, atol=1.e-6)
    assert np.isclose(error, 0, rtol=1.e-5, atol=1.e-6)


@pytest.mark.parametrize('gate', list(UnivGate))
def test_lookup_std_kdtree(gate):
    gate = UnivGate.Z
    matrix = np.array(gate)
    nn = KDTreeNN(cliffordt_seqs(5))
    node, lookup_error = nn.lookup(matrix)
    seq = [n.data for n in node.children]
    v = reduce(lambda a, b: a @ b, seq, np.eye(2))
    mvec = vec(matrix)
    vvec = vec(v)
    print(mvec)
    print(vvec)
    assert np.allclose(np.array(v), node.data)  # self-consistent
    error = dist(np.array(v), np.array(matrix))
    assert np.isclose(error, lookup_error)
    assert np.isclose(error, 0)
