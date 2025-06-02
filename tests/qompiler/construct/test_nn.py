from functools import reduce

import numpy as np
import pytest

from quompiler.construct.nn.nn import BruteNN, AutoNN
from quompiler.construct.types import UnivGate
from quompiler.utils.group_su2 import dist, vec
from quompiler.utils.std_decompose import cliffordt_seqs


@pytest.mark.parametrize('gate', list(UnivGate))
def test_lookup_std_autonn(gate):
    gate = UnivGate.Z
    matrix = gate.matrix
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
    assert np.allclose(mvec, vvec)
    assert np.isclose(error, lookup_error) and np.isclose(error, 0)

@pytest.mark.parametrize('gate', list(UnivGate))
def test_lookup_std(gate):
    gate = UnivGate.Z
    matrix = gate.matrix
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
    assert np.allclose(mvec, vvec)
    assert np.isclose(error, lookup_error) and np.isclose(error, 0)
