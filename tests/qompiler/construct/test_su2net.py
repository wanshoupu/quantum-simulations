import random
from functools import reduce

import numpy as np
import pytest

from quompiler.construct.su2net import SU2Net
from quompiler.construct.types import UnivGate
from quompiler.optimize.code_analyze import tree_stats
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mfun import dist, gphase
from quompiler.utils.mgen import random_unitary, random_su2

formatter = MatrixFormatter(precision=2)


def test_init_lazy():
    error = .2
    su2net = SU2Net(error)
    assert su2net.error == error
    # tree not constructed until first lookup
    assert not su2net.constructed


@pytest.mark.parametrize('gate', list(UnivGate))
def test_init_lookup_std(gate):
    su2net = SU2Net(.2)
    gph = gphase(gate.matrix)
    matrix = gate.matrix / gph
    node = su2net.lookup(matrix)
    seq = [n.data for n in node.children]
    v = reduce(lambda a, b: a @ b, seq, np.eye(2))
    assert np.allclose(np.array(v), node.data)  # self-consistent
    error = dist(np.array(v), np.array(matrix))
    print(f'{gate} ~{seq} lookup error: {error}')
    assert error <= su2net.error * 10


@pytest.mark.parametrize("seed", [42, 123, 999, 2023])
def test_init_lookup_terminates(seed):
    su2net = SU2Net(.3)
    random.seed(seed)
    np.random.seed(seed)
    u = random_su2()
    node = su2net.lookup(u)
    v = reduce(lambda a, b: a @ b, [n.data for n in node.children], np.eye(2))
    assert np.allclose(np.array(v), node.data)  # self-consistent
    error = dist(np.array(v), np.array(u))
    print(f'\n{formatter.tostr(u)}\nlookup error: {error}')
    assert error < su2net.error * 10


@pytest.mark.parametrize("seed", [42, 123, 999, 2023])
def test_init_lookup_errors(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    su2net = SU2Net(.5)
    u = random_su2()
    su2net.lookup(u)
