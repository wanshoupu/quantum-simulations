import random
from functools import reduce

from quompiler.construct.cmat import CUnitary
from quompiler.construct.quompiler import quompile
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mgen import cyclic_matrix, random_unitary
import numpy as np

random.seed(42)
np.random.seed(42)
formatter = MatrixFormatter(precision=2)


def test_compile_identity_matrix():
    n = 3
    dim = 1 << n
    u = np.eye(dim)
    bc = quompile(u)
    assert bc is not None
    assert np.array_equal(bc.data.matrix, np.eye(bc.data.matrix.shape[0]))
    assert bc.children == []


def test_compile_sing_qubit_circuit():
    n = 1
    dim = 1 << n
    u = random_unitary(dim)
    bc = quompile(u)
    # print(bc)
    assert bc is not None
    assert 1 == len([a for a in bc])
    assert isinstance(bc.data, CUnitary)


def test_compile_cyclic():
    u = cyclic_matrix(8, 1)
    bc = quompile(u)
    # print(bc)
    assert bc is not None
    assert 18 == len([a for a in bc])
    leaves = [a.data.inflate() for a in bc if isinstance(a.data, CUnitary)]
    v = reduce(lambda a, b: a @ b, leaves)
    assert np.allclose(v, u), f'circuit != input:\ncompiled=\n{formatter.tostr(v)},\ninput=\n{formatter.tostr(u)}'


def test_interp_random_unitary():
    for _ in range(10):
        print(f'Test {_}th round')
        n = random.randint(1, 4)
        dim = 1 << n
        m = random_unitary(dim)
        bc = quompile(m)
        leaves = [a.data.inflate() for a in bc if isinstance(a.data, CUnitary)]
        v = reduce(lambda a, b: a @ b, leaves)
        assert np.allclose(v, m), f'circuit != input:\ncompiled=\n{formatter.tostr(v)},\ninput=\n{formatter.tostr(m)}'
