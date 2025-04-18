from quompiler.construct.cmat import CUnitary
from quompiler.construct.quompiler import quompile
from quompiler.utils.mgen import cyclic_matrix, random_unitary
import numpy as np


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


def test_compile():
    u = cyclic_matrix(8, 1)
    bc = quompile(u)
    # print(bc)
    assert bc is not None
    assert 18 == len([a for a in bc])
