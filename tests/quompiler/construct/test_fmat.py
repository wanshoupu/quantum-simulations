import numpy as np
from sympy import partition

from quompiler.construct.fmat import FactorMat
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.inter_product import mykron, inter_product, mesh_product
from quompiler.utils.mgen import random_unitary

formatter = MatrixFormatter(precision=2)


def test_FactorMat_init():
    fu = FactorMat([random_unitary(6), np.eye(3)], (3, 6))
    assert fu


def test_FactorMat_inflate():
    n = 6
    matrices = [random_unitary(n), np.eye(3)]
    partition = 3
    fu = FactorMat(matrices, (partition, n))
    # print()
    # print(formatter.tostr(mat))
    expected = inter_product(*matrices, n // partition)
    assert np.allclose(fu.inflate(), expected)


def test_FactorMat_deflate_kron_product():
    coms = random_unitary(2), np.eye(3), random_unitary(2), np.eye(3), random_unitary(2)
    mat = mykron(*coms)
    fu = FactorMat.deflate(mat)
    assert len(fu.matrices) == len(coms)
    assert np.allclose(fu.inflate(), mat)
    assert all(np.allclose(a @ a.conj().T, np.eye(a.shape[0])) for a in fu.matrices)


def test_FactorMat_deflate_inter_product():
    coms = random_unitary(8), np.eye(2), random_unitary(2)
    partitions = 2, 4, 8
    mat = mesh_product(coms, partitions)
    fu = FactorMat.deflate(mat)
    assert len(fu.matrices) == len(coms)
    assert np.allclose(fu.inflate(), mat)
    assert all(np.allclose(a @ a.conj().T, np.eye(a.shape[0])) for a in fu.matrices)
