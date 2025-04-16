from common.construct.quompiler import quompile
from common.utils.mgen import cyclic_matrix
import numpy as np


def test_compile_identity_matrix():
    n = 3
    dim = 1 << n
    u = np.eye(dim)
    bc = quompile(u)
    print(bc)


def test_compile():
    u = cyclic_matrix(8, 1)
    bc = quompile(u)
    print(bc)
