import random

import numpy as np
from numpy import kron

from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.inter_product import inter_product, mykron
from quompiler.utils.mgen import random_unitary

formatter = MatrixFormatter(precision=2)


def test_inter_product_reshaping():
    """
    given an inter_product, two different reshaping methods result in different matrix
    1. reshape(a,b,c,a,b,c)->transpose(0,3,1,4,2,5)->reshape(a*a,b*b,c*c)->transpose(0,2,1)
    2. reshape(a*b,c,a*b,c)->transpose(0,2,1,3)->reshape(a*b*a*b,c*c)
    """
    a, b, c = 2, 3, 2
    A = kron(np.arange(1, 1 + 2 ** 2).reshape(2, 2), np.ones((3, 3)))
    B = np.array([[1, 1j], [113, 113j]])
    M = inter_product(A, B, b)
    print()
    print(formatter.tostr(M))
    M1 = rearrange1(M, a, b, c)
    print('rearrange1')
    print(formatter.tostr(M1))

    M2 = rearrange2(M, a * b, c)
    print('rearrange2')
    print(formatter.tostr(M2))


def rearrange1(M, a, b, c):
    """
    reshape(a,b,c,a,b,c)->transpose(0,3,1,4,2,5)->reshape(a*a,b*b,c*c)->transpose(0,2,1)->reshape(a*a*c*c,b*b)
    """
    m2 = (M.reshape(a, b, c, a, b, c)
          .transpose(0, 3, 1, 4, 2, 5)
          .reshape(a * a, b * b, c * c)
          .transpose(0, 2, 1)
          .reshape(a * a * c * c, b * b))
    return m2


def rearrange2(M, a, c):
    """
    reshape(a,c,a,c)->transpose(0,2,1,3)->reshape(a*a,c*c)
    """
    m2 = np.reshape(M, (a, c, a, c)).transpose(0, 2, 1, 3).reshape(a * a, c * c)
    return m2


def test_inter_product_singlet_qubit():
    """
    given a unitary matrix of singlet qubit, study the inter_product of it with a number of eyes.
    """
    matrix = random_unitary(2)
    n = 2
    mats = [np.eye(2) for _ in range(n)]
    k = random.randrange(n)
    mats.insert(k, matrix)
    for m in mats:
        print()
        print(formatter.tostr(m))
    result = mykron(*mats)
    print()
    print(formatter.tostr(result))
