from functools import reduce

import numpy as np
import pytest
import sympy

from quompiler.utils.format_matrix import MatrixFormatter

formatter = MatrixFormatter(precision=2)


def mat(multiplier):
    return multiplier * np.array([[1, 3], [2, 4]])


def kron(mats):
    return reduce(lambda a, b: np.kron(a, b), mats)


def printm(m):
    print(formatter.tostr(m))


@pytest.mark.parametrize("n,k", [
    (4, 0),
    (4, 1),
    (4, 2),
    (4, 3),
])
def test_kron(n, k):
    print()
    print(n, k)
    ms = [mat(1) if i == k else np.eye(2) for i in range(n)]
    # for m in ms:
    #     printm(m)
    k = kron(ms)
    print()
    printm(k)


def test_foo():
    a = sympy.symarray(prefix='a', shape=(2, 2))
    print()
    print(a)
    b = sympy.kronecker_product(a, sympy.eye(2))
    print()
    print(b)
