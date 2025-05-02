import random
from itertools import product

import numpy as np
import pytest
from numpy import kron

from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.inter_product import inter_product, mesh_product
from quompiler.utils.inter_product import kron_factor, mykron, mesh_factor, recursive_kron_factor, inter_factor, int_factors
from quompiler.utils.mfun import allprop
from quompiler.utils.mgen import random_unitary

formatter = MatrixFormatter(precision=2)


def manual_inter_product(A, B, m):
    """
    same as quompiler.utils.inter_product.inter_product but for verification purposes
    """
    assert len(A.shape) == 2 and A.shape[0] == A.shape[1]
    N = A.shape[0]
    assert len(B.shape) == 2 and B.shape[0] == B.shape[1]
    n = B.shape[0]

    # no change to A
    if n == 1:
        return A * B[0, 0]
    if N % m:
        raise ValueError(f'The dimension of A must be divisible by m but got dim={N} and m={m}')
    if m == 1:
        return kron(A, B)

    if N == m:
        return kron(B, A)

    C = np.zeros((N * n, N * n), dtype=np.complexfloating)
    for i, j in product(range(0, N, m), range(0, N, m)):
        for k, l in product(range(n), range(n)):
            C[n * i + k * m:n * i + (k + 1) * m, n * j + l * m:n * j + (l + 1) * m] = A[i:i + m, j:j + m] * B[k, l]
    return C


def test_mykron():
    ms = random_unitary(3), random_unitary(2), random_unitary(5), random_unitary(3),
    kk = mykron(*ms)
    assert np.allclose(kk, kron(kron(kron(ms[0], ms[1]), ms[2]), ms[3]))


def test_int_factors():
    n = 3 * 2 * 6 * 5 * 7
    factors = int_factors(n)
    assert all(a * b == n for a, b in factors)
    assert sorted(factors) == factors
    assert (1, n) not in factors
    assert (n, 1) not in factors


def test_inter_product():
    """
    test the configuration of Kronecker product A ⨁ I ⨁ C
    """
    coms = random_unitary(5), random_unitary(3)
    A = kron(*coms)
    B = np.eye(2)

    expected = kron(kron(coms[0], B), coms[1])
    # pprint(expected, num_columns=10000)
    actual = inter_product(A, B, 3)
    assert np.allclose(actual, expected)


def test_sandwich_product_arbitray_matrix():
    """
        let m = len(A), n = len(I), l = len(C)
        then the Kproduct, A ⨁ I ⨁ C, is formed by
        1. get the Kproduct K = A ⨁ C
        2. divide up K
    """
    A = random_unitary(15)
    B = random_unitary(2)

    actual = inter_product(A, B, 5)
    expected = manual_inter_product(A, B, 5)
    assert np.allclose(actual, expected)


def test_inter_product_8_2_4():
    """
        let m = len(A), n = len(I), l = len(C)
        then the Kproduct, A ⨁ I ⨁ C, is formed by
        1. get the Kproduct K = A ⨁ C
        2. divide up K
    """
    A = random_unitary(8)
    B = random_unitary(3)

    m = 2
    actual = inter_product(A, B, m)
    expected = manual_inter_product(A, B, m)
    assert np.allclose(actual, expected)


def test_inter_product_left_kron():
    coms = random_unitary(5), random_unitary(2)
    C = mykron(*coms)

    E = random_unitary(3)

    # execute
    actual = inter_product(C, E, 10)
    expected = mykron(E, coms[0], coms[1])
    assert np.allclose(actual, expected)


def test_inter_product_right_kron():
    coms = random_unitary(5), random_unitary(2)
    C = mykron(*coms)
    E = random_unitary(3)

    # execute
    actual = inter_product(C, E, 1)

    expected = mykron(coms[0], coms[1], E)
    assert np.allclose(actual, expected)


def test_inter_product_5_3_2():
    coms = random_unitary(5), random_unitary(2)
    C = mykron(*coms)
    E = random_unitary(3)

    # execute
    actual = inter_product(C, E, 2)

    # print('Z', flush=True)
    # pprint(Z, num_columns=10000)
    expected = mykron(coms[0], E, coms[1])
    assert np.allclose(actual, expected)


def test_mesh_product_2_3_4():
    coms = random_unitary(2), random_unitary(2), random_unitary(2)
    A = mykron(*coms)

    E = random_unitary(3)

    # execute
    actual = mesh_product([A, E], [2, 8])

    expected = mykron(coms[0], E, coms[1], coms[2])

    assert np.allclose(actual, expected)


def test_mesh_product_4_3_2():
    coms = random_unitary(2), random_unitary(2), random_unitary(2)
    A = mykron(*coms)

    E = random_unitary(3)

    # execute
    actual = mesh_product([A, E], [4, 8])

    expected = mykron(coms[0], coms[1], E, coms[2])
    assert np.allclose(actual, expected)


def test_mesh_product_equiv_kron_reverse_order():
    n = 5
    A = random_unitary(n)
    E = random_unitary(3)
    # execute
    actual = mesh_product([A, E], [1, n])

    expected = kron(E, A)
    assert np.allclose(actual, expected)


def test_mesh_product_inter_product_1():
    coms = np.array([[2, 3], [4, 5]]), np.array([[6, 7], [8, 9]]), np.array([[10, 11], [12, 13]])
    A = mykron(*coms)
    E = np.eye(2)

    # execute
    actual = mesh_product([A, E], [4, 8])
    # print('actual')
    # print(formatter.tostr(actual))

    expected = mykron(coms[0], coms[1], E, coms[2])
    # print('expected')
    # print(formatter.tostr(expected))
    assert np.allclose(actual, expected)


def test_mesh_product_inter_product_2():
    coms = np.array([[2, 3], [4, 5]]), np.array([[6, 7], [8, 9]]), np.array([[10, 11], [12, 13]])
    A = inter_product(mykron(*coms), np.eye(2), 2)
    E = np.eye(2)
    # execute
    actual = mesh_product([A, E], [2, 16])

    expected = mykron(coms[0], E, mykron(coms[1], np.eye(2), coms[2]))
    # print('expected')
    # print(formatter.tostr(expected))
    assert np.allclose(actual, expected)


def test_mesh_product_eyes_16_3_2_3_2():
    coms = np.array([[2, 3], [4, 5]]), np.array([[6, 7], [8, 9]]), np.array([[10, 11], [12, 13]])
    A = mykron(*coms)
    E = np.eye(2)
    F = np.eye(2)

    # execute
    actual = mesh_product([A, E, F], [2, 4, 8])
    # print('actual')
    # print(formatter.tostr(actual))

    expected = mykron(coms[0], E, coms[1], F, coms[2])
    # print('expected')
    # print(formatter.tostr(expected))
    assert np.allclose(actual, expected)


def test_mesh_product_16_3_2_3_2():
    coms = random_unitary(2), random_unitary(2), random_unitary(2)
    A = mykron(*coms)

    E = random_unitary(2)

    F = random_unitary(2)
    G = random_unitary(2)

    # execute
    actual = mesh_product((A, E, F, G), (2, 4, 4, 8))
    # print('actual')
    # print(formatter.tostr(actual))

    expected = mykron(coms[0], E, coms[1], F, G, coms[2])
    # print('expected')
    # print(formatter.tostr(expected))
    assert np.allclose(actual, expected)


def test_kron_factor_singleton():
    n = 3 * 2
    m = random_unitary(n)
    bfactors = kron_factor(m)
    assert len(bfactors) == 1
    assert np.array_equal(bfactors[0], m)


@pytest.mark.parametrize("a,b", [
    (2, 2),
    (2, 3),
    (3, 5),
])
def test_kron_factor_kron_two(a, b):
    m = np.kron(random_unitary(a), random_unitary(b))
    bfactors = kron_factor(m)
    assert len(bfactors) == 2
    assert bfactors[0].shape == (a, a)
    assert bfactors[1].shape == (b, b)


@pytest.mark.parametrize("a,b", [
    (2, 2),
    (2, 3),
    (3, 5),
])
def test_kron_factor_with_right_identity(a, b):
    m = np.kron(random_unitary(a), np.eye(b))
    bfactors = kron_factor(m)
    assert len(bfactors) == 2
    assert bfactors[0].shape == (a, a)
    assert bfactors[1].shape == (b, b)


@pytest.mark.parametrize("a,b", [
    (2, 2),
    (2, 3),
    (3, 5),
])
def test_kron_factor_with_left_identity(a, b):
    m = np.kron(np.eye(a), random_unitary(b))
    bfactors = kron_factor(m)
    assert len(bfactors) == 2
    assert bfactors[0].shape == (a, a)
    assert bfactors[1].shape == (b, b)


def test_recursive_kron_factor():
    dims = 2, 3, 5, 4
    m = mykron(*[random_unitary(d) for d in dims])
    bfactors = recursive_kron_factor(m)
    assert len(bfactors) == 4
    assert all(bfactors[i].shape == (d, d) for i, d in enumerate(dims))
    assert np.allclose(mykron(*bfactors), m)


def test_recursive_kron_factor_random():
    dims = [2, 3, 5, 4]
    for _ in range(10):
        random.shuffle(dims)
        m = mykron(*[random_unitary(d) for d in dims])
        bfactors = recursive_kron_factor(m)
        assert len(bfactors) == len(dims)
        assert all(bfactors[i].shape == (d, d) for i, d in enumerate(dims))
        assert np.allclose(mykron(*bfactors), m)


def test_inter_factor_random():
    a, b = random_unitary(6), random_unitary(2)
    # print(f'a=\n{formatter.tostr(a)}')
    # print(f'b=\n{formatter.tostr(b)}')
    m = inter_product(a, b, 2)
    ms, factors = inter_factor(m)
    assert len(factors) == 1
    dough, matrices = ms
    # print(f'dough=\n{formatter.tostr(dough)}')
    # print(f'matrices=\n{formatter.tostr(matrices)}')
    p, _ = allprop(inter_product(dough, matrices, 2), m)
    assert p
    dp, _ = allprop(dough, a)
    assert dp, f'dough=\n{formatter.tostr(dough)}\nexpected=\n{formatter.tostr(a)}'
    yp, _ = allprop(matrices, b)
    assert yp, f'matrices=\n{formatter.tostr(matrices)}\nexpected=\n{formatter.tostr(b)}'


def test_inter_factor_identity_factors():
    a, b = random_unitary(6), np.eye(2)
    # print(f'a=\n{formatter.tostr(a)}')
    # print(f'b=\n{formatter.tostr(b)}')
    m = inter_product(a, b, 2)
    ms, factors = inter_factor(m)
    assert len(factors) == 1
    dough, matrices = ms
    # print(f'dough=\n{formatter.tostr(dough)}')
    # print(f'matrices=\n{formatter.tostr(matrices)}')
    p, _ = allprop(inter_product(dough, matrices, 2), m)
    assert p
    dp, _ = allprop(dough, a)
    assert dp, f'dough=\n{formatter.tostr(dough)}\nexpected=\n{formatter.tostr(a)}'
    yp, _ = allprop(matrices, b)
    assert yp, f'matrices=\n{formatter.tostr(matrices)}\nexpected=\n{formatter.tostr(b)}'


def test_inter_factor_all_identities():
    a, b = np.eye(6), np.eye(2)
    # print(f'a=\n{formatter.tostr(a)}')
    # print(f'b=\n{formatter.tostr(b)}')
    m = inter_product(a, b, 2)
    ms, factors = inter_factor(m)
    assert len(factors) == 1
    dough, matrices = ms
    # print(f'dough=\n{formatter.tostr(dough)}')
    # print(f'matrices=\n{formatter.tostr(matrices)}')
    p, _ = allprop(inter_product(dough, matrices, 2), m)
    assert p
    dp, _ = allprop(dough, a)
    assert dp, f'dough=\n{formatter.tostr(dough)}\nexpected=\n{formatter.tostr(a)}'
    yp, _ = allprop(matrices, b)
    assert yp, f'matrices=\n{formatter.tostr(matrices)}\nexpected=\n{formatter.tostr(b)}'


def test_mesh_product_u_eye3():
    """
    This test is to demonstrate the effect of order between dough and matrices
    When factor = 1, it means divide the dough into one block (that is, dough, the matrix itself), and multiplying all the matrices on the left.
    """
    n = 3
    coms = [np.array([[2, 3], [4, 5]])] + [np.eye(2) for _ in range(n)]
    expected = mykron(*coms[1:], coms[0])
    # print('expected')
    # print(formatter.tostr(expected))

    # execute
    factors = [1] * 3 + [2]
    u = mesh_product(coms, factors)
    assert np.allclose(u, expected)


def test_mesh_factor_inter_product_1():
    A = mykron(np.array([[2, 3], [4, 5]]), np.array([[6, 7], [8, 9]]), np.array([[10, 11], [12, 13]]))
    E = np.eye(2)
    expected = inter_product(A, E, 2)
    # print('expected')
    # print(formatter.tostr(expected))

    # execute
    matrices, factors = mesh_factor(expected)
    assert len(factors) == 4 == len(matrices)
    u = mesh_product(matrices, factors)
    assert np.allclose(u, expected)


def test_mesh_factor_inter_product_2():
    coms = np.array([[2, 3], [4, 5]]), np.array([[6, 7], [8, 9]]), np.array([[10, 11], [12, 13]])
    A = inter_product(mykron(*coms), np.eye(2), 2)
    E = np.eye(2)
    test = inter_product(A, E, 8)

    # execute
    matrices, factors = mesh_factor(test)

    assert len(factors) == 5 == len(matrices)
    u = mesh_product(matrices, factors)
    assert np.allclose(u, test)


def test_mesh_factor_eyes_16_3_2_3_2():
    coms = np.array([[2, 3], [4, 5]]), np.array([[6, 7], [8, 9]]), np.array([[10, 11], [12, 13]])
    A = mykron(*coms)
    E = np.eye(2)
    F = np.eye(2)
    test = mesh_product((A, E, F), (2, 4, 8))
    # print('test')
    # print(formatter.tostr(test))

    # execute

    matrices, factors = mesh_factor(test)

    assert len(factors) == 5 == len(matrices)
    u = mesh_product(matrices, factors)
    assert np.allclose(u, test)


def test_mesh_factor_16_3_2_3_2():
    coms = random_unitary(2), random_unitary(2), random_unitary(2)
    A = mykron(*coms)

    E = random_unitary(2)

    F = random_unitary(2)
    test = mesh_product((A, E, F), (2, 4, 8))
    # print('test')
    # print(formatter.tostr(test))

    # execute
    matrices, factors = mesh_factor(test)

    assert len(factors) == 5 == len(matrices)
    u = mesh_product(matrices, factors)
    assert np.allclose(u, test)


def test_mesh_factor_3_yeast():
    a, b, c, d = random_unitary(6), random_unitary(2), np.eye(3), random_unitary(3)
    test = mesh_product([a, b, c, d], [2, 2, 6, 6])

    # execute
    matrices, factors = mesh_factor(test)

    assert len(factors) == 4 == len(matrices)
    u = mesh_product(matrices, factors)
    assert np.allclose(u, test)


def test_mesh_factor_eye_yeast():
    a, b, c, d = random_unitary(6), np.eye(2), np.eye(4), np.eye(4)
    test = mesh_product([a, b, c, d], [2, 2, 6, 6])

    # execute
    matrices, factors = mesh_factor(test)

    # The np.eye(4) are also factored into np.eye(2). Therefore we have 6 factors
    assert len(factors) == 6 == len(matrices)
    assert factors[-1] == matrices[0].shape[0]
    u = mesh_product(matrices, factors)
    assert np.allclose(u, test)
    assert all(np.allclose(y, np.eye(y.shape[0])) for y in matrices[1:])
