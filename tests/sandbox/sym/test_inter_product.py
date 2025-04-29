from itertools import product

import sympy
from sympy import Matrix, symbols, kronecker_product as kron, pprint

from sandbox.sym.inter_product import inter_product, mesh_product
from sandbox.sym.sym_gen import symmat


def another_inter_product(A, B, m):
    """
    same as sandbox.sym.inter_product.inter_product but for verification purposes
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

    C = sympy.zeros(N * n)
    for i, j in product(range(0, N, m), range(0, N, m)):
        for k, l in product(range(n), range(n)):
            C[n * i + k * m:n * i + (k + 1) * m, n * j + l * m:n * j + (l + 1) * m] = A[i:i + m, j:j + m] * B[k, l]
    return C


def test_kron_equivalence():
    # Define symbolic variables
    a, b, c, d = symbols('a b c d')
    e, f, g, h = symbols('e f g h')

    # Define two symbolic matrices
    A = Matrix([[a, b], [c, d]])
    B = sympy.eye(2)
    C = Matrix([[e, f], [g, h]])

    # Compute the Kronecker product
    expected = kron(A, B, C)
    print('expected')
    pprint(expected)

    actual = inter_product(kron(A, C), B, 2)
    print('actual')
    pprint(actual)
    assert actual == expected


def test_kron_4qubit_sandwich():
    """
    test the configuration of unitary matrix acting on first and third qubit with the second qubit unchanged.
    """
    # Define symbolic variables
    A = symmat(2)
    # print('A')
    # pprint(A)
    B = sympy.eye(8)
    swap = Matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    K = kron(A, B)
    # print('K')
    # pprint(K)

    s12 = kron(sympy.eye(2), sympy.eye(2), swap, )
    final = s12 * K * s12

    # print('final')
    # pprint(final)
    assert K == final


def test_inter_product():
    """
    test the configuration of Kronecker product A ⨁ I ⨁ C
    """
    coms = symmat(5, 'a'), symmat(3, 'c')
    A = kron(*coms)
    B = sympy.eye(2)

    expected = kron(coms[0], B, coms[1])
    # pprint(expected, num_columns=10000)
    assert inter_product(A, B, 3) == expected


def test_sandwich_product_arbitray_matrix():
    """
        let m = len(A), n = len(I), l = len(C)
        then the Kproduct, A ⨁ I ⨁ C, is formed by
        1. get the Kproduct K = A ⨁ C
        2. divide up K
    """
    # Define symbolic variables
    A = symmat(15, 'a')
    # print('A')
    # pprint(A, num_columns=10000)

    B = symmat(2, 'b')
    # print('B')
    # pprint(B)

    C = inter_product(A, B, 5)
    # print('C')
    # pprint(C, num_columns=10000)

    expected = another_inter_product(A, B, 5)
    # print('expected')
    # pprint(expected, num_columns=10000)
    assert C == expected


def test_inter_product_8_2_4():
    """
        let m = len(A), n = len(I), l = len(C)
        then the Kproduct, A ⨁ I ⨁ C, is formed by
        1. get the Kproduct K = A ⨁ C
        2. divide up K
    """
    # Define symbolic variables
    A = symmat(8, 'a')
    # print('A')
    # pprint(A, num_columns=10000)

    B = symmat(3, 'b')
    # print('B')
    # pprint(B)

    m = 2

    actual = inter_product(A, B, m)
    # print('C')
    # pprint(C, num_columns=10000)

    expected = another_inter_product(A, B, m)
    # print('expected')
    # pprint(expected, num_columns=10000)
    assert actual == expected


def test_inter_product_left_kron():
    # Define symbolic variables
    coms = symmat(5, 'a'), symmat(2, 'b')
    C = kron(*coms)
    # print('C')
    # pprint(C, num_columns=10000)

    E = symmat(3, 'e')
    # print('E')
    # pprint(E)

    # execute
    actual = inter_product(C, E, 10)
    # print('actual', flush=True)
    # pprint(actual, num_columns=10000)
    expected = kron(E, coms[0], coms[1])
    assert actual == expected


def test_inter_product_right_kron():
    # Define symbolic variables
    coms = symmat(5, 'a'), symmat(2, 'b')
    C = kron(*coms)
    # print('C')
    # pprint(C, num_columns=10000)

    E = symmat(3, 'e')
    # print('E')
    # pprint(E)

    # execute
    actual = inter_product(C, E, 1)
    # print('actual', flush=True)
    # pprint(actual, num_columns=10000)

    expected = kron(coms[0], coms[1], E)
    assert actual == expected

    expected2 = another_inter_product(C, E, 1)
    assert actual == expected2


def test_inter_product_5_3_2():
    coms = symmat(5, 'a'), symmat(2, 'b')
    C = kron(*coms)
    # print('C')
    # pprint(C, num_columns=10000)

    E = symmat(3, 'e')
    # print('E')
    # pprint(E)

    # execute
    actual = inter_product(C, E, 2)
    # print('actual', flush=True)
    # pprint(actual, num_columns=10000)

    expected = kron(coms[0], E, coms[1])
    assert actual == expected

    expected2 = another_inter_product(C, E, 2)
    assert actual == expected2


def test_mesh_product_2_3_4():
    coms = symmat(2, 'a'), symmat(2, 'b'), symmat(2, 'c')
    A = kron(*coms)
    # print('A')
    # pprint(A, num_columns=10000)

    E = symmat(3, 'e')
    # print('E')
    # pprint(E)

    # execute
    actual = mesh_product(A, (E,), (4,))
    # print('actual', flush=True)
    # pprint(actual, num_columns=10000)

    expected = kron(coms[0], E, coms[1], coms[2])
    # print('expected')
    # pprint(expected, num_columns=10000)

    assert actual == expected


def test_inter_product_4_3_2():
    coms = symmat(2, 'a'), symmat(2, 'b'), symmat(2, 'c')
    A = kron(*coms)
    # print('A')
    # pprint(A, num_columns=10000)

    E = symmat(3, 'e')
    # print('E')
    # pprint(E)

    # execute
    actual = mesh_product(A, (E,), (2,))
    # print('actual', flush=True)
    # pprint(actual, num_columns=10000)

    expected = kron(coms[0], coms[1], E, coms[2])
    # print('expected')
    # pprint(expected, num_columns=10000)

    assert actual == expected


def test_mesh_product_82_2_2():
    coms = symmat(2, 'a'), symmat(2, 'b'), symmat(2, 'c')
    A = kron(*coms)
    # print('\nA')
    # pprint(A, num_columns=10000)

    E = symmat(2, 'e')
    # print('\nE')
    # pprint(E)

    F = symmat(2, 'f')
    # print('\nF')
    # pprint(F, num_columns=10000)

    # execute
    actual = mesh_product(A, (E, F), (4, 2))
    # print('\nactual', flush=True)
    # pprint(actual, num_columns=10000)

    expected = kron(coms[0], E, coms[1], F, coms[2])
    # print('\nexpected')
    # pprint(expected, num_columns=10000)
    assert actual == expected


def test_mesh_product_84_2():
    coms = symmat(2, 'a'), symmat(2, 'b'), symmat(2, 'c')
    A = kron(*coms)
    # print('\nA')
    # pprint(A, num_columns=10000)

    E = symmat(2, 'e')
    # print('\nE')
    # pprint(E)

    F = symmat(2, 'f')
    # print('\nF')
    # pprint(F, num_columns=10000)

    # execute
    actual = mesh_product(A, (E, F), (2, 2))
    # print('\nactual', flush=True)
    # pprint(actual, num_columns=10000)

    expected = kron(coms[0], coms[1], E, F, coms[2])
    # print('\nexpected')
    # pprint(expected, num_columns=10000)
    assert actual == expected
