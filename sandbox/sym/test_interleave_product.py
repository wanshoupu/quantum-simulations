import textwrap
from itertools import product

import sympy
from sympy import Matrix, symbols, kronecker_product as kron
from sympy import pprint
from sympy.printing.pretty import pretty


def test_kron():
    # Define symbolic variables
    a, b, c, d = symbols('a b c d')
    e, f, g, h = symbols('e f g h')

    # Define two symbolic matrices
    A = Matrix([[a, b], [c, d]])
    B = sympy.eye(2)
    C = Matrix([[e, f], [g, h]])

    # Compute the Kronecker product
    K = kron(A, B, C)

    print()
    pprint(K)
    expected = """
        ⎡a⋅e  a⋅f   0    0   b⋅e  b⋅f   0    0 ⎤
        ⎢                                      ⎥
        ⎢a⋅g  a⋅h   0    0   b⋅g  b⋅h   0    0 ⎥
        ⎢                                      ⎥
        ⎢ 0    0   a⋅e  a⋅f   0    0   b⋅e  b⋅f⎥
        ⎢                                      ⎥
        ⎢ 0    0   a⋅g  a⋅h   0    0   b⋅g  b⋅h⎥
        ⎢                                      ⎥
        ⎢c⋅e  c⋅f   0    0   d⋅e  d⋅f   0    0 ⎥
        ⎢                                      ⎥
        ⎢c⋅g  c⋅h   0    0   d⋅g  d⋅h   0    0 ⎥
        ⎢                                      ⎥
        ⎢ 0    0   c⋅e  c⋅f   0    0   d⋅e  d⋅f⎥
        ⎢                                      ⎥
        ⎣ 0    0   c⋅g  c⋅h   0    0   d⋅g  d⋅h⎦
    """
    assert pretty(K) == textwrap.dedent(expected).strip()


def test_kron_3qubit_sandwich():
    """
    test the configuration of unitary matrix acting on first and third qubit with the second qubit unchanged.
    """
    # Define symbolic variables
    a, b, c, d = symbols('a b c d')
    e, f, g, h = symbols('e f g h')
    i, j, k, l = symbols('i j k l')
    m, n, o, p = symbols('m n o p')
    A = Matrix([[a, b, c, d], [e, f, g, h], [i, j, k, l], [m, n, o, p]])
    B = sympy.eye(2)
    swap = Matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    # Compute the Kronecker product
    K = kron(A, B)
    print()
    pprint(K)

    s12 = kron(sympy.eye(2), swap)
    final = s12 * K * s12

    print()
    pprint(final)


def test_swap_higher_d():
    swap = Matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    I = sympy.eye(2)
    print('x ⨁ I')
    pprint(kron(I, swap))
    print('I ⨁ x')
    pprint(kron(swap, I))


def test_kron_4qubit_sandwich():
    """
    test the configuration of unitary matrix acting on first and third qubit with the second qubit unchanged.
    """
    # Define symbolic variables
    A = square_m(2)
    print('A')
    pprint(A)
    B = sympy.eye(5)
    swap = Matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    K = kron(A, B)
    print('K')
    pprint(K)

    s12 = kron(sympy.eye(2), sympy.eye(2), swap, )
    final = s12 * K * s12

    print('final')
    pprint(final)
    assert K == final


def test_inspiration_sandwich_product():
    """
    test the configuration of Kronecker product A ⨁ I ⨁ C
    """
    # Define symbolic variables
    A = square_m(5, 'a')
    print('A')
    pprint(A)

    B = sympy.eye(2)
    print('B')
    pprint(B)

    C = square_m(3, 'c')
    print('C')
    pprint(C)
    D = kron(A, C)
    print('D')
    pprint(D, num_columns=10000)
    E = kron(A, B, C)
    print('E')
    pprint(E, num_columns=10000)


def test_interleave_product():
    """
    test the configuration of Kronecker product A ⨁ I ⨁ C
    """

    A = kron(square_m(5, 'a'), square_m(3, 'c'))
    B = sympy.eye(2)

    expected = kron(square_m(5, 'a'), B, square_m(3, 'c'))
    # pprint(expected, num_columns=10000)
    assert interleave_product(A, B, 5, 3) == expected


def test_sandwich_product_arbitray_matrix():
    """
        let m = len(A), n = len(I), l = len(C)
        then the Kproduct, A ⨁ I ⨁ C, is formed by
        1. get the Kproduct K = A ⨁ C
        2. divide up K
    """
    # Define symbolic variables
    A = square_m(15, 'a')
    print('A')
    pprint(A, num_columns=10000)

    B = square_m(2, 'b')
    print('B')
    pprint(B)

    C = interleave_product(A, B, 5, 3)
    print('C')
    pprint(C, num_columns=10000)

def test_sandwich_product_8_2_4():
    """
        let m = len(A), n = len(I), l = len(C)
        then the Kproduct, A ⨁ I ⨁ C, is formed by
        1. get the Kproduct K = A ⨁ C
        2. divide up K
    """
    # Define symbolic variables
    A = square_m(8, 'a')
    print('A')
    pprint(A, num_columns=10000)

    B = square_m(3, 'b')
    print('B')
    pprint(B)

    C = interleave_product(A, B, 2, 4)
    print('C')
    pprint(C, num_columns=10000)


def interleave_product(A, B, m, n):
    """
    A matrix product similar to Kronecker product. But matrix A is first divided up into m x m n-sized blocks, then B is Kronecker multiplied between the m x m n-sized blocks.
    For example,
    A = [[
    :param A: square matrix of shape (m*n,m*n).
    :param B: square matrix of shape (k, k), with k > 0.
    :param m: int, denotes the number of blocks to divide the matrix A into.
    :param n: int, denotes the block size to be divided up out of A.
    :return: The interleaved product with A(m) ⨁ B ⨁ A(n)
    """
    sa = A.shape
    assert sa[0] == sa[1]
    sb = B.shape
    assert sb[0] == sb[1]
    assert sa[0] == m * n
    C = sympy.zeros(sa[0] * sb[0])
    for i, j in product(range(0, sa[0], n), range(0, sa[0], n)):
        for k, l in product(range(sb[0]), range(sb[0])):
            C[sb[0] * i+k*n:sb[0] * i + (k+1)*n, sb[0] * j+l*n:sb[0] * j + (l+1)*n] = A[i:i + n, j:j + n] * B[k, l]
    return C


def square_m(n, prefix: str = None):
    if prefix:
        syms = symbols(f'{prefix}:{n}(:{n})')
    else:
        syms = symbols(f':{n}(:{n})')
    A = Matrix(n, n, syms, complex=True)
    return A
