import textwrap
from itertools import product
from textwrap import dedent

import sympy
from quimb import expec
from sympy import Matrix, symbols, kronecker_product as kron
from sympy import pprint
from sympy.printing.pretty import pretty
import numpy as np


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

    # print()
    # pprint(K)
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


def test_kron_4qubit_sandwich():
    """
    test the configuration of unitary matrix acting on first and third qubit with the second qubit unchanged.
    """
    # Define symbolic variables
    A = square_m(2)
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
    coms = square_m(5, 'a'), square_m(3, 'c')
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
    A = square_m(15, 'a')
    # print('A')
    # pprint(A, num_columns=10000)

    B = square_m(2, 'b')
    # print('B')
    # pprint(B)

    C = inter_product(A, B, 5)
    # print('C')
    # pprint(C, num_columns=10000)

    expected = """
        ⎡a₀₀⋅b₀₀   a₀₁⋅b₀₀   a₀₂⋅b₀₀   a₀₃⋅b₀₀   a₀₄⋅b₀₀   a₀₀⋅b₀₁   a₀₁⋅b₀₁   a₀₂⋅b₀₁   a₀₃⋅b₀₁   a₀₄⋅b₀₁   a₀₅⋅b₀₀   a₀₆⋅b₀₀   a₀₇⋅b₀₀   a₀₈⋅b₀₀   a₀₉⋅b₀₀   a₀₅⋅b₀₁   a₀₆⋅b₀₁   a₀₇⋅b₀₁   a₀₈⋅b₀₁   a₀₉⋅b₀₁   a₀₁₀⋅b₀₀   a₀₁₁⋅b₀₀   a₀₁₂⋅b₀₀   a₀₁₃⋅b₀₀   a₀₁₄⋅b₀₀   a₀₁₀⋅b₀₁   a₀₁₁⋅b₀₁   a₀₁₂⋅b₀₁   a₀₁₃⋅b₀₁   a₀₁₄⋅b₀₁ ⎤
        ⎢                                                                                                                                                                                                                                                                                                                    ⎥
        ⎢a₁₀⋅b₀₀   a₁₁⋅b₀₀   a₁₂⋅b₀₀   a₁₃⋅b₀₀   a₁₄⋅b₀₀   a₁₀⋅b₀₁   a₁₁⋅b₀₁   a₁₂⋅b₀₁   a₁₃⋅b₀₁   a₁₄⋅b₀₁   a₁₅⋅b₀₀   a₁₆⋅b₀₀   a₁₇⋅b₀₀   a₁₈⋅b₀₀   a₁₉⋅b₀₀   a₁₅⋅b₀₁   a₁₆⋅b₀₁   a₁₇⋅b₀₁   a₁₈⋅b₀₁   a₁₉⋅b₀₁   a₁₁₀⋅b₀₀   a₁₁₁⋅b₀₀   a₁₁₂⋅b₀₀   a₁₁₃⋅b₀₀   a₁₁₄⋅b₀₀   a₁₁₀⋅b₀₁   a₁₁₁⋅b₀₁   a₁₁₂⋅b₀₁   a₁₁₃⋅b₀₁   a₁₁₄⋅b₀₁ ⎥
        ⎢                                                                                                                                                                                                                                                                                                                    ⎥
        ⎢a₂₀⋅b₀₀   a₂₁⋅b₀₀   a₂₂⋅b₀₀   a₂₃⋅b₀₀   a₂₄⋅b₀₀   a₂₀⋅b₀₁   a₂₁⋅b₀₁   a₂₂⋅b₀₁   a₂₃⋅b₀₁   a₂₄⋅b₀₁   a₂₅⋅b₀₀   a₂₆⋅b₀₀   a₂₇⋅b₀₀   a₂₈⋅b₀₀   a₂₉⋅b₀₀   a₂₅⋅b₀₁   a₂₆⋅b₀₁   a₂₇⋅b₀₁   a₂₈⋅b₀₁   a₂₉⋅b₀₁   a₂₁₀⋅b₀₀   a₂₁₁⋅b₀₀   a₂₁₂⋅b₀₀   a₂₁₃⋅b₀₀   a₂₁₄⋅b₀₀   a₂₁₀⋅b₀₁   a₂₁₁⋅b₀₁   a₂₁₂⋅b₀₁   a₂₁₃⋅b₀₁   a₂₁₄⋅b₀₁ ⎥
        ⎢                                                                                                                                                                                                                                                                                                                    ⎥
        ⎢a₃₀⋅b₀₀   a₃₁⋅b₀₀   a₃₂⋅b₀₀   a₃₃⋅b₀₀   a₃₄⋅b₀₀   a₃₀⋅b₀₁   a₃₁⋅b₀₁   a₃₂⋅b₀₁   a₃₃⋅b₀₁   a₃₄⋅b₀₁   a₃₅⋅b₀₀   a₃₆⋅b₀₀   a₃₇⋅b₀₀   a₃₈⋅b₀₀   a₃₉⋅b₀₀   a₃₅⋅b₀₁   a₃₆⋅b₀₁   a₃₇⋅b₀₁   a₃₈⋅b₀₁   a₃₉⋅b₀₁   a₃₁₀⋅b₀₀   a₃₁₁⋅b₀₀   a₃₁₂⋅b₀₀   a₃₁₃⋅b₀₀   a₃₁₄⋅b₀₀   a₃₁₀⋅b₀₁   a₃₁₁⋅b₀₁   a₃₁₂⋅b₀₁   a₃₁₃⋅b₀₁   a₃₁₄⋅b₀₁ ⎥
        ⎢                                                                                                                                                                                                                                                                                                                    ⎥
        ⎢a₄₀⋅b₀₀   a₄₁⋅b₀₀   a₄₂⋅b₀₀   a₄₃⋅b₀₀   a₄₄⋅b₀₀   a₄₀⋅b₀₁   a₄₁⋅b₀₁   a₄₂⋅b₀₁   a₄₃⋅b₀₁   a₄₄⋅b₀₁   a₄₅⋅b₀₀   a₄₆⋅b₀₀   a₄₇⋅b₀₀   a₄₈⋅b₀₀   a₄₉⋅b₀₀   a₄₅⋅b₀₁   a₄₆⋅b₀₁   a₄₇⋅b₀₁   a₄₈⋅b₀₁   a₄₉⋅b₀₁   a₄₁₀⋅b₀₀   a₄₁₁⋅b₀₀   a₄₁₂⋅b₀₀   a₄₁₃⋅b₀₀   a₄₁₄⋅b₀₀   a₄₁₀⋅b₀₁   a₄₁₁⋅b₀₁   a₄₁₂⋅b₀₁   a₄₁₃⋅b₀₁   a₄₁₄⋅b₀₁ ⎥
        ⎢                                                                                                                                                                                                                                                                                                                    ⎥
        ⎢a₀₀⋅b₁₀   a₀₁⋅b₁₀   a₀₂⋅b₁₀   a₀₃⋅b₁₀   a₀₄⋅b₁₀   a₀₀⋅b₁₁   a₀₁⋅b₁₁   a₀₂⋅b₁₁   a₀₃⋅b₁₁   a₀₄⋅b₁₁   a₀₅⋅b₁₀   a₀₆⋅b₁₀   a₀₇⋅b₁₀   a₀₈⋅b₁₀   a₀₉⋅b₁₀   a₀₅⋅b₁₁   a₀₆⋅b₁₁   a₀₇⋅b₁₁   a₀₈⋅b₁₁   a₀₉⋅b₁₁   a₀₁₀⋅b₁₀   a₀₁₁⋅b₁₀   a₀₁₂⋅b₁₀   a₀₁₃⋅b₁₀   a₀₁₄⋅b₁₀   a₀₁₀⋅b₁₁   a₀₁₁⋅b₁₁   a₀₁₂⋅b₁₁   a₀₁₃⋅b₁₁   a₀₁₄⋅b₁₁ ⎥
        ⎢                                                                                                                                                                                                                                                                                                                    ⎥
        ⎢a₁₀⋅b₁₀   a₁₁⋅b₁₀   a₁₂⋅b₁₀   a₁₃⋅b₁₀   a₁₄⋅b₁₀   a₁₀⋅b₁₁   a₁₁⋅b₁₁   a₁₂⋅b₁₁   a₁₃⋅b₁₁   a₁₄⋅b₁₁   a₁₅⋅b₁₀   a₁₆⋅b₁₀   a₁₇⋅b₁₀   a₁₈⋅b₁₀   a₁₉⋅b₁₀   a₁₅⋅b₁₁   a₁₆⋅b₁₁   a₁₇⋅b₁₁   a₁₈⋅b₁₁   a₁₉⋅b₁₁   a₁₁₀⋅b₁₀   a₁₁₁⋅b₁₀   a₁₁₂⋅b₁₀   a₁₁₃⋅b₁₀   a₁₁₄⋅b₁₀   a₁₁₀⋅b₁₁   a₁₁₁⋅b₁₁   a₁₁₂⋅b₁₁   a₁₁₃⋅b₁₁   a₁₁₄⋅b₁₁ ⎥
        ⎢                                                                                                                                                                                                                                                                                                                    ⎥
        ⎢a₂₀⋅b₁₀   a₂₁⋅b₁₀   a₂₂⋅b₁₀   a₂₃⋅b₁₀   a₂₄⋅b₁₀   a₂₀⋅b₁₁   a₂₁⋅b₁₁   a₂₂⋅b₁₁   a₂₃⋅b₁₁   a₂₄⋅b₁₁   a₂₅⋅b₁₀   a₂₆⋅b₁₀   a₂₇⋅b₁₀   a₂₈⋅b₁₀   a₂₉⋅b₁₀   a₂₅⋅b₁₁   a₂₆⋅b₁₁   a₂₇⋅b₁₁   a₂₈⋅b₁₁   a₂₉⋅b₁₁   a₂₁₀⋅b₁₀   a₂₁₁⋅b₁₀   a₂₁₂⋅b₁₀   a₂₁₃⋅b₁₀   a₂₁₄⋅b₁₀   a₂₁₀⋅b₁₁   a₂₁₁⋅b₁₁   a₂₁₂⋅b₁₁   a₂₁₃⋅b₁₁   a₂₁₄⋅b₁₁ ⎥
        ⎢                                                                                                                                                                                                                                                                                                                    ⎥
        ⎢a₃₀⋅b₁₀   a₃₁⋅b₁₀   a₃₂⋅b₁₀   a₃₃⋅b₁₀   a₃₄⋅b₁₀   a₃₀⋅b₁₁   a₃₁⋅b₁₁   a₃₂⋅b₁₁   a₃₃⋅b₁₁   a₃₄⋅b₁₁   a₃₅⋅b₁₀   a₃₆⋅b₁₀   a₃₇⋅b₁₀   a₃₈⋅b₁₀   a₃₉⋅b₁₀   a₃₅⋅b₁₁   a₃₆⋅b₁₁   a₃₇⋅b₁₁   a₃₈⋅b₁₁   a₃₉⋅b₁₁   a₃₁₀⋅b₁₀   a₃₁₁⋅b₁₀   a₃₁₂⋅b₁₀   a₃₁₃⋅b₁₀   a₃₁₄⋅b₁₀   a₃₁₀⋅b₁₁   a₃₁₁⋅b₁₁   a₃₁₂⋅b₁₁   a₃₁₃⋅b₁₁   a₃₁₄⋅b₁₁ ⎥
        ⎢                                                                                                                                                                                                                                                                                                                    ⎥
        ⎢a₄₀⋅b₁₀   a₄₁⋅b₁₀   a₄₂⋅b₁₀   a₄₃⋅b₁₀   a₄₄⋅b₁₀   a₄₀⋅b₁₁   a₄₁⋅b₁₁   a₄₂⋅b₁₁   a₄₃⋅b₁₁   a₄₄⋅b₁₁   a₄₅⋅b₁₀   a₄₆⋅b₁₀   a₄₇⋅b₁₀   a₄₈⋅b₁₀   a₄₉⋅b₁₀   a₄₅⋅b₁₁   a₄₆⋅b₁₁   a₄₇⋅b₁₁   a₄₈⋅b₁₁   a₄₉⋅b₁₁   a₄₁₀⋅b₁₀   a₄₁₁⋅b₁₀   a₄₁₂⋅b₁₀   a₄₁₃⋅b₁₀   a₄₁₄⋅b₁₀   a₄₁₀⋅b₁₁   a₄₁₁⋅b₁₁   a₄₁₂⋅b₁₁   a₄₁₃⋅b₁₁   a₄₁₄⋅b₁₁ ⎥
        ⎢                                                                                                                                                                                                                                                                                                                    ⎥
        ⎢a₅₀⋅b₀₀   a₅₁⋅b₀₀   a₅₂⋅b₀₀   a₅₃⋅b₀₀   a₅₄⋅b₀₀   a₅₀⋅b₀₁   a₅₁⋅b₀₁   a₅₂⋅b₀₁   a₅₃⋅b₀₁   a₅₄⋅b₀₁   a₅₅⋅b₀₀   a₅₆⋅b₀₀   a₅₇⋅b₀₀   a₅₈⋅b₀₀   a₅₉⋅b₀₀   a₅₅⋅b₀₁   a₅₆⋅b₀₁   a₅₇⋅b₀₁   a₅₈⋅b₀₁   a₅₉⋅b₀₁   a₅₁₀⋅b₀₀   a₅₁₁⋅b₀₀   a₅₁₂⋅b₀₀   a₅₁₃⋅b₀₀   a₅₁₄⋅b₀₀   a₅₁₀⋅b₀₁   a₅₁₁⋅b₀₁   a₅₁₂⋅b₀₁   a₅₁₃⋅b₀₁   a₅₁₄⋅b₀₁ ⎥
        ⎢                                                                                                                                                                                                                                                                                                                    ⎥
        ⎢a₆₀⋅b₀₀   a₆₁⋅b₀₀   a₆₂⋅b₀₀   a₆₃⋅b₀₀   a₆₄⋅b₀₀   a₆₀⋅b₀₁   a₆₁⋅b₀₁   a₆₂⋅b₀₁   a₆₃⋅b₀₁   a₆₄⋅b₀₁   a₆₅⋅b₀₀   a₆₆⋅b₀₀   a₆₇⋅b₀₀   a₆₈⋅b₀₀   a₆₉⋅b₀₀   a₆₅⋅b₀₁   a₆₆⋅b₀₁   a₆₇⋅b₀₁   a₆₈⋅b₀₁   a₆₉⋅b₀₁   a₆₁₀⋅b₀₀   a₆₁₁⋅b₀₀   a₆₁₂⋅b₀₀   a₆₁₃⋅b₀₀   a₆₁₄⋅b₀₀   a₆₁₀⋅b₀₁   a₆₁₁⋅b₀₁   a₆₁₂⋅b₀₁   a₆₁₃⋅b₀₁   a₆₁₄⋅b₀₁ ⎥
        ⎢                                                                                                                                                                                                                                                                                                                    ⎥
        ⎢a₇₀⋅b₀₀   a₇₁⋅b₀₀   a₇₂⋅b₀₀   a₇₃⋅b₀₀   a₇₄⋅b₀₀   a₇₀⋅b₀₁   a₇₁⋅b₀₁   a₇₂⋅b₀₁   a₇₃⋅b₀₁   a₇₄⋅b₀₁   a₇₅⋅b₀₀   a₇₆⋅b₀₀   a₇₇⋅b₀₀   a₇₈⋅b₀₀   a₇₉⋅b₀₀   a₇₅⋅b₀₁   a₇₆⋅b₀₁   a₇₇⋅b₀₁   a₇₈⋅b₀₁   a₇₉⋅b₀₁   a₇₁₀⋅b₀₀   a₇₁₁⋅b₀₀   a₇₁₂⋅b₀₀   a₇₁₃⋅b₀₀   a₇₁₄⋅b₀₀   a₇₁₀⋅b₀₁   a₇₁₁⋅b₀₁   a₇₁₂⋅b₀₁   a₇₁₃⋅b₀₁   a₇₁₄⋅b₀₁ ⎥
        ⎢                                                                                                                                                                                                                                                                                                                    ⎥
        ⎢a₈₀⋅b₀₀   a₈₁⋅b₀₀   a₈₂⋅b₀₀   a₈₃⋅b₀₀   a₈₄⋅b₀₀   a₈₀⋅b₀₁   a₈₁⋅b₀₁   a₈₂⋅b₀₁   a₈₃⋅b₀₁   a₈₄⋅b₀₁   a₈₅⋅b₀₀   a₈₆⋅b₀₀   a₈₇⋅b₀₀   a₈₈⋅b₀₀   a₈₉⋅b₀₀   a₈₅⋅b₀₁   a₈₆⋅b₀₁   a₈₇⋅b₀₁   a₈₈⋅b₀₁   a₈₉⋅b₀₁   a₈₁₀⋅b₀₀   a₈₁₁⋅b₀₀   a₈₁₂⋅b₀₀   a₈₁₃⋅b₀₀   a₈₁₄⋅b₀₀   a₈₁₀⋅b₀₁   a₈₁₁⋅b₀₁   a₈₁₂⋅b₀₁   a₈₁₃⋅b₀₁   a₈₁₄⋅b₀₁ ⎥
        ⎢                                                                                                                                                                                                                                                                                                                    ⎥
        ⎢a₉₀⋅b₀₀   a₉₁⋅b₀₀   a₉₂⋅b₀₀   a₉₃⋅b₀₀   a₉₄⋅b₀₀   a₉₀⋅b₀₁   a₉₁⋅b₀₁   a₉₂⋅b₀₁   a₉₃⋅b₀₁   a₉₄⋅b₀₁   a₉₅⋅b₀₀   a₉₆⋅b₀₀   a₉₇⋅b₀₀   a₉₈⋅b₀₀   a₉₉⋅b₀₀   a₉₅⋅b₀₁   a₉₆⋅b₀₁   a₉₇⋅b₀₁   a₉₈⋅b₀₁   a₉₉⋅b₀₁   a₉₁₀⋅b₀₀   a₉₁₁⋅b₀₀   a₉₁₂⋅b₀₀   a₉₁₃⋅b₀₀   a₉₁₄⋅b₀₀   a₉₁₀⋅b₀₁   a₉₁₁⋅b₀₁   a₉₁₂⋅b₀₁   a₉₁₃⋅b₀₁   a₉₁₄⋅b₀₁ ⎥
        ⎢                                                                                                                                                                                                                                                                                                                    ⎥
        ⎢a₅₀⋅b₁₀   a₅₁⋅b₁₀   a₅₂⋅b₁₀   a₅₃⋅b₁₀   a₅₄⋅b₁₀   a₅₀⋅b₁₁   a₅₁⋅b₁₁   a₅₂⋅b₁₁   a₅₃⋅b₁₁   a₅₄⋅b₁₁   a₅₅⋅b₁₀   a₅₆⋅b₁₀   a₅₇⋅b₁₀   a₅₈⋅b₁₀   a₅₉⋅b₁₀   a₅₅⋅b₁₁   a₅₆⋅b₁₁   a₅₇⋅b₁₁   a₅₈⋅b₁₁   a₅₉⋅b₁₁   a₅₁₀⋅b₁₀   a₅₁₁⋅b₁₀   a₅₁₂⋅b₁₀   a₅₁₃⋅b₁₀   a₅₁₄⋅b₁₀   a₅₁₀⋅b₁₁   a₅₁₁⋅b₁₁   a₅₁₂⋅b₁₁   a₅₁₃⋅b₁₁   a₅₁₄⋅b₁₁ ⎥
        ⎢                                                                                                                                                                                                                                                                                                                    ⎥
        ⎢a₆₀⋅b₁₀   a₆₁⋅b₁₀   a₆₂⋅b₁₀   a₆₃⋅b₁₀   a₆₄⋅b₁₀   a₆₀⋅b₁₁   a₆₁⋅b₁₁   a₆₂⋅b₁₁   a₆₃⋅b₁₁   a₆₄⋅b₁₁   a₆₅⋅b₁₀   a₆₆⋅b₁₀   a₆₇⋅b₁₀   a₆₈⋅b₁₀   a₆₉⋅b₁₀   a₆₅⋅b₁₁   a₆₆⋅b₁₁   a₆₇⋅b₁₁   a₆₈⋅b₁₁   a₆₉⋅b₁₁   a₆₁₀⋅b₁₀   a₆₁₁⋅b₁₀   a₆₁₂⋅b₁₀   a₆₁₃⋅b₁₀   a₆₁₄⋅b₁₀   a₆₁₀⋅b₁₁   a₆₁₁⋅b₁₁   a₆₁₂⋅b₁₁   a₆₁₃⋅b₁₁   a₆₁₄⋅b₁₁ ⎥
        ⎢                                                                                                                                                                                                                                                                                                                    ⎥
        ⎢a₇₀⋅b₁₀   a₇₁⋅b₁₀   a₇₂⋅b₁₀   a₇₃⋅b₁₀   a₇₄⋅b₁₀   a₇₀⋅b₁₁   a₇₁⋅b₁₁   a₇₂⋅b₁₁   a₇₃⋅b₁₁   a₇₄⋅b₁₁   a₇₅⋅b₁₀   a₇₆⋅b₁₀   a₇₇⋅b₁₀   a₇₈⋅b₁₀   a₇₉⋅b₁₀   a₇₅⋅b₁₁   a₇₆⋅b₁₁   a₇₇⋅b₁₁   a₇₈⋅b₁₁   a₇₉⋅b₁₁   a₇₁₀⋅b₁₀   a₇₁₁⋅b₁₀   a₇₁₂⋅b₁₀   a₇₁₃⋅b₁₀   a₇₁₄⋅b₁₀   a₇₁₀⋅b₁₁   a₇₁₁⋅b₁₁   a₇₁₂⋅b₁₁   a₇₁₃⋅b₁₁   a₇₁₄⋅b₁₁ ⎥
        ⎢                                                                                                                                                                                                                                                                                                                    ⎥
        ⎢a₈₀⋅b₁₀   a₈₁⋅b₁₀   a₈₂⋅b₁₀   a₈₃⋅b₁₀   a₈₄⋅b₁₀   a₈₀⋅b₁₁   a₈₁⋅b₁₁   a₈₂⋅b₁₁   a₈₃⋅b₁₁   a₈₄⋅b₁₁   a₈₅⋅b₁₀   a₈₆⋅b₁₀   a₈₇⋅b₁₀   a₈₈⋅b₁₀   a₈₉⋅b₁₀   a₈₅⋅b₁₁   a₈₆⋅b₁₁   a₈₇⋅b₁₁   a₈₈⋅b₁₁   a₈₉⋅b₁₁   a₈₁₀⋅b₁₀   a₈₁₁⋅b₁₀   a₈₁₂⋅b₁₀   a₈₁₃⋅b₁₀   a₈₁₄⋅b₁₀   a₈₁₀⋅b₁₁   a₈₁₁⋅b₁₁   a₈₁₂⋅b₁₁   a₈₁₃⋅b₁₁   a₈₁₄⋅b₁₁ ⎥
        ⎢                                                                                                                                                                                                                                                                                                                    ⎥
        ⎢a₉₀⋅b₁₀   a₉₁⋅b₁₀   a₉₂⋅b₁₀   a₉₃⋅b₁₀   a₉₄⋅b₁₀   a₉₀⋅b₁₁   a₉₁⋅b₁₁   a₉₂⋅b₁₁   a₉₃⋅b₁₁   a₉₄⋅b₁₁   a₉₅⋅b₁₀   a₉₆⋅b₁₀   a₉₇⋅b₁₀   a₉₈⋅b₁₀   a₉₉⋅b₁₀   a₉₅⋅b₁₁   a₉₆⋅b₁₁   a₉₇⋅b₁₁   a₉₈⋅b₁₁   a₉₉⋅b₁₁   a₉₁₀⋅b₁₀   a₉₁₁⋅b₁₀   a₉₁₂⋅b₁₀   a₉₁₃⋅b₁₀   a₉₁₄⋅b₁₀   a₉₁₀⋅b₁₁   a₉₁₁⋅b₁₁   a₉₁₂⋅b₁₁   a₉₁₃⋅b₁₁   a₉₁₄⋅b₁₁ ⎥
        ⎢                                                                                                                                                                                                                                                                                                                    ⎥
        ⎢a₁₀₀⋅b₀₀  a₁₀₁⋅b₀₀  a₁₀₂⋅b₀₀  a₁₀₃⋅b₀₀  a₁₀₄⋅b₀₀  a₁₀₀⋅b₀₁  a₁₀₁⋅b₀₁  a₁₀₂⋅b₀₁  a₁₀₃⋅b₀₁  a₁₀₄⋅b₀₁  a₁₀₅⋅b₀₀  a₁₀₆⋅b₀₀  a₁₀₇⋅b₀₀  a₁₀₈⋅b₀₀  a₁₀₉⋅b₀₀  a₁₀₅⋅b₀₁  a₁₀₆⋅b₀₁  a₁₀₇⋅b₀₁  a₁₀₈⋅b₀₁  a₁₀₉⋅b₀₁  a₁₀₁₀⋅b₀₀  a₁₀₁₁⋅b₀₀  a₁₀₁₂⋅b₀₀  a₁₀₁₃⋅b₀₀  a₁₀₁₄⋅b₀₀  a₁₀₁₀⋅b₀₁  a₁₀₁₁⋅b₀₁  a₁₀₁₂⋅b₀₁  a₁₀₁₃⋅b₀₁  a₁₀₁₄⋅b₀₁⎥
        ⎢                                                                                                                                                                                                                                                                                                                    ⎥
        ⎢a₁₁₀⋅b₀₀  a₁₁₁⋅b₀₀  a₁₁₂⋅b₀₀  a₁₁₃⋅b₀₀  a₁₁₄⋅b₀₀  a₁₁₀⋅b₀₁  a₁₁₁⋅b₀₁  a₁₁₂⋅b₀₁  a₁₁₃⋅b₀₁  a₁₁₄⋅b₀₁  a₁₁₅⋅b₀₀  a₁₁₆⋅b₀₀  a₁₁₇⋅b₀₀  a₁₁₈⋅b₀₀  a₁₁₉⋅b₀₀  a₁₁₅⋅b₀₁  a₁₁₆⋅b₀₁  a₁₁₇⋅b₀₁  a₁₁₈⋅b₀₁  a₁₁₉⋅b₀₁  a₁₁₁₀⋅b₀₀  a₁₁₁₁⋅b₀₀  a₁₁₁₂⋅b₀₀  a₁₁₁₃⋅b₀₀  a₁₁₁₄⋅b₀₀  a₁₁₁₀⋅b₀₁  a₁₁₁₁⋅b₀₁  a₁₁₁₂⋅b₀₁  a₁₁₁₃⋅b₀₁  a₁₁₁₄⋅b₀₁⎥
        ⎢                                                                                                                                                                                                                                                                                                                    ⎥
        ⎢a₁₂₀⋅b₀₀  a₁₂₁⋅b₀₀  a₁₂₂⋅b₀₀  a₁₂₃⋅b₀₀  a₁₂₄⋅b₀₀  a₁₂₀⋅b₀₁  a₁₂₁⋅b₀₁  a₁₂₂⋅b₀₁  a₁₂₃⋅b₀₁  a₁₂₄⋅b₀₁  a₁₂₅⋅b₀₀  a₁₂₆⋅b₀₀  a₁₂₇⋅b₀₀  a₁₂₈⋅b₀₀  a₁₂₉⋅b₀₀  a₁₂₅⋅b₀₁  a₁₂₆⋅b₀₁  a₁₂₇⋅b₀₁  a₁₂₈⋅b₀₁  a₁₂₉⋅b₀₁  a₁₂₁₀⋅b₀₀  a₁₂₁₁⋅b₀₀  a₁₂₁₂⋅b₀₀  a₁₂₁₃⋅b₀₀  a₁₂₁₄⋅b₀₀  a₁₂₁₀⋅b₀₁  a₁₂₁₁⋅b₀₁  a₁₂₁₂⋅b₀₁  a₁₂₁₃⋅b₀₁  a₁₂₁₄⋅b₀₁⎥
        ⎢                                                                                                                                                                                                                                                                                                                    ⎥
        ⎢a₁₃₀⋅b₀₀  a₁₃₁⋅b₀₀  a₁₃₂⋅b₀₀  a₁₃₃⋅b₀₀  a₁₃₄⋅b₀₀  a₁₃₀⋅b₀₁  a₁₃₁⋅b₀₁  a₁₃₂⋅b₀₁  a₁₃₃⋅b₀₁  a₁₃₄⋅b₀₁  a₁₃₅⋅b₀₀  a₁₃₆⋅b₀₀  a₁₃₇⋅b₀₀  a₁₃₈⋅b₀₀  a₁₃₉⋅b₀₀  a₁₃₅⋅b₀₁  a₁₃₆⋅b₀₁  a₁₃₇⋅b₀₁  a₁₃₈⋅b₀₁  a₁₃₉⋅b₀₁  a₁₃₁₀⋅b₀₀  a₁₃₁₁⋅b₀₀  a₁₃₁₂⋅b₀₀  a₁₃₁₃⋅b₀₀  a₁₃₁₄⋅b₀₀  a₁₃₁₀⋅b₀₁  a₁₃₁₁⋅b₀₁  a₁₃₁₂⋅b₀₁  a₁₃₁₃⋅b₀₁  a₁₃₁₄⋅b₀₁⎥
        ⎢                                                                                                                                                                                                                                                                                                                    ⎥
        ⎢a₁₄₀⋅b₀₀  a₁₄₁⋅b₀₀  a₁₄₂⋅b₀₀  a₁₄₃⋅b₀₀  a₁₄₄⋅b₀₀  a₁₄₀⋅b₀₁  a₁₄₁⋅b₀₁  a₁₄₂⋅b₀₁  a₁₄₃⋅b₀₁  a₁₄₄⋅b₀₁  a₁₄₅⋅b₀₀  a₁₄₆⋅b₀₀  a₁₄₇⋅b₀₀  a₁₄₈⋅b₀₀  a₁₄₉⋅b₀₀  a₁₄₅⋅b₀₁  a₁₄₆⋅b₀₁  a₁₄₇⋅b₀₁  a₁₄₈⋅b₀₁  a₁₄₉⋅b₀₁  a₁₄₁₀⋅b₀₀  a₁₄₁₁⋅b₀₀  a₁₄₁₂⋅b₀₀  a₁₄₁₃⋅b₀₀  a₁₄₁₄⋅b₀₀  a₁₄₁₀⋅b₀₁  a₁₄₁₁⋅b₀₁  a₁₄₁₂⋅b₀₁  a₁₄₁₃⋅b₀₁  a₁₄₁₄⋅b₀₁⎥
        ⎢                                                                                                                                                                                                                                                                                                                    ⎥
        ⎢a₁₀₀⋅b₁₀  a₁₀₁⋅b₁₀  a₁₀₂⋅b₁₀  a₁₀₃⋅b₁₀  a₁₀₄⋅b₁₀  a₁₀₀⋅b₁₁  a₁₀₁⋅b₁₁  a₁₀₂⋅b₁₁  a₁₀₃⋅b₁₁  a₁₀₄⋅b₁₁  a₁₀₅⋅b₁₀  a₁₀₆⋅b₁₀  a₁₀₇⋅b₁₀  a₁₀₈⋅b₁₀  a₁₀₉⋅b₁₀  a₁₀₅⋅b₁₁  a₁₀₆⋅b₁₁  a₁₀₇⋅b₁₁  a₁₀₈⋅b₁₁  a₁₀₉⋅b₁₁  a₁₀₁₀⋅b₁₀  a₁₀₁₁⋅b₁₀  a₁₀₁₂⋅b₁₀  a₁₀₁₃⋅b₁₀  a₁₀₁₄⋅b₁₀  a₁₀₁₀⋅b₁₁  a₁₀₁₁⋅b₁₁  a₁₀₁₂⋅b₁₁  a₁₀₁₃⋅b₁₁  a₁₀₁₄⋅b₁₁⎥
        ⎢                                                                                                                                                                                                                                                                                                                    ⎥
        ⎢a₁₁₀⋅b₁₀  a₁₁₁⋅b₁₀  a₁₁₂⋅b₁₀  a₁₁₃⋅b₁₀  a₁₁₄⋅b₁₀  a₁₁₀⋅b₁₁  a₁₁₁⋅b₁₁  a₁₁₂⋅b₁₁  a₁₁₃⋅b₁₁  a₁₁₄⋅b₁₁  a₁₁₅⋅b₁₀  a₁₁₆⋅b₁₀  a₁₁₇⋅b₁₀  a₁₁₈⋅b₁₀  a₁₁₉⋅b₁₀  a₁₁₅⋅b₁₁  a₁₁₆⋅b₁₁  a₁₁₇⋅b₁₁  a₁₁₈⋅b₁₁  a₁₁₉⋅b₁₁  a₁₁₁₀⋅b₁₀  a₁₁₁₁⋅b₁₀  a₁₁₁₂⋅b₁₀  a₁₁₁₃⋅b₁₀  a₁₁₁₄⋅b₁₀  a₁₁₁₀⋅b₁₁  a₁₁₁₁⋅b₁₁  a₁₁₁₂⋅b₁₁  a₁₁₁₃⋅b₁₁  a₁₁₁₄⋅b₁₁⎥
        ⎢                                                                                                                                                                                                                                                                                                                    ⎥
        ⎢a₁₂₀⋅b₁₀  a₁₂₁⋅b₁₀  a₁₂₂⋅b₁₀  a₁₂₃⋅b₁₀  a₁₂₄⋅b₁₀  a₁₂₀⋅b₁₁  a₁₂₁⋅b₁₁  a₁₂₂⋅b₁₁  a₁₂₃⋅b₁₁  a₁₂₄⋅b₁₁  a₁₂₅⋅b₁₀  a₁₂₆⋅b₁₀  a₁₂₇⋅b₁₀  a₁₂₈⋅b₁₀  a₁₂₉⋅b₁₀  a₁₂₅⋅b₁₁  a₁₂₆⋅b₁₁  a₁₂₇⋅b₁₁  a₁₂₈⋅b₁₁  a₁₂₉⋅b₁₁  a₁₂₁₀⋅b₁₀  a₁₂₁₁⋅b₁₀  a₁₂₁₂⋅b₁₀  a₁₂₁₃⋅b₁₀  a₁₂₁₄⋅b₁₀  a₁₂₁₀⋅b₁₁  a₁₂₁₁⋅b₁₁  a₁₂₁₂⋅b₁₁  a₁₂₁₃⋅b₁₁  a₁₂₁₄⋅b₁₁⎥
        ⎢                                                                                                                                                                                                                                                                                                                    ⎥
        ⎢a₁₃₀⋅b₁₀  a₁₃₁⋅b₁₀  a₁₃₂⋅b₁₀  a₁₃₃⋅b₁₀  a₁₃₄⋅b₁₀  a₁₃₀⋅b₁₁  a₁₃₁⋅b₁₁  a₁₃₂⋅b₁₁  a₁₃₃⋅b₁₁  a₁₃₄⋅b₁₁  a₁₃₅⋅b₁₀  a₁₃₆⋅b₁₀  a₁₃₇⋅b₁₀  a₁₃₈⋅b₁₀  a₁₃₉⋅b₁₀  a₁₃₅⋅b₁₁  a₁₃₆⋅b₁₁  a₁₃₇⋅b₁₁  a₁₃₈⋅b₁₁  a₁₃₉⋅b₁₁  a₁₃₁₀⋅b₁₀  a₁₃₁₁⋅b₁₀  a₁₃₁₂⋅b₁₀  a₁₃₁₃⋅b₁₀  a₁₃₁₄⋅b₁₀  a₁₃₁₀⋅b₁₁  a₁₃₁₁⋅b₁₁  a₁₃₁₂⋅b₁₁  a₁₃₁₃⋅b₁₁  a₁₃₁₄⋅b₁₁⎥
        ⎢                                                                                                                                                                                                                                                                                                                    ⎥
        ⎣a₁₄₀⋅b₁₀  a₁₄₁⋅b₁₀  a₁₄₂⋅b₁₀  a₁₄₃⋅b₁₀  a₁₄₄⋅b₁₀  a₁₄₀⋅b₁₁  a₁₄₁⋅b₁₁  a₁₄₂⋅b₁₁  a₁₄₃⋅b₁₁  a₁₄₄⋅b₁₁  a₁₄₅⋅b₁₀  a₁₄₆⋅b₁₀  a₁₄₇⋅b₁₀  a₁₄₈⋅b₁₀  a₁₄₉⋅b₁₀  a₁₄₅⋅b₁₁  a₁₄₆⋅b₁₁  a₁₄₇⋅b₁₁  a₁₄₈⋅b₁₁  a₁₄₉⋅b₁₁  a₁₄₁₀⋅b₁₀  a₁₄₁₁⋅b₁₀  a₁₄₁₂⋅b₁₀  a₁₄₁₃⋅b₁₀  a₁₄₁₄⋅b₁₀  a₁₄₁₀⋅b₁₁  a₁₄₁₁⋅b₁₁  a₁₄₁₂⋅b₁₁  a₁₄₁₃⋅b₁₁  a₁₄₁₄⋅b₁₁⎦
    """
    assert pretty(C, num_columns=10000) == dedent(expected).strip()


def test_inter_product_8_2_4():
    """
        let m = len(A), n = len(I), l = len(C)
        then the Kproduct, A ⨁ I ⨁ C, is formed by
        1. get the Kproduct K = A ⨁ C
        2. divide up K
    """
    # Define symbolic variables
    A = square_m(8, 'a')
    # print('A')
    # pprint(A, num_columns=10000)

    B = square_m(3, 'b')
    # print('B')
    # pprint(B)

    C = inter_product(A, B, 2)
    # print('C')
    # pprint(C, num_columns=10000)
    expected = '''
        ⎡a₀₀⋅b₀₀  a₀₁⋅b₀₀  a₀₀⋅b₀₁  a₀₁⋅b₀₁  a₀₀⋅b₀₂  a₀₁⋅b₀₂  a₀₂⋅b₀₀  a₀₃⋅b₀₀  a₀₂⋅b₀₁  a₀₃⋅b₀₁  a₀₂⋅b₀₂  a₀₃⋅b₀₂  a₀₄⋅b₀₀  a₀₅⋅b₀₀  a₀₄⋅b₀₁  a₀₅⋅b₀₁  a₀₄⋅b₀₂  a₀₅⋅b₀₂  a₀₆⋅b₀₀  a₀₇⋅b₀₀  a₀₆⋅b₀₁  a₀₇⋅b₀₁  a₀₆⋅b₀₂  a₀₇⋅b₀₂⎤
        ⎢                                                                                                                                                                                                                      ⎥
        ⎢a₁₀⋅b₀₀  a₁₁⋅b₀₀  a₁₀⋅b₀₁  a₁₁⋅b₀₁  a₁₀⋅b₀₂  a₁₁⋅b₀₂  a₁₂⋅b₀₀  a₁₃⋅b₀₀  a₁₂⋅b₀₁  a₁₃⋅b₀₁  a₁₂⋅b₀₂  a₁₃⋅b₀₂  a₁₄⋅b₀₀  a₁₅⋅b₀₀  a₁₄⋅b₀₁  a₁₅⋅b₀₁  a₁₄⋅b₀₂  a₁₅⋅b₀₂  a₁₆⋅b₀₀  a₁₇⋅b₀₀  a₁₆⋅b₀₁  a₁₇⋅b₀₁  a₁₆⋅b₀₂  a₁₇⋅b₀₂⎥
        ⎢                                                                                                                                                                                                                      ⎥
        ⎢a₀₀⋅b₁₀  a₀₁⋅b₁₀  a₀₀⋅b₁₁  a₀₁⋅b₁₁  a₀₀⋅b₁₂  a₀₁⋅b₁₂  a₀₂⋅b₁₀  a₀₃⋅b₁₀  a₀₂⋅b₁₁  a₀₃⋅b₁₁  a₀₂⋅b₁₂  a₀₃⋅b₁₂  a₀₄⋅b₁₀  a₀₅⋅b₁₀  a₀₄⋅b₁₁  a₀₅⋅b₁₁  a₀₄⋅b₁₂  a₀₅⋅b₁₂  a₀₆⋅b₁₀  a₀₇⋅b₁₀  a₀₆⋅b₁₁  a₀₇⋅b₁₁  a₀₆⋅b₁₂  a₀₇⋅b₁₂⎥
        ⎢                                                                                                                                                                                                                      ⎥
        ⎢a₁₀⋅b₁₀  a₁₁⋅b₁₀  a₁₀⋅b₁₁  a₁₁⋅b₁₁  a₁₀⋅b₁₂  a₁₁⋅b₁₂  a₁₂⋅b₁₀  a₁₃⋅b₁₀  a₁₂⋅b₁₁  a₁₃⋅b₁₁  a₁₂⋅b₁₂  a₁₃⋅b₁₂  a₁₄⋅b₁₀  a₁₅⋅b₁₀  a₁₄⋅b₁₁  a₁₅⋅b₁₁  a₁₄⋅b₁₂  a₁₅⋅b₁₂  a₁₆⋅b₁₀  a₁₇⋅b₁₀  a₁₆⋅b₁₁  a₁₇⋅b₁₁  a₁₆⋅b₁₂  a₁₇⋅b₁₂⎥
        ⎢                                                                                                                                                                                                                      ⎥
        ⎢a₀₀⋅b₂₀  a₀₁⋅b₂₀  a₀₀⋅b₂₁  a₀₁⋅b₂₁  a₀₀⋅b₂₂  a₀₁⋅b₂₂  a₀₂⋅b₂₀  a₀₃⋅b₂₀  a₀₂⋅b₂₁  a₀₃⋅b₂₁  a₀₂⋅b₂₂  a₀₃⋅b₂₂  a₀₄⋅b₂₀  a₀₅⋅b₂₀  a₀₄⋅b₂₁  a₀₅⋅b₂₁  a₀₄⋅b₂₂  a₀₅⋅b₂₂  a₀₆⋅b₂₀  a₀₇⋅b₂₀  a₀₆⋅b₂₁  a₀₇⋅b₂₁  a₀₆⋅b₂₂  a₀₇⋅b₂₂⎥
        ⎢                                                                                                                                                                                                                      ⎥
        ⎢a₁₀⋅b₂₀  a₁₁⋅b₂₀  a₁₀⋅b₂₁  a₁₁⋅b₂₁  a₁₀⋅b₂₂  a₁₁⋅b₂₂  a₁₂⋅b₂₀  a₁₃⋅b₂₀  a₁₂⋅b₂₁  a₁₃⋅b₂₁  a₁₂⋅b₂₂  a₁₃⋅b₂₂  a₁₄⋅b₂₀  a₁₅⋅b₂₀  a₁₄⋅b₂₁  a₁₅⋅b₂₁  a₁₄⋅b₂₂  a₁₅⋅b₂₂  a₁₆⋅b₂₀  a₁₇⋅b₂₀  a₁₆⋅b₂₁  a₁₇⋅b₂₁  a₁₆⋅b₂₂  a₁₇⋅b₂₂⎥
        ⎢                                                                                                                                                                                                                      ⎥
        ⎢a₂₀⋅b₀₀  a₂₁⋅b₀₀  a₂₀⋅b₀₁  a₂₁⋅b₀₁  a₂₀⋅b₀₂  a₂₁⋅b₀₂  a₂₂⋅b₀₀  a₂₃⋅b₀₀  a₂₂⋅b₀₁  a₂₃⋅b₀₁  a₂₂⋅b₀₂  a₂₃⋅b₀₂  a₂₄⋅b₀₀  a₂₅⋅b₀₀  a₂₄⋅b₀₁  a₂₅⋅b₀₁  a₂₄⋅b₀₂  a₂₅⋅b₀₂  a₂₆⋅b₀₀  a₂₇⋅b₀₀  a₂₆⋅b₀₁  a₂₇⋅b₀₁  a₂₆⋅b₀₂  a₂₇⋅b₀₂⎥
        ⎢                                                                                                                                                                                                                      ⎥
        ⎢a₃₀⋅b₀₀  a₃₁⋅b₀₀  a₃₀⋅b₀₁  a₃₁⋅b₀₁  a₃₀⋅b₀₂  a₃₁⋅b₀₂  a₃₂⋅b₀₀  a₃₃⋅b₀₀  a₃₂⋅b₀₁  a₃₃⋅b₀₁  a₃₂⋅b₀₂  a₃₃⋅b₀₂  a₃₄⋅b₀₀  a₃₅⋅b₀₀  a₃₄⋅b₀₁  a₃₅⋅b₀₁  a₃₄⋅b₀₂  a₃₅⋅b₀₂  a₃₆⋅b₀₀  a₃₇⋅b₀₀  a₃₆⋅b₀₁  a₃₇⋅b₀₁  a₃₆⋅b₀₂  a₃₇⋅b₀₂⎥
        ⎢                                                                                                                                                                                                                      ⎥
        ⎢a₂₀⋅b₁₀  a₂₁⋅b₁₀  a₂₀⋅b₁₁  a₂₁⋅b₁₁  a₂₀⋅b₁₂  a₂₁⋅b₁₂  a₂₂⋅b₁₀  a₂₃⋅b₁₀  a₂₂⋅b₁₁  a₂₃⋅b₁₁  a₂₂⋅b₁₂  a₂₃⋅b₁₂  a₂₄⋅b₁₀  a₂₅⋅b₁₀  a₂₄⋅b₁₁  a₂₅⋅b₁₁  a₂₄⋅b₁₂  a₂₅⋅b₁₂  a₂₆⋅b₁₀  a₂₇⋅b₁₀  a₂₆⋅b₁₁  a₂₇⋅b₁₁  a₂₆⋅b₁₂  a₂₇⋅b₁₂⎥
        ⎢                                                                                                                                                                                                                      ⎥
        ⎢a₃₀⋅b₁₀  a₃₁⋅b₁₀  a₃₀⋅b₁₁  a₃₁⋅b₁₁  a₃₀⋅b₁₂  a₃₁⋅b₁₂  a₃₂⋅b₁₀  a₃₃⋅b₁₀  a₃₂⋅b₁₁  a₃₃⋅b₁₁  a₃₂⋅b₁₂  a₃₃⋅b₁₂  a₃₄⋅b₁₀  a₃₅⋅b₁₀  a₃₄⋅b₁₁  a₃₅⋅b₁₁  a₃₄⋅b₁₂  a₃₅⋅b₁₂  a₃₆⋅b₁₀  a₃₇⋅b₁₀  a₃₆⋅b₁₁  a₃₇⋅b₁₁  a₃₆⋅b₁₂  a₃₇⋅b₁₂⎥
        ⎢                                                                                                                                                                                                                      ⎥
        ⎢a₂₀⋅b₂₀  a₂₁⋅b₂₀  a₂₀⋅b₂₁  a₂₁⋅b₂₁  a₂₀⋅b₂₂  a₂₁⋅b₂₂  a₂₂⋅b₂₀  a₂₃⋅b₂₀  a₂₂⋅b₂₁  a₂₃⋅b₂₁  a₂₂⋅b₂₂  a₂₃⋅b₂₂  a₂₄⋅b₂₀  a₂₅⋅b₂₀  a₂₄⋅b₂₁  a₂₅⋅b₂₁  a₂₄⋅b₂₂  a₂₅⋅b₂₂  a₂₆⋅b₂₀  a₂₇⋅b₂₀  a₂₆⋅b₂₁  a₂₇⋅b₂₁  a₂₆⋅b₂₂  a₂₇⋅b₂₂⎥
        ⎢                                                                                                                                                                                                                      ⎥
        ⎢a₃₀⋅b₂₀  a₃₁⋅b₂₀  a₃₀⋅b₂₁  a₃₁⋅b₂₁  a₃₀⋅b₂₂  a₃₁⋅b₂₂  a₃₂⋅b₂₀  a₃₃⋅b₂₀  a₃₂⋅b₂₁  a₃₃⋅b₂₁  a₃₂⋅b₂₂  a₃₃⋅b₂₂  a₃₄⋅b₂₀  a₃₅⋅b₂₀  a₃₄⋅b₂₁  a₃₅⋅b₂₁  a₃₄⋅b₂₂  a₃₅⋅b₂₂  a₃₆⋅b₂₀  a₃₇⋅b₂₀  a₃₆⋅b₂₁  a₃₇⋅b₂₁  a₃₆⋅b₂₂  a₃₇⋅b₂₂⎥
        ⎢                                                                                                                                                                                                                      ⎥
        ⎢a₄₀⋅b₀₀  a₄₁⋅b₀₀  a₄₀⋅b₀₁  a₄₁⋅b₀₁  a₄₀⋅b₀₂  a₄₁⋅b₀₂  a₄₂⋅b₀₀  a₄₃⋅b₀₀  a₄₂⋅b₀₁  a₄₃⋅b₀₁  a₄₂⋅b₀₂  a₄₃⋅b₀₂  a₄₄⋅b₀₀  a₄₅⋅b₀₀  a₄₄⋅b₀₁  a₄₅⋅b₀₁  a₄₄⋅b₀₂  a₄₅⋅b₀₂  a₄₆⋅b₀₀  a₄₇⋅b₀₀  a₄₆⋅b₀₁  a₄₇⋅b₀₁  a₄₆⋅b₀₂  a₄₇⋅b₀₂⎥
        ⎢                                                                                                                                                                                                                      ⎥
        ⎢a₅₀⋅b₀₀  a₅₁⋅b₀₀  a₅₀⋅b₀₁  a₅₁⋅b₀₁  a₅₀⋅b₀₂  a₅₁⋅b₀₂  a₅₂⋅b₀₀  a₅₃⋅b₀₀  a₅₂⋅b₀₁  a₅₃⋅b₀₁  a₅₂⋅b₀₂  a₅₃⋅b₀₂  a₅₄⋅b₀₀  a₅₅⋅b₀₀  a₅₄⋅b₀₁  a₅₅⋅b₀₁  a₅₄⋅b₀₂  a₅₅⋅b₀₂  a₅₆⋅b₀₀  a₅₇⋅b₀₀  a₅₆⋅b₀₁  a₅₇⋅b₀₁  a₅₆⋅b₀₂  a₅₇⋅b₀₂⎥
        ⎢                                                                                                                                                                                                                      ⎥
        ⎢a₄₀⋅b₁₀  a₄₁⋅b₁₀  a₄₀⋅b₁₁  a₄₁⋅b₁₁  a₄₀⋅b₁₂  a₄₁⋅b₁₂  a₄₂⋅b₁₀  a₄₃⋅b₁₀  a₄₂⋅b₁₁  a₄₃⋅b₁₁  a₄₂⋅b₁₂  a₄₃⋅b₁₂  a₄₄⋅b₁₀  a₄₅⋅b₁₀  a₄₄⋅b₁₁  a₄₅⋅b₁₁  a₄₄⋅b₁₂  a₄₅⋅b₁₂  a₄₆⋅b₁₀  a₄₇⋅b₁₀  a₄₆⋅b₁₁  a₄₇⋅b₁₁  a₄₆⋅b₁₂  a₄₇⋅b₁₂⎥
        ⎢                                                                                                                                                                                                                      ⎥
        ⎢a₅₀⋅b₁₀  a₅₁⋅b₁₀  a₅₀⋅b₁₁  a₅₁⋅b₁₁  a₅₀⋅b₁₂  a₅₁⋅b₁₂  a₅₂⋅b₁₀  a₅₃⋅b₁₀  a₅₂⋅b₁₁  a₅₃⋅b₁₁  a₅₂⋅b₁₂  a₅₃⋅b₁₂  a₅₄⋅b₁₀  a₅₅⋅b₁₀  a₅₄⋅b₁₁  a₅₅⋅b₁₁  a₅₄⋅b₁₂  a₅₅⋅b₁₂  a₅₆⋅b₁₀  a₅₇⋅b₁₀  a₅₆⋅b₁₁  a₅₇⋅b₁₁  a₅₆⋅b₁₂  a₅₇⋅b₁₂⎥
        ⎢                                                                                                                                                                                                                      ⎥
        ⎢a₄₀⋅b₂₀  a₄₁⋅b₂₀  a₄₀⋅b₂₁  a₄₁⋅b₂₁  a₄₀⋅b₂₂  a₄₁⋅b₂₂  a₄₂⋅b₂₀  a₄₃⋅b₂₀  a₄₂⋅b₂₁  a₄₃⋅b₂₁  a₄₂⋅b₂₂  a₄₃⋅b₂₂  a₄₄⋅b₂₀  a₄₅⋅b₂₀  a₄₄⋅b₂₁  a₄₅⋅b₂₁  a₄₄⋅b₂₂  a₄₅⋅b₂₂  a₄₆⋅b₂₀  a₄₇⋅b₂₀  a₄₆⋅b₂₁  a₄₇⋅b₂₁  a₄₆⋅b₂₂  a₄₇⋅b₂₂⎥
        ⎢                                                                                                                                                                                                                      ⎥
        ⎢a₅₀⋅b₂₀  a₅₁⋅b₂₀  a₅₀⋅b₂₁  a₅₁⋅b₂₁  a₅₀⋅b₂₂  a₅₁⋅b₂₂  a₅₂⋅b₂₀  a₅₃⋅b₂₀  a₅₂⋅b₂₁  a₅₃⋅b₂₁  a₅₂⋅b₂₂  a₅₃⋅b₂₂  a₅₄⋅b₂₀  a₅₅⋅b₂₀  a₅₄⋅b₂₁  a₅₅⋅b₂₁  a₅₄⋅b₂₂  a₅₅⋅b₂₂  a₅₆⋅b₂₀  a₅₇⋅b₂₀  a₅₆⋅b₂₁  a₅₇⋅b₂₁  a₅₆⋅b₂₂  a₅₇⋅b₂₂⎥
        ⎢                                                                                                                                                                                                                      ⎥
        ⎢a₆₀⋅b₀₀  a₆₁⋅b₀₀  a₆₀⋅b₀₁  a₆₁⋅b₀₁  a₆₀⋅b₀₂  a₆₁⋅b₀₂  a₆₂⋅b₀₀  a₆₃⋅b₀₀  a₆₂⋅b₀₁  a₆₃⋅b₀₁  a₆₂⋅b₀₂  a₆₃⋅b₀₂  a₆₄⋅b₀₀  a₆₅⋅b₀₀  a₆₄⋅b₀₁  a₆₅⋅b₀₁  a₆₄⋅b₀₂  a₆₅⋅b₀₂  a₆₆⋅b₀₀  a₆₇⋅b₀₀  a₆₆⋅b₀₁  a₆₇⋅b₀₁  a₆₆⋅b₀₂  a₆₇⋅b₀₂⎥
        ⎢                                                                                                                                                                                                                      ⎥
        ⎢a₇₀⋅b₀₀  a₇₁⋅b₀₀  a₇₀⋅b₀₁  a₇₁⋅b₀₁  a₇₀⋅b₀₂  a₇₁⋅b₀₂  a₇₂⋅b₀₀  a₇₃⋅b₀₀  a₇₂⋅b₀₁  a₇₃⋅b₀₁  a₇₂⋅b₀₂  a₇₃⋅b₀₂  a₇₄⋅b₀₀  a₇₅⋅b₀₀  a₇₄⋅b₀₁  a₇₅⋅b₀₁  a₇₄⋅b₀₂  a₇₅⋅b₀₂  a₇₆⋅b₀₀  a₇₇⋅b₀₀  a₇₆⋅b₀₁  a₇₇⋅b₀₁  a₇₆⋅b₀₂  a₇₇⋅b₀₂⎥
        ⎢                                                                                                                                                                                                                      ⎥
        ⎢a₆₀⋅b₁₀  a₆₁⋅b₁₀  a₆₀⋅b₁₁  a₆₁⋅b₁₁  a₆₀⋅b₁₂  a₆₁⋅b₁₂  a₆₂⋅b₁₀  a₆₃⋅b₁₀  a₆₂⋅b₁₁  a₆₃⋅b₁₁  a₆₂⋅b₁₂  a₆₃⋅b₁₂  a₆₄⋅b₁₀  a₆₅⋅b₁₀  a₆₄⋅b₁₁  a₆₅⋅b₁₁  a₆₄⋅b₁₂  a₆₅⋅b₁₂  a₆₆⋅b₁₀  a₆₇⋅b₁₀  a₆₆⋅b₁₁  a₆₇⋅b₁₁  a₆₆⋅b₁₂  a₆₇⋅b₁₂⎥
        ⎢                                                                                                                                                                                                                      ⎥
        ⎢a₇₀⋅b₁₀  a₇₁⋅b₁₀  a₇₀⋅b₁₁  a₇₁⋅b₁₁  a₇₀⋅b₁₂  a₇₁⋅b₁₂  a₇₂⋅b₁₀  a₇₃⋅b₁₀  a₇₂⋅b₁₁  a₇₃⋅b₁₁  a₇₂⋅b₁₂  a₇₃⋅b₁₂  a₇₄⋅b₁₀  a₇₅⋅b₁₀  a₇₄⋅b₁₁  a₇₅⋅b₁₁  a₇₄⋅b₁₂  a₇₅⋅b₁₂  a₇₆⋅b₁₀  a₇₇⋅b₁₀  a₇₆⋅b₁₁  a₇₇⋅b₁₁  a₇₆⋅b₁₂  a₇₇⋅b₁₂⎥
        ⎢                                                                                                                                                                                                                      ⎥
        ⎢a₆₀⋅b₂₀  a₆₁⋅b₂₀  a₆₀⋅b₂₁  a₆₁⋅b₂₁  a₆₀⋅b₂₂  a₆₁⋅b₂₂  a₆₂⋅b₂₀  a₆₃⋅b₂₀  a₆₂⋅b₂₁  a₆₃⋅b₂₁  a₆₂⋅b₂₂  a₆₃⋅b₂₂  a₆₄⋅b₂₀  a₆₅⋅b₂₀  a₆₄⋅b₂₁  a₆₅⋅b₂₁  a₆₄⋅b₂₂  a₆₅⋅b₂₂  a₆₆⋅b₂₀  a₆₇⋅b₂₀  a₆₆⋅b₂₁  a₆₇⋅b₂₁  a₆₆⋅b₂₂  a₆₇⋅b₂₂⎥
        ⎢                                                                                                                                                                                                                      ⎥
        ⎣a₇₀⋅b₂₀  a₇₁⋅b₂₀  a₇₀⋅b₂₁  a₇₁⋅b₂₁  a₇₀⋅b₂₂  a₇₁⋅b₂₂  a₇₂⋅b₂₀  a₇₃⋅b₂₀  a₇₂⋅b₂₁  a₇₃⋅b₂₁  a₇₂⋅b₂₂  a₇₃⋅b₂₂  a₇₄⋅b₂₀  a₇₅⋅b₂₀  a₇₄⋅b₂₁  a₇₅⋅b₂₁  a₇₄⋅b₂₂  a₇₅⋅b₂₂  a₇₆⋅b₂₀  a₇₇⋅b₂₀  a₇₆⋅b₂₁  a₇₇⋅b₂₁  a₇₆⋅b₂₂  a₇₇⋅b₂₂⎦
    '''
    assert pretty(C, num_columns=10000) == dedent(expected).strip()


def test_inter_product_left_kron():
    # Define symbolic variables
    coms = square_m(5, 'a'), square_m(2, 'b')
    C = kron(*coms)
    # print('C')
    # pprint(C, num_columns=10000)

    E = square_m(3, 'e')
    # print('E')
    # pprint(E)

    # execute
    Z = inter_product(C, E, 10)
    # print('Z', flush=True)
    # pprint(Z, num_columns=10000)
    expected = kron(E, coms[0], coms[1])
    assert Z == expected


def test_inter_product_right_kron():
    # Define symbolic variables
    coms = square_m(5, 'a'), square_m(2, 'b')
    C = kron(*coms)
    # print('C')
    # pprint(C, num_columns=10000)

    E = square_m(3, 'e')
    # print('E')
    # pprint(E)

    # execute
    Z = inter_product(C, E, 1)
    # print('Z', flush=True)
    # pprint(Z, num_columns=10000)

    expected = kron(coms[0], coms[1], E)
    assert Z == expected


def test_inter_product_5_3_2():
    coms = square_m(5, 'a'), square_m(2, 'b')
    C = kron(*coms)
    # print('C')
    # pprint(C, num_columns=10000)

    E = square_m(3, 'e')
    # print('E')
    # pprint(E)

    # execute
    Z = inter_product(C, E, 2)

    # print('Z', flush=True)
    # pprint(Z, num_columns=10000)
    expected = kron(coms[0], E, coms[1])
    assert Z == expected


def test_inter_product_2_3_4():
    coms = square_m(2, 'a'), square_m(2, 'b'), square_m(2, 'c')
    A = kron(*coms)
    # print('A')
    # pprint(A, num_columns=10000)

    E = square_m(3, 'e')
    # print('E')
    # pprint(E)

    # execute
    Z = mesh_product(A, (E,), (4,))
    # print('Z', flush=True)
    # pprint(Z, num_columns=10000)

    expected = kron(coms[0], E, coms[1], coms[2])
    # print('expected')
    # pprint(expected, num_columns=10000)

    assert Z == expected


def test_inter_product_4_3_2():
    coms = square_m(2, 'a'), square_m(2, 'b'), square_m(2, 'c')
    A = kron(*coms)
    # print('A')
    # pprint(A, num_columns=10000)

    E = square_m(3, 'e')
    # print('E')
    # pprint(E)

    # execute
    Z = mesh_product(A, (E,), (2,))
    # print('Z', flush=True)
    # pprint(Z, num_columns=10000)

    expected = kron(coms[0], coms[1], E, coms[2])
    # print('expected')
    # pprint(expected, num_columns=10000)

    assert Z == expected


def test_mesh_product_16_3_2_3_2():
    coms = square_m(2, 'a'), square_m(2, 'b'), square_m(2, 'c')
    A = kron(*coms)
    # print('\nA')
    # pprint(A, num_columns=10000)

    E = square_m(2, 'e')
    # print('\nE')
    # pprint(E)

    F = square_m(2, 'f')
    # print('\nF')
    # pprint(F, num_columns=10000)

    # execute
    Z = mesh_product(A, (E, F), (4, 2))
    # print('\nZ', flush=True)
    # pprint(Z, num_columns=10000)

    expected = kron(coms[0], E, coms[1], F, coms[2])
    # print('\nexpected')
    # pprint(expected, num_columns=10000)

    assert Z == expected


def mesh_product(bread, yeast, factors):
    """
    This method is a shorthand for inter_product(inter_product(... inter_product(A, seeds[0]), seeds[1]), ...)
    and carries out inter_product recursively.
    :param bread: square matrix of shape (m, m).
    :param yeast: list of square matrices of shape S(k1, k1), S(k2, k2), etc.
    :param factors: list of int, (f1,f2,...) denotes the sizes of blocks to divide the subsequent resultant matrix into.
           bed will be divided recursively by the factors into blocks.
           seeds matrices will be mesh multiplied in Kronecker fashion.
           The size of factors == size of seeds and also m must be divisible by product(f0, f1, ...).
    :return: The inter product with A(f0) ⨁ S1 ⨁ A(f1) ⨁ ... ⨁ Sn ⨁ A(fn).
    """
    for s, f in zip(yeast, factors):
        bread = inter_product(bread, s, f)
        # print(f'\nbread {s[0, 0]}', flush=True)
        # pprint(bread, num_columns=10000)
    return bread


def inter_product(A, B, m):
    """
    A matrix multiplication operation of special Tracy–Singh product type.
    https://en.wikipedia.org/wiki/Kronecker_product#Tracy%E2%80%93Singh_product
    Matrix A is first divided up into n x n number of m x m sized blocks;
    Then B is mesh multiplied between the n x n number of m x m sized blocks.
    For example,
        A =
        ⎡a₀₀  a₀₁  a₀₂  a₀₃⎤
        ⎢                  ⎥
        ⎢a₁₀  a₁₁  a₁₂  a₁₃⎥
        ⎢                  ⎥
        ⎢a₂₀  a₂₁  a₂₂  a₂₃⎥
        ⎢                  ⎥
        ⎣a₃₀  a₃₁  a₃₂  a₃₃⎦
        B =
        ⎡b₀₀  b₀₁⎤
        ⎢        ⎥
        ⎣b₁₀  b₁₁⎦
        inter_product(A, B) =
        ⎡a₀₀⋅b₀₀  a₀₁⋅b₀₀  a₀₀⋅b₀₁  a₀₁⋅b₀₁  a₀₂⋅b₀₀  a₀₃⋅b₀₀  a₀₂⋅b₀₁  a₀₃⋅b₀₁⎤
        ⎢                                                                      ⎥
        ⎢a₁₀⋅b₀₀  a₁₁⋅b₀₀  a₁₀⋅b₀₁  a₁₁⋅b₀₁  a₁₂⋅b₀₀  a₁₃⋅b₀₀  a₁₂⋅b₀₁  a₁₃⋅b₀₁⎥
        ⎢                                                                      ⎥
        ⎢a₀₀⋅b₁₀  a₀₁⋅b₁₀  a₀₀⋅b₁₁  a₀₁⋅b₁₁  a₀₂⋅b₁₀  a₀₃⋅b₁₀  a₀₂⋅b₁₁  a₀₃⋅b₁₁⎥
        ⎢                                                                      ⎥
        ⎢a₁₀⋅b₁₀  a₁₁⋅b₁₀  a₁₀⋅b₁₁  a₁₁⋅b₁₁  a₁₂⋅b₁₀  a₁₃⋅b₁₀  a₁₂⋅b₁₁  a₁₃⋅b₁₁⎥
        ⎢                                                                      ⎥
        ⎢a₂₀⋅b₀₀  a₂₁⋅b₀₀  a₂₀⋅b₀₁  a₂₁⋅b₀₁  a₂₂⋅b₀₀  a₂₃⋅b₀₀  a₂₂⋅b₀₁  a₂₃⋅b₀₁⎥
        ⎢                                                                      ⎥
        ⎢a₃₀⋅b₀₀  a₃₁⋅b₀₀  a₃₀⋅b₀₁  a₃₁⋅b₀₁  a₃₂⋅b₀₀  a₃₃⋅b₀₀  a₃₂⋅b₀₁  a₃₃⋅b₀₁⎥
        ⎢                                                                      ⎥
        ⎢a₂₀⋅b₁₀  a₂₁⋅b₁₀  a₂₀⋅b₁₁  a₂₁⋅b₁₁  a₂₂⋅b₁₀  a₂₃⋅b₁₀  a₂₂⋅b₁₁  a₂₃⋅b₁₁⎥
        ⎢                                                                      ⎥
        ⎣a₃₀⋅b₁₀  a₃₁⋅b₁₀  a₃₀⋅b₁₁  a₃₁⋅b₁₁  a₃₂⋅b₁₀  a₃₃⋅b₁₀  a₃₂⋅b₁₁  a₃₃⋅b₁₁⎦
    :param A: square matrix of shape (N, N).
    :param B: square matrix of shape (k, k), with k > 0.
    :param m: an integer factor of N, denotes the size of blocks to divide the matrix A into.
    :return: The inter product with A(n) ⨁ B ⨁ A(m) with N = n*m
    """
    assert len(A.shape) == 2 and A.shape[0] == A.shape[1]
    N = A.shape[0]
    assert len(B.shape) == 2 and B.shape[0] == B.shape[1]
    n = B.shape[0]

    # no change to A
    if n == 1:
        return A * B[0, 0]

    assert N % m == 0
    if m == 1:
        return kron(A, B)

    if N == m:
        return kron(B, A)

    C = sympy.zeros(N * n)
    for i, j in product(range(0, N, m), range(0, N, m)):
        for k, l in product(range(n), range(n)):
            C[n * i + k * m:n * i + (k + 1) * m, n * j + l * m:n * j + (l + 1) * m] = A[i:i + m, j:j + m] * B[k, l]
    return C


def square_m(n, prefix: str = None):
    if prefix:
        syms = symbols(f'{prefix}:{n}(:{n})')
    else:
        syms = symbols(f':{n}(:{n})')
    A = Matrix(n, n, syms, complex=True)
    return A
