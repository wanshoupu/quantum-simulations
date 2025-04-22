import random
import textwrap

import sympy
from sympy import Matrix, symbols, kronecker_product as kron, pretty

from quompiler.construct.cmat import QubitClass

random.seed(3)


def random_control2(n) -> tuple[QubitClass, ...]:
    """
    Generate a random control sequence with total n qubits, k target qubits, (n-k) control qubits
    :param n: positive integer
    :param k: 0< k <= n
    :return: Control sequence
    """
    mid = [q.id for q in QubitClass]
    result = [QubitClass.get(random.choice(mid)) for _ in range(n)]
    return tuple(result)




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
