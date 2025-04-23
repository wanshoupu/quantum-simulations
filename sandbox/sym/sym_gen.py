import string
from itertools import product

from sympy import symbols, Matrix

alphabeta = list(string.ascii_lowercase)
alphamax = len(alphabeta)


def symmat(n, prefix: str = None):
    """
    Create a square symbolic matrix of dimension n x n.
    When n*n <= 26, single Latin alphabets will be used for symbols.
    When n <= 26, double Latin alphabets will be used for symbols.
    Otherwise, symbols of the format '{prefix}:m(:n)', such as, 'a₀₁', 'b₁₁', 'c₁₂', will be used.
    :param n: dimension
    :param prefix: optional prefix of the symbols used. When n*n <= 26, prefix is not used because single letter will be automatically used.
    :return: a matrix with symbols.
    """
    prefix = prefix or 'a'

    if n * n <= alphamax:
        return symmat_26(n)
    if n <= alphamax:
        return symmat_26_26(n)
    return symmat_sub(n, prefix)


def symmat_26(n):
    assert n * n <= alphamax
    syms = symbols(' '.join(alphabeta))
    return Matrix(n, n, syms[:n * n], complex=True)


def symmat_26_26(n):
    assert n <= alphamax
    names = [f"{s}{t}" for s, t in product(alphabeta, repeat=2)]
    syms = symbols(names)
    return Matrix(n, n, syms[:n * n], complex=True)


def symmat_sub(n, prefix):
    syms = symbols(f'{prefix}:{n}(:{n})')
    return Matrix(n, n, syms[:n * n], complex=True)
