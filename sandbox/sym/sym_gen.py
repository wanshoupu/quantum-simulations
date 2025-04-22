import string
from itertools import product

from sympy import symbols, Matrix

alphabeta = list(string.ascii_lowercase)
alphamax = len(alphabeta)


def square_m(n, prefix: str = None):
    prefix = prefix or 'a'

    if n * n <= alphamax:
        return square_26(n)
    if n <= alphamax:
        return square_26_26(n)
    syms = symbols(f'{prefix}:{n}(:{n})')
    return Matrix(n, n, syms[:n * n], complex=True)


def square_26(n):
    assert n * n <= alphamax
    syms = symbols(' '.join(alphabeta))
    return Matrix(n, n, syms[:n * n], complex=True)


def square_26_26(n):
    assert n <= alphamax
    names = [f"{s}{t}" for s, t in product(alphabeta, repeat=2)]
    syms = symbols(names)
    return Matrix(n, n, syms[:n * n], complex=True)
