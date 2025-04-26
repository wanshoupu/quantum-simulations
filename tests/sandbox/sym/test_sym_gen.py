import re

import pytest
from sympy import pprint

from sandbox.sym.sym_gen import symmat, alphabeta


@pytest.mark.parametrize("n", [2, 3, 5])
def test_syms_le_26(n):
    m = symmat(n)
    # print(f'\n{n} x {n}')
    syms = m.free_symbols
    assert set(s.name for s in syms) <= set(alphabeta)


@pytest.mark.parametrize("n", [2, 3, 5])
def test_syms_le_26_prefix(n):
    prefix = 'b'
    m = symmat(n, prefix)
    # print(f'\n{n} x {n}')
    syms = m.free_symbols
    assert all(re.fullmatch(f'{prefix}\d+', s.name) for s in syms)


@pytest.mark.parametrize("n", [6, ])
def test_syms_le_26_26(n):
    m = symmat(n)
    # print(f'\n{n} x {n}')
    # pprint(m)
    syms = m.free_symbols
    assert all(set(s.name) <= set(alphabeta) for s in syms)


@pytest.mark.parametrize("n", [27])
def test_syms_ge_26_26(n):
    prefix = 'a'
    m = symmat(n, prefix)
    # print(f'\n{n} x {n}')
    # pprint(m)
    syms = m.free_symbols
    assert all(s.name[len(prefix):].isdigit() for s in syms)
