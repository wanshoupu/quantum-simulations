import numpy as np
import sympy
from sympy import pprint

from sandbox.sym.inter_product import inter_product
from sandbox.sym.sym_gen import symmat
from sandbox.sym.symmat_format import mat_print


def test_kron_equivalence():
    m, n = 2, 3
    A = symmat(m * n)
    pprint(A)
    tensor = sympy.reshape(A, (m, n, m, n))
    pprint(tensor)


def test_nested_inter_product():
    """
    In general, there are nested inter_products as shown in this example.
    But in the special case of yeasts being identity matrices, nesting may be treated on the first level only.
    """
    a, b, c = symmat(4, 'a'), symmat(6, 'b'), symmat(2, 'c')
    m = inter_product(b, c, 3)
    nested = inter_product(a, m, 2)
    mat_print(nested)


def test_reshape_higher_dim_tensor():
    """
    Understand the reshape and transpose of numpy.
    The rightmost dimension is the 'most elementary' one, meaning, they are close to the building blocks - numbers.
    Moving left we ascend in the hierarchy of composition and the leftmost dimension is the highest dimension.
    So in analogy: the dimensions are like [library, floor, isle, shelf, book, page, row, word]
    """
    a, b, c, d = 2, 3, 2, 3
    m = np.arange(a * b * c * d).reshape((a * b, c * d))
    print(f'\nshape{a * b, c * d}')
    print(m)
    m2 = np.reshape(m, (a, b, c, d))
    print(f'shape{a, b, c, d}')
    print(m2)
    m3 = np.transpose(m2, [0, 2, 1, 3])
    print(f'transpose {0, 2, 1, 3}')
    print(m3)
    m4 = np.reshape(m3, [a * a, b * b])
    print(f'shape {a * a, b * b}')
    print(m4)
