import random

from numpy import kron

from quompiler.construct.types import UnivGate
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.inter_product import inter_product, mykron
from quompiler.utils.mgen import random_unitary, random_su2

formatter = MatrixFormatter(precision=2)


def test_inter_product_reshaping():
    a = (0.3342544441814146 - 0.942482873343051j)
    b = (0.40632805094123764 + 0.9137272651170562j)
    c = a / b
    d = a + b
    e = a - b
    f = a * b
    print()
    print(c)
    print(d)
    print(e)
    print(f)
