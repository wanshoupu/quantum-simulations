from functools import reduce

import numpy as np

from quompiler.construct.bytecode import Bytecode, BytecodeIter, BytecodeRevIter
from quompiler.utils.mgen import random_UnitaryM_2l, random_unitary

randu = lambda: random_UnitaryM_2l(4, 0, 1)
make_bc = lambda children: Bytecode(reduce(lambda a, b: a @ b, [c.data for c in children]), children=children)


def test_init():
    m = random_UnitaryM_2l(3, 0, 1)
    bc = Bytecode(m)
    print(bc)


def test_init_with_children():
    children = [Bytecode(randu()), Bytecode(randu())]
    bc = Bytecode(randu(), children)
    assert bc is not None


def test_iter():
    children = [Bytecode(randu()), Bytecode(randu())]
    bc = Bytecode(randu(), children)
    nodes = [n for n in BytecodeIter(bc) if n.is_leaf()]
    assert all(a == b for a, b in zip(nodes, children))


def test_reverse_iter():
    children = [Bytecode(randu()), Bytecode(randu())]
    bc = Bytecode(randu(), children)
    nodes = [n for n in BytecodeRevIter(bc) if n.is_leaf()]
    assert all(a == b for a, b in zip(nodes, children[::-1]))


def test_herm_verify_identity():
    nodes = [make_bc([Bytecode(random_unitary(2)) for _ in range(3)]) for _ in range(4)]
    root = make_bc(nodes)
    root_herm = root.herm()
    leaves = [n.data for n in BytecodeRevIter(root) if n.is_leaf()]
    leaves_herm = [n.data for n in BytecodeRevIter(root_herm) if n.is_leaf()]
    prod = reduce(lambda a, b: a @ b, leaves)
    prod_herm = reduce(lambda a, b: a @ b, leaves_herm)
    assert np.allclose(prod @ prod_herm, np.eye(2))
