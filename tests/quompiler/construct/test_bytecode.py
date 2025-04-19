from quompiler.construct.bytecode import Bytecode, BytecodeIter, ReverseBytecodeIter
from quompiler.utils.mgen import random_UnitaryM_2l


def test_init():
    m = random_UnitaryM_2l(3, 0, 1)
    bc = Bytecode(m)
    print(bc)


def test_init_with_children():
    rm = lambda: random_UnitaryM_2l(4, 0, 1)
    children = [Bytecode(rm()), Bytecode(rm())]
    bc = Bytecode(rm(), children)
    assert bc is not None


def test_iter():
    rm = lambda: random_UnitaryM_2l(4, 0, 1)
    children = [Bytecode(rm()), Bytecode(rm())]
    bc = Bytecode(rm(), children)
    nodes = [n for n in BytecodeIter(bc) if not n.children]
    assert all(a == b for a, b in zip(nodes, children))


def test_reverse_iter():
    rm = lambda: random_UnitaryM_2l(4, 0, 1)
    children = [Bytecode(rm()), Bytecode(rm())]
    bc = Bytecode(rm(), children)
    nodes = [n for n in ReverseBytecodeIter(bc) if not n.children]
    assert all(a == b for a, b in zip(nodes, children[::-1]))
