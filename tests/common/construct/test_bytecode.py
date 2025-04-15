from common.construct.bytecode import Bytecode
from common.utils.mgen import random_UnitaryM_2l


def test_init():
    m = random_UnitaryM_2l(3, 0, 1)
    bc = Bytecode(m)
    print(bc)


def test_init_with_children():
    m = random_UnitaryM_2l(4, 0, 1)
    children = [Bytecode(random_UnitaryM_2l(4, 0, 1)), random_UnitaryM_2l(4, 0, 1)]
    bc = Bytecode(m, children)
    print(bc)
