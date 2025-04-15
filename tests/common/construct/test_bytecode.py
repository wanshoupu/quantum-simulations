from common.construct.bytecode import Bytecode
from tests.common.utils.mgen import random_UnitaryM_2l


def test_init():
    m = random_UnitaryM_2l(3, 0, 1)
    bc = Bytecode(m)
    print(bc)
