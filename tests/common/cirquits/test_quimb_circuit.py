from quompiler.circuits.quimb_circuit import QuimbBuilder
from quompiler.utils.mgen import random_UnitaryM_2l


def test_create_builder():
    n = 3
    dim = 1 << n
    array = random_UnitaryM_2l(dim, 3, 4)

    quimbC = QuimbBuilder(n)
    quimbC.build_gate(array)
