from common.construct.qiskit_circuit import QiskitBuilder
from common.utils.mgen import random_UnitaryM_2l


def test_create_builder():
    n = 3
    dim = 1 << n
    array = random_UnitaryM_2l(dim, 3, 4)

    qiskitC = QiskitBuilder(n)
    qiskitC.build_gate(array)
