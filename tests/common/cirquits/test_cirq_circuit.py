from quompiler.circuits.cirq_circuit import CirqBuilder
from quompiler.construct.cmat import UnitaryM, UnivGate, CUnitary
from quompiler.utils.mgen import random_UnitaryM_2l


def test_create_builder():
    n = 3
    dim = 1 << n
    array = random_UnitaryM_2l(dim, 3, 4)

    cirqC = CirqBuilder(n)
    cirqC.build_gate(array)
    phase = CUnitary(UnivGate.S.mat, (None, False, True))
    cirqC.build_gate(phase)
    circuit = cirqC.finish()
    print(circuit)
