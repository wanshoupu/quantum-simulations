from quompiler.circuits.cirq_circuit import CirqBuilder
from quompiler.construct.cmat import UnitaryM, UnivGate, CUnitary
from quompiler.utils.format_matrix import MatrixFormatter
from quompiler.utils.mgen import random_UnitaryM_2l


def test_create_builder():
    n = 3
    dim = 1 << n
    array = random_UnitaryM_2l(dim, 3, 4)
    formatter = MatrixFormatter(precision=2)
    cirqC = CirqBuilder(n)
    cirqC.build_gate(array)
    phase = CUnitary(UnivGate.S.mat, (None, False, True))
    print()
    print(formatter.tostr(phase.inflate()))
    cirqC.build_gate(phase)
    circuit = cirqC.finish()
    print(circuit.all_qubits())
    print(circuit)
    u = circuit.unitary(circuit.all_qubits())
    print(formatter.tostr(u))
