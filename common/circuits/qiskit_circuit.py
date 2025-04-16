from typing_extensions import override

from common.construct.cmat import UnitaryM, validm2l

from common.circuits.circuit_builder import CircuitBuilder
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate


class QiskitBuilder(CircuitBuilder):
    def __init__(self, dimension: int):
        self.qubits = []
        self.circuit = qiskit.QuantumCircuit()

    @override
    def get_unigate(self, m: UnitaryM) -> object:
        pass

    @override
    def build_gate(self, m: UnitaryM):
        if not validm2l(m.matrix):
            custom_gate = UnitaryGate(m.matrix)
            self.circuit.append(custom_gate, self.qubits)

        # if isinstance(m, CUnitary):

    @override
    def finish(self) -> QuantumCircuit:
        pass
